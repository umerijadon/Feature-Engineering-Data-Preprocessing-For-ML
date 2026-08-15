# Feature-Engineering-Data-Preprocessing-For-ML

**Data Science Internship Programme | Week 2**
**AnalystLab Africa Consulting**

## Project Overview

For Week 2 of my internship, I worked on the preprocessing stage of a heart disease classification problem.

The goal was not to build a prediction model yet. Instead, I focused on taking a raw dataset and getting it into a condition where it could be used for machine learning without carrying obvious data quality problems into the next stage.

I used the Heart Disease Prediction dataset from Kaggle by fedesoriano as a stand-in for employee health screening data at a fictional manufacturing company, ABC Manufacturing Ltd.

The main areas I worked on were data quality checks, missing-value handling, feature engineering, categorical encoding, scaling, outlier treatment, and feature selection.

## Business Context

The project was framed around an employee wellness screening programme.

The idea was to prepare health-related employee data for a future model that could help identify patterns associated with heart disease risk. The model would be a screening aid, not a replacement for medical diagnosis.

The preprocessing work was guided by a few questions:

* Which variables are useful for the prediction problem?
* Which variables need to be encoded?
* Which numerical variables need to be scaled?
* Are any features redundant?
* Are there missing values that are not being detected normally?
* How should unusual values and outliers be handled?
* Is the resulting dataset suitable for the next stage of model development?

## Dataset

**Source:** [Heart Disease Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)

The dataset contains:

* **918 records**
* **12 original attributes**
* **Target:** `HeartDisease`

  * `1` = heart disease
  * `0` = no heart disease
* **508 positive cases (55.3%)**
* **410 negative cases (44.7%)**

The target was reasonably balanced, so there was no major class imbalance to address during preprocessing.

### The data quality issue I found

One of the more useful findings came from simply looking at the descriptive statistics.

`df.isnull().sum()` showed **zero missing values** across the dataset. At first, this suggested that there were no missing observations.

However, `df.describe()` showed that:

* **172 patients (18.7%) had `Cholesterol = 0`**
* **1 patient had `RestingBP = 0`**

A cholesterol reading of zero is not a plausible measurement in this dataset, so treating those values as valid would have introduced bad information into the model.

I converted those zeros to `NaN` and handled them as missing values.

This was a good reminder that checking for `NaN` values alone is not enough. Some missing data can be hidden behind placeholder values.

## Preprocessing Workflow

### 1. Data inspection

I started by checking:

* Dataset dimensions
* Data types
* Missing values
* Duplicate records
* Descriptive statistics
* Minimum and maximum values
* Category distributions

This was where the zero-value issue in `Cholesterol` and `RestingBP` was identified.

### 2. Missing-value treatment

After identifying the invalid zero values, I replaced them with `NaN`.

For imputation:

* `Cholesterol` was median-imputed within `ChestPainType` groups.
* `RestingBP` was imputed using the overall median.

Using the median helped avoid letting extreme values have too much influence on the replacement values.

### 3. Feature engineering

I created two additional features:

**`AgeGroup`**

Age was grouped into categories to make it easier to examine differences between age ranges.

**`RiskFactorCount`**

A simple count of selected risk-related indicators, producing a score from 0 to 4.

I also renamed some of the original columns to make the dataset easier to read:

| Original    | New name                |
| ----------- | ----------------------- |
| `Oldpeak`   | `ST_Depression`         |
| `FastingBS` | `FastingBloodSugarHigh` |
| `MaxHR`     | `MaxHeartRate`          |

### 4. Categorical encoding

The categorical variables were converted into numerical representations for machine learning.

For binary variables such as:

* `Sex`
* `ExerciseAngina`

I used binary encoding.

For variables with multiple categories, I used one-hot encoding with `drop_first=True`.

These included:

* `ChestPainType`
* `RestingECG`
* `ST_Slope`
* `AgeGroup`

### 5. Scaling

`StandardScaler` was applied to the continuous numerical variables.

I did not scale:

* Binary indicators
* The target variable

This keeps the binary features interpretable while putting the continuous variables on comparable scales.

### 6. Outlier treatment

I used the IQR method to identify potential outliers.

I chose this approach instead of relying on z-scores because some of the clinical variables were skewed.

The main variables where high-end values were capped were:

* `RestingBP`
* `Cholesterol`
* `ST_Depression`

I chose capping instead of deleting the observations because the dataset only contains 918 records, and removing valid patient records simply because they contained extreme values could unnecessarily reduce the dataset.

### 7. Feature selection and redundancy checks

I looked at both:

* Correlation between features and the target
* Random Forest feature importance

I also checked for highly correlated feature pairs.

No major redundancy was found beyond relationships that were expected because of feature construction, particularly:

* `Age` and `AgeGroup_60+`
* `ST_Slope_Flat` and `ST_Slope_Up`

These relationships are worth being aware of when building the eventual model.

## Key Findings

### Hidden missing values were the biggest data quality issue

The most obvious issue wasn't detected by `isnull()`.

**18.7% of the dataset had `Cholesterol = 0`.**

Those records would have been treated as real measurements if I had relied only on a standard missing-value check.

### The strongest relationships with the target

The features with the strongest target correlations included:

| Feature           | Correlation |
| ----------------- | ----------: |
| `ST_Slope_Up`     |       -0.62 |
| `ST_Slope_Flat`   |       +0.55 |
| `ExerciseAngina`  |       +0.49 |
| `RiskFactorCount` |       +0.49 |
| `ST_Depression`   |       +0.41 |

Correlation was used as an exploratory measure, not as proof that any of these variables independently cause heart disease.

### Random Forest feature importance

The Random Forest analysis ranked these among the most important features:

1. `ST_Slope_Up`
2. `ST_Slope_Flat`
3. `ST_Depression`
4. `MaxHeartRate`
5. `ExerciseAngina`

The fact that several of these also appeared strongly in the correlation analysis gave me a useful cross-check between the two approaches.

### Chest pain patterns

`ASY` (asymptomatic) had the highest observed heart disease rate among the chest pain categories in this dataset.

This was one of the more interesting patterns I found during the exploratory analysis and is something I would investigate further rather than treating it as a standalone medical conclusion.

## Output Files

### `heart_cleaned.csv`

**918 × 12**

A cleaned and human-readable version of the original dataset.

It contains:

* Corrected missing values
* Renamed columns
* Original categorical values
* Unscaled numerical values

### `heart_ml_ready.csv`

**918 × 20**

The machine-learning-oriented version of the dataset after:

* Feature engineering
* Categorical encoding
* Scaling
* Outlier treatment

This dataset is structured for the next stage of model development. For proper model evaluation, preprocessing parameters should be fitted on the training data and then applied to validation/test data rather than fitting them on the complete dataset first.

## Tools Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Jupyter Notebook

## What I Took Away From the Project

The biggest lesson from this project was that preprocessing is not just about converting strings to numbers and filling in `NaN` values.

A dataset can report zero missing values and still contain missing or invalid information.

In this case, simply looking at the minimum values in the descriptive statistics revealed a problem that a standard null check missed.

I also got more practice making preprocessing decisions based on the characteristics of the data rather than applying the same technique to every variable.

The next step would be to use the prepared data to build and evaluate classification models, while making sure that the preprocessing pipeline is fitted only on the training data to avoid leakage.

## Author

**Jadon Umeri**
Data Science Intern, AnalystLab Africa Consulting.
