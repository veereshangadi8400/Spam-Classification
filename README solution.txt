# Email Spam Classifier using Machine Learning

## 1. Project Overview

This project builds a machine learning classifier that identifies whether an email is **Spam** or **Ham (legitimate email)** using Apache SpamAssassin's public email dataset.

The objective is to develop and evaluate multiple classification models, compare their performance using precision, recall, and F1-score, and tune the selected model to improve spam detection.

**Final model:** Logistic Regression with hyperparameter tuning using GridSearchCV.

**Best tuned recall:** 99.04%

## 2. Problem Statement

Spam emails can contain unwanted advertisements, fraudulent offers, phishing attempts, and other potentially harmful content. Manually identifying spam is inefficient, so an automated classification system can help distinguish spam from legitimate emails.

The challenge is to build a model that detects spam reliably while minimizing the number of legitimate emails incorrectly classified as spam.

## 3. Dataset

The project uses the public email datasets provided by Apache SpamAssassin.

| Dataset | Number of emails |
|---|---:|
| Easy Ham | 2,501 |
| Spam | 501 |
| **Total** | **3,002** |

The dataset contains raw email messages, including headers and message bodies. Some emails also contain HTML content.

### Target labels

- `0` — Ham (legitimate email)
- `1` — Spam

The raw email contents are stored as input features, and the corresponding binary labels are stored separately.

## 4. Project Workflow

1. Download and extract the SpamAssassin datasets.
2. Inspect raw email messages and understand their structure.
3. Load email contents and assign binary labels.
4. Split the dataset into training and testing sets.
5. Convert email text into sparse numerical feature vectors.
6. Train and compare multiple classification algorithms.
7. Evaluate model performance using precision, recall, and F1-score.
8. Tune Logistic Regression using GridSearchCV.
9. Evaluate the tuned model on the test set.

## 5. Train/Test Split

The dataset was divided using Scikit-learn's `train_test_split()` with a test size of 20% and `random_state=42`.

| Split | Number of emails |
|---|---:|
| Training set | 2,401 |
| Testing set | 601 |
| Total | 3,002 |

The training set is used to learn patterns from the email data, while the test set is reserved for evaluating performance on unseen examples.

## 6. Text Feature Extraction

Machine learning classifiers require numerical input, so the email text was converted into numerical word-count features using Scikit-learn's `CountVectorizer`.

The vectorizer builds a vocabulary from the training emails and represents each email using counts of vocabulary words.

The vectorization process produced:

- Training matrix shape: `(2401, 78927)`
- Testing matrix shape: `(601, 78927)`

Each row represents an email, and each column represents a vocabulary term. The matrices are sparse because most emails contain only a small fraction of the vocabulary.

**Leakage prevention:** The vectorizer was fitted on the training data and then used to transform the test data. The test set was not used to learn the vocabulary.

## 7. Models Evaluated

Multiple classification algorithms were experimented with:

- Multinomial Naive Bayes
- Logistic Regression
- Support Vector Classifier (SVC)
- Random Forest Classifier

These models were compared to understand how different classification approaches performed on the email dataset.

Logistic Regression was selected for further tuning based on the project's emphasis on spam recall and its observed performance.

## 8. Evaluation Metrics

The following metrics were used:

**Precision**

Measures how many emails predicted as spam were actually spam.

`Precision = TP / (TP + FP)`

Higher precision means fewer legitimate emails are incorrectly flagged as spam.

**Recall**

Measures how many actual spam emails the model successfully detects.

`Recall = TP / (TP + FN)`

Higher recall means fewer spam emails escape detection.

**F1-score**

The harmonic mean of precision and recall. It provides a single measure of the balance between these two metrics.

`F1-score = 2 × (Precision × Recall) / (Precision + Recall)`

Recall was prioritized during hyperparameter tuning because missing actual spam emails was considered more costly than incorrectly flagging some legitimate emails.

## 9. Hyperparameter Tuning

`GridSearchCV` was used to tune the Logistic Regression model.

Configuration:

- Parameter tuned: `C`
- Candidate values: `[0.01, 0.1, 1, 10]`
- Cross-validation folds: `5`
- Scoring metric: `recall`

The best parameter reported was:

`C = 10`

The `C` parameter controls the strength of regularization in Logistic Regression. Smaller values apply stronger regularization, while larger values apply weaker regularization.

GridSearchCV evaluates the candidate settings using cross-validation and selects the setting with the best mean recall.

## 10. Final Results

The following results were obtained from the tuned Logistic Regression model on the test set.

| Metric | Before tuning | After tuning |
|---|---:|---:|
| Precision | ~98% | 97.17% |
| Recall | ~98% | 99.04% |
| F1-score | ~98% | 98.10% |

The tuned model improved recall and slightly improved F1-score, while precision decreased slightly. This represents a trade-off: the model catches more spam but may incorrectly flag a few more legitimate emails.

The final results suggest that the tuned model meets the project's priority of minimizing missed spam.

*Note: The before-tuning values are approximate as recorded during experimentation. Replace them with the exact values printed by your notebook before publishing.*

## 11. Key Learnings

- Working with real-world email data and understanding raw email structure.
- Loading and labeling text datasets using Python.
- Splitting data into training and test sets.
- Converting unstructured text into sparse numerical features.
- Training and comparing different machine learning classifiers.
- Evaluating classification models using precision, recall, and F1-score.
- Using cross-validation and GridSearchCV for hyperparameter tuning.
- Understanding the trade-off between false positives and false negatives.
- Preventing test-set vocabulary leakage during feature extraction.

## 12. Technologies Used

- Python
- Scikit-learn
- NumPy
- Apache SpamAssassin public email dataset
- `CountVectorizer`
- `train_test_split`
- `GridSearchCV`
- Logistic Regression, Multinomial Naive Bayes, SVC, and Random Forest

## 13. Limitations and Future Improvements

- The dataset is relatively small and contains substantially more ham than spam.
- The current text representation uses word counts and may not capture semantic meaning.
- Further experiments could compare `CountVectorizer` with `TfidfVectorizer`.
- HTML content, email headers, URLs, and other email-specific patterns could be processed more deliberately.
- Additional evaluation could include stratified splitting, class-wise metrics, and testing on a separate or newer email dataset.
- A future version could expose the trained model through a simple API or web interface.

## 14. Conclusion

This project demonstrates an end-to-end text classification workflow, from loading real email data to training, evaluating, and tuning machine learning models.

Logistic Regression was selected for further optimization, and GridSearchCV improved test-set recall to 99.04%, with 97.17% precision and a 98.10% F1-score.

The project provided practical experience with text feature extraction, model comparison, evaluation metrics, and hyperparameter optimization.
