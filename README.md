# Text Generation: Story Ending Prediction
## Could T5 model understand the causal relationship and generate reasonable story endings?
### Ming Chen, Fengyao Luo
Introducing a data pipeline with fine-tuned T5 model to predict story endings for inputed stories

Research Paper [NLP Story Ending Prediction](Story_Ending_Prediction.pdf)

Presentation [NLP Story Ending Prediction PPT](https://github.com/xiaowanzio8/NLP-Story-Ending-Prediction-with-T5/blob/main/Final%20Presentation%20.pptx)

## Goal
- Build an end to end data pipeline to predict story endings
- fine tune T5 on 50K+ ROC stories  

## Abstract

![Image of T5 Function Image](https://github.com/xiaowanzio8/NLP-Story-Ending-Prediction-with-T5/blob/main/images/intro.jpg)

Story telling by a machine has fascinated many
science fiction writers. With the development
of technology such as GPT-3, BERT, and T5,
machines can generate reasonable and fluent
sentences with certain guidance. However, it is
still a challenge for machines to understand the
causal relationship between events and understand
the related ideas within sentences.
This research explored the generation function
from the latest Encoder-Decoder Model
T5, and applied 2 different sentence similarity
methods (T5 sentence similarity, Universal
Sentence Encoder) to evaluate the model performance
on the Story Cloze Test. We achieved
the baseline val accuracy as 71.4%. Error analysis
revealed that story ending generation varied
and similarity scores between output and ending
1 or ending 2 are very close. Furthermore,
we trained the model to output 5 endings and
applied a Simi-Senti score (Sentiment consistency
indicator * 1 + similarity score) to the
model, which improved model performance by
6.6%, and reached the final validation accuracy
of 76.1%. We reached a test accuracy of 74.5%
on the leaderboard of Story Cloze Test Winter
2018.


## Data Pipeline

![Image of Data Pipeline](https://github.com/xiaowanzio8/NLP-Story-Ending-Prediction-with-T5/blob/main/images/pipeline.jpg)

## Dataset

![Image of Dataset](https://github.com/xiaowanzio8/NLP-Story-Ending-Prediction-with-T5/blob/main/images/dataset.jpg)

**ROCStories Corpora**
- It is a  new corpus of five-sentence commonsense stories.This corpus icaptures a rich set of causal and temporal common sense relations between daily events, and it is a high quality collection of everyday life stories that can also be used for story generation.

**Story Cloze Test**
- It is a new commonsense reasoning framework for evaluating story understanding, story generation, and script learning. This test requires us to choose the correct ending to a four-sentence story.

## Evaluation Metrics

After obtaining the output sentences from the fine-tuned T5 model, we used two different ways to measure sentence similarity. We measured the similarity between output sentence and Ending 1 and also between output sentence and Ending 2 and then voted for the sentence that had the greater sentence similarity score. 

**T5 Sentence Similarity**
- We used the pre-trained T5 model to measure sentence similarity with the prefix “stsb sentence 1: ..., sentence 2: ...’. The embedding is from the Pre-trained T5 base model. 
**Universal Sentence Encoder**
- We also used the Universal Sentence Encoder large model to transfer the word text to embeddings, and then compared the similarity between output and two ending options. 

In the Story Cloze Test, the right ending usually goes with the flow of the previous 4 sentences, which means that the right ending shares the similar sentiment with the input 4 sentences. Other than similarity scores, we used two ways to generate the sentiment score of our model output, ending 1, ending 2. 

**VADER (Valence Aware Dictionary for Sentiment Reasoning)**
- It is a model which is specifically attuned to the sentiments expressed in social media. It is sensitive to polarity (positive/negative) and intensity (strength) of emotion 
**Flair - NLP sentiment analysis**
- Flair is a sentiment classifier model which pretrained on IMDb movies reviews and based on a character-level LSTM neural network. Flair takes the whole sentence into account and outputs “positive” or “negative” to label the sentence. This model has reached the state of arts in various datasets, which also provided us the best result for our SCT dataset. 

**Simi-Senti SCORE** 
- (To evaluate our model performance in the second evaluation layer, we firstly got the sentiment classes(positive, negative, neutral) for each set of output, ending 1 and ending 2. If the sentiment class matched between output and endings, we created a dummy variable to indicate that the ending sentiment is consistent with the output. Next, we calculated the similarity scores between output and two endings. We measured the similarity between output sentence and Ending 1 and also between output sentence and Ending 2. We used T5 similarity score as our main method and used Universal Sentence Encoder when the T5 similarity score is 0.0.
The distribution of the difference between two similarity scores (output and ending 1 similarity score; output and ending 2 similarity score) is likely following the normal distribution and more than half scores fall into [-1, 1]Therefore, we decided to set weight = 1 )


## T5 Model Architecture

![Image of model arch](https://github.com/xiaowanzio8/NLP-Story-Ending-Prediction-with-T5/blob/main/images/model_arch.jpg)

## Baseline 

![Image of baseline](https://github.com/xiaowanzio8/NLP-Story-Ending-Prediction-with-T5/blob/main/images/baseline.jpg)

For the baseline model, we used the pretrained text-to-text T5-base. Our model architecture was built on PyTorch-lightning and Transformers. For the ROC story dataset, we formatted the model input as a small paragraph, which concatenated the first 4 sentences and the corresponding output as the 5th sentence. We then further breakdown sentences into tokens and then we pad all the sentences to the same length, which is 256.  We passed the tokens into the base model, through back propagation, our loss kept decreasing. In the end, we output the prediction of sentence 5 to compare with the target sentence. 
![image](https://user-images.githubusercontent.com/59941969/147163994-8878d081-3b90-4ff8-8f17-a90addc14741.png)



## Experiments

![Image of Experiment](https://github.com/xiaowanzio8/NLP-Story-Ending-Prediction-with-T5/blob/main/images/experiments.jpg)

We fine-tuned our models in AWS by creating a G-2xlarge instance with Nvidia deep learning AMI and conducted limited parameter tuning (batch size: 8,16; learning rate: 1e-4, 1e-5, 5e-4; precision: 16, 32) to find the model that had the best performance. We used AdamW optimizer to adjust weight decay and learning rate separately. 

Firstly, we fine-tuned a pre-trained T5-small model on the ROCstories dataset. We studied some recent research papers. In addition to the T5-small model, we experimented with the T5-base model since the training dataset size is over 50,000. The T5-base model has 220 million parameters, which could understand the sentence complexity better. See Appendix \ref{sec:appendix D} for the prediction result examples from T5-base and T5-small models.

In our baseline model, we only generated one ending for each story. In fact, one story could have many possible endings. In order to add some variance to the final output, we tried to generate 5 outputs for each story by setting Number of Return Sequence = 5 &  Beam Size = 10. 


## Experiment Results

![Image of small vs base 1](https://github.com/xiaowanzio8/NLP-Story-Ending-Prediction-with-T5/blob/main/images/small%20vs%20base.jpg)

![Image of small vs base 2](https://github.com/xiaowanzio8/NLP-Story-Ending-Prediction-with-T5/blob/main/images/small%20vs%20base2.jpg)


For the experiment results, we found out that T5 base worked better than T5 small in our case, because we have over 50 thousands training examples. 

Therefore, we experimented rest of our enhancement on the T5 base model. The result is present on the screen. 

After adding the 5 outputs, and simi-senti scores, we reached an accuracy of 76% eventually, which has increased 6.6% compared to baseline.

Overall, we found out that increasing the number of sequence did not help much while applying sentiment analysis helps quite bit to increase the accuracy.


![Image of result](https://github.com/xiaowanzio8/NLP-Story-Ending-Prediction-with-T5/blob/main/images/result.jpg)


## Test Result

After we fine tuned our model, and applied the enhancement for evaluation process, we are happy to reach a test accuracy of 74.5% on the leaderboard of Story Cloze Test Winter 2018, which is the 5th position of the leaderboard.

## Future Work

In terms of future work for this study, it can be done through separating the input 4 sentences into “beginning”, “middle”, and then add attentions between two parts. and using both of them to predict the ending. 

