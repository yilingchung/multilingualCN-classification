# Multilingual Counter Narrative Type Classification
This work aims at classifying counter narrative types given a hate speech for English, Italian and French.

## Requirements
```
pip install -r requirements.txt
```


## Data
__CONAN__: We use the hate countering dataset __CONAN__ from the paper [CONAN - COunter NArratives through Nichesourcing: a Multilingual Dataset of Responses to Fight Online Hate Speech](https://github.com/marcoguerini/CONAN).

__WikiLingua__: To create unrelated pairs with repect to islamophobia, we use the WikiLingua data from the paper [WikiLingua: A New Benchmark Dataset for Cross-Lingual Abstractive Summarization](https://github.com/esdurmus/Wikilingua).

In our experiments, we focus on pairs that are annotated with just one counter narrative type. The data partition used in our experiments can be found under ```./data/```. For more details on data partition procedure, please see our paper submitted.
The data is stored in csv format, with 3 columns: __sentence1__ respresenting hate speech, __sentence2__ as corresponding counter narrative, and __label__ as counter narrative type.


## Training
For counter narrative type classification, we use the implementation [run_glue.py](https://github.com/huggingface/transformers/blob/v4.3.0.rc1/examples/text-classification/run_glue.py) from Transformers library that can be adapted to various classification tasks.
 
To reproduce our results of training data on XLM-RoBERTa-base, please run: 

```
python run_glue.py \
  --model_name_or_path xlm-roberta-base \
  --train_file path_to_train_file \
  --test_file path_to_test_file \
  --learning_rate 2e-5 \
  --num_train_epochs 10 \
  --do_train \
  --do_predict \
  --max_seq_length 256 \
  --per_device_train_batch_size 32 \
  --report_to none \
  --logging_steps 30 \
  --evaluation_strategy steps \
  --eval_steps 5000 \
  --load_best_model_at_end True \
  --metric_for_best_model 'f1' \
  --output_dir path_to_output_file
  ...
```

## Citation
You can find further details in our paper:

Yi-Ling Chung, Marco Guerini, and Rodrigo Agerri. 2021. <em>Multilingual Counter Narrative Type Classification.</em> In Proceedings of the 8th Workshop on Argument Mining.


```bibtex
@inproceedings{chung-etal-2021-multilingual,
    title = "{Multilingual Counter Narrative Type Classification",
    author = "Chung, Yi-Ling and Guerini, Marco and Agerri, Rodrigo ",
    booktitle = "Proceedings of the 8th Workshop on Argument Mining",
    month = nov,
    year = "2021",
    url = "https://arxiv.org/abs/2109.13664",
}
```

