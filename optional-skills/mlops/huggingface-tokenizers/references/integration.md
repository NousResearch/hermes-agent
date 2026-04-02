# Transformers Integration

Complete guide to using HuggingFace Tokenizers with the Transformers library.

## AutoTokenizer

The easiest way to load tokenizers.

### Loading pretrained tokenizers

```python
from transformers import AutoTokenizer

# Load from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Check if using fast tokenizer (Rust-based)
print(tokenizer.is_fast)  # True

# Access underlying tokenizers.Tokenizer
if tokenizer.is_fast:
    fast_tokenizer = tokenizer.backend_tokenizer
    print(type(fast_tokenizer))  # <class 'tokenizers.Tokenizer'>
```

### Fast vs slow tokenizers

| Feature                  | Fast (Rust)    | Slow (Python) |
|--------------------------|----------------|---------------|
| Speed                    | 5-10× faster   | Baseline      |
| Alignment tracking       | ✅ Full support | ❌ Limited     |
| Batch processing         | ✅ Optimized    | ⚠️ Slower      |
| Offset mapping           | ✅ Yes          | ❌ No          |
| Installation             | `tokenizers`   | Built-in      |

**Always use fast tokenizers when available.**

### Check available tokenizers

```python
from transformers import TOKENIZER_MAPPING

# List all fast tokenizers
for config_class, (slow, fast) in TOKENIZER_MAPPING.items():
    if fast is not None:
        print(f"{config_class.__name__}: {fast.__name__}")
```

## PreTrainedTokenizerFast

Wrap custom tokenizers for transformers.

### Convert custom tokenizer

```python
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from transformers import PreTrainedTokenizerFast

# Train custom tokenizer
tokenizer = Tokenizer(BPE())
trainer = BpeTrainer(
    vocab_size=30000,
    special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
)
tokenizer.train(files=["corpus.txt"], trainer=trainer)

# Save tokenizer
tokenizer.save("my-tokenizer.json")

# Wrap for transformers
transformers_tokenizer = PreTrainedTokenizerFast(
    tokenizer_file="my-tokenizer.json",
    unk_token="[UNK]",
    sep_token="[SEP]",
    pad_token="[PAD]",
    cls_token="[CLS]",
    mask_token="[MASK]"
)

# Save in transformers format
transformers_tokenizer.save_pretrained("my-tokenizer")
```

**Result**: Directory with `tokenizer.json` + `tokenizer_config.json` + `special_tokens_map.json`

### Use like any transformers tokenizer

```python
# Load
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("my-tokenizer")

# Encode with all transformers features
outputs = tokenizer(
    "Hello world",
    padding="max_length",
    truncation=True,
    max_length=128,
    return_tensors="pt"
)

print(outputs.keys())
# dict_keys(['input_ids', 'token_type_ids', 'attention_mask'])
```

## Special tokens

### Default special tokens

| Model Family | CLS/BOS | SEP/EOS       | PAD     | UNK     | MASK    |
|--------------|---------|---------------|---------|---------|---------|
| BERT         | [CLS]   | [SEP]         | [PAD]   | [UNK]   | [MASK]  |
| GPT-2        | -       | <\|endoftext\|> | <\|endoftext\|> | <\|endoftext\|> | -       |
| RoBERTa      | <s>     | </s>          | <pad>   | <unk>   | <mask>  |
| T5           | -       | </s>          | <pad>   | <unk>   | -       |

### Adding special tokens

```python
# Add new special tokens
special_tokens_dict = {
    "additional_special_tokens": ["<|image|>", "<|video|>", "<|audio|>"]
}

num_added_tokens = tokenizer.add_special_tokens(special_tokens_dict)
print(f"Added {num_added_tokens} tokens")

# Resize model embeddings
model.resize_token_embeddings(len(tokenizer))

# Use new tokens
text = "This is an image: <|image|>"
tokens = tokenizer.encode(text)
```

### Adding regular tokens

```python
# Add domain-specific tokens
new_tokens = ["COVID-19", "mRNA", "vaccine"]
num_added = tokenizer.add_tokens(new_tokens)

# These are NOT special tokens (can be split if needed)
tokenizer.add_tokens(new_tokens, special_tokens=False)

# These ARE special tokens (never split)
tokenizer.add_tokens(new_tokens, special_tokens=True)
```

## Encoding and decoding

### Basic encoding

```python
# Single sentence
text = "Hello, how are you?"
encoded = tokenizer(text)

print(encoded)
# {'input_ids': [101, 7592, 1010, 2129, 2024, 2017, 1029, 102],
#  'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0],
#  'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1]}
```

### Batch encoding

```python
# Multiple sentences
texts = ["Hello world", "How are you?", "I am fine"]
encoded = tokenizer(texts, padding=True, truncation=True, max_length=10)

print(encoded['input_ids'])
# [[101, 7592, 2088, 102, 0, 0, 0, 0, 0, 0],
#  [101, 2129, 2024, 2017, 1029, 102, 0, 0, 0, 0],
#  [101, 1045, 2572, 2986, 102, 0, 0, 0, 0, 0]]
```

### Return tensors

```python
# Return PyTorch tensors
outputs = tokenizer("Hello world", return_tensors="pt")
print(outputs['input_ids'].shape)  # torch.Size([1, 5])

# Return TensorFlow tensors
outputs = tokenizer("Hello world", return_tensors="tf")

# Return NumPy arrays
outputs = tokenizer("Hello world", return_tensors="np")

# Return lists (default)
outputs = tokenizer("Hello world", return_tensors=None)
```

### Decoding

```python
# Decode token IDs
ids = [101, 7592, 2088, 102]
text = tokenizer.decode(ids)
print(text)  # "[CLS] hello world [SEP]"

# Skip special tokens
text = tokenizer.decode(ids, skip_special_tokens=True)
print(text)  # "hello world"

# Batch decode
batch_ids = [[101, 7592, 102], [101, 2088, 102]]
texts = tokenizer.batch_decode(batch_ids, skip_special_tokens=True)
print(texts)  # ["hello", "world"]
```

## Padding and truncation

### Padding strategies

```python
# Pad to max length in batch
tokenizer(texts, padding="longest")

# Pad to model max length
tokenizer(texts, padding="max_length", max_length=128)

# No padding
tokenizer(texts, padding=False)

# Pad to multiple of value (for efficient computation)
tokenizer(texts, padding="max_length", max_length=128, pad_to_multiple_of=8)
# Result: length will be 128 (already multiple of 8)
```

### Truncation strategies

```python
# Truncate to max length
tokenizer(text, truncation=True, max_length=10)

# Only truncate first sequence (for pairs)
tokenizer(text1, text2, truncation="only_first", max_length=20)

# Only truncate second sequence
tokenizer(text1, text2, truncation="only_second", max_length=20)

# Truncate longest first (default for pairs)
tokenizer(text1, text2, truncation="longest_first", max_length=20)

# No truncation (error if too long)
tokenizer(text, truncation=False)
```

### Stride for long documents

```python
# For documents longer than max_length
text = "Very long document " * 1000

# Encode with overlap
encodings = tokenizer(
    text,
    max_length=512,
    stride=128,          # Overlap between chunks
    truncation=True,
    return_overflowing_tokens=True,
    return_offsets_mapping=True
)

# Get all chunks
num_chunks = len(encodings['input_ids'])
print(f"Split into {num_chunks} chunks")

# Each chunk overlaps by stride tokens
for i, chunk in enumerate(encodings['input_ids']):
    print(f"Chunk {i}: {len(chunk)} tokens")
```

**Use case**: Long document QA, sliding window inference

## Alignment and offsets

### Offset mapping

```python
# Get character offsets for each token
encoded = tokenizer("Hello, world!", return_offsets_mapping=True)

for token, (start, end) in zip(
    encoded.tokens(),
    encoded['offset_mapping'][0]
):
    print(f"{token:10s} → [{start:2d}, {end:2d})")

# Output:
# [CLS]      → [ 0,  0)
# Hello      → [ 0,  5)
# ,          → [ 5,  6)
# world      → [ 7, 12)
# !          → [12, 13)
# [SEP]      → [ 0,  0)
```

### Word IDs

```python
# Get word index for each token
encoded = tokenizer("Hello world", return_offsets_mapping=True)
word_ids = encoded.word_ids()

print(word_ids)
# [None, 0, 1, None]
# None = special token, 0 = first word, 1 = second word
```

**Use case**: Token classification (NER, POS tagging)

### Character to token mapping

```python
text = "Machine learning is awesome"
encoded = tokenizer(text, return_offsets_mapping=True)

# Find token for character position
char_pos = 8  # "l" in "learning"
token_idx = encoded.char_to_token(char_pos)

print(f"Character {char_pos} is in token {token_idx}: {encoded.tokens()[token_idx]}")
# Character 8 is in token 2: learning
```

**Use case**: Question answering (map answer character span to tokens)

### Sequence pairs

```python
# Encode sentence pair
encoded = tokenizer("Question here", "Answer here", return_offsets_mapping=True)

# Get sequence IDs (which sequence each token belongs to)
sequence_ids = encoded.sequence_ids()
print(sequence_ids)
# [None, 0, 0, 0, None, 1, 1, 1, None]
# None = special token, 0 = question, 1 = answer
```

## Model integration

### Use with transformers models

```python
from transformers import AutoModel, AutoTokenizer
import torch

# Load model and tokenizer
model = AutoModel.from_pretrained("bert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Tokenize
text = "Hello world"
inputs = tokenizer(text, return_tensors="pt")

# Forward pass
with torch.no_grad():
    outputs = model(**inputs)

# Get embeddings
last_hidden_state = outputs.last_hidden_state
print(last_hidden_state.shape)  # [1, seq_len, hidden_size]
```

### Custom model with custom tokenizer

```python
from transformers import BertConfig, BertModel

# Train custom tokenizer
from tokenizers import Tokenizer, models, trainers
tokenizer = Tokenizer(models.BPE())
trainer = trainers.BpeTrainer(vocab_size=30000)
tokenizer.train(files=["data.txt"], trainer=trainer)

# Wrap for transformers
from transformers import PreTrainedTokenizerFast
fast_tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer,
    unk_token="[UNK]",
    pad_token="[PAD]"
)

# Create model with custom vocab size
config = BertConfig(vocab_size=30000)
model = BertModel(config)

# Use together
inputs = fast_tokenizer("Hello world", return_tensors="pt")
outputs = model(**inputs)
```

### Save and load together

```python
# Save both
model.save_pretrained("my-model")
tokenizer.save_pretrained("my-model")

# Directory structure:
# my-model/
#   ├── config.json
#   ├── pytorch_model.bin
#   ├── tokenizer.json
#   ├── tokenizer_config.json
#   └── special_tokens_map.json

# Load both
from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained("my-model")
tokenizer = AutoTokenizer.from_pretrained("my-model")
```

## Advanced features

### Multimodal tokenization

```python
from transformers import AutoTokenizer

# LLaVA-style (image + text)
tokenizer = AutoTokenizer.from_pretrained("llava-hf/llava-1.5-7b-hf")

# Add image placeholder token
tokenizer.add_special_tokens({"additional_special_tokens": ["<image>"]})

# Use in prompt
text = "Describe this image: <image>"
inputs = tokenizer(text, return_tensors="pt")
```

### Template formatting

```python
# Chat template
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"},
    {"role": "assistant", "content": "Hi! How can I help?"},
    {"role": "user", "content": "What's the weather?"}
]

# Apply chat template (if tokenizer has one)
if hasattr(tokenizer, "apply_chat_template"):
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    inputs = tokenizer(text, return_tensors="pt")
```

### Custom template

```python
from transformers import PreTrainedTokenizerFast

tokenizer = PreTrainedTokenizerFast(tokenizer_file="tokenizer.json")

# Define chat template
tokenizer.chat_template = """
{%- for message in messages %}
    {%- if message['role'] == 'system' %}
        System: {{ message['content'] }}\\n
    {%- elif message['role'] == 'user' %}
        User: {{ message['content'] }}\\n
    {%- elif message['role'] == 'assistant' %}
        Assistant: {{ message['content'] }}\\n
    {%- endif %}
{%- endfor %}
Assistant:
"""

# Use template
text = tokenizer.apply_chat_template(messages, tokenize=False)
```

## Performance optimization

### Batch processing

```python
# Process large datasets efficiently
from datasets import load_dataset

dataset = load_dataset("imdb", split="train[:1000]")

# Tokenize in batches
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=512
    )

# Map over dataset (batched)
tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    batch_size=1000,
    num_proc=4  # Parallel processing
)
```

### Caching

```python
# Enable caching for repeated tokenization
tokenizer = AutoTokenizer.from_pretrained(
    "bert-base-uncased",
    use_fast=True,
    cache_dir="./cache"  # Cache tokenizer files
)

# Tokenize with caching
from functools import lru_cache

@lru_cache(maxsize=10000)
def cached_tokenize(text):
    return tuple(tokenizer.encode(text))

# Reuses cached results for repeated inputs
```

### Memory efficiency

```python
# For very large datasets, use streaming
from datasets import load_dataset

dataset = load_dataset("pile", split="train", streaming=True)

def process_batch(batch):
    # Tokenize
    tokens = tokenizer(batch["text"], truncation=True, max_length=512)

    # Process tokens...

    return tokens

# Process in chunks (memory efficient)
for batch in dataset.batch(batch_size=1000):
    processed = process_batch(batch)
```

## Troubleshooting

### Issue: Tokenizer not fast

**Symptom**:
```python
tokenizer.is_fast  # False
```

**Solution**: Install tokenizers library
```bash
pip install tokenizers
```

### Issue: Special tokens not working

**Symptom**: Special tokens are split into subwords

**Solution**: Add as special tokens, not regular tokens
```python
# Wrong
tokenizer.add_tokens(["<|image|>"])

# Correct
tokenizer.add_special_tokens({"additional_special_tokens": ["<|image|>"]})
```

### Issue: Offset mapping not available

**Symptom**:
```python
tokenizer("text", return_offsets_mapping=True)
# Error: return_offsets_mapping not supported
```

**Solution**: Use fast tokenizer
```python
from transformers import AutoTokenizer

# Load fast version
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=True)
```

### Issue: Padding inconsistent

**Symptom**: Some sequences padded, others not

**Solution**: Specify padding strategy
```python
# Explicit padding
tokenizer(
    texts,
    padding="max_length",  # or "longest"
    max_length=128
)
```

## Best practices

1. **Always use fast tokenizers**:
   - 5-10× faster
   - Full alignment tracking
   - Better batch processing

2. **Save tokenizer with model**:
   - Ensures reproducibility
   - Prevents version mismatches

3. **Use batch processing for datasets**:
   - Tokenize with `.map(batched=True)`
   - Set `num_proc` for parallelism

4. **Enable caching for repeated inputs**:
   - Use `lru_cache` for inference
   - Cache tokenizer files with `cache_dir`

5. **Handle special tokens properly**:
   - Use `add_special_tokens()` for never-split tokens
   - Resize embeddings after adding tokens

6. **Test alignment for downstream tasks**:
   - Verify `offset_mapping` is correct
   - Test `char_to_token()` on samples

7. **Version control tokenizer config**:
   - Save `tokenizer_config.json`
   - Document custom templates
   - Track vocabulary changes
