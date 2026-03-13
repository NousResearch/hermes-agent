# Training Custom Tokenizers

Complete guide to training tokenizers from scratch.

## Training workflow

### Step 1: Choose tokenization algorithm

**Decision tree**:
- **GPT-style model** → BPE
- **BERT-style model** → WordPiece
- **Multilingual/No word boundaries** → Unigram

### Step 2: Prepare training data

```python
# Option 1: From files
files = ["train.txt", "validation.txt"]

# Option 2: From Python list
texts = [
    "This is the first sentence.",
    "This is the second sentence.",
    # ... more texts
]

# Option 3: From dataset iterator
from datasets import load_dataset

dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")

def batch_iterator(batch_size=1000):
    for i in range(0, len(dataset), batch_size):
        yield dataset[i:i + batch_size]["text"]
```

### Step 3: Initialize tokenizer

**BPE example**:
```python
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

tokenizer = Tokenizer(BPE())
tokenizer.pre_tokenizer = ByteLevel()
tokenizer.decoder = ByteLevelDecoder()

trainer = BpeTrainer(
    vocab_size=50000,
    min_frequency=2,
    special_tokens=["<|endoftext|>", "<|padding|>"],
    show_progress=True
)
```

**WordPiece example**:
```python
from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer
from tokenizers.normalizers import BertNormalizer
from tokenizers.pre_tokenizers import BertPreTokenizer

tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
tokenizer.normalizer = BertNormalizer(lowercase=True)
tokenizer.pre_tokenizer = BertPreTokenizer()

trainer = WordPieceTrainer(
    vocab_size=30522,
    min_frequency=2,
    special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
    continuing_subword_prefix="##",
    show_progress=True
)
```

**Unigram example**:
```python
from tokenizers.models import Unigram
from tokenizers.trainers import UnigramTrainer

tokenizer = Tokenizer(Unigram())

trainer = UnigramTrainer(
    vocab_size=8000,
    special_tokens=["<unk>", "<s>", "</s>", "<pad>"],
    unk_token="<unk>",
    show_progress=True
)
```

### Step 4: Train

```python
# From files
tokenizer.train(files=files, trainer=trainer)

# From iterator (recommended for large datasets)
tokenizer.train_from_iterator(
    batch_iterator(),
    trainer=trainer,
    length=len(dataset)  # Optional, for progress bar
)
```

**Training time** (30k vocab on 16-core CPU):
- 10 MB: 15-30 seconds
- 100 MB: 1-3 minutes
- 1 GB: 15-30 minutes
- 10 GB: 2-4 hours

### Step 5: Add post-processing

```python
from tokenizers.processors import TemplateProcessing

# BERT-style
tokenizer.post_processor = TemplateProcessing(
    single="[CLS] $A [SEP]",
    pair="[CLS] $A [SEP] $B [SEP]",
    special_tokens=[
        ("[CLS]", tokenizer.token_to_id("[CLS]")),
        ("[SEP]", tokenizer.token_to_id("[SEP]")),
    ],
)

# GPT-2 style
tokenizer.post_processor = TemplateProcessing(
    single="$A <|endoftext|>",
    special_tokens=[
        ("<|endoftext|>", tokenizer.token_to_id("<|endoftext|>")),
    ],
)
```

### Step 6: Save

```python
# Save to JSON
tokenizer.save("my-tokenizer.json")

# Save to directory (for transformers)
tokenizer.save("my-tokenizer-dir/tokenizer.json")

# Convert to transformers format
from transformers import PreTrainedTokenizerFast

transformers_tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer,
    unk_token="[UNK]",
    pad_token="[PAD]",
    cls_token="[CLS]",
    sep_token="[SEP]",
    mask_token="[MASK]"
)

transformers_tokenizer.save_pretrained("my-tokenizer-dir")
```

## Trainer configuration

### BpeTrainer parameters

```python
from tokenizers.trainers import BpeTrainer

trainer = BpeTrainer(
    vocab_size=30000,              # Target vocabulary size
    min_frequency=2,               # Minimum frequency for merges
    special_tokens=["[UNK]"],      # Special tokens (added first)
    limit_alphabet=1000,           # Limit initial alphabet size
    initial_alphabet=[],           # Pre-defined initial characters
    show_progress=True,            # Show progress bar
    continuing_subword_prefix="",  # Prefix for continuing subwords
    end_of_word_suffix=""          # Suffix for end of words
)
```

**Parameter tuning**:
- **vocab_size**: Start with 30k for English, 50k for multilingual
- **min_frequency**: 2-5 for large corpora, 1 for small
- **limit_alphabet**: Reduce for non-English (CJK languages)

### WordPieceTrainer parameters

```python
from tokenizers.trainers import WordPieceTrainer

trainer = WordPieceTrainer(
    vocab_size=30522,              # BERT uses 30,522
    min_frequency=2,
    special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
    limit_alphabet=1000,
    continuing_subword_prefix="##", # BERT-style prefix
    show_progress=True
)
```

### UnigramTrainer parameters

```python
from tokenizers.trainers import UnigramTrainer

trainer = UnigramTrainer(
    vocab_size=8000,               # Typically smaller than BPE/WordPiece
    special_tokens=["<unk>", "<s>", "</s>"],
    unk_token="<unk>",
    max_piece_length=16,           # Maximum token length
    n_sub_iterations=2,            # EM algorithm iterations
    shrinking_factor=0.75,         # Vocabulary reduction rate
    show_progress=True
)
```

## Training from large datasets

### Memory-efficient training

```python
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

# Load dataset
dataset = load_dataset("wikipedia", "20220301.en", split="train", streaming=True)

# Create iterator (yields batches)
def batch_iterator(batch_size=1000):
    batch = []
    for sample in dataset:
        batch.append(sample["text"])
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch

# Initialize tokenizer
tokenizer = Tokenizer(BPE())
trainer = BpeTrainer(vocab_size=50000, special_tokens=["<|endoftext|>"])

# Train (memory efficient - streams data)
tokenizer.train_from_iterator(
    batch_iterator(),
    trainer=trainer
)
```

**Memory usage**: ~200 MB (vs 10+ GB loading full dataset)

### Multi-file training

```python
import glob

# Find all training files
files = glob.glob("data/train/*.txt")
print(f"Training on {len(files)} files")

# Train on all files
tokenizer.train(files=files, trainer=trainer)
```

### Parallel training (multi-processing)

```python
from multiprocessing import Pool, cpu_count
import os

def train_shard(shard_files):
    """Train tokenizer on a shard of files."""
    tokenizer = Tokenizer(BPE())
    trainer = BpeTrainer(vocab_size=50000)
    tokenizer.train(files=shard_files, trainer=trainer)
    return tokenizer.get_vocab()

# Split files into shards
num_shards = cpu_count()
file_shards = [files[i::num_shards] for i in range(num_shards)]

# Train shards in parallel
with Pool(num_shards) as pool:
    vocab_shards = pool.map(train_shard, file_shards)

# Merge vocabularies (custom logic needed)
# This is a simplified example - real implementation would merge intelligently
final_vocab = {}
for vocab in vocab_shards:
    final_vocab.update(vocab)
```

## Domain-specific tokenizers

### Code tokenizer

```python
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.normalizers import Sequence, NFC

# Code-optimized configuration
tokenizer = Tokenizer(BPE())

# Minimal normalization (preserve case, whitespace)
tokenizer.normalizer = NFC()  # Only normalize Unicode

# Byte-level pre-tokenization (handles all characters)
tokenizer.pre_tokenizer = ByteLevel()

# Train on code corpus
trainer = BpeTrainer(
    vocab_size=50000,
    special_tokens=["<|endoftext|>", "<|pad|>"],
    min_frequency=2
)

tokenizer.train(files=["code_corpus.txt"], trainer=trainer)
```

### Medical/scientific tokenizer

```python
# Preserve case and special characters
from tokenizers.normalizers import NFKC
from tokenizers.pre_tokenizers import Whitespace, Punctuation, Sequence

tokenizer = Tokenizer(BPE())

# Minimal normalization
tokenizer.normalizer = NFKC()

# Preserve medical terms
tokenizer.pre_tokenizer = Sequence([
    Whitespace(),
    Punctuation(behavior="isolated")  # Keep punctuation separate
])

trainer = BpeTrainer(
    vocab_size=50000,
    special_tokens=["[UNK]", "[CLS]", "[SEP]"],
    min_frequency=3  # Higher threshold for rare medical terms
)

tokenizer.train(files=["pubmed_corpus.txt"], trainer=trainer)
```

### Multilingual tokenizer

```python
# Handle multiple scripts
from tokenizers.normalizers import NFKC, Lowercase, Sequence

tokenizer = Tokenizer(BPE())

# Normalize but don't lowercase (preserves script differences)
tokenizer.normalizer = NFKC()

# Byte-level handles all Unicode
from tokenizers.pre_tokenizers import ByteLevel
tokenizer.pre_tokenizer = ByteLevel()

trainer = BpeTrainer(
    vocab_size=100000,  # Larger vocab for multiple languages
    special_tokens=["<unk>", "<s>", "</s>"],
    limit_alphabet=None  # No limit (handles all scripts)
)

# Train on multilingual corpus
tokenizer.train(files=["multilingual_corpus.txt"], trainer=trainer)
```

## Vocabulary size selection

### Guidelines by task

| Task                  | Recommended Vocab Size | Rationale |
|-----------------------|------------------------|-----------|
| English (monolingual) | 30,000 - 50,000       | Balanced coverage |
| Multilingual          | 50,000 - 250,000      | More languages = more tokens |
| Code                  | 30,000 - 50,000       | Similar to English |
| Domain-specific       | 10,000 - 30,000       | Smaller, focused vocabulary |
| Character-level tasks | 1,000 - 5,000         | Only characters + subwords |

### Vocabulary size impact

**Small vocab (10k)**:
- Pros: Faster training, smaller model, less memory
- Cons: More tokens per sentence, worse OOV handling

**Medium vocab (30k-50k)**:
- Pros: Good balance, standard choice
- Cons: None (recommended default)

**Large vocab (100k+)**:
- Pros: Fewer tokens per sentence, better OOV
- Cons: Slower training, larger embedding table

### Empirical testing

```python
# Train multiple tokenizers with different vocab sizes
vocab_sizes = [10000, 30000, 50000, 100000]

for vocab_size in vocab_sizes:
    tokenizer = Tokenizer(BPE())
    trainer = BpeTrainer(vocab_size=vocab_size)
    tokenizer.train(files=["sample.txt"], trainer=trainer)

    # Evaluate on test set
    test_text = "Test sentence for evaluation..."
    tokens = tokenizer.encode(test_text).ids

    print(f"Vocab: {vocab_size:6d} | Tokens: {len(tokens):3d} | Avg: {len(test_text)/len(tokens):.2f} chars/token")

# Example output:
# Vocab:  10000 | Tokens:  12 | Avg: 2.33 chars/token
# Vocab:  30000 | Tokens:   8 | Avg: 3.50 chars/token
# Vocab:  50000 | Tokens:   7 | Avg: 4.00 chars/token
# Vocab: 100000 | Tokens:   6 | Avg: 4.67 chars/token
```

## Testing tokenizer quality

### Coverage test

```python
# Test on held-out data
test_corpus = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")

total_tokens = 0
unk_tokens = 0
unk_id = tokenizer.token_to_id("[UNK]")

for text in test_corpus["text"]:
    if text.strip():
        encoding = tokenizer.encode(text)
        total_tokens += len(encoding.ids)
        unk_tokens += encoding.ids.count(unk_id)

unk_rate = unk_tokens / total_tokens
print(f"Unknown token rate: {unk_rate:.2%}")

# Good quality: <1% unknown tokens
# Acceptable: 1-5%
# Poor: >5%
```

### Compression test

```python
# Measure tokenization efficiency
import numpy as np

token_lengths = []

for text in test_corpus["text"][:1000]:
    if text.strip():
        encoding = tokenizer.encode(text)
        chars_per_token = len(text) / len(encoding.ids)
        token_lengths.append(chars_per_token)

avg_chars_per_token = np.mean(token_lengths)
print(f"Average characters per token: {avg_chars_per_token:.2f}")

# Good: 4-6 chars/token (English)
# Acceptable: 3-4 chars/token
# Poor: <3 chars/token (under-compression)
```

### Semantic test

```python
# Manually inspect tokenization of common words/phrases
test_phrases = [
    "tokenization",
    "machine learning",
    "artificial intelligence",
    "preprocessing",
    "hello world"
]

for phrase in test_phrases:
    tokens = tokenizer.encode(phrase).tokens
    print(f"{phrase:25s} → {tokens}")

# Good tokenization:
# tokenization              → ['token', 'ization']
# machine learning          → ['machine', 'learning']
# artificial intelligence   → ['artificial', 'intelligence']
```

## Troubleshooting

### Issue: Training too slow

**Solutions**:
1. Reduce vocabulary size
2. Increase `min_frequency`
3. Use `limit_alphabet` to reduce initial alphabet
4. Train on subset first

```python
# Fast training configuration
trainer = BpeTrainer(
    vocab_size=20000,      # Smaller vocab
    min_frequency=5,       # Higher threshold
    limit_alphabet=500,    # Limit alphabet
    show_progress=True
)
```

### Issue: High unknown token rate

**Solutions**:
1. Increase vocabulary size
2. Decrease `min_frequency`
3. Check normalization (might be too aggressive)

```python
# Better coverage configuration
trainer = BpeTrainer(
    vocab_size=50000,      # Larger vocab
    min_frequency=1,       # Lower threshold
)
```

### Issue: Poor quality tokenization

**Solutions**:
1. Verify normalization matches your use case
2. Check pre-tokenization splits correctly
3. Ensure training data is representative
4. Try different algorithm (BPE vs WordPiece vs Unigram)

```python
# Debug tokenization pipeline
text = "Sample text to debug"

# Check normalization
normalized = tokenizer.normalizer.normalize_str(text)
print(f"Normalized: {normalized}")

# Check pre-tokenization
pre_tokens = tokenizer.pre_tokenizer.pre_tokenize_str(text)
print(f"Pre-tokens: {pre_tokens}")

# Check final tokenization
tokens = tokenizer.encode(text).tokens
print(f"Tokens: {tokens}")
```

## Best practices

1. **Use representative training data** - Match your target domain
2. **Start with standard configs** - BERT WordPiece or GPT-2 BPE
3. **Test on held-out data** - Measure unknown token rate
4. **Iterate on vocabulary size** - Test 30k, 50k, 100k
5. **Save tokenizer with model** - Ensure reproducibility
6. **Version your tokenizers** - Track changes for reproducibility
7. **Document special tokens** - Critical for model training
