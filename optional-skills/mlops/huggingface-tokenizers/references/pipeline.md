# Tokenization Pipeline Components

Complete guide to normalizers, pre-tokenizers, models, post-processors, and decoders.

## Pipeline overview

**Full tokenization pipeline**:
```
Raw Text
  ↓
Normalization (cleaning, lowercasing)
  ↓
Pre-tokenization (split into words)
  ↓
Model (apply BPE/WordPiece/Unigram)
  ↓
Post-processing (add special tokens)
  ↓
Token IDs
```

**Decoding reverses the process**:
```
Token IDs
  ↓
Decoder (handle special encodings)
  ↓
Raw Text
```

## Normalizers

Clean and standardize input text.

### Common normalizers

**Lowercase**:
```python
from tokenizers.normalizers import Lowercase

tokenizer.normalizer = Lowercase()

# Input: "Hello WORLD"
# Output: "hello world"
```

**Unicode normalization**:
```python
from tokenizers.normalizers import NFD, NFC, NFKD, NFKC

# NFD: Canonical decomposition
tokenizer.normalizer = NFD()
# "é" → "e" + "́" (separate characters)

# NFC: Canonical composition (default)
tokenizer.normalizer = NFC()
# "e" + "́" → "é" (composed)

# NFKD: Compatibility decomposition
tokenizer.normalizer = NFKD()
# "ﬁ" → "f" + "i"

# NFKC: Compatibility composition
tokenizer.normalizer = NFKC()
# Most aggressive normalization
```

**Strip accents**:
```python
from tokenizers.normalizers import StripAccents

tokenizer.normalizer = StripAccents()

# Input: "café"
# Output: "cafe"
```

**Whitespace handling**:
```python
from tokenizers.normalizers import Strip, StripAccents

# Remove leading/trailing whitespace
tokenizer.normalizer = Strip()

# Input: "  hello  "
# Output: "hello"
```

**Replace patterns**:
```python
from tokenizers.normalizers import Replace

# Replace newlines with spaces
tokenizer.normalizer = Replace("\\n", " ")

# Input: "hello\\nworld"
# Output: "hello world"
```

### Combining normalizers

```python
from tokenizers.normalizers import Sequence, NFD, Lowercase, StripAccents

# BERT-style normalization
tokenizer.normalizer = Sequence([
    NFD(),           # Unicode decomposition
    Lowercase(),     # Convert to lowercase
    StripAccents()   # Remove accents
])

# Input: "Café au Lait"
# After NFD: "Café au Lait" (e + ́)
# After Lowercase: "café au lait"
# After StripAccents: "cafe au lait"
```

### Use case examples

**Case-insensitive model (BERT)**:
```python
from tokenizers.normalizers import BertNormalizer

# All-in-one BERT normalization
tokenizer.normalizer = BertNormalizer(
    clean_text=True,        # Remove control characters
    handle_chinese_chars=True,  # Add spaces around Chinese
    strip_accents=True,     # Remove accents
    lowercase=True          # Lowercase
)
```

**Case-sensitive model (GPT-2)**:
```python
# Minimal normalization
tokenizer.normalizer = NFC()  # Only normalize Unicode
```

**Multilingual (mBERT)**:
```python
# Preserve scripts, normalize form
tokenizer.normalizer = NFKC()
```

## Pre-tokenizers

Split text into word-like units before tokenization.

### Whitespace splitting

```python
from tokenizers.pre_tokenizers import Whitespace

tokenizer.pre_tokenizer = Whitespace()

# Input: "Hello world! How are you?"
# Output: [("Hello", (0, 5)), ("world!", (6, 12)), ("How", (13, 16)), ("are", (17, 20)), ("you?", (21, 25))]
```

### Punctuation isolation

```python
from tokenizers.pre_tokenizers import Punctuation

tokenizer.pre_tokenizer = Punctuation()

# Input: "Hello, world!"
# Output: [("Hello", ...), (",", ...), ("world", ...), ("!", ...)]
```

### Byte-level (GPT-2)

```python
from tokenizers.pre_tokenizers import ByteLevel

tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=True)

# Input: "Hello world"
# Output: Byte-level tokens with Ġ prefix for spaces
# [("ĠHello", ...), ("Ġworld", ...)]
```

**Key feature**: Handles ALL Unicode characters (256 byte combinations)

### Metaspace (SentencePiece)

```python
from tokenizers.pre_tokenizers import Metaspace

tokenizer.pre_tokenizer = Metaspace(replacement="▁", add_prefix_space=True)

# Input: "Hello world"
# Output: [("▁Hello", ...), ("▁world", ...)]
```

**Used by**: T5, ALBERT (via SentencePiece)

### Digits splitting

```python
from tokenizers.pre_tokenizers import Digits

# Split digits individually
tokenizer.pre_tokenizer = Digits(individual_digits=True)

# Input: "Room 123"
# Output: [("Room", ...), ("1", ...), ("2", ...), ("3", ...)]

# Keep digits together
tokenizer.pre_tokenizer = Digits(individual_digits=False)

# Input: "Room 123"
# Output: [("Room", ...), ("123", ...)]
```

### BERT pre-tokenizer

```python
from tokenizers.pre_tokenizers import BertPreTokenizer

tokenizer.pre_tokenizer = BertPreTokenizer()

# Splits on whitespace and punctuation, preserves CJK
# Input: "Hello, 世界!"
# Output: [("Hello", ...), (",", ...), ("世", ...), ("界", ...), ("!", ...)]
```

### Combining pre-tokenizers

```python
from tokenizers.pre_tokenizers import Sequence, Whitespace, Punctuation

tokenizer.pre_tokenizer = Sequence([
    Whitespace(),     # Split on whitespace first
    Punctuation()     # Then isolate punctuation
])

# Input: "Hello, world!"
# After Whitespace: [("Hello,", ...), ("world!", ...)]
# After Punctuation: [("Hello", ...), (",", ...), ("world", ...), ("!", ...)]
```

### Pre-tokenizer comparison

| Pre-tokenizer     | Use Case                        | Example                                    |
|-------------------|---------------------------------|--------------------------------------------|
| Whitespace        | Simple English                  | "Hello world" → ["Hello", "world"]         |
| Punctuation       | Isolate symbols                 | "world!" → ["world", "!"]                  |
| ByteLevel         | Multilingual, emojis            | "🌍" → byte tokens                          |
| Metaspace         | SentencePiece-style             | "Hello" → ["▁Hello"]                       |
| BertPreTokenizer  | BERT-style (CJK aware)          | "世界" → ["世", "界"]                        |
| Digits            | Handle numbers                  | "123" → ["1", "2", "3"] or ["123"]        |

## Models

Core tokenization algorithms.

### BPE Model

```python
from tokenizers.models import BPE

model = BPE(
    vocab=None,           # Or provide pre-built vocab
    merges=None,          # Or provide merge rules
    unk_token="[UNK]",    # Unknown token
    continuing_subword_prefix="",
    end_of_word_suffix="",
    fuse_unk=False        # Keep unknown tokens separate
)

tokenizer = Tokenizer(model)
```

**Parameters**:
- `vocab`: Dict of token → id
- `merges`: List of merge rules `["a b", "ab c"]`
- `unk_token`: Token for unknown words
- `continuing_subword_prefix`: Prefix for subwords (empty for GPT-2)
- `end_of_word_suffix`: Suffix for last subword (empty for GPT-2)

### WordPiece Model

```python
from tokenizers.models import WordPiece

model = WordPiece(
    vocab=None,
    unk_token="[UNK]",
    max_input_chars_per_word=100,  # Max word length
    continuing_subword_prefix="##"  # BERT-style prefix
)

tokenizer = Tokenizer(model)
```

**Key difference**: Uses `##` prefix for continuing subwords.

### Unigram Model

```python
from tokenizers.models import Unigram

model = Unigram(
    vocab=None,  # List of (token, score) tuples
    unk_id=0,    # ID for unknown token
    byte_fallback=False  # Fall back to bytes if no match
)

tokenizer = Tokenizer(model)
```

**Probabilistic**: Selects tokenization with highest probability.

### WordLevel Model

```python
from tokenizers.models import WordLevel

# Simple word-to-ID mapping (no subwords)
model = WordLevel(
    vocab=None,
    unk_token="[UNK]"
)

tokenizer = Tokenizer(model)
```

**Warning**: Requires huge vocabulary (one token per word).

## Post-processors

Add special tokens and format output.

### Template processing

**BERT-style** (`[CLS] sentence [SEP]`):
```python
from tokenizers.processors import TemplateProcessing

tokenizer.post_processor = TemplateProcessing(
    single="[CLS] $A [SEP]",
    pair="[CLS] $A [SEP] $B [SEP]",
    special_tokens=[
        ("[CLS]", 101),
        ("[SEP]", 102),
    ],
)

# Single sentence
output = tokenizer.encode("Hello world")
# [101, ..., 102]  ([CLS] hello world [SEP])

# Sentence pair
output = tokenizer.encode("Hello", "world")
# [101, ..., 102, ..., 102]  ([CLS] hello [SEP] world [SEP])
```

**GPT-2 style** (`sentence <|endoftext|>`):
```python
tokenizer.post_processor = TemplateProcessing(
    single="$A <|endoftext|>",
    special_tokens=[
        ("<|endoftext|>", 50256),
    ],
)
```

**RoBERTa style** (`<s> sentence </s>`):
```python
tokenizer.post_processor = TemplateProcessing(
    single="<s> $A </s>",
    pair="<s> $A </s> </s> $B </s>",
    special_tokens=[
        ("<s>", 0),
        ("</s>", 2),
    ],
)
```

**T5 style** (no special tokens):
```python
# T5 doesn't add special tokens via post-processor
tokenizer.post_processor = None
```

### RobertaProcessing

```python
from tokenizers.processors import RobertaProcessing

tokenizer.post_processor = RobertaProcessing(
    sep=("</s>", 2),
    cls=("<s>", 0),
    add_prefix_space=True,  # Add space before first token
    trim_offsets=True       # Trim leading space from offsets
)
```

### ByteLevelProcessing

```python
from tokenizers.processors import ByteLevel as ByteLevelProcessing

tokenizer.post_processor = ByteLevelProcessing(
    trim_offsets=True  # Remove Ġ from offsets
)
```

## Decoders

Convert token IDs back to text.

### ByteLevel decoder

```python
from tokenizers.decoders import ByteLevel

tokenizer.decoder = ByteLevel()

# Handles byte-level tokens
# ["ĠHello", "Ġworld"] → "Hello world"
```

### WordPiece decoder

```python
from tokenizers.decoders import WordPiece

tokenizer.decoder = WordPiece(prefix="##")

# Removes ## prefix and concatenates
# ["token", "##ization"] → "tokenization"
```

### Metaspace decoder

```python
from tokenizers.decoders import Metaspace

tokenizer.decoder = Metaspace(replacement="▁", add_prefix_space=True)

# Converts ▁ back to spaces
# ["▁Hello", "▁world"] → "Hello world"
```

### BPEDecoder

```python
from tokenizers.decoders import BPEDecoder

tokenizer.decoder = BPEDecoder(suffix="</w>")

# Removes suffix and concatenates
# ["token", "ization</w>"] → "tokenization"
```

### Sequence decoder

```python
from tokenizers.decoders import Sequence, ByteLevel, Strip

tokenizer.decoder = Sequence([
    ByteLevel(),      # Decode byte-level first
    Strip(' ', 1, 1)  # Strip leading/trailing spaces
])
```

## Complete pipeline examples

### BERT tokenizer

```python
from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.normalizers import BertNormalizer
from tokenizers.pre_tokenizers import BertPreTokenizer
from tokenizers.processors import TemplateProcessing
from tokenizers.decoders import WordPiece as WordPieceDecoder

# Model
tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))

# Normalization
tokenizer.normalizer = BertNormalizer(lowercase=True)

# Pre-tokenization
tokenizer.pre_tokenizer = BertPreTokenizer()

# Post-processing
tokenizer.post_processor = TemplateProcessing(
    single="[CLS] $A [SEP]",
    pair="[CLS] $A [SEP] $B [SEP]",
    special_tokens=[("[CLS]", 101), ("[SEP]", 102)],
)

# Decoder
tokenizer.decoder = WordPieceDecoder(prefix="##")

# Enable padding
tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")

# Enable truncation
tokenizer.enable_truncation(max_length=512)
```

### GPT-2 tokenizer

```python
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.normalizers import NFC
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.processors import TemplateProcessing

# Model
tokenizer = Tokenizer(BPE())

# Normalization (minimal)
tokenizer.normalizer = NFC()

# Byte-level pre-tokenization
tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)

# Post-processing
tokenizer.post_processor = TemplateProcessing(
    single="$A <|endoftext|>",
    special_tokens=[("<|endoftext|>", 50256)],
)

# Byte-level decoder
tokenizer.decoder = ByteLevelDecoder()
```

### T5 tokenizer (SentencePiece-style)

```python
from tokenizers import Tokenizer
from tokenizers.models import Unigram
from tokenizers.normalizers import NFKC
from tokenizers.pre_tokenizers import Metaspace
from tokenizers.decoders import Metaspace as MetaspaceDecoder

# Model
tokenizer = Tokenizer(Unigram())

# Normalization
tokenizer.normalizer = NFKC()

# Metaspace pre-tokenization
tokenizer.pre_tokenizer = Metaspace(replacement="▁", add_prefix_space=True)

# No post-processing (T5 doesn't add CLS/SEP)
tokenizer.post_processor = None

# Metaspace decoder
tokenizer.decoder = MetaspaceDecoder(replacement="▁", add_prefix_space=True)
```

## Alignment tracking

Track token positions in original text.

### Basic alignment

```python
text = "Hello, world!"
output = tokenizer.encode(text)

for token, (start, end) in zip(output.tokens, output.offsets):
    print(f"{token:10s} → [{start:2d}, {end:2d}): {text[start:end]!r}")

# Output:
# [CLS]      → [ 0,  0): ''
# hello      → [ 0,  5): 'Hello'
# ,          → [ 5,  6): ','
# world      → [ 7, 12): 'world'
# !          → [12, 13): '!'
# [SEP]      → [ 0,  0): ''
```

### Word-level alignment

```python
# Get word_ids (which word each token belongs to)
encoding = tokenizer.encode("Hello world")
word_ids = encoding.word_ids

print(word_ids)
# [None, 0, 0, 1, None]
# None = special token, 0 = first word, 1 = second word
```

**Use case**: Token classification (NER)
```python
# Align predictions to words
predictions = ["O", "B-PER", "I-PER", "O", "O"]
word_predictions = {}

for token_idx, word_idx in enumerate(encoding.word_ids):
    if word_idx is not None and word_idx not in word_predictions:
        word_predictions[word_idx] = predictions[token_idx]

print(word_predictions)
# {0: "B-PER", 1: "O"}  # First word is PERSON, second is OTHER
```

### Span alignment

```python
# Find token span for character span
text = "Machine learning is awesome"
char_start, char_end = 8, 16  # "learning"

encoding = tokenizer.encode(text)

# Find token span
token_start = encoding.char_to_token(char_start)
token_end = encoding.char_to_token(char_end - 1) + 1

print(f"Tokens {token_start}:{token_end} = {encoding.tokens[token_start:token_end]}")
# Tokens 2:3 = ['learning']
```

**Use case**: Question answering (extract answer span)

## Custom components

### Custom normalizer

```python
from tokenizers import NormalizedString, Normalizer

class CustomNormalizer:
    def normalize(self, normalized: NormalizedString):
        # Custom normalization logic
        normalized.lowercase()
        normalized.replace("  ", " ")  # Replace double spaces

# Use custom normalizer
tokenizer.normalizer = CustomNormalizer()
```

### Custom pre-tokenizer

```python
from tokenizers import PreTokenizedString

class CustomPreTokenizer:
    def pre_tokenize(self, pretok: PreTokenizedString):
        # Custom pre-tokenization logic
        pretok.split(lambda i, char: char.isspace())

tokenizer.pre_tokenizer = CustomPreTokenizer()
```

## Troubleshooting

### Issue: Misaligned offsets

**Symptom**: Offsets don't match original text
```python
text = "  hello"  # Leading spaces
offsets = [(0, 5)]  # Expects "  hel"
```

**Solution**: Check normalization strips spaces
```python
# Preserve offsets
tokenizer.normalizer = Sequence([
    Strip(),  # This changes offsets!
])

# Use trim_offsets in post-processor instead
tokenizer.post_processor = ByteLevelProcessing(trim_offsets=True)
```

### Issue: Special tokens not added

**Symptom**: No [CLS] or [SEP] in output

**Solution**: Check post-processor is set
```python
tokenizer.post_processor = TemplateProcessing(
    single="[CLS] $A [SEP]",
    special_tokens=[("[CLS]", 101), ("[SEP]", 102)],
)
```

### Issue: Incorrect decoding

**Symptom**: Decoded text has ## or ▁

**Solution**: Set correct decoder
```python
# For WordPiece
tokenizer.decoder = WordPieceDecoder(prefix="##")

# For SentencePiece
tokenizer.decoder = MetaspaceDecoder(replacement="▁")
```

## Best practices

1. **Match pipeline to model architecture**:
   - BERT → BertNormalizer + BertPreTokenizer + WordPiece
   - GPT-2 → NFC + ByteLevel + BPE
   - T5 → NFKC + Metaspace + Unigram

2. **Test pipeline on sample inputs**:
   - Check normalization doesn't over-normalize
   - Verify pre-tokenization splits correctly
   - Ensure decoding reconstructs text

3. **Preserve alignment for downstream tasks**:
   - Use `trim_offsets` instead of stripping in normalizer
   - Test `char_to_token()` on sample spans

4. **Document your pipeline**:
   - Save complete tokenizer config
   - Document special tokens
   - Note any custom components
