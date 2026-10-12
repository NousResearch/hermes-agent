# Datasets

Complete guide to preference datasets for SimPO training.

## Dataset Format

### Required Fields

Preference datasets must contain:
```json
{
  "prompt": "User question or instruction",
  "chosen": "Better/preferred response",
  "rejected": "Worse/rejected response"
}
```

**Alternative field names** (auto-detected):
- `prompt` → `question`, `instruction`, `input`
- `chosen` → `response_chosen`, `winner`, `preferred`
- `rejected` → `response_rejected`, `loser`

### Example Entry

```json
{
  "prompt": "Explain quantum computing in simple terms.",
  "chosen": "Quantum computing uses quantum bits (qubits) that can exist in multiple states simultaneously through superposition. This allows quantum computers to process many possibilities at once, making them potentially much faster than classical computers for specific tasks like cryptography and optimization.",
  "rejected": "It's like regular computing but quantum."
}
```

## Popular Datasets

### 1. UltraFeedback (Recommended)

**HuggingFaceH4/ultrafeedback_binarized**:
- **Size**: 60K preference pairs
- **Quality**: High (GPT-4 annotations)
- **Domain**: General instruction following
- **Format**: Clean, ready-to-use

**Config**:
```yaml
dataset_mixer:
  HuggingFaceH4/ultrafeedback_binarized: 1.0
dataset_splits:
  - train_prefs
  - test_prefs
```

### 2. Argilla UltraFeedback (Cleaned)

**argilla/ultrafeedback-binarized-preferences-cleaned**:
- **Size**: 50K pairs (filtered)
- **Quality**: Very high (deduped, cleaned)
- **Domain**: General
- **Format**: Clean

**Config**:
```yaml
dataset_mixer:
  argilla/ultrafeedback-binarized-preferences-cleaned: 1.0
```

### 3. Distilabel Math

**argilla/distilabel-math-preference-dpo**:
- **Size**: 30K pairs
- **Quality**: High (GSM8K, MATH)
- **Domain**: Math reasoning
- **Format**: Math-specific

**Config**:
```yaml
dataset_mixer:
  argilla/distilabel-math-preference-dpo: 1.0
```

### 4. HelpSteer

**nvidia/HelpSteer**:
- **Size**: 38K samples
- **Quality**: High (human ratings)
- **Domain**: Helpfulness alignment
- **Format**: Multi-attribute ratings

**Config**:
```yaml
dataset_mixer:
  nvidia/HelpSteer: 1.0
```

### 5. Anthropic HH-RLHF

**Anthropic/hh-rlhf**:
- **Size**: 161K samples
- **Quality**: High (human preferences)
- **Domain**: Harmless + helpful
- **Format**: Conversational

**Config**:
```yaml
dataset_mixer:
  Anthropic/hh-rlhf: 1.0
```

## Dataset Mixing

### Multiple Datasets

**Equal mix**:
```yaml
dataset_mixer:
  HuggingFaceH4/ultrafeedback_binarized: 0.5
  Anthropic/hh-rlhf: 0.5
```

**Weighted mix**:
```yaml
dataset_mixer:
  HuggingFaceH4/ultrafeedback_binarized: 0.7
  argilla/distilabel-math-preference-dpo: 0.2
  nvidia/HelpSteer: 0.1
```

**Domain-specific emphasis**:
```yaml
# 80% general + 20% math
dataset_mixer:
  HuggingFaceH4/ultrafeedback_binarized: 0.8
  argilla/distilabel-math-preference-dpo: 0.2
```

## Data Quality

### Quality Indicators

**Good preference data**:
- ✅ Clear quality difference between chosen/rejected
- ✅ Diverse prompts
- ✅ Minimal noise/annotation errors
- ✅ Appropriate difficulty level

**Poor preference data**:
- ❌ Ambiguous preferences
- ❌ Repetitive prompts
- ❌ Annotation noise
- ❌ Too easy/hard prompts

### Quality Filtering

**Filter by length difference**:
```python
def filter_by_length(example):
    chosen_len = len(example['chosen'].split())
    rejected_len = len(example['rejected'].split())
    # Reject if chosen is much shorter (potential low-effort)
    return chosen_len >= rejected_len * 0.5

dataset = dataset.filter(filter_by_length)
```

**Filter by diversity**:
```python
seen_prompts = set()

def filter_duplicates(example):
    prompt = example['prompt']
    if prompt in seen_prompts:
        return False
    seen_prompts.add(prompt)
    return True

dataset = dataset.filter(filter_duplicates)
```

## Custom Dataset Creation

### Format 1: JSON Lines

**File** (`preferences.jsonl`):
```jsonl
{"prompt": "What is Python?", "chosen": "Python is a high-level programming language...", "rejected": "It's a snake."}
{"prompt": "Explain AI.", "chosen": "AI refers to systems that can...", "rejected": "It's computers that think."}
```

**Load**:
```yaml
dataset_mixer:
  json:
    data_files: preferences.jsonl
```

### Format 2: HuggingFace Dataset

**Create from dict**:
```python
from datasets import Dataset

data = {
    "prompt": ["What is Python?", "Explain AI."],
    "chosen": ["Python is...", "AI refers to..."],
    "rejected": ["It's a snake.", "It's computers..."]
}

dataset = Dataset.from_dict(data)
dataset.push_to_hub("username/my-preferences")
```

**Use in config**:
```yaml
dataset_mixer:
  username/my-preferences: 1.0
```

### Format 3: ChatML

**For conversational data**:
```json
{
  "prompt": [
    {"role": "user", "content": "What is quantum computing?"}
  ],
  "chosen": [
    {"role": "assistant", "content": "Quantum computing uses qubits..."}
  ],
  "rejected": [
    {"role": "assistant", "content": "It's like regular computing but quantum."}
  ]
}
```

**Apply chat template**:
```yaml
dataset_text_field: null  # Will apply chat template
```

## Synthetic Data Generation

### Using GPT-4

**Prompt template**:
```
Given the following question:
{prompt}

Generate two responses:
1. A high-quality, detailed response (chosen)
2. A low-quality, brief response (rejected)

Format as JSON with "chosen" and "rejected" fields.
```

**Example code**:
```python
import openai

def generate_pair(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{
            "role": "user",
            "content": f"Given: {prompt}\n\nGenerate chosen/rejected pair in JSON."
        }]
    )
    return json.loads(response.choices[0].message.content)

# Generate dataset
prompts = load_prompts()
dataset = [generate_pair(p) for p in prompts]
```

### Using Local Model

**With vLLM**:
```python
from vllm import LLM

llm = LLM(model="meta-llama/Meta-Llama-3-70B-Instruct")

def generate_variations(prompt):
    # Generate multiple completions
    outputs = llm.generate(
        [prompt] * 4,
        sampling_params={
            "temperature": 0.8,
            "top_p": 0.9,
            "max_tokens": 512
        }
    )

    # Select best/worst
    chosen = max(outputs, key=lambda x: len(x.outputs[0].text))
    rejected = min(outputs, key=lambda x: len(x.outputs[0].text))

    return {
        "prompt": prompt,
        "chosen": chosen.outputs[0].text,
        "rejected": rejected.outputs[0].text
    }
```

## Data Preprocessing

### Truncation

**Limit sequence length**:
```yaml
max_prompt_length: 512
max_completion_length: 512
max_length: 1024  # Total
```

**Implementation**:
```python
def truncate_example(example):
    tokenizer.truncation_side = "left"  # For prompts
    prompt_tokens = tokenizer(
        example['prompt'],
        max_length=512,
        truncation=True
    )

    tokenizer.truncation_side = "right"  # For completions
    chosen_tokens = tokenizer(
        example['chosen'],
        max_length=512,
        truncation=True
    )

    return {
        "prompt": tokenizer.decode(prompt_tokens['input_ids']),
        "chosen": tokenizer.decode(chosen_tokens['input_ids'])
    }

dataset = dataset.map(truncate_example)
```

### Deduplication

**Remove exact duplicates**:
```python
dataset = dataset.unique('prompt')
```

**Remove near-duplicates** (MinHash):
```python
from datasketch import MinHash, MinHashLSH

def deduplicate_lsh(dataset, threshold=0.8):
    lsh = MinHashLSH(threshold=threshold, num_perm=128)
    seen = []

    for i, example in enumerate(dataset):
        m = MinHash(num_perm=128)
        for word in example['prompt'].split():
            m.update(word.encode('utf8'))

        if not lsh.query(m):
            lsh.insert(i, m)
            seen.append(example)

    return Dataset.from_list(seen)

dataset = deduplicate_lsh(dataset)
```

## Data Augmentation

### Paraphrasing Prompts

```python
def paraphrase_prompt(example):
    # Use paraphrasing model
    paraphrased = paraphrase_model(example['prompt'])

    return [
        example,  # Original
        {
            "prompt": paraphrased,
            "chosen": example['chosen'],
            "rejected": example['rejected']
        }
    ]

dataset = dataset.map(paraphrase_prompt, batched=False, remove_columns=[])
```

### Difficulty Balancing

**Mix easy/medium/hard**:
```python
def categorize_difficulty(example):
    prompt_len = len(example['prompt'].split())
    if prompt_len < 20:
        return "easy"
    elif prompt_len < 50:
        return "medium"
    else:
        return "hard"

dataset = dataset.map(lambda x: {"difficulty": categorize_difficulty(x)})

# Sample balanced dataset
easy = dataset.filter(lambda x: x['difficulty'] == 'easy').shuffle().select(range(1000))
medium = dataset.filter(lambda x: x['difficulty'] == 'medium').shuffle().select(range(1000))
hard = dataset.filter(lambda x: x['difficulty'] == 'hard').shuffle().select(range(1000))

balanced = concatenate_datasets([easy, medium, hard]).shuffle()
```

## Dataset Statistics

### Compute Stats

```python
def compute_stats(dataset):
    prompt_lens = [len(x['prompt'].split()) for x in dataset]
    chosen_lens = [len(x['chosen'].split()) for x in dataset]
    rejected_lens = [len(x['rejected'].split()) for x in dataset]

    print(f"Dataset size: {len(dataset)}")
    print(f"Avg prompt length: {np.mean(prompt_lens):.1f} words")
    print(f"Avg chosen length: {np.mean(chosen_lens):.1f} words")
    print(f"Avg rejected length: {np.mean(rejected_lens):.1f} words")
    print(f"Chosen > Rejected: {sum(c > r for c, r in zip(chosen_lens, rejected_lens)) / len(dataset):.1%}")

compute_stats(dataset)
```

**Expected output**:
```
Dataset size: 50000
Avg prompt length: 45.2 words
Avg chosen length: 180.5 words
Avg rejected length: 120.3 words
Chosen > Rejected: 85.2%
```

## Best Practices

### 1. Data Quality Over Quantity

- **Prefer**: 10K high-quality pairs
- **Over**: 100K noisy pairs

### 2. Clear Preference Signals

- Chosen should be noticeably better
- Avoid marginal differences
- Remove ambiguous pairs

### 3. Domain Matching

- Match dataset domain to target use case
- Mix datasets for broader coverage
- Include safety-filtered data

### 4. Validate Before Training

```python
# Sample 10 random examples
samples = dataset.shuffle().select(range(10))

for ex in samples:
    print(f"Prompt: {ex['prompt']}")
    print(f"Chosen: {ex['chosen'][:100]}...")
    print(f"Rejected: {ex['rejected'][:100]}...")
    print(f"Preference clear: {'✓' if len(ex['chosen']) > len(ex['rejected']) else '?'}")
    print()
```

## References

- HuggingFace Datasets: https://huggingface.co/datasets
- Alignment Handbook: https://github.com/huggingface/alignment-handbook
- UltraFeedback: https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized
