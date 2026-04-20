# GRPO (Group Relative Policy Optimization) — Deep Guide

Expert-level patterns, critical insights, and production-ready workflows for fine-tuning language models with custom reward functions using TRL's `GRPOTrainer`. This is the deep reference for the GRPO workflow summarized in the main skill.

## When to use GRPO

Use GRPO when you need to:
- **Enforce specific output formats** (XML tags, JSON, structured reasoning)
- **Teach verifiable tasks** with objective correctness metrics (math, coding, fact-checking)
- **Improve reasoning capabilities** by rewarding chain-of-thought patterns
- **Align models to domain-specific behaviors** without labeled preference data
- **Optimize for multiple objectives** simultaneously (format + correctness + style)

**Do NOT use GRPO for:**
- Simple supervised fine-tuning tasks → use SFT
- Tasks without clear reward signals
- When you already have high-quality preference pairs → use DPO/PPO

## Core concepts

### 1. GRPO algorithm fundamentals

**Key mechanism:**
- Generates **multiple completions** per prompt (group size: 4–16)
- Compares completions within each group using reward functions
- Updates policy to favor higher-rewarded responses relative to the group

**Critical differences from PPO:**
- No separate reward model needed
- More sample-efficient (learns from within-group comparisons)
- Simpler to implement and debug

**Mathematical intuition:**
```
For each prompt p:
  1. Generate N completions: {c₁, c₂, ..., cₙ}
  2. Compute rewards: {r₁, r₂, ..., rₙ}
  3. Learn to increase probability of high-reward completions
     relative to low-reward ones in the same group
```

### 2. Reward function design philosophy

**Golden rules:**
1. **Compose multiple reward functions** — each handles one aspect (format, correctness, style)
2. **Scale rewards appropriately** — higher weight = stronger signal
3. **Use incremental rewards** — partial credit for partial compliance
4. **Test rewards independently** — debug each reward function in isolation

**Reward function types:**

| Type | Use Case | Example Weight |
|------|----------|----------------|
| **Correctness** | Verifiable tasks (math, code) | 2.0 (highest) |
| **Format** | Strict structure enforcement | 0.5–1.0 |
| **Length** | Encourage verbosity/conciseness | 0.1–0.5 |
| **Style** | Penalize unwanted patterns | −0.5 to 0.5 |

## Implementation workflow

### Step 1: Dataset preparation

**Critical requirements:**
- Prompts in chat format (list of dicts with `role` and `content`)
- Include system prompts to set expectations
- For verifiable tasks, include ground truth answers as additional columns

```python
from datasets import load_dataset, Dataset

SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
[Your step-by-step thinking]
</reasoning>
<answer>
[Final answer]
</answer>
"""

def prepare_dataset(raw_data):
    """Transform raw data into GRPO-compatible format.

    Returns: Dataset with columns:
    - 'prompt': List[Dict] with role/content (system + user messages)
    - 'answer': str (ground truth, optional but recommended)
    """
    return raw_data.map(lambda x: {
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': x['question']}
        ],
        'answer': extract_answer(x['raw_answer'])
    })
```

**Pro tips:**
- Use one-shot or few-shot examples in the system prompt for complex formats
- Keep prompts concise (max_prompt_length: 256–512 tokens)
- Validate data quality before training (garbage in = garbage out)

### Step 2: Reward function implementation

**Template structure:**
```python
def reward_function_name(
    prompts,        # List[List[Dict]]: Original prompts
    completions,    # List[List[Dict]]: Model generations
    answer=None,    # Optional: Ground truth from dataset
    **kwargs        # Additional dataset columns
) -> list[float]:
    """Evaluate completions and return rewards (one per completion)."""
    responses = [comp[0]['content'] for comp in completions]
    rewards = []
    for response in responses:
        score = compute_score(response)
        rewards.append(score)
    return rewards
```

**Example 1: correctness reward (math/coding)**
```python
def correctness_reward(prompts, completions, answer, **kwargs):
    """Reward correct answers with high score."""
    responses = [comp[0]['content'] for comp in completions]
    extracted = [extract_final_answer(r) for r in responses]
    return [2.0 if ans == gt else 0.0
            for ans, gt in zip(extracted, answer)]
```

**Example 2: format reward (structured output)**
```python
import re

def format_reward(completions, **kwargs):
    """Reward XML-like structured format."""
    pattern = r'<reasoning>.*?</reasoning>\s*<answer>.*?</answer>'
    responses = [comp[0]['content'] for comp in completions]
    return [1.0 if re.search(pattern, r, re.DOTALL) else 0.0
            for r in responses]
```

**Example 3: incremental format reward (partial credit)**
```python
def incremental_format_reward(completions, **kwargs):
    """Award partial credit for format compliance."""
    responses = [comp[0]['content'] for comp in completions]
    rewards = []

    for r in responses:
        score = 0.0
        if '<reasoning>' in r:  score += 0.25
        if '</reasoning>' in r: score += 0.25
        if '<answer>' in r:     score += 0.25
        if '</answer>' in r:    score += 0.25
        # Penalize extra text after closing tag
        if r.count('</answer>') == 1:
            extra_text = r.split('</answer>')[-1].strip()
            score -= len(extra_text) * 0.001
        rewards.append(score)

    return rewards
```

**Critical insight:** Combine 3–5 reward functions for robust training. Order matters less than diversity of signals.

### Step 3: Training configuration

**Memory-optimized config (small GPU)**
```python
from trl import GRPOConfig

training_args = GRPOConfig(
    output_dir="outputs/grpo-model",

    # Learning rate
    learning_rate=5e-6,          # Lower = more stable
    adam_beta1=0.9,
    adam_beta2=0.99,
    weight_decay=0.1,
    warmup_ratio=0.1,
    lr_scheduler_type='cosine',

    # Batch settings
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,  # Effective batch = 4

    # GRPO-specific
    num_generations=8,            # Group size: 8–16 recommended
    max_prompt_length=256,
    max_completion_length=512,

    # Training duration
    num_train_epochs=1,
    max_steps=None,

    # Optimization
    bf16=True,                    # Faster on A100/H100
    optim="adamw_8bit",          # Memory-efficient optimizer
    max_grad_norm=0.1,

    # Logging
    logging_steps=1,
    save_steps=100,
    report_to="wandb",
)
```

**High-performance config (large GPU)**
```python
training_args = GRPOConfig(
    output_dir="outputs/grpo-model",
    learning_rate=1e-5,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    num_generations=16,           # Larger groups = better signal
    max_prompt_length=512,
    max_completion_length=1024,
    num_train_epochs=1,
    bf16=True,
    use_vllm=True,                # Fast generation with vLLM
    logging_steps=10,
)
```

**Critical hyperparameters:**

| Parameter | Impact | Tuning Advice |
|-----------|--------|---------------|
| `num_generations` | Group size for comparison | Start 8, increase to 16 if GPU allows |
| `learning_rate` | Convergence speed/stability | 5e-6 (safe), 1e-5 (faster, riskier) |
| `max_completion_length` | Output verbosity | Match your task (512 reasoning, 256 short answers) |
| `gradient_accumulation_steps` | Effective batch size | Increase if GPU memory limited |

### Step 4: Model setup and training

**Standard setup (Transformers + TRL)**
```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from trl import GRPOTrainer

model_name = "Qwen/Qwen2.5-1.5B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",  # 2–3× faster
    device_map="auto",
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Optional: LoRA for parameter-efficient training
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    task_type="CAUSAL_LM",
    lora_dropout=0.05,
)

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[
        incremental_format_reward,
        format_reward,
        correctness_reward,
    ],
    args=training_args,
    train_dataset=dataset,
    peft_config=peft_config,   # Remove for full fine-tuning
)

trainer.train()
trainer.save_model("final_model")
```

**Unsloth setup (2–3× faster)**
```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="google/gemma-3-1b-it",
    max_seq_length=1024,
    load_in_4bit=True,
    fast_inference=True,
    max_lora_rank=32,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=32,
    use_gradient_checkpointing="unsloth",
)

# Rest is identical to the standard setup
trainer = GRPOTrainer(model=model, ...)
trainer.train()
```

## Critical training insights

### 1. Loss behavior (EXPECTED pattern)
- **Loss starts near 0 and INCREASES during training** — this is CORRECT
- Loss measures KL divergence from initial policy; the model is learning (diverging from original behavior to optimize rewards)
- **Monitor reward metrics, not loss, for progress**

### 2. Reward tracking

Key metrics to watch:
- `reward` — average across all completions
- `reward_std` — diversity within groups (should remain > 0)
- `kl` — KL divergence from reference (should grow moderately)

**Healthy pattern:**
```
Step   Reward    Reward_Std   KL
100    0.5       0.3          0.02
200    0.8       0.25         0.05
300    1.2       0.2          0.08  ← Good progression
400    1.5       0.15         0.12
```

**Warning signs:**
- `reward_std` → 0 (model collapsing to a single response)
- `kl` exploding (> 0.5) — diverging too much, reduce LR
- Reward stuck — reward functions too harsh or model capacity issue

### 3. Common pitfalls and solutions

| Problem | Symptom | Solution |
|---------|---------|----------|
| **Mode collapse** | All completions identical | Increase `num_generations`, add diversity penalty |
| **No learning** | Flat rewards | Check reward function logic, increase LR |
| **OOM errors** | GPU memory exceeded | Reduce `num_generations`, enable gradient checkpointing |
| **Slow training** | < 1 it/s | Enable `use_vllm=True`, use Unsloth, reduce seq length |
| **Format ignored** | Model doesn't follow structure | Increase format reward weight, add incremental rewards |

## Advanced patterns

### 1. Multi-stage training

For complex tasks, train in stages:

```python
# Stage 1: Format compliance
trainer_stage1 = GRPOTrainer(
    model=model,
    reward_funcs=[incremental_format_reward, format_reward],
    ...
)
trainer_stage1.train()

# Stage 2: Correctness
trainer_stage2 = GRPOTrainer(
    model=model,
    reward_funcs=[format_reward, correctness_reward],
    ...
)
trainer_stage2.train()
```

### 2. Adaptive reward scaling

```python
class AdaptiveReward:
    def __init__(self, base_reward_func, initial_weight=1.0):
        self.func = base_reward_func
        self.weight = initial_weight

    def __call__(self, *args, **kwargs):
        rewards = self.func(*args, **kwargs)
        return [r * self.weight for r in rewards]

    def adjust_weight(self, success_rate):
        """Increase weight if model struggling, decrease if succeeding."""
        if success_rate < 0.3:
            self.weight *= 1.2
        elif success_rate > 0.8:
            self.weight *= 0.9
```

### 3. Custom dataset integration

```python
def load_custom_knowledge_base(csv_path):
    import pandas as pd
    df = pd.read_csv(csv_path)
    return Dataset.from_pandas(df).map(lambda x: {
        'prompt': [
            {'role': 'system', 'content': CUSTOM_SYSTEM_PROMPT},
            {'role': 'user', 'content': x['question']}
        ],
        'answer': x['expert_answer']
    })
```

## Deployment and inference

### Save and merge LoRA
```python
if hasattr(trainer.model, 'merge_and_unload'):
    merged_model = trainer.model.merge_and_unload()
    merged_model.save_pretrained("production_model")
    tokenizer.save_pretrained("production_model")
```

### Inference
```python
from transformers import pipeline

generator = pipeline("text-generation", model="production_model", tokenizer=tokenizer)

result = generator(
    [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': "What is 15 + 27?"},
    ],
    max_new_tokens=256,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
)
print(result[0]['generated_text'])
```

## Best practices checklist

**Before training:**
- [ ] Validate dataset format (prompts as List[Dict])
- [ ] Test reward functions on sample data
- [ ] Calculate expected `max_prompt_length` from data
- [ ] Choose `num_generations` based on GPU memory
- [ ] Set up logging (wandb recommended)

**During training:**
- [ ] Monitor reward progression (should increase)
- [ ] Check `reward_std` (should stay > 0.1)
- [ ] Watch for OOM errors (reduce batch size if needed)
- [ ] Sample generations every 50–100 steps
- [ ] Validate format compliance on holdout set

**After training:**
- [ ] Merge LoRA weights if using PEFT
- [ ] Test on diverse prompts
- [ ] Compare to baseline model
- [ ] Document reward weights and hyperparameters
- [ ] Save reproducibility config

## Troubleshooting

### Debugging workflow
1. **Isolate reward functions** — test each independently
2. **Check data distribution** — ensure diversity in prompts
3. **Reduce complexity** — start with single reward, add gradually
4. **Monitor generations** — print samples every N steps
5. **Validate extraction logic** — ensure answer parsing works

### Quick debug reward
```python
def debug_reward(completions, **kwargs):
    responses = [comp[0]['content'] for comp in completions]
    for i, r in enumerate(responses[:2]):
        print(f"Response {i}: {r[:200]}...")
    return [1.0] * len(responses)

# Test without training
trainer = GRPOTrainer(..., reward_funcs=[debug_reward])
trainer.generate_completions(dataset[:1])
```

## Template

A production-ready training script lives at **`../templates/basic_grpo_training.py`**. It uses Qwen 2.5-1.5B-Instruct with LoRA and three reward functions (incremental format, strict format, correctness) on GSM8K. Copy and adapt:
1. `get_dataset()` — swap in your data loader
2. Reward functions — tune to your task
3. `SYSTEM_PROMPT` — match your output format
4. `GRPOConfig` — adjust hyperparameters for your GPU

## References and resources

- TRL GRPO Trainer: https://huggingface.co/docs/trl/grpo_trainer
- GRPO paper (DeepSeek): https://arxiv.org/abs/2402.03300
- DeepSeek R1 paper: https://arxiv.org/abs/2501.12948
- Open R1 implementation: https://github.com/huggingface/open-r1
- TRL examples: https://github.com/huggingface/trl/tree/main/examples
- Unsloth (faster training): https://docs.unsloth.ai/

## Critical reminders

- **Loss goes UP during training** — this is normal (it's KL divergence)
- **Use 3–5 reward functions** — single rewards often fail
- **Test rewards before training** — debug each function independently
- **Monitor `reward_std`** — should stay > 0.1 (avoid mode collapse)
- **Start with `num_generations=4–8`** — scale up if GPU allows
