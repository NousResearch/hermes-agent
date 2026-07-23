# TRL Fine-Tuning Reference

TRL (Transformer Reinforcement Learning) provides post-training methods for aligning language models: SFT, DPO, PPO, GRPO, and reward modeling.

## Installation

```bash
pip install trl transformers datasets peft accelerate
```

## Core Trainers

### SFTTrainer — Supervised Fine-Tuning

Instruction tuning with prompt-completion pairs.

```python
from trl import SFTTrainer, SFTConfig

training_args = SFTConfig(
    output_dir="model-sft",
    per_device_train_batch_size=4,
    num_train_epochs=1,
    learning_rate=2e-5,
)
trainer = SFTTrainer(model=model, args=training_args, train_dataset=dataset, tokenizer=tokenizer)
trainer.train()
```

### DPOTrainer — Direct Preference Optimization

Align model with preferences (chosen/rejected pairs) without reward model.

```python
from trl import DPOTrainer, DPOConfig

config = DPOConfig(output_dir="model-dpo", beta=0.1, max_length=1024)
trainer = DPOTrainer(model=model, args=config, train_dataset=preference_dataset, processing_class=tokenizer)
trainer.train()
```

### RewardTrainer — Reward Modeling

Train model to predict human preferences.

```python
from trl import RewardTrainer, RewardConfig

model = AutoModelForSequenceClassification.from_pretrained("sft-model", num_labels=1)
config = RewardConfig(output_dir="reward-model", per_device_train_batch_size=2, learning_rate=1e-5)
trainer = RewardTrainer(model=model, args=config, processing_class=tokenizer, train_dataset=dataset)
trainer.train()
```

### GRPOTrainer — Group Relative Policy Optimization

Memory-efficient online RL. Generate completions, compute rewards, optimize policy.

```python
from trl import GRPOTrainer, GRPOConfig

config = GRPOConfig(output_dir="model-grpo", num_generations=4, max_new_tokens=128)
trainer = GRPOTrainer(
    model="Qwen/Qwen2-0.5B-Instruct",
    reward_funcs=reward_function,  # Your reward function
    args=config,
    train_dataset=dataset
)
trainer.train()
```

## Full RLHF Pipeline

```
RLHF Training:
- [ ] Step 1: Supervised fine-tuning (SFT)
- [ ] Step 2: Train reward model
- [ ] Step 3: PPO reinforcement learning
- [ ] Step 4: Evaluate aligned model
```

CLI for PPO:
```bash
python -m trl.scripts.ppo \
    --model_name_or_path Qwen2.5-0.5B-SFT \
    --reward_model_path Qwen2.5-0.5B-Reward \
    --dataset_name trl-internal-testing/descriptiveness-sentiment-trl-style \
    --output_dir Qwen2.5-0.5B-PPO \
    --learning_rate 3e-6 \
    --per_device_train_batch_size 64 \
    --total_episodes 10000
```

## Method Selection Guide

| Method | Data Required | Use Case |
|--------|--------------|----------|
| **SFT** | Prompt-completion pairs | Basic instruction following |
| **DPO** | Chosen/rejected pairs | Simple preference alignment |
| **Reward Model** | Preference pairs | RLHF pipeline |
| **PPO** | Reward model | Maximum RL control |
| **GRPO** | Custom reward function | Memory-efficient online RL |

## Common Issues

### OOM during DPO
```python
config = DPOConfig(
    per_device_train_batch_size=1,
    max_length=512,
    gradient_accumulation_steps=8
)
model.gradient_checkpointing_enable()
```

### Poor alignment quality — tune beta
```python
config = DPOConfig(beta=0.5)   # More conservative (default: 0.1)
config = DPOConfig(beta=0.01)   # More aggressive
```

### Reward model not learning
```python
config = RewardConfig(learning_rate=1e-5, num_train_epochs=3)
```

### PPO training unstable
```python
config = PPOConfig(kl_coef=0.1, cliprange=0.1)
```

## VRAM Requirements (7B model)

| Method | VRAM |
|--------|------|
| SFT (LoRA) | ~16GB |
| DPO (LoRA) | ~24GB |
| PPO | ~40GB+ |
| GRPO (LoRA) | ~24GB |

## Resources

- Docs: https://huggingface.co/docs/trl/
- GitHub: https://github.com/huggingface/trl
- Papers: InstructGPT (2022), DPO (2023), GRPO (2024)