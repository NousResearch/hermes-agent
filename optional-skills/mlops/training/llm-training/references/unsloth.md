# Unsloth Reference

Unsloth provides 2-5x faster LoRA/QLoRA fine-tuning with 50% less VRAM usage. Optimized for Llama, Mistral, Gemma, and Qwen models.

## Installation

```bash
pip install unsloth
# Also install flash-attn for maximum speed
pip install flash-attn
```

## Core Features

- **2-5x faster** training compared to standard libraries
- **50% less VRAM** usage
- **Free** Kaggle and Google Colab notebooks
- **No quality loss** — mathematically equivalent to standard LoRA
- Supports Llama, Mistral, Gemma, Qwen, Phi, and more

## Quick Start

### Load a Model

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.2-1B",
    max_seq_length=2048,
    dtype=None,  # Auto-detect (float16, bfloat16, or float32)
    load_in_4bit=True,  # Use QLoRA (4-bit quantization)
)
```

### Add LoRA Adapters

```python
model = FastLanguageModel.get_peft_model(
    model,
    r=16,  # LoRA rank (8, 16, 32, 64)
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    gradient_checkpointing=True,
    use_gradient_checkpointing="unsloth",  # Unsloth's memory-efficient checkpointing
)
```

### Fine-Tune with SFTTrainer

```python
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig

training_args = SFTConfig(
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    warmup_steps=10,
    max_steps=60,
    learning_rate=2e-4,
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    logging_steps=10,
    optim="adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="linear",
    seed=3407,
    output_dir="outputs",
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    processing_class=tokenizer,
)
trainer.train()
```

## Supported Models

| Family | Variants |
|--------|----------|
| **Llama** | Llama 2, Llama 3, Llama 3.2, Llama 3.3 |
| **Mistral** | Mistral, Mistral-Nemo, Mixtral |
| **Gemma** | Gemma, Gemma 2 |
| **Qwen** | Qwen 2, Qwen 2.5 |
| **Phi** | Phi 3 |

## Memory Requirements

| Model Size | Standard QLoRA | Unsloth QLoRA |
|------------|----------------|---------------|
| 1B | ~8GB | ~4GB |
| 3B | ~16GB | ~8GB |
| 7B | ~24GB | ~12GB |
| 14B | ~48GB | ~24GB |

## Best Practices

- Use `load_in_4bit=True` for QLoRA (recommended)
- Set `gradient_checkpointing=True` for memory efficiency
- Use `optim="adamw_8bit"` to reduce optimizer memory
- Keep `max_seq_length` as short as possible for your task
- Use `unsloth` gradient checkpointing for 30% less memory than standard

## Resources

- GitHub: https://github.com/unsloth/unsloth
- Docs: https://unsloth.ai/docs
- Free notebooks: https://kaggle.com/organizations/unsloth