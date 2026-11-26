---
name: llm-training
description: "LLM fine-tuning: Axolotl (YAML), TRL (SFT/DPO/PPO/GRPO), Unsloth (fast LoRA/QLoRA)."
version: 1.0.0
author: Orchestra Research
license: MIT
dependencies: [torch, transformers, datasets, peft, accelerate, trl, axolotl, unsloth]
metadata:
  hermes:
    tags: [Fine-Tuning, LLM, LoRA, QLoRA, SFT, DPO, PPO, GRPO, RLHF, Axolotl, TRL, Unsloth, HuggingFace, DeepSpeed]

---

# llm-training: LLM Fine-Tuning

## Overview

This umbrella skill consolidates three fine-tuning tools for language models. Choose the right tool based on your workflow and constraints.

### Tool Comparison

| Tool | Strength | Methods | Config Style | Speed |
|------|----------|---------|-------------|-------|
| **Axolotl** | YAML-based, 100+ models, DeepSpeed, multimodal | LoRA, QLoRA, DPO, KTO, ORPO, GRPO | YAML declarative | Standard |
| **TRL** | Full RLHF pipeline, programmatic control | SFT, DPO, PPO, GRPO, Reward Modeling | Python API / CLI | Standard |
| **Unsloth** | 2-5x faster, 50% less VRAM | LoRA, QLoRA (Llama, Mistral, Gemma, Qwen) | Python API | 2-5x faster |

## When to Use Each

### Use Axolotl when:
- You prefer declarative YAML configs over code
- You need DeepSpeed ZeRO or FSDP distributed training
- You want multimodal fine-tuning support
- You need context parallelism or advanced multi-GPU strategies
- You want to save models in compressed format for vLLM/llmcompressor

### Use TRL when:
- You need the full RLHF pipeline (SFT → Reward Model → PPO)
- You prefer programmatic Python control
- You have preference data (chosen/rejected pairs) for DPO
- You need GRPO with custom reward functions
- You want built-in CLI scripts for quick training

### Use Unsloth when:
- You need maximum training speed (2-5x faster)
- You have limited VRAM (50% reduction)
- You are fine-tuning Llama, Mistral, Gemma, or Qwen
- You want free Kaggle/Google Colab support
- Memory efficiency is critical

## Quick Start

### Axolotl (YAML)
```bash
pip install axolotl
axolotl train config.yml
```

### TRL (Python)
```python
from trl import SFTTrainer, SFTConfig
trainer = SFTTrainer(model="Qwen/Qwen2.5-0.5B", train_dataset=data)
trainer.train()
```

### Unsloth (Python)
```python
from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained("unsloth/Llama-3.2-1B")
```

## Reference Files

This skill includes detailed documentation for each absorbed tool:

- **[references/axolotl.md](references/axolotl.md)** — Axolotl: YAML configs, LoRA/QLoRA, DPO/GRPO, DeepSpeed, FSDP, multimodal, compressed saving
- **[references/trl-fine-tuning.md](references/trl-fine-tuning.md)** — TRL: SFT, DPO, PPO, GRPO, reward modeling, RLHF pipelines, advanced configurations
- **[references/unsloth.md](references/unsloth.md)** — Unsloth: 2-5x faster LoRA/QLoRA, VRAM reduction, supported models, usage patterns

## Hardware Guidance

| Method | 7B Model VRAM | Recommendation |
|--------|--------------|----------------|
| SFT (LoRA) | ~16GB | Any modern GPU |
| DPO (LoRA) | ~24GB | A10G+ |
| PPO | ~40GB+ | A100/H100 |
| GRPO | ~24GB | A10G+ |
| Unsloth QLoRA | ~6-8GB | Consumer GPU |

Use LoRA/QLoRA for memory efficiency across all tools.

## Notes

- This skill was created by consolidating axolotl, trl-fine-tuning, and unsloth
- All absorbed tools are archived at `.archive/mlops/`
- Reference files contain the complete documentation from each absorbed skill
- Quick reference patterns from each tool are preserved in their respective reference files