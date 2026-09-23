---
name: unsloth
description: "Unsloth: 2-5x faster LoRA/QLoRA fine-tuning, less VRAM."
version: 1.0.0
author: Orchestra Research
license: MIT
dependencies: [unsloth, torch, transformers, trl, datasets, peft]
platforms: [linux, macos]
metadata:
  hermes:
    tags: [Fine-Tuning, Unsloth, Fast Training, LoRA, QLoRA, Memory-Efficient, Optimization, Llama, Mistral, Gemma, Qwen]

---

# Unsloth Skill

Assistance with unsloth development, generated from official documentation.

## When to Use This Skill

This skill should be triggered when:
- Working with unsloth
- Asking about unsloth features or APIs
- Implementing unsloth solutions
- Debugging unsloth code
- Learning unsloth best practices

## Quick Reference

### Common Patterns

**Pattern 1:** Load a 4-bit model with `FastLanguageModel.from_pretrained`:

```python
from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Llama-3.3-70B-Instruct",
    max_seq_length = 2048,
    load_in_4bit = True,
)
```

**Pattern 2:** Attach LoRA adapters with `get_peft_model` (Unsloth gradient checkpointing):

```python
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16,
    use_gradient_checkpointing = "unsloth",
)
```

**Pattern 3:** Fine-tune with TRL `SFTTrainer`:

```python
from trl import SFTTrainer, SFTConfig
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    args = SFTConfig(output_dir = "outputs"),
)
trainer.train()
```

**Pattern 4:** Enable native 2x faster Unsloth inference before `generate`:

```python
FastLanguageModel.for_inference(model)
_ = model.generate(**inputs, max_new_tokens = 64)
```

**Pattern 5:** Merge and save 16-bit weights for vLLM:

```python
model.save_pretrained_merged("model", tokenizer, save_method = "merged_16bit")
```

**Pattern 6:** Export GGUF for Ollama / llama.cpp:

```python
model.save_pretrained_gguf("directory", tokenizer, quantization_method = "q4_k_m")
```

## Reference Files

This skill includes full documentation in `references/`:

- **llms-txt.md** - Llms-Txt documentation
- **llms-full.md** - Full Unsloth documentation
- **llms.md** - Documentation link index

Use `view` to read specific reference files when detailed information is needed.

## Working with This Skill

### For Beginners
Start with the getting_started or tutorials reference files for foundational concepts.

### For Specific Features
Use the appropriate category reference file (api, guides, etc.) for detailed information.

### For Code Examples
The quick reference section above contains common patterns extracted from the official docs.

## Resources

### references/
Organized documentation extracted from official sources. These files contain:
- Detailed explanations
- Code examples with language annotations
- Links to original documentation
- Table of contents for quick navigation

### scripts/
Add helper scripts here for common automation tasks.

### assets/
Add templates, boilerplate, or example projects here.

## Notes

- This skill was automatically generated from official documentation
- Reference files preserve the structure and examples from source docs
- Code examples include language detection for better syntax highlighting
- Quick reference patterns are extracted from common usage examples in the docs

## Updating

To refresh this skill with updated documentation:
1. Re-run the scraper with the same configuration
2. The skill will be rebuilt with the latest information

<!-- Trigger re-upload 1763621536 -->



