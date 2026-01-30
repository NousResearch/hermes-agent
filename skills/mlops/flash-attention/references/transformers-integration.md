# HuggingFace Transformers Integration

## Contents
- Enabling Flash Attention in Transformers
- Supported model architectures
- Configuration examples
- Performance comparisons
- Troubleshooting model-specific issues

## Enabling Flash Attention in Transformers

HuggingFace Transformers (v4.36+) supports Flash Attention 2 natively.

**Simple enable for any supported model**:
```python
from transformers import AutoModel

model = AutoModel.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    attn_implementation="flash_attention_2",
    torch_dtype=torch.float16,
    device_map="auto"
)
```

**Install requirements**:
```bash
pip install transformers>=4.36
pip install flash-attn --no-build-isolation
```

## Supported model architectures

As of Transformers 4.40:

**Fully supported**:
- Llama / Llama 2 / Llama 3
- Mistral / Mixtral
- Falcon
- GPT-NeoX
- Phi / Phi-2 / Phi-3
- Qwen / Qwen2
- Gemma
- Starcoder2
- GPT-J
- OPT
- BLOOM

**Partially supported** (encoder-decoder):
- BART
- T5 / Flan-T5
- Whisper

**Check support**:
```python
from transformers import AutoConfig

config = AutoConfig.from_pretrained("model-name")
print(config._attn_implementation_internal)
# 'flash_attention_2' if supported
```

## Configuration examples

### Llama 2 with Flash Attention

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_id = "meta-llama/Llama-2-7b-hf"

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    attn_implementation="flash_attention_2",
    torch_dtype=torch.float16,
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(model_id)

# Generate
inputs = tokenizer("Once upon a time", return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_length=100)
print(tokenizer.decode(outputs[0]))
```

### Mistral with Flash Attention for long context

```python
from transformers import AutoModelForCausalLM
import torch

model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-v0.1",
    attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16,  # Better for long context
    device_map="auto",
    max_position_embeddings=32768  # Extended context
)

# Process long document (32K tokens)
long_text = "..." * 10000
inputs = tokenizer(long_text, return_tensors="pt", truncation=False).to("cuda")
outputs = model.generate(**inputs, max_new_tokens=512)
```

### Fine-tuning with Flash Attention

```python
from transformers import Trainer, TrainingArguments
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    attn_implementation="flash_attention_2",
    torch_dtype=torch.float16
)

training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    fp16=True,  # Must match model dtype
    optim="adamw_torch_fused"  # Fast optimizer
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset
)

trainer.train()
```

### Multi-GPU training

```python
from transformers import AutoModelForCausalLM
import torch

# Model parallelism with Flash Attention
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-13b-hf",
    attn_implementation="flash_attention_2",
    torch_dtype=torch.float16,
    device_map="auto",  # Automatic multi-GPU placement
    max_memory={0: "20GB", 1: "20GB"}  # Limit per GPU
)
```

## Performance comparisons

### Memory usage (Llama 2 7B, batch=1)

| Sequence Length | Standard Attention | Flash Attention 2 | Reduction |
|-----------------|-------------------|-------------------|-----------|
| 512 | 1.2 GB | 0.9 GB | 25% |
| 2048 | 3.8 GB | 1.4 GB | 63% |
| 8192 | 14.2 GB | 3.2 GB | 77% |
| 32768 | OOM (>24GB) | 10.8 GB | Fits! |

### Speed (tokens/sec, A100 80GB)

| Model | Standard | Flash Attn 2 | Speedup |
|-------|----------|--------------|---------|
| Llama 2 7B (seq=2048) | 42 | 118 | 2.8x |
| Llama 2 13B (seq=4096) | 18 | 52 | 2.9x |
| Llama 2 70B (seq=2048) | 4 | 11 | 2.75x |

### Training throughput (samples/sec)

| Model | Batch Size | Standard | Flash Attn 2 | Speedup |
|-------|------------|----------|--------------|---------|
| Llama 2 7B | 4 | 1.2 | 3.1 | 2.6x |
| Llama 2 7B | 8 | 2.1 | 5.8 | 2.8x |
| Llama 2 13B | 2 | 0.6 | 1.7 | 2.8x |

## Troubleshooting model-specific issues

### Issue: Model doesn't support Flash Attention

Check support list above. If not supported, use PyTorch SDPA as fallback:

```python
model = AutoModelForCausalLM.from_pretrained(
    "model-name",
    attn_implementation="sdpa",  # PyTorch native (still faster)
    torch_dtype=torch.float16
)
```

### Issue: CUDA out of memory during loading

Reduce memory footprint:

```python
model = AutoModelForCausalLM.from_pretrained(
    "model-name",
    attn_implementation="flash_attention_2",
    torch_dtype=torch.float16,
    device_map="auto",
    max_memory={0: "18GB"},  # Reserve memory for KV cache
    low_cpu_mem_usage=True
)
```

### Issue: Slower inference than expected

Ensure dtype matches:

```python
# Model and inputs must both be float16/bfloat16
model = model.to(torch.float16)
inputs = tokenizer(..., return_tensors="pt").to("cuda")
inputs = {k: v.to(torch.float16) if v.dtype == torch.float32 else v
          for k, v in inputs.items()}
```

### Issue: Different outputs vs standard attention

Flash Attention is numerically equivalent but uses different computation order. Small differences (<1e-3) are normal:

```python
# Compare outputs
model_standard = AutoModelForCausalLM.from_pretrained("model-name", torch_dtype=torch.float16)
model_flash = AutoModelForCausalLM.from_pretrained(
    "model-name",
    attn_implementation="flash_attention_2",
    torch_dtype=torch.float16
)

inputs = tokenizer("Test", return_tensors="pt").to("cuda")

with torch.no_grad():
    out_standard = model_standard(**inputs).logits
    out_flash = model_flash(**inputs).logits

diff = (out_standard - out_flash).abs().max()
print(f"Max diff: {diff:.6f}")  # Should be ~1e-3 to 1e-4
```

### Issue: ImportError during model loading

Install flash-attn:
```bash
pip install flash-attn --no-build-isolation
```

Or disable Flash Attention:
```python
model = AutoModelForCausalLM.from_pretrained(
    "model-name",
    attn_implementation="eager",  # Standard PyTorch
    torch_dtype=torch.float16
)
```

## Best practices

1. **Always use float16/bfloat16** with Flash Attention (not float32)
2. **Set device_map="auto"** for automatic memory management
3. **Use bfloat16 for long context** (better numerical stability)
4. **Enable gradient checkpointing** for training large models
5. **Monitor memory** with `torch.cuda.max_memory_allocated()`

**Example with all best practices**:
```python
from transformers import AutoModelForCausalLM, TrainingArguments

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16,  # Better for training
    device_map="auto",
    low_cpu_mem_usage=True
)

# Enable gradient checkpointing for memory
model.gradient_checkpointing_enable()

# Training with optimizations
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    bf16=True,  # Match model dtype
    optim="adamw_torch_fused",
    gradient_checkpointing=True
)
```
