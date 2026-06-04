---
name: llm-inference
description: "LLM inference: local GGUF (llama.cpp), server-side (vLLM), structured output (Outlines), safety/abliteration (Obliteratus)"
version: 1.0.0
author: Orchestra Research
license: MIT
dependencies: [llama-cpp-python, vllm, outlines, obliteratus]
metadata:
  hermes:
    tags: [llm-inference, llama-cpp, vllm, outlines, obliteratus, GGUF, PagedAttention, Structured-Generation, Abliteration, Quantization]
---

# LLM Inference

Unified skill covering the full spectrum of LLM inference approaches.

## Tools Overview

| Tool | When to Use | Key Feature |
|------|-------------|-------------|
| **[llama-cpp](references/llama-cpp.md)** | Local GGUF inference, edge/CPU deployment, Apple Silicon | Run GGUF models from Hugging Face Hub directly |
| **[vllm](references/vllm.md)** | Production serving, high-throughput APIs, tensor parallelism | 24x throughput via PagedAttention + continuous batching |
| **[outlines](references/outlines.md)** | Structured JSON/Pydantic output, grammar-constrained generation | Zero-overhead FSM-based constraint at token level |
| **[obliteratus](references/obliteratus.md)** | Remove refusal behaviors from open-weight LLMs | Diff-in-means, SVD, LEACE surgical weight projection |

## Quick Reference

### Local GGUF (llama.cpp)
```bash
# Install
brew install llama.cpp

# Run from Hugging Face Hub
llama-cli -hf bartowski/Llama-3.2-3B-Instruct-GGUF:Q8_0

# OpenAI-compatible server
llama-server -hf bartowski/Llama-3.2-3B-Instruct-GGUF:Q8_0
```

### Server-Side Serving (vLLM)
```bash
pip install vllm

# OpenAI-compatible server
vllm serve meta-llama/Llama-3-8B-Instruct

# With quantization
vllm serve TheBloke/Llama-2-70B-AWQ --quantization awq
```

### Structured Output (Outlines)
```python
from pydantic import BaseModel
import outlines

model = outlines.models.transformers("microsoft/Phi-3-mini-4k-instruct")

class Person(BaseModel):
    name: str
    age: int

generator = outlines.generate.json(model, Person)
person = generator("Generate: John, 30 years old")
```

### Abliteration (Obliteratus)
```bash
pip install -e git+https://github.com/elder-plinius/OBLITERATUS.git

# Abliterate a model (CLI only - AGPL license)
obliteratus obliterate <model_name> --method advanced --output-dir ./abliterated-models
```

## References

- **[llama-cpp](references/llama-cpp.md)** — GGUF local inference, HF Hub discovery, quant selection, llama-server
- **[vllm](references/vllm.md)** — Production serving, PagedAttention, continuous batching, tensor parallelism
- **[outlines](references/outlines.md)** — Structured generation, Pydantic integration, grammar-based constraints
- **[obliteratus](references/obliteratus.md)** — Refusal removal, weight projection, ablation strategies

## Decision Guide

| Need | Tool |
|------|------|
| CPU/edge inference, single user | llama.cpp |
| High-throughput production API | vLLM |
| Guaranteed valid JSON/Pydantic output | Outlines |
| Remove refusal behaviors | Obliteratus |
| Combination of above | Use the specific reference |

## Archived Skills

This skill absorbed and replaces the following skills (moved to `.archive/mlops/`):
- llama-cpp
- vllm
- outlines
- obliteratus