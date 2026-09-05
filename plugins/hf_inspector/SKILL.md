---
name: hf-inspector
description: "Inspect Hugging Face models, architecture, context, and GGUF quants."
version: 1.0.0
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [huggingface, models, llm, gguf, architecture]
    category: research
---

# Hugging Face Model & GGUF Quant Explorer

## When to Use
- When asked to inspect a Hugging Face model ID (e.g. `NousResearch/Hermes-3-Llama-3.1-8B`, `Qwen/Qwen2.5-Coder-32B-Instruct`).
- When checking parameter counts, context window limits, or architecture of an open-weight LLM.
- When searching for GGUF / AWQ / GPTQ quantization files for local inference with llama.cpp or Ollama.

## How to Run
Invoke through the `hf_inspect_model` or `hf_list_quants` tools:
- `hf_inspect_model(model_id="NousResearch/Hermes-3-Llama-3.1-8B")`
- `hf_list_quants(model_id="bartowski/Meta-Llama-3.1-8B-Instruct-GGUF")`

## Procedure
1. If the user provides a Hugging Face repo or model name, run `hf_inspect_model(model_id)`.
2. To find quantized files or download links, run `hf_list_quants(model_id)`.
3. Present the parameters, context length, architecture, and quants in a clear summary.

## Pitfalls
- Base models and their GGUF quants are often in separate repos (e.g. `NousResearch/Hermes-3-Llama-3.1-8B` vs `bartowski/Hermes-3-Llama-3.1-8B-GGUF`). Check companion GGUF repos if `hf_list_quants` returns no quant files in the base repository.
