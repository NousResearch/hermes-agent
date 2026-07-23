---
name: local-model-debug
description: Debug common issues with locally-run LLMs (llama.cpp, Ollama, etc.). Covers think mode quirks (reasoning_content vs content), chat template mismatches, CPU inference tuning, context window limits, and model file corruption.
version: 1.0.0
author: ligl0325
license: MIT
platforms: [linux, macos]
metadata:
  hermes:
    tags: [local-model, llama.cpp, debugging, llm, inference, miniCPM, gguf]
    category: mlops
triggers: ['model not responding', 'empty response', 'think mode issue', 'model debug', 'llama server problem', 'chat template error', 'local LLM issue']
toolsets: [terminal]
---

# Local Model Debugging Guide

## Phase 1: Quick Health Check

- Is the server running? `ss -tlnp | grep <port>`
- Is the model loaded? `curl http://127.0.0.1:<port>/v1/models`
- Basic inference test: curl with simple prompt, check response time
- Memory check: `free -h` (model should fit comfortably)

## Phase 2: Empty/Failed Response Diagnosis

- Check response fields: if content='' but reasoning_content has text → think mode issue
- Fix: restart with `--reasoning off` (llama.cpp) or equivalent parameter
- Check finish_reason: 'length' means output truncated (increase max_tokens); 'stop' is normal
- Check HTTP status: 500 means server error (check log), 200 with empty body → response parsing mismatch

## Phase 3: Chat Template Issues

- Symptom: gibberish or repetitive output → wrong chat template
- Common fix: try `--chat-template` parameter with model-specific template
- For llama.cpp: the right template matters especially for MiniCPM, DeepSeek, Qwen
- Check known-good templates in llama.cpp source or model card

## Phase 4: Performance Tuning

- Check tok/s: compare with expected for model size (1B: 15-25 tok/s, 3B: 8-15 tok/s, 7B: 3-8 tok/s on modern CPU)
- Increase threads: `-t N` (N = core count)
- Context size: `-c 2048` or `4096` (larger = slower but more memory)
- Batch size: `-b 512` (larger batch = faster prompt processing)
- Flash attention: `--flash-attn` (faster but memory tradeoff)
- KV cache quantization: `--cache-type-k q8_0` (reduces memory)

## Pitfalls

- **Think mode**: many modern models (MiniCPM5, DeepSeek-R1, Qwen-QwQ) use think mode — content goes in reasoning_content, not content field
- **CPU only**: models > 7B are painful on CPU (sub-2 tok/s)
- **Swap thrashing**: if model + context exceeds RAM, performance collapses
- **CUDA vs CPU**: some llama.cpp builds lack GPU support; check build info
- **Context overflow**: long conversations exceed ctx size; use smaller context or summarization
