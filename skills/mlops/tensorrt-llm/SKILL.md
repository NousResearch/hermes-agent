---
name: tensorrt-llm
description: Optimizes LLM inference with NVIDIA TensorRT for maximum throughput and lowest latency. Use for production deployment on NVIDIA GPUs (A100/H100), when you need 10-100x faster inference than PyTorch, or for serving models with quantization (FP8/INT4), in-flight batching, and multi-GPU scaling.
version: 1.0.0
author: Orchestra Research
license: MIT
tags: [Inference Serving, TensorRT-LLM, NVIDIA, Inference Optimization, High Throughput, Low Latency, Production, FP8, INT4, In-Flight Batching, Multi-GPU]
dependencies: [tensorrt-llm, torch]
---

# TensorRT-LLM

NVIDIA's open-source library for optimizing LLM inference with state-of-the-art performance on NVIDIA GPUs.

## When to use TensorRT-LLM

**Use TensorRT-LLM when:**
- Deploying on NVIDIA GPUs (A100, H100, GB200)
- Need maximum throughput (24,000+ tokens/sec on Llama 3)
- Require low latency for real-time applications
- Working with quantized models (FP8, INT4, FP4)
- Scaling across multiple GPUs or nodes

**Use vLLM instead when:**
- Need simpler setup and Python-first API
- Want PagedAttention without TensorRT compilation
- Working with AMD GPUs or non-NVIDIA hardware

**Use llama.cpp instead when:**
- Deploying on CPU or Apple Silicon
- Need edge deployment without NVIDIA GPUs
- Want simpler GGUF quantization format

## Quick start

### Installation

```bash
# Docker (recommended)
docker pull nvidia/tensorrt_llm:latest

# pip install
pip install tensorrt_llm==1.2.0rc3

# Requires CUDA 13.0.0, TensorRT 10.13.2, Python 3.10-3.12
```

### Basic inference

```python
from tensorrt_llm import LLM, SamplingParams

# Initialize model
llm = LLM(model="meta-llama/Meta-Llama-3-8B")

# Configure sampling
sampling_params = SamplingParams(
    max_tokens=100,
    temperature=0.7,
    top_p=0.9
)

# Generate
prompts = ["Explain quantum computing"]
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(output.text)
```

### Serving with trtllm-serve

```bash
# Start server (automatic model download and compilation)
trtllm-serve meta-llama/Meta-Llama-3-8B \
    --tp_size 4 \              # Tensor parallelism (4 GPUs)
    --max_batch_size 256 \
    --max_num_tokens 4096

# Client request
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Meta-Llama-3-8B",
    "messages": [{"role": "user", "content": "Hello!"}],
    "temperature": 0.7,
    "max_tokens": 100
  }'
```

## Key features

### Performance optimizations
- **In-flight batching**: Dynamic batching during generation
- **Paged KV cache**: Efficient memory management
- **Flash Attention**: Optimized attention kernels
- **Quantization**: FP8, INT4, FP4 for 2-4× faster inference
- **CUDA graphs**: Reduced kernel launch overhead

### Parallelism
- **Tensor parallelism (TP)**: Split model across GPUs
- **Pipeline parallelism (PP)**: Layer-wise distribution
- **Expert parallelism**: For Mixture-of-Experts models
- **Multi-node**: Scale beyond single machine

### Advanced features
- **Speculative decoding**: Faster generation with draft models
- **LoRA serving**: Efficient multi-adapter deployment
- **Disaggregated serving**: Separate prefill and generation

## Common patterns

### Quantized model (FP8)

```python
from tensorrt_llm import LLM

# Load FP8 quantized model (2× faster, 50% memory)
llm = LLM(
    model="meta-llama/Meta-Llama-3-70B",
    dtype="fp8",
    max_num_tokens=8192
)

# Inference same as before
outputs = llm.generate(["Summarize this article..."])
```

### Multi-GPU deployment

```python
# Tensor parallelism across 8 GPUs
llm = LLM(
    model="meta-llama/Meta-Llama-3-405B",
    tensor_parallel_size=8,
    dtype="fp8"
)
```

### Batch inference

```python
# Process 100 prompts efficiently
prompts = [f"Question {i}: ..." for i in range(100)]

outputs = llm.generate(
    prompts,
    sampling_params=SamplingParams(max_tokens=200)
)

# Automatic in-flight batching for maximum throughput
```

## Performance benchmarks

**Meta Llama 3-8B** (H100 GPU):
- Throughput: 24,000 tokens/sec
- Latency: ~10ms per token
- vs PyTorch: **100× faster**

**Llama 3-70B** (8× A100 80GB):
- FP8 quantization: 2× faster than FP16
- Memory: 50% reduction with FP8

## Supported models

- **LLaMA family**: Llama 2, Llama 3, CodeLlama
- **GPT family**: GPT-2, GPT-J, GPT-NeoX
- **Qwen**: Qwen, Qwen2, QwQ
- **DeepSeek**: DeepSeek-V2, DeepSeek-V3
- **Mixtral**: Mixtral-8x7B, Mixtral-8x22B
- **Vision**: LLaVA, Phi-3-vision
- **100+ models** on HuggingFace

## References

- **[Optimization Guide](references/optimization.md)** - Quantization, batching, KV cache tuning
- **[Multi-GPU Setup](references/multi-gpu.md)** - Tensor/pipeline parallelism, multi-node
- **[Serving Guide](references/serving.md)** - Production deployment, monitoring, autoscaling

## Resources

- **Docs**: https://nvidia.github.io/TensorRT-LLM/
- **GitHub**: https://github.com/NVIDIA/TensorRT-LLM
- **Models**: https://huggingface.co/models?library=tensorrt_llm


