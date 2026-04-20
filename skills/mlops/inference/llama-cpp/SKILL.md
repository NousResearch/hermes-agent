---
name: llama-cpp
description: Run LLM inference with llama.cpp on CPU, Apple Silicon, AMD/Intel GPUs, or NVIDIA — plus GGUF model conversion and quantization (2–8 bit with K-quants and imatrix). Covers CLI, Python bindings, OpenAI-compatible server, and Ollama/LM Studio integration. Use for edge deployment, M1/M2/M3/M4 Macs, CUDA-less environments, or flexible local quantization.
version: 2.0.0
author: Orchestra Research
license: MIT
dependencies: [llama-cpp-python>=0.2.0]
metadata:
  hermes:
    tags: [llama.cpp, GGUF, Quantization, CPU Inference, Apple Silicon, Edge Deployment, Non-NVIDIA, AMD GPUs, Intel GPUs, Embedded, Model Compression]
---

# llama.cpp + GGUF

Pure C/C++ LLM inference with minimal dependencies, plus the GGUF (GPT-Generated Unified Format) standard used for quantized weights. One toolchain covers conversion, quantization, and serving.

## When to use

**Use llama.cpp + GGUF when:**
- Running on CPU-only machines or Apple Silicon (M1/M2/M3/M4) with Metal acceleration
- Using AMD (ROCm) or Intel GPUs where CUDA isn't available
- Edge deployment (Raspberry Pi, embedded systems, consumer laptops)
- Need flexible quantization (2–8 bit with K-quants)
- Want local AI tools (LM Studio, Ollama, text-generation-webui, koboldcpp)
- Want a single binary deploy without Docker/Python

**Key advantages:**
- Universal hardware: CPU, Apple Silicon, NVIDIA, AMD, Intel
- No Python runtime required (pure C/C++)
- K-quants + imatrix for better low-bit quality
- OpenAI-compatible server built in
- Rich ecosystem (Ollama, LM Studio, llama-cpp-python)

**Use alternatives instead:**
- **vLLM** — NVIDIA GPUs, PagedAttention, Python-first, max throughput
- **TensorRT-LLM** — Production NVIDIA (A100/H100), maximum speed
- **AWQ/GPTQ** — Calibrated quantization for NVIDIA-only deployments
- **bitsandbytes** — Simple HuggingFace transformers integration
- **HQQ** — Fast calibration-free quantization

## Quick start

### Install

```bash
# macOS / Linux (simplest)
brew install llama.cpp

# Or build from source
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
make                        # CPU
make GGML_METAL=1           # Apple Silicon
make GGML_CUDA=1            # NVIDIA CUDA
make LLAMA_HIP=1            # AMD ROCm

# Python bindings (optional)
pip install llama-cpp-python
# With CUDA:   CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --force-reinstall --no-cache-dir
# With Metal:  CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python --force-reinstall --no-cache-dir
```

### Download a pre-quantized GGUF

```bash
# TheBloke hosts most popular models pre-quantized
huggingface-cli download \
    TheBloke/Llama-2-7B-Chat-GGUF \
    llama-2-7b-chat.Q4_K_M.gguf \
    --local-dir models/
```

### Or convert a HuggingFace model to GGUF

```bash
# 1. Download HF model
huggingface-cli download meta-llama/Llama-3.1-8B --local-dir ./llama-3.1-8b

# 2. Convert to FP16 GGUF
python convert_hf_to_gguf.py ./llama-3.1-8b \
    --outfile llama-3.1-8b-f16.gguf \
    --outtype f16

# 3. Quantize to Q4_K_M
./llama-quantize llama-3.1-8b-f16.gguf llama-3.1-8b-q4_k_m.gguf Q4_K_M
```

### Run inference

```bash
# One-shot prompt
./llama-cli -m model.Q4_K_M.gguf -p "Explain quantum computing" -n 256

# Interactive chat
./llama-cli -m model.Q4_K_M.gguf --interactive

# With GPU offload
./llama-cli -m model.Q4_K_M.gguf -ngl 35 -p "Hello!"
```

### Serve an OpenAI-compatible API

```bash
./llama-server \
    -m model.Q4_K_M.gguf \
    --host 0.0.0.0 \
    --port 8080 \
    -ngl 35 \
    -c 4096 \
    --parallel 4 \
    --cont-batching
```

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "local",
    "messages": [{"role": "user", "content": "Hello!"}],
    "temperature": 0.7,
    "max_tokens": 100
  }'
```

## Quantization formats (GGUF)

### K-quant methods (recommended)

| Type | Bits | Size (7B) | Quality | Use Case |
|------|------|-----------|---------|----------|
| Q2_K | 2.5 | ~2.8 GB | Low | Extreme compression (testing only) |
| Q3_K_S | 3.0 | ~3.0 GB | Low-Med | Memory constrained |
| Q3_K_M | 3.3 | ~3.3 GB | Medium | Fits small devices |
| Q4_K_S | 4.0 | ~3.8 GB | Med-High | Speed critical |
| **Q4_K_M** | 4.5 | ~4.1 GB | High | **Recommended default** |
| Q5_K_S | 5.0 | ~4.6 GB | High | Quality focused |
| Q5_K_M | 5.5 | ~4.8 GB | Very High | High quality |
| Q6_K | 6.0 | ~5.5 GB | Excellent | Near-original |
| Q8_0 | 8.0 | ~7.2 GB | Best | Maximum quality, minimal degradation |

**Variant suffixes** — `_S` (Small, faster, lower quality), `_M` (Medium, balanced), `_L` (Large, better quality).

**Legacy (Q4_0/Q4_1/Q5_0/Q5_1) exist** but always prefer K-quants for better quality/size ratio.

**IQ quantization** — ultra-low-bit with importance-aware methods: IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_XS, IQ3_S, IQ4_XS. Require `--imatrix`.

**Task-specific defaults:**
- General chat / assistants: Q4_K_M, or Q5_K_M if RAM allows
- Code generation: Q5_K_M or Q6_K (higher precision helps)
- Technical / medical: Q6_K or Q8_0
- Very large (70B, 405B) on consumer hardware: Q3_K_M or Q4_K_S
- Raspberry Pi / edge: Q2_K or Q3_K_S

## Conversion workflows

### Basic: HF → GGUF → quantized

```bash
python convert_hf_to_gguf.py ./model --outfile model-f16.gguf --outtype f16
./llama-quantize model-f16.gguf model-q4_k_m.gguf Q4_K_M
./llama-cli -m model-q4_k_m.gguf -p "Hello!" -n 50
```

### With importance matrix (imatrix) — better low-bit quality

`imatrix` gives 10–20% perplexity improvement at Q4, essential at Q3 and below.

```bash
# 1. Convert to FP16 GGUF
python convert_hf_to_gguf.py ./model --outfile model-f16.gguf

# 2. Prepare calibration data (diverse text, ~100MB is ideal)
cat > calibration.txt << 'EOF'
The quick brown fox jumps over the lazy dog.
Machine learning is a subset of artificial intelligence.
# Add more diverse text samples...
EOF

# 3. Generate importance matrix
./llama-imatrix -m model-f16.gguf \
    -f calibration.txt \
    --chunk 512 \
    -o model.imatrix \
    -ngl 35

# 4. Quantize with imatrix
./llama-quantize --imatrix model.imatrix \
    model-f16.gguf model-q4_k_m.gguf Q4_K_M
```

### Multi-quant batch

```bash
#!/bin/bash
MODEL="llama-3.1-8b-f16.gguf"
IMATRIX="llama-3.1-8b.imatrix"

./llama-imatrix -m $MODEL -f wiki.txt -o $IMATRIX -ngl 35

for QUANT in Q4_K_M Q5_K_M Q6_K Q8_0; do
    OUTPUT="llama-3.1-8b-${QUANT,,}.gguf"
    ./llama-quantize --imatrix $IMATRIX $MODEL $OUTPUT $QUANT
    echo "Created: $OUTPUT ($(du -h $OUTPUT | cut -f1))"
done
```

### Quality testing (perplexity)

```bash
./llama-perplexity -m model.gguf -f wikitext-2-raw/wiki.test.raw -c 512
# Baseline FP16: ~5.96  |  Q4_K_M: ~6.06 (+1.7%)  |  Q2_K: ~6.87 (+15.3%)
```

## Python bindings (llama-cpp-python)

### Basic generation

```python
from llama_cpp import Llama

llm = Llama(
    model_path="./model-q4_k_m.gguf",
    n_ctx=4096,
    n_gpu_layers=35,     # 0 for CPU only, 99 to offload everything
    n_threads=8,
)

output = llm(
    "What is machine learning?",
    max_tokens=256,
    temperature=0.7,
    stop=["</s>", "\n\n"],
)
print(output["choices"][0]["text"])
```

### Chat completion + streaming

```python
llm = Llama(
    model_path="./model-q4_k_m.gguf",
    n_ctx=4096,
    n_gpu_layers=35,
    chat_format="llama-3",    # Or "chatml", "mistral", etc.
)

# Non-streaming
response = llm.create_chat_completion(
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is Python?"},
    ],
    max_tokens=256,
    temperature=0.7,
)
print(response["choices"][0]["message"]["content"])

# Streaming
for chunk in llm("Explain quantum computing:", max_tokens=256, stream=True):
    print(chunk["choices"][0]["text"], end="", flush=True)
```

### Embeddings

```python
llm = Llama(model_path="./model-q4_k_m.gguf", embedding=True, n_gpu_layers=35)
vec = llm.embed("This is a test sentence.")
print(f"Embedding dimension: {len(vec)}")
```

## Hardware acceleration

### Apple Silicon (Metal)

```bash
make clean && make GGML_METAL=1
./llama-cli -m model.gguf -ngl 99 -p "Hello"   # offload all layers
```

```python
llm = Llama(
    model_path="model.gguf",
    n_gpu_layers=99,     # Offload everything
    n_threads=1,         # Metal handles parallelism
)
```

Performance: M3 Max ~40–60 tok/s on Llama 2-7B Q4_K_M.

### NVIDIA (CUDA)

```bash
make clean && make GGML_CUDA=1
./llama-cli -m model.gguf -ngl 35 -p "Hello"

# Hybrid for large models
./llama-cli -m llama-70b.Q4_K_M.gguf -ngl 20   # GPU: 20 layers, CPU: rest

# Multi-GPU split
./llama-cli -m large-model.gguf --tensor-split 0.5,0.5 -ngl 60
```

### AMD (ROCm)

```bash
make LLAMA_HIP=1
./llama-cli -m model.gguf -ngl 999
```

### CPU

```bash
# Match PHYSICAL cores, not logical
./llama-cli -m model.gguf -t 8 -p "Hello"

# BLAS acceleration (2–3× speedup)
make LLAMA_OPENBLAS=1
```

```python
llm = Llama(
    model_path="model.gguf",
    n_gpu_layers=0,
    n_threads=8,
    n_batch=512,         # Larger batch = faster prompt processing
)
```

## Performance benchmarks

### CPU (Llama 2-7B Q4_K_M)

| CPU | Threads | Speed |
|-----|---------|-------|
| Apple M3 Max (Metal) | 16 | 50 tok/s |
| AMD Ryzen 9 7950X | 32 | 35 tok/s |
| Intel i9-13900K | 32 | 30 tok/s |

### GPU offloading on RTX 4090

| Layers GPU | Speed | VRAM |
|------------|-------|------|
| 0 (CPU only) | 30 tok/s | 0 GB |
| 20 (hybrid) | 80 tok/s | 8 GB |
| 35 (all) | 120 tok/s | 12 GB |

## Supported models

- **LLaMA family**: Llama 2 (7B/13B/70B), Llama 3 (8B/70B/405B), Code Llama
- **Mistral family**: Mistral 7B, Mixtral 8x7B/8x22B
- **Other**: Falcon, BLOOM, GPT-J, Phi-3, Gemma, Qwen, LLaVA (vision), Whisper (audio)

Find GGUF models: https://huggingface.co/models?library=gguf

## Ecosystem integrations

### Ollama

```bash
cat > Modelfile << 'EOF'
FROM ./model-q4_k_m.gguf
TEMPLATE """{{ .System }}
{{ .Prompt }}"""
PARAMETER temperature 0.7
PARAMETER num_ctx 4096
EOF

ollama create mymodel -f Modelfile
ollama run mymodel "Hello!"
```

### LM Studio

1. Place GGUF file in `~/.cache/lm-studio/models/`
2. Open LM Studio and select the model
3. Configure context length and GPU offload, start inference

### text-generation-webui

```bash
cp model-q4_k_m.gguf text-generation-webui/models/
python server.py --model model-q4_k_m.gguf --loader llama.cpp --n-gpu-layers 35
```

### OpenAI client → llama-server

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="not-needed")
response = client.chat.completions.create(
    model="local-model",
    messages=[{"role": "user", "content": "Hello!"}],
    max_tokens=256,
)
print(response.choices[0].message.content)
```

## Best practices

1. **Use K-quants** — Q4_K_M is the recommended default
2. **Use imatrix** for Q4 and below (calibration improves quality substantially)
3. **Offload as many layers as VRAM allows** — start high, reduce by 5 on OOM
4. **Thread count** — match physical cores, not logical
5. **Batch size** — increase `n_batch` (e.g. 512) for faster prompt processing
6. **Context** — start at 4096, grow only as needed (memory scales with ctx)
7. **Flash Attention** — add `--flash-attn` if your build supports it

## Common issues (quick fixes)

**Model loads slowly** — use `--mmap` for memory-mapped loading.

**Out of memory (GPU)** — reduce `-ngl`, use a smaller quant (Q4_K_S / Q3_K_M), or quantize the KV cache:
```python
Llama(model_path="...", type_k=2, type_v=2, n_gpu_layers=35)  # Q4_0 KV cache
```

**Garbage output** — wrong `chat_format`, temperature too high, or model file corrupted. Test with `temperature=0.1` and verify FP16 baseline works.

**Connection refused (server)** — bind to `--host 0.0.0.0`, check `lsof -i :8080`.

See `references/troubleshooting.md` for the full playbook.

## References

- **[advanced-usage.md](references/advanced-usage.md)** — speculative decoding, batched inference, grammar-constrained generation, LoRA, multi-GPU, custom builds, benchmark scripts
- **[quantization.md](references/quantization.md)** — perplexity tables, use-case guide, model size scaling (7B/13B/70B RAM needs), imatrix deep dive
- **[server.md](references/server.md)** — OpenAI API endpoints, Docker deployment, NGINX load balancing, monitoring
- **[optimization.md](references/optimization.md)** — CPU threading, BLAS, GPU offload heuristics, batch tuning, benchmarks
- **[troubleshooting.md](references/troubleshooting.md)** — install/convert/quantize/inference/server issues, Apple Silicon, debugging

## Resources

- **GitHub**: https://github.com/ggml-org/llama.cpp
- **Python bindings**: https://github.com/abetlen/llama-cpp-python
- **Pre-quantized models**: https://huggingface.co/TheBloke
- **GGUF converter Space**: https://huggingface.co/spaces/ggml-org/gguf-my-repo
- **License**: MIT
