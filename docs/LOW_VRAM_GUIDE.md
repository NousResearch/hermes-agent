# Running Hermes on Low Resource Hardware

If you are running Hermes on a machine with limited VRAM (e.g., 4GB–8GB) or a CPU-only setup, you can still achieve excellent performance by using quantized models and memory-efficient inference engines.

Hermes is an agent framework that connects to external inference engines. To optimize for low resources, the optimization must happen at the engine level (e.g., Ollama, llama.cpp).

## 1. Recommended Inference Engines

### Option A: Ollama (Recommended for ease of use)
Ollama automatically handles 4-bit quantization (Q4_K_M) for most models, drastically reducing VRAM usage.
1. Install [Ollama](https://ollama.com/).
2. Pull a small, quantized model: 
   `ollama run hermes2pro:latest` (requires ~4.5GB VRAM)
3. Point Hermes to your local Ollama instance (see config example below).

### Option B: llama.cpp (Recommended for CPU-only or mixed inference)
If you have no dedicated GPU, `llama.cpp` allows you to offload layers to your CPU and standard RAM.
1. Download a `.gguf` quantized model (Q4_K_M or Q5_K_M) from HuggingFace.
2. Run the llama.cpp server:
   `./server -m hermes-model.gguf -c 4096 --n-gpu-layers 10` (Adjust layers based on your VRAM).

## 2. VRAM & Performance Expectations

| Hardware | Model Size | Quantization | Context Length | Expected Speed |
|----------|------------|--------------|----------------|----------------|
| 4GB VRAM | 7B / 8B    | 4-bit (Q4)   | 4096           | 15-25 tokens/s |
| 8GB VRAM | 7B / 8B    | 8-bit (Q8)   | 8192           | 20-35 tokens/s |
| CPU Only | 7B / 8B    | 4-bit (Q4)   | 4096           | 5-10 tokens/s  |

*Tip: Reducing your `context_length` in the Hermes config is the most effective way to prevent Out-Of-Memory (OOM) errors on low VRAM.*
