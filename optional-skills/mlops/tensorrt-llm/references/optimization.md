# TensorRT-LLM Optimization Guide

Comprehensive guide to optimizing LLM inference with TensorRT-LLM.

## Quantization

### FP8 Quantization (Recommended for H100)

**Benefits**:
- 2× faster inference
- 50% memory reduction
- Minimal accuracy loss (<1% perplexity degradation)

**Usage**:
```python
from tensorrt_llm import LLM

# Automatic FP8 quantization
llm = LLM(
    model="meta-llama/Meta-Llama-3-70B",
    dtype="fp8",
    quantization="fp8"
)
```

**Performance** (Llama 3-70B on 8× H100):
- FP16: 5,000 tokens/sec
- FP8: **10,000 tokens/sec** (2× speedup)
- Memory: 140GB → 70GB

### INT4 Quantization (Maximum compression)

**Benefits**:
- 4× memory reduction
- 3-4× faster inference
- Fits larger models on same hardware

**Usage**:
```python
# INT4 with AWQ calibration
llm = LLM(
    model="meta-llama/Meta-Llama-3-405B",
    dtype="int4_awq",
    quantization="awq"
)

# INT4 with GPTQ calibration
llm = LLM(
    model="meta-llama/Meta-Llama-3-405B",
    dtype="int4_gptq",
    quantization="gptq"
)
```

**Trade-offs**:
- Accuracy: 1-3% perplexity increase
- Speed: 3-4× faster than FP16
- Use case: When memory is critical

## In-Flight Batching

**What it does**: Dynamically batches requests during generation instead of waiting for all sequences to finish.

**Configuration**:
```python
# Server configuration
trtllm-serve meta-llama/Meta-Llama-3-8B \
    --max_batch_size 256 \           # Maximum concurrent sequences
    --max_num_tokens 4096 \           # Total tokens in batch
    --enable_chunked_context \        # Split long prompts
    --scheduler_policy max_utilization
```

**Performance**:
- Throughput: **4-8× higher** vs static batching
- Latency: Lower P50/P99 for mixed workloads
- GPU utilization: 80-95% vs 40-60%

## Paged KV Cache

**What it does**: Manages KV cache memory like OS manages virtual memory (paging).

**Benefits**:
- 40-60% higher throughput
- No memory fragmentation
- Supports longer sequences

**Configuration**:
```python
# Automatic paged KV cache (default)
llm = LLM(
    model="meta-llama/Meta-Llama-3-8B",
    kv_cache_free_gpu_mem_fraction=0.9,  # Use 90% GPU mem for cache
    enable_prefix_caching=True            # Cache common prefixes
)
```

## Speculative Decoding

**What it does**: Uses small draft model to predict multiple tokens, verified by target model in parallel.

**Speedup**: 2-3× faster for long generations

**Usage**:
```python
from tensorrt_llm import LLM

# Target model (Llama 3-70B)
llm = LLM(
    model="meta-llama/Meta-Llama-3-70B",
    speculative_model="meta-llama/Meta-Llama-3-8B",  # Draft model
    num_speculative_tokens=5                          # Tokens to predict ahead
)

# Same API, 2-3× faster
outputs = llm.generate(prompts)
```

**Best models for drafting**:
- Target: Llama 3-70B → Draft: Llama 3-8B
- Target: Qwen2-72B → Draft: Qwen2-7B
- Same family, 8-10× smaller

## CUDA Graphs

**What it does**: Reduces kernel launch overhead by recording GPU operations.

**Benefits**:
- 10-20% lower latency
- More stable P99 latency
- Better for small batch sizes

**Configuration** (automatic by default):
```python
llm = LLM(
    model="meta-llama/Meta-Llama-3-8B",
    enable_cuda_graph=True,  # Default: True
    cuda_graph_cache_size=2  # Cache 2 graph variants
)
```

## Chunked Context

**What it does**: Splits long prompts into chunks to reduce memory spikes.

**Use case**: Prompts >8K tokens with limited GPU memory

**Configuration**:
```bash
trtllm-serve meta-llama/Meta-Llama-3-8B \
    --max_num_tokens 4096 \
    --enable_chunked_context \
    --max_chunked_prefill_length 2048  # Process 2K tokens at a time
```

## Overlap Scheduling

**What it does**: Overlaps compute and memory operations.

**Benefits**:
- 15-25% higher throughput
- Better GPU utilization
- Default in v1.2.0+

**No configuration needed** - enabled automatically.

## Quantization Comparison Table

| Method | Memory | Speed | Accuracy | Use Case |
|--------|--------|-------|----------|----------|
| FP16 | 1× (baseline) | 1× | Best | High accuracy needed |
| FP8 | 0.5× | 2× | -0.5% ppl | **H100 default** |
| INT4 AWQ | 0.25× | 3-4× | -1.5% ppl | Memory critical |
| INT4 GPTQ | 0.25× | 3-4× | -2% ppl | Maximum speed |

## Tuning Workflow

1. **Start with defaults**:
   ```python
   llm = LLM(model="meta-llama/Meta-Llama-3-70B")
   ```

2. **Enable FP8** (if H100):
   ```python
   llm = LLM(model="...", dtype="fp8")
   ```

3. **Tune batch size**:
   ```python
   # Increase until OOM, then reduce 20%
   trtllm-serve ... --max_batch_size 256
   ```

4. **Enable chunked context** (if long prompts):
   ```bash
   --enable_chunked_context --max_chunked_prefill_length 2048
   ```

5. **Try speculative decoding** (if latency critical):
   ```python
   llm = LLM(model="...", speculative_model="...")
   ```

## Benchmarking

```bash
# Install benchmark tool
pip install tensorrt_llm[benchmark]

# Run benchmark
python benchmarks/python/benchmark.py \
    --model meta-llama/Meta-Llama-3-8B \
    --batch_size 64 \
    --input_len 128 \
    --output_len 256 \
    --dtype fp8
```

**Metrics to track**:
- Throughput (tokens/sec)
- Latency P50/P90/P99 (ms)
- GPU memory usage (GB)
- GPU utilization (%)

## Common Issues

**OOM errors**:
- Reduce `max_batch_size`
- Reduce `max_num_tokens`
- Enable INT4 quantization
- Increase `tensor_parallel_size`

**Low throughput**:
- Increase `max_batch_size`
- Enable in-flight batching
- Verify CUDA graphs enabled
- Check GPU utilization

**High latency**:
- Try speculative decoding
- Reduce `max_batch_size` (less queueing)
- Use FP8 instead of FP16
