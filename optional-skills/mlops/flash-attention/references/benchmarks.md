# Performance Benchmarks

## Contents
- Speed comparisons across GPUs
- Memory usage analysis
- Scaling with sequence length
- Training vs inference performance
- Flash Attention versions comparison

## Speed comparisons across GPUs

### A100 80GB (Ampere)

**Forward pass time** (milliseconds, batch=8, heads=32, dim=64):

| Seq Length | Standard | Flash Attn 2 | Flash Attn 3 | Speedup (FA2) |
|------------|----------|--------------|--------------|---------------|
| 512 | 1.2 | 0.9 | N/A | 1.3x |
| 1024 | 3.8 | 1.4 | N/A | 2.7x |
| 2048 | 14.2 | 4.8 | N/A | 3.0x |
| 4096 | 55.1 | 17.3 | N/A | 3.2x |
| 8192 | 218.5 | 66.2 | N/A | 3.3x |

### H100 80GB (Hopper)

**Forward pass time** (milliseconds, same config):

| Seq Length | Standard | Flash Attn 2 | Flash Attn 3 (FP16) | Flash Attn 3 (FP8) | Best Speedup |
|------------|----------|--------------|---------------------|--------------------|--------------|
| 512 | 0.8 | 0.6 | 0.4 | 0.3 | 2.7x |
| 1024 | 2.6 | 1.0 | 0.6 | 0.4 | 6.5x |
| 2048 | 9.8 | 3.4 | 2.0 | 1.3 | 7.5x |
| 4096 | 38.2 | 12.5 | 7.2 | 4.8 | 8.0x |
| 8192 | 151.4 | 47.8 | 27.1 | 18.2 | 8.3x |

**Key insight**: Flash Attention 3 on H100 with FP8 achieves ~1.2 PFLOPS (75% of theoretical max).

### A10G 24GB (Ampere)

**Forward pass time** (milliseconds, batch=4):

| Seq Length | Standard | Flash Attn 2 | Speedup |
|------------|----------|--------------|---------|
| 512 | 2.1 | 1.6 | 1.3x |
| 1024 | 6.8 | 2.8 | 2.4x |
| 2048 | 25.9 | 9.4 | 2.8x |
| 4096 | 102.1 | 35.2 | 2.9x |

## Memory usage analysis

### GPU memory consumption (batch=8, heads=32, dim=64)

**Standard attention memory**:

| Seq Length | Attention Matrix | KV Cache | Total | Notes |
|------------|------------------|----------|-------|-------|
| 512 | 8 MB | 32 MB | 40 MB | Manageable |
| 2048 | 128 MB | 128 MB | 256 MB | Growing |
| 8192 | 2048 MB (2 GB) | 512 MB | 2.5 GB | Large |
| 32768 | 32768 MB (32 GB) | 2048 MB | 34 GB | OOM on 24GB GPUs |

**Flash Attention 2 memory**:

| Seq Length | Attention (on-chip) | KV Cache | Total | Reduction |
|------------|---------------------|----------|-------|-----------|
| 512 | 0 MB (recomputed) | 32 MB | 32 MB | 20% |
| 2048 | 0 MB | 128 MB | 128 MB | 50% |
| 8192 | 0 MB | 512 MB | 512 MB | 80% |
| 32768 | 0 MB | 2048 MB | 2 GB | 94% |

**Key insight**: Flash Attention doesn't materialize attention matrix, saving O(N²) memory.

### Memory scaling comparison

**Llama 2 7B model memory** (float16, batch=1):

| Context Length | Standard Attention | Flash Attention 2 | Can Fit 24GB GPU? |
|----------------|-------------------|-------------------|-------------------|
| 2K | 3.2 GB | 2.1 GB | Both: Yes |
| 4K | 5.8 GB | 2.8 GB | Both: Yes |
| 8K | 12.1 GB | 4.2 GB | Both: Yes |
| 16K | 26.3 GB (OOM) | 7.8 GB | Only Flash: Yes |
| 32K | OOM | 14.2 GB | Only Flash: Yes |

### Training memory (Llama 2 7B, batch=4)

| Context | Standard (GB) | Flash Attn (GB) | Reduction |
|---------|---------------|-----------------|-----------|
| 2K | 18.2 | 12.4 | 32% |
| 4K | 34.8 | 16.8 | 52% |
| 8K | OOM (>40GB) | 26.2 | Fits! |

## Scaling with sequence length

### Computational complexity

**Standard attention**:
- Time: O(N² × d)
- Memory: O(N² + N × d)

**Flash Attention**:
- Time: O(N² × d) (same, but with better constants)
- Memory: O(N × d) (linear!)

### Empirical scaling (A100, batch=1, heads=32, dim=64)

**Time per token (milliseconds)**:

| Sequence | 512 | 1K | 2K | 4K | 8K | 16K |
|----------|-----|-----|-----|-----|-----|------|
| Standard | 0.15 | 0.37 | 1.11 | 3.44 | 13.4 | 52.8 |
| Flash Attn 2 | 0.11 | 0.14 | 0.24 | 0.43 | 0.83 | 1.64 |
| Speedup | 1.4x | 2.6x | 4.6x | 8.0x | 16.1x | 32.2x |

**Observation**: Speedup increases quadratically with sequence length!

### Memory per token (MB)

| Sequence | 512 | 1K | 2K | 4K | 8K | 16K |
|----------|-----|-----|-----|-----|-----|------|
| Standard | 0.08 | 0.13 | 0.25 | 0.64 | 2.05 | 8.13 |
| Flash Attn 2 | 0.06 | 0.06 | 0.06 | 0.06 | 0.06 | 0.06 |

**Observation**: Flash Attention memory per token is constant!

## Training vs inference performance

### Training (forward + backward, Llama 2 7B, A100)

| Batch × Seq | Standard (samples/sec) | Flash Attn (samples/sec) | Speedup |
|-------------|------------------------|--------------------------|---------|
| 4 × 2K | 1.2 | 3.1 | 2.6x |
| 8 × 2K | 2.1 | 5.8 | 2.8x |
| 4 × 4K | 0.4 | 1.3 | 3.3x |
| 8 × 4K | OOM | 2.4 | Enabled |
| 2 × 8K | 0.1 | 0.4 | 4.0x |

### Inference (generation, Llama 2 7B, A100)

| Context Length | Standard (tokens/sec) | Flash Attn (tokens/sec) | Speedup |
|----------------|----------------------|-------------------------|---------|
| 512 | 48 | 52 | 1.1x |
| 2K | 42 | 62 | 1.5x |
| 4K | 31 | 58 | 1.9x |
| 8K | 18 | 51 | 2.8x |
| 16K | OOM | 42 | Enabled |

**Note**: Inference speedup less dramatic than training because generation is memory-bound (KV cache accesses).

## Flash Attention versions comparison

### Flash Attention 1 vs 2 vs 3 (H100, seq=4096, batch=8)

| Metric | FA1 | FA2 | FA3 (FP16) | FA3 (FP8) |
|--------|-----|-----|------------|-----------|
| Forward time (ms) | 28.4 | 12.5 | 7.2 | 4.8 |
| Memory (GB) | 4.8 | 4.2 | 4.2 | 2.8 |
| TFLOPS | 180 | 420 | 740 | 1150 |
| GPU util % | 35% | 55% | 75% | 82% |

**Key improvements**:
- FA2: 2.3x faster than FA1 (better parallelism)
- FA3 (FP16): 1.7x faster than FA2 (H100 async optimizations)
- FA3 (FP8): 2.6x faster than FA2 (low precision)

### Features by version

| Feature | FA1 | FA2 | FA3 |
|---------|-----|-----|-----|
| Basic attention | ✅ | ✅ | ✅ |
| Causal masking | ✅ | ✅ | ✅ |
| Multi-query attention | ❌ | ✅ | ✅ |
| Sliding window | ❌ | ✅ | ✅ |
| Paged KV cache | ❌ | ✅ | ✅ |
| FP8 support | ❌ | ❌ | ✅ (H100 only) |
| Work partitioning | Basic | Advanced | Optimal |

## Real-world model benchmarks

### Llama 2 models (A100 80GB, batch=4, seq=2048)

| Model | Params | Standard (samples/sec) | Flash Attn (samples/sec) | Speedup |
|-------|--------|------------------------|--------------------------|---------|
| Llama 2 7B | 7B | 1.2 | 3.1 | 2.6x |
| Llama 2 13B | 13B | 0.6 | 1.7 | 2.8x |
| Llama 2 70B | 70B | 0.12 | 0.34 | 2.8x |

### GPT-style models (seq=1024)

| Model | Standard (tokens/sec) | Flash Attn (tokens/sec) | Speedup |
|-------|----------------------|-------------------------|---------|
| GPT-2 (124M) | 520 | 680 | 1.3x |
| GPT-J (6B) | 42 | 98 | 2.3x |
| GPT-NeoX (20B) | 8 | 22 | 2.75x |

## Recommendations by use case

**Training large models (>7B parameters)**:
- Use Flash Attention 2 on A100
- Use Flash Attention 3 FP8 on H100 for maximum speed
- Expected: 2.5-3x speedup

**Long context inference (>4K tokens)**:
- Flash Attention essential (enables contexts standard attention can't handle)
- Expected: 2-4x speedup, 5-10x memory reduction

**Short sequences (<512 tokens)**:
- Flash Attention provides 1.2-1.5x speedup
- Minimal memory benefit
- Still worth enabling (no downside)

**Multi-user serving**:
- Flash Attention reduces per-request memory
- Allows higher concurrent batch sizes
- Can serve 2-3x more users on same hardware
