---
name: optimizing-attention-flash
description: Optimizes transformer attention with Flash Attention for 2-4x speedup and 10-20x memory reduction. Use when training/running transformers with long sequences (>512 tokens), encountering GPU memory issues with attention, or need faster inference. Supports PyTorch native SDPA, flash-attn library, H100 FP8, and sliding window attention.
version: 1.0.0
author: Orchestra Research
license: MIT
tags: [Optimization, Flash Attention, Attention Optimization, Memory Efficiency, Speed Optimization, Long Context, PyTorch, SDPA, H100, FP8, Transformers]
dependencies: [flash-attn, torch, transformers]
---

# Flash Attention - Fast Memory-Efficient Attention

## Quick start

Flash Attention provides 2-4x speedup and 10-20x memory reduction for transformer attention through IO-aware tiling and recomputation.

**PyTorch native (easiest, PyTorch 2.2+)**:
```python
import torch
import torch.nn.functional as F

q = torch.randn(2, 8, 512, 64, device='cuda', dtype=torch.float16)  # [batch, heads, seq, dim]
k = torch.randn(2, 8, 512, 64, device='cuda', dtype=torch.float16)
v = torch.randn(2, 8, 512, 64, device='cuda', dtype=torch.float16)

# Automatically uses Flash Attention if available
out = F.scaled_dot_product_attention(q, k, v)
```

**flash-attn library (more features)**:
```bash
pip install flash-attn --no-build-isolation
```

```python
from flash_attn import flash_attn_func

# q, k, v: [batch, seqlen, nheads, headdim]
out = flash_attn_func(q, k, v, dropout_p=0.0, causal=True)
```

## Common workflows

### Workflow 1: Enable in existing PyTorch model

Copy this checklist:

```
Flash Attention Integration:
- [ ] Step 1: Check PyTorch version (≥2.2)
- [ ] Step 2: Enable Flash Attention backend
- [ ] Step 3: Verify speedup with profiling
- [ ] Step 4: Test accuracy matches baseline
```

**Step 1: Check PyTorch version**

```bash
python -c "import torch; print(torch.__version__)"
# Should be ≥2.2.0
```

If <2.2, upgrade:
```bash
pip install --upgrade torch
```

**Step 2: Enable Flash Attention backend**

Replace standard attention:
```python
# Before (standard attention)
attn_weights = torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(d_k), dim=-1)
out = attn_weights @ v

# After (Flash Attention)
import torch.nn.functional as F
out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
```

Force Flash Attention backend:
```python
with torch.backends.cuda.sdp_kernel(
    enable_flash=True,
    enable_math=False,
    enable_mem_efficient=False
):
    out = F.scaled_dot_product_attention(q, k, v)
```

**Step 3: Verify speedup with profiling**

```python
import torch.utils.benchmark as benchmark

def test_attention(use_flash):
    q, k, v = [torch.randn(2, 8, 2048, 64, device='cuda', dtype=torch.float16) for _ in range(3)]

    if use_flash:
        with torch.backends.cuda.sdp_kernel(enable_flash=True):
            return F.scaled_dot_product_attention(q, k, v)
    else:
        attn = (q @ k.transpose(-2, -1) / 8.0).softmax(dim=-1)
        return attn @ v

# Benchmark
t_flash = benchmark.Timer(stmt='test_attention(True)', globals=globals())
t_standard = benchmark.Timer(stmt='test_attention(False)', globals=globals())

print(f"Flash: {t_flash.timeit(100).mean:.3f}s")
print(f"Standard: {t_standard.timeit(100).mean:.3f}s")
```

Expected: 2-4x speedup for sequences >512 tokens.

**Step 4: Test accuracy matches baseline**

```python
# Compare outputs
q, k, v = [torch.randn(1, 8, 512, 64, device='cuda', dtype=torch.float16) for _ in range(3)]

# Flash Attention
out_flash = F.scaled_dot_product_attention(q, k, v)

# Standard attention
attn_weights = torch.softmax(q @ k.transpose(-2, -1) / 8.0, dim=-1)
out_standard = attn_weights @ v

# Check difference
diff = (out_flash - out_standard).abs().max()
print(f"Max difference: {diff:.6f}")
# Should be <1e-3 for float16
```

### Workflow 2: Use flash-attn library for advanced features

For multi-query attention, sliding window, or H100 FP8.

Copy this checklist:

```
flash-attn Library Setup:
- [ ] Step 1: Install flash-attn library
- [ ] Step 2: Modify attention code
- [ ] Step 3: Enable advanced features
- [ ] Step 4: Benchmark performance
```

**Step 1: Install flash-attn library**

```bash
# NVIDIA GPUs (CUDA 12.0+)
pip install flash-attn --no-build-isolation

# Verify installation
python -c "from flash_attn import flash_attn_func; print('Success')"
```

**Step 2: Modify attention code**

```python
from flash_attn import flash_attn_func

# Input: [batch_size, seq_len, num_heads, head_dim]
# Transpose from [batch, heads, seq, dim] if needed
q = q.transpose(1, 2)  # [batch, seq, heads, dim]
k = k.transpose(1, 2)
v = v.transpose(1, 2)

out = flash_attn_func(
    q, k, v,
    dropout_p=0.1,
    causal=True,  # For autoregressive models
    window_size=(-1, -1),  # No sliding window
    softmax_scale=None  # Auto-scale
)

out = out.transpose(1, 2)  # Back to [batch, heads, seq, dim]
```

**Step 3: Enable advanced features**

Multi-query attention (shared K/V across heads):
```python
from flash_attn import flash_attn_func

# q: [batch, seq, num_q_heads, dim]
# k, v: [batch, seq, num_kv_heads, dim]  # Fewer KV heads
out = flash_attn_func(q, k, v)  # Automatically handles MQA
```

Sliding window attention (local attention):
```python
# Only attend to window of 256 tokens before/after
out = flash_attn_func(
    q, k, v,
    window_size=(256, 256),  # (left, right) window
    causal=True
)
```

**Step 4: Benchmark performance**

```python
import torch
from flash_attn import flash_attn_func
import time

q, k, v = [torch.randn(4, 4096, 32, 64, device='cuda', dtype=torch.float16) for _ in range(3)]

# Warmup
for _ in range(10):
    _ = flash_attn_func(q, k, v)

# Benchmark
torch.cuda.synchronize()
start = time.time()
for _ in range(100):
    out = flash_attn_func(q, k, v)
    torch.cuda.synchronize()
end = time.time()

print(f"Time per iteration: {(end-start)/100*1000:.2f}ms")
print(f"Memory allocated: {torch.cuda.max_memory_allocated()/1e9:.2f}GB")
```

### Workflow 3: H100 FP8 optimization (FlashAttention-3)

For maximum performance on H100 GPUs.

```
FP8 Setup:
- [ ] Step 1: Verify H100 GPU available
- [ ] Step 2: Install flash-attn with FP8 support
- [ ] Step 3: Convert inputs to FP8
- [ ] Step 4: Run with FP8 attention
```

**Step 1: Verify H100 GPU**

```bash
nvidia-smi --query-gpu=name --format=csv
# Should show "H100" or "H800"
```

**Step 2: Install flash-attn with FP8 support**

```bash
pip install flash-attn --no-build-isolation
# FP8 support included for H100
```

**Step 3: Convert inputs to FP8**

```python
import torch

q = torch.randn(2, 4096, 32, 64, device='cuda', dtype=torch.float16)
k = torch.randn(2, 4096, 32, 64, device='cuda', dtype=torch.float16)
v = torch.randn(2, 4096, 32, 64, device='cuda', dtype=torch.float16)

# Convert to float8_e4m3 (FP8)
q_fp8 = q.to(torch.float8_e4m3fn)
k_fp8 = k.to(torch.float8_e4m3fn)
v_fp8 = v.to(torch.float8_e4m3fn)
```

**Step 4: Run with FP8 attention**

```python
from flash_attn import flash_attn_func

# FlashAttention-3 automatically uses FP8 kernels on H100
out = flash_attn_func(q_fp8, k_fp8, v_fp8)
# Result: ~1.2 PFLOPS, 1.5-2x faster than FP16
```

## When to use vs alternatives

**Use Flash Attention when:**
- Training transformers with sequences >512 tokens
- Running inference with long context (>2K tokens)
- GPU memory constrained (OOM with standard attention)
- Need 2-4x speedup without accuracy loss
- Using PyTorch 2.2+ or can install flash-attn

**Use alternatives instead:**
- **Standard attention**: Sequences <256 tokens (overhead not worth it)
- **xFormers**: Need more attention variants (not just speed)
- **Memory-efficient attention**: CPU inference (Flash Attention needs GPU)

## Common issues

**Issue: ImportError: cannot import flash_attn**

Install with no-build-isolation flag:
```bash
pip install flash-attn --no-build-isolation
```

Or install CUDA toolkit first:
```bash
conda install cuda -c nvidia
pip install flash-attn --no-build-isolation
```

**Issue: Slower than expected (no speedup)**

Flash Attention benefits increase with sequence length:
- <512 tokens: Minimal speedup (10-20%)
- 512-2K tokens: 2-3x speedup
- >2K tokens: 3-4x speedup

Check sequence length is sufficient.

**Issue: RuntimeError: CUDA error**

Verify GPU supports Flash Attention:
```python
import torch
print(torch.cuda.get_device_capability())
# Should be ≥(7, 5) for Turing+
```

Flash Attention requires:
- Ampere (A100, A10): ✅ Full support
- Turing (T4): ✅ Supported
- Volta (V100): ❌ Not supported

**Issue: Accuracy degradation**

Check dtype is float16 or bfloat16 (not float32):
```python
q = q.to(torch.float16)  # Or torch.bfloat16
```

Flash Attention uses float16/bfloat16 for speed. Float32 not supported.

## Advanced topics

**Integration with HuggingFace Transformers**: See [references/transformers-integration.md](references/transformers-integration.md) for enabling Flash Attention in BERT, GPT, Llama models.

**Performance benchmarks**: See [references/benchmarks.md](references/benchmarks.md) for detailed speed and memory comparisons across GPUs and sequence lengths.

**Algorithm details**: See [references/algorithm.md](references/algorithm.md) for tiling strategy, recomputation, and IO complexity analysis.

**Advanced features**: See [references/advanced-features.md](references/advanced-features.md) for rotary embeddings, ALiBi, paged KV cache, and custom attention masks.

## Hardware requirements

- **GPU**: NVIDIA Ampere+ (A100, A10, A30) or AMD MI200+
- **VRAM**: Same as standard attention (Flash Attention doesn't increase memory)
- **CUDA**: 12.0+ (11.8 minimum)
- **PyTorch**: 2.2+ for native support

**Not supported**: V100 (Volta), CPU inference

## Resources

- Paper: "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness" (NeurIPS 2022)
- Paper: "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning" (ICLR 2024)
- Blog: https://tridao.me/blog/2024/flash3/
- GitHub: https://github.com/Dao-AILab/flash-attention
- PyTorch docs: https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html



