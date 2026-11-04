# Multi-GPU Deployment Guide

Comprehensive guide to scaling TensorRT-LLM across multiple GPUs and nodes.

## Parallelism Strategies

### Tensor Parallelism (TP)

**What it does**: Splits model layers across GPUs horizontally.

**Use case**:
- Model fits in total GPU memory but not single GPU
- Need low latency (single forward pass)
- GPUs on same node (NVLink required for best performance)

**Example** (Llama 3-70B on 4× A100):
```python
from tensorrt_llm import LLM

llm = LLM(
    model="meta-llama/Meta-Llama-3-70B",
    tensor_parallel_size=4,  # Split across 4 GPUs
    dtype="fp16"
)

# Model automatically sharded across GPUs
# Single forward pass, low latency
```

**Performance**:
- Latency: ~Same as single GPU
- Throughput: 4× higher (4 GPUs)
- Communication: High (activations synced every layer)

### Pipeline Parallelism (PP)

**What it does**: Splits model layers across GPUs vertically (layer-wise).

**Use case**:
- Very large models (175B+)
- Can tolerate higher latency
- GPUs across multiple nodes

**Example** (Llama 3-405B on 8× H100):
```python
llm = LLM(
    model="meta-llama/Meta-Llama-3-405B",
    tensor_parallel_size=4,   # TP=4 within nodes
    pipeline_parallel_size=2, # PP=2 across nodes
    dtype="fp8"
)

# Total: 8 GPUs (4×2)
# Layers 0-40: Node 1 (4 GPUs with TP)
# Layers 41-80: Node 2 (4 GPUs with TP)
```

**Performance**:
- Latency: Higher (sequential through pipeline)
- Throughput: High with micro-batching
- Communication: Lower than TP

### Expert Parallelism (EP)

**What it does**: Distributes MoE experts across GPUs.

**Use case**: Mixture-of-Experts models (Mixtral, DeepSeek-V2)

**Example** (Mixtral-8x22B on 8× A100):
```python
llm = LLM(
    model="mistralai/Mixtral-8x22B",
    tensor_parallel_size=4,
    expert_parallel_size=2,  # Distribute 8 experts across 2 groups
    dtype="fp8"
)
```

## Configuration Examples

### Small model (7-13B) - Single GPU

```python
# Llama 3-8B on 1× A100 80GB
llm = LLM(
    model="meta-llama/Meta-Llama-3-8B",
    dtype="fp16"  # or fp8 for H100
)
```

**Resources**:
- GPU: 1× A100 80GB
- Memory: ~16GB model + 30GB KV cache
- Throughput: 3,000-5,000 tokens/sec

### Medium model (70B) - Multi-GPU same node

```python
# Llama 3-70B on 4× A100 80GB (NVLink)
llm = LLM(
    model="meta-llama/Meta-Llama-3-70B",
    tensor_parallel_size=4,
    dtype="fp8"  # 70GB → 35GB per GPU
)
```

**Resources**:
- GPU: 4× A100 80GB with NVLink
- Memory: ~35GB per GPU (FP8)
- Throughput: 10,000-15,000 tokens/sec
- Latency: 15-20ms per token

### Large model (405B) - Multi-node

```python
# Llama 3-405B on 2 nodes × 8 H100 = 16 GPUs
llm = LLM(
    model="meta-llama/Meta-Llama-3-405B",
    tensor_parallel_size=8,    # TP within each node
    pipeline_parallel_size=2,  # PP across 2 nodes
    dtype="fp8"
)
```

**Resources**:
- GPU: 2 nodes × 8 H100 80GB
- Memory: ~25GB per GPU (FP8)
- Throughput: 20,000-30,000 tokens/sec
- Network: InfiniBand recommended

## Server Deployment

### Single-node multi-GPU

```bash
# Llama 3-70B on 4 GPUs (automatic TP)
trtllm-serve meta-llama/Meta-Llama-3-70B \
    --tp_size 4 \
    --max_batch_size 256 \
    --dtype fp8

# Listens on http://localhost:8000
```

### Multi-node with Ray

```bash
# Node 1 (head node)
ray start --head --port=6379

# Node 2 (worker)
ray start --address='node1:6379'

# Deploy across cluster
trtllm-serve meta-llama/Meta-Llama-3-405B \
    --tp_size 8 \
    --pp_size 2 \
    --num_workers 2 \  # 2 nodes
    --dtype fp8
```

### Kubernetes deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: tensorrt-llm-llama3-70b
spec:
  replicas: 1
  template:
    spec:
      containers:
      - name: trtllm
        image: nvidia/tensorrt_llm:latest
        command:
          - trtllm-serve
          - meta-llama/Meta-Llama-3-70B
          - --tp_size=4
          - --max_batch_size=256
        resources:
          limits:
            nvidia.com/gpu: 4  # Request 4 GPUs
```

## Parallelism Decision Tree

```
Model size < 20GB?
├─ YES: Single GPU (no parallelism)
└─ NO: Model size < 80GB?
    ├─ YES: TP=2 or TP=4 (same node)
    └─ NO: Model size < 320GB?
        ├─ YES: TP=4 or TP=8 (same node, NVLink required)
        └─ NO: TP=8 + PP=2 (multi-node)
```

## Communication Optimization

### NVLink vs PCIe

**NVLink** (DGX A100, HGX H100):
- Bandwidth: 600 GB/s (A100), 900 GB/s (H100)
- Ideal for TP (high communication)
- **Recommended for all multi-GPU setups**

**PCIe**:
- Bandwidth: 64 GB/s (PCIe 4.0 x16)
- 10× slower than NVLink
- Avoid TP, use PP instead

### InfiniBand for multi-node

**HDR InfiniBand** (200 Gb/s):
- Required for multi-node TP or PP
- Latency: <1μs
- **Essential for 405B+ models**

## Monitoring Multi-GPU

```python
# Monitor GPU utilization
nvidia-smi dmon -s u

# Monitor memory
nvidia-smi dmon -s m

# Monitor NVLink utilization
nvidia-smi nvlink --status

# TensorRT-LLM built-in metrics
curl http://localhost:8000/metrics
```

**Key metrics**:
- GPU utilization: Target 80-95%
- Memory usage: Should be balanced across GPUs
- NVLink traffic: High for TP, low for PP
- Throughput: Tokens/sec across all GPUs

## Common Issues

### Imbalanced GPU memory

**Symptom**: GPU 0 has 90% memory, GPU 3 has 40%

**Solutions**:
- Verify TP/PP configuration
- Check model sharding (should be equal)
- Restart server to reset state

### Low NVLink utilization

**Symptom**: NVLink bandwidth <100 GB/s with TP=4

**Solutions**:
- Verify NVLink topology: `nvidia-smi topo -m`
- Check for PCIe fallback
- Ensure GPUs are on same NVSwitch

### OOM with multi-GPU

**Solutions**:
- Increase TP size (more GPUs)
- Reduce batch size
- Enable FP8 quantization
- Use pipeline parallelism

## Performance Scaling

### TP Scaling (Llama 3-70B, FP8)

| GPUs | TP Size | Throughput | Latency | Efficiency |
|------|---------|------------|---------|------------|
| 1 | 1 | OOM | - | - |
| 2 | 2 | 6,000 tok/s | 18ms | 85% |
| 4 | 4 | 11,000 tok/s | 16ms | 78% |
| 8 | 8 | 18,000 tok/s | 15ms | 64% |

**Note**: Efficiency drops with more GPUs due to communication overhead.

### PP Scaling (Llama 3-405B, FP8)

| Nodes | TP | PP | Total GPUs | Throughput |
|-------|----|----|------------|------------|
| 1 | 8 | 1 | 8 | OOM |
| 2 | 8 | 2 | 16 | 25,000 tok/s |
| 4 | 8 | 4 | 32 | 45,000 tok/s |

## Best Practices

1. **Prefer TP over PP** when possible (lower latency)
2. **Use NVLink** for all TP deployments
3. **Use InfiniBand** for multi-node deployments
4. **Start with smallest TP** that fits model in memory
5. **Monitor GPU balance** - all GPUs should have similar utilization
6. **Test with benchmark** before production
7. **Use FP8** on H100 for 2× speedup
