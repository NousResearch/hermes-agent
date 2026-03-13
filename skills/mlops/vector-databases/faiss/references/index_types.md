# FAISS Index Types Guide

Complete guide to choosing and using FAISS index types.

## Index selection guide

| Dataset Size | Index Type | Training | Accuracy | Speed |
|--------------|------------|----------|----------|-------|
| < 10K | Flat | No | 100% | Slow |
| 10K-1M | IVF | Yes | 95-99% | Fast |
| 1M-10M | HNSW | No | 99% | Fastest |
| > 10M | IVF+PQ | Yes | 90-95% | Fast, low memory |

## Flat indices (exact search)

### IndexFlatL2 - L2 (Euclidean) distance

```python
import faiss
import numpy as np

d = 128  # Dimension
index = faiss.IndexFlatL2(d)

# Add vectors
vectors = np.random.random((1000, d)).astype('float32')
index.add(vectors)

# Search
k = 5
query = np.random.random((1, d)).astype('float32')
distances, indices = index.search(query, k)
```

**Use when:**
- Dataset < 10,000 vectors
- Need 100% accuracy
- Serving as baseline

### IndexFlatIP - Inner product (cosine similarity)

```python
# For cosine similarity, normalize vectors first
import faiss

d = 128
index = faiss.IndexFlatIP(d)

# Normalize vectors (required for cosine similarity)
faiss.normalize_L2(vectors)
index.add(vectors)

# Search
faiss.normalize_L2(query)
distances, indices = index.search(query, k)
```

**Use when:**
- Need cosine similarity
- Recommendation systems
- Text embeddings

## IVF indices (inverted file)

### IndexIVFFlat - Cluster-based search

```python
# Create quantizer
quantizer = faiss.IndexFlatL2(d)

# Create IVF index with 100 clusters
nlist = 100  # Number of clusters
index = faiss.IndexIVFFlat(quantizer, d, nlist)

# Train on data (required!)
index.train(vectors)

# Add vectors
index.add(vectors)

# Search (nprobe = clusters to search)
index.nprobe = 10  # Search 10 closest clusters
distances, indices = index.search(query, k)
```

**Parameters:**
- `nlist`: Number of clusters (√N to 4√N recommended)
- `nprobe`: Clusters to search (1-nlist, higher = more accurate)

**Use when:**
- Dataset 10K-1M vectors
- Need fast approximate search
- Can afford training time

### Tuning nprobe

```python
# Test different nprobe values
for nprobe in [1, 5, 10, 20, 50]:
    index.nprobe = nprobe
    distances, indices = index.search(query, k)
    # Measure recall/speed trade-off
```

**Guidelines:**
- `nprobe=1`: Fastest, ~50% recall
- `nprobe=10`: Good balance, ~95% recall
- `nprobe=nlist`: Exact search (same as Flat)

## HNSW indices (graph-based)

### IndexHNSWFlat - Hierarchical NSW

```python
# HNSW index
M = 32  # Number of connections per layer (16-64)
index = faiss.IndexHNSWFlat(d, M)

# Optional: Set ef_construction (build time parameter)
index.hnsw.efConstruction = 40  # Higher = better quality, slower build

# Add vectors (no training needed!)
index.add(vectors)

# Search
index.hnsw.efSearch = 16  # Search time parameter
distances, indices = index.search(query, k)
```

**Parameters:**
- `M`: Connections per layer (16-64, default 32)
- `efConstruction`: Build quality (40-200, higher = better)
- `efSearch`: Search quality (16-512, higher = more accurate)

**Use when:**
- Need best quality approximate search
- Can afford higher memory (more connections)
- Dataset 1M-10M vectors

## PQ indices (product quantization)

### IndexPQ - Memory-efficient

```python
# PQ reduces memory by 16-32×
m = 8   # Number of subquantizers (divides d)
nbits = 8  # Bits per subquantizer

index = faiss.IndexPQ(d, m, nbits)

# Train (required!)
index.train(vectors)

# Add vectors
index.add(vectors)

# Search
distances, indices = index.search(query, k)
```

**Parameters:**
- `m`: Subquantizers (d must be divisible by m)
- `nbits`: Bits per code (8 or 16)

**Memory savings:**
- Original: d × 4 bytes (float32)
- PQ: m bytes
- Compression ratio: 4d/m

**Use when:**
- Limited memory
- Large datasets (> 10M vectors)
- Can accept ~90-95% accuracy

### IndexIVFPQ - IVF + PQ combined

```python
# Best for very large datasets
nlist = 4096
m = 8
nbits = 8

quantizer = faiss.IndexFlatL2(d)
index = faiss.IndexIVFPQ(quantizer, d, nlist, m, nbits)

# Train
index.train(vectors)
index.add(vectors)

# Search
index.nprobe = 32
distances, indices = index.search(query, k)
```

**Use when:**
- Dataset > 10M vectors
- Need fast search + low memory
- Can accept 90-95% accuracy

## GPU indices

### Single GPU

```python
import faiss

# Create CPU index
index_cpu = faiss.IndexFlatL2(d)

# Move to GPU
res = faiss.StandardGpuResources()  # GPU resources
index_gpu = faiss.index_cpu_to_gpu(res, 0, index_cpu)  # GPU 0

# Use normally
index_gpu.add(vectors)
distances, indices = index_gpu.search(query, k)
```

### Multi-GPU

```python
# Use all available GPUs
index_gpu = faiss.index_cpu_to_all_gpus(index_cpu)

# Or specific GPUs
gpus = [0, 1, 2, 3]  # Use GPUs 0-3
index_gpu = faiss.index_cpu_to_gpus_list(index_cpu, gpus)
```

**Speedup:**
- Single GPU: 10-50× faster than CPU
- Multi-GPU: Near-linear scaling

## Index factory

```python
# Easy index creation with string descriptors
index = faiss.index_factory(d, "IVF100,Flat")
index = faiss.index_factory(d, "HNSW32")
index = faiss.index_factory(d, "IVF4096,PQ8")

# Train and use
index.train(vectors)
index.add(vectors)
```

**Common descriptors:**
- `"Flat"`: Exact search
- `"IVF100,Flat"`: IVF with 100 clusters
- `"HNSW32"`: HNSW with M=32
- `"IVF4096,PQ8"`: IVF + PQ compression

## Performance comparison

### Search speed (1M vectors, k=10)

| Index | Build Time | Search Time | Memory | Recall |
|-------|------------|-------------|--------|--------|
| Flat | 0s | 50ms | 512 MB | 100% |
| IVF100 | 5s | 2ms | 512 MB | 95% |
| HNSW32 | 60s | 1ms | 1GB | 99% |
| IVF4096+PQ8 | 30s | 3ms | 32 MB | 90% |

*CPU (16 cores), 128-dim vectors*

## Best practices

1. **Start with Flat** - Baseline for comparison
2. **Use IVF for medium datasets** - Good balance
3. **Use HNSW for best quality** - If memory allows
4. **Add PQ for memory savings** - Large datasets
5. **GPU for > 100K vectors** - 10-50× speedup
6. **Tune nprobe/efSearch** - Trade-off speed/accuracy
7. **Train on representative data** - Better clustering
8. **Save trained indices** - Avoid retraining

## Resources

- **Wiki**: https://github.com/facebookresearch/faiss/wiki
- **Paper**: https://arxiv.org/abs/1702.08734
