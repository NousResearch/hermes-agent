# Pinecone Deployment Guide

Production deployment patterns for Pinecone.

## Serverless vs Pod-based

### Serverless (Recommended)

```python
from pinecone import Pinecone, ServerlessSpec

pc = Pinecone(api_key="your-key")

# Create serverless index
pc.create_index(
    name="my-index",
    dimension=1536,
    metric="cosine",
    spec=ServerlessSpec(
        cloud="aws",  # or "gcp", "azure"
        region="us-east-1"
    )
)
```

**Benefits:**
- Auto-scaling
- Pay per usage
- No infrastructure management
- Cost-effective for variable load

**Use when:**
- Variable traffic
- Cost optimization important
- Don't need consistent latency

### Pod-based

```python
from pinecone import PodSpec

pc.create_index(
    name="my-index",
    dimension=1536,
    metric="cosine",
    spec=PodSpec(
        environment="us-east1-gcp",
        pod_type="p1.x1",  # or p1.x2, p1.x4, p1.x8
        pods=2,  # Number of pods
        replicas=2  # High availability
    )
)
```

**Benefits:**
- Consistent performance
- Predictable latency
- Higher throughput
- Dedicated resources

**Use when:**
- Production workloads
- Need consistent p95 latency
- High throughput required

## Hybrid search

### Dense + Sparse vectors

```python
# Upsert with both dense and sparse vectors
index.upsert(vectors=[
    {
        "id": "doc1",
        "values": [0.1, 0.2, ...],  # Dense (semantic)
        "sparse_values": {
            "indices": [10, 45, 123],  # Token IDs
            "values": [0.5, 0.3, 0.8]   # TF-IDF/BM25 scores
        },
        "metadata": {"text": "..."}
    }
])

# Hybrid query
results = index.query(
    vector=[0.1, 0.2, ...],  # Dense query
    sparse_vector={
        "indices": [10, 45],
        "values": [0.5, 0.3]
    },
    top_k=10,
    alpha=0.5  # 0=sparse only, 1=dense only, 0.5=balanced
)
```

**Benefits:**
- Best of both worlds
- Semantic + keyword matching
- Better recall than either alone

## Namespaces for multi-tenancy

```python
# Separate data by user/tenant
index.upsert(
    vectors=[{"id": "doc1", "values": [...]}],
    namespace="user-123"
)

# Query specific namespace
results = index.query(
    vector=[...],
    namespace="user-123",
    top_k=5
)

# List namespaces
stats = index.describe_index_stats()
print(stats['namespaces'])
```

**Use cases:**
- Multi-tenant SaaS
- User-specific data isolation
- A/B testing (prod/staging namespaces)

## Metadata filtering

### Exact match

```python
results = index.query(
    vector=[...],
    filter={"category": "tutorial"},
    top_k=5
)
```

### Range queries

```python
results = index.query(
    vector=[...],
    filter={"price": {"$gte": 100, "$lte": 500}},
    top_k=5
)
```

### Complex filters

```python
results = index.query(
    vector=[...],
    filter={
        "$and": [
            {"category": {"$in": ["tutorial", "guide"]}},
            {"difficulty": {"$lte": 3}},
            {"published": {"$gte": "2024-01-01"}}
        ]
    },
    top_k=5
)
```

## Best practices

1. **Use serverless for development** - Cost-effective
2. **Switch to pods for production** - Consistent performance
3. **Implement namespaces** - Multi-tenancy
4. **Add metadata strategically** - Enable filtering
5. **Use hybrid search** - Better quality
6. **Batch upserts** - 100-200 vectors per batch
7. **Monitor usage** - Check Pinecone dashboard
8. **Set up alerts** - Usage/cost thresholds
9. **Regular backups** - Export important data
10. **Test filters** - Verify performance

## Resources

- **Docs**: https://docs.pinecone.io
- **Console**: https://app.pinecone.io
