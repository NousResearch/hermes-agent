# Qdrant Troubleshooting Guide

## Installation Issues

### Docker Issues

**Error**: `Cannot connect to Docker daemon`

**Fix**:
```bash
# Start Docker daemon
sudo systemctl start docker

# Or use Docker Desktop on Mac/Windows
open -a Docker
```

**Error**: `Port 6333 already in use`

**Fix**:
```bash
# Find process using port
lsof -i :6333

# Kill process or use different port
docker run -p 6334:6333 qdrant/qdrant
```

### Python Client Issues

**Error**: `ModuleNotFoundError: No module named 'qdrant_client'`

**Fix**:
```bash
pip install qdrant-client

# With specific version
pip install qdrant-client>=1.12.0
```

**Error**: `grpc._channel._InactiveRpcError`

**Fix**:
```bash
# Install with gRPC support
pip install 'qdrant-client[grpc]'

# Or disable gRPC
client = QdrantClient(host="localhost", port=6333, prefer_grpc=False)
```

## Connection Issues

### Cannot Connect to Server

**Error**: `ConnectionRefusedError: [Errno 111] Connection refused`

**Solutions**:

1. **Check server is running**:
```bash
docker ps | grep qdrant
curl http://localhost:6333/healthz
```

2. **Verify port binding**:
```bash
# Check listening ports
netstat -tlnp | grep 6333

# Docker port mapping
docker port <container_id>
```

3. **Use correct host**:
```python
# Docker on Linux
client = QdrantClient(host="localhost", port=6333)

# Docker on Mac/Windows with networking issues
client = QdrantClient(host="127.0.0.1", port=6333)

# Inside Docker network
client = QdrantClient(host="qdrant", port=6333)
```

### Timeout Errors

**Error**: `TimeoutError: Connection timed out`

**Fix**:
```python
# Increase timeout
client = QdrantClient(
    host="localhost",
    port=6333,
    timeout=60  # seconds
)

# For large operations
client.upsert(
    collection_name="documents",
    points=large_batch,
    wait=False  # Don't wait for indexing
)
```

### SSL/TLS Errors

**Error**: `ssl.SSLCertVerificationError`

**Fix**:
```python
# Qdrant Cloud
client = QdrantClient(
    url="https://cluster.cloud.qdrant.io",
    api_key="your-api-key"
)

# Self-signed certificate
client = QdrantClient(
    host="localhost",
    port=6333,
    https=True,
    verify=False  # Disable verification (not recommended for production)
)
```

## Collection Issues

### Collection Already Exists

**Error**: `ValueError: Collection 'documents' already exists`

**Fix**:
```python
# Check before creating
collections = client.get_collections().collections
names = [c.name for c in collections]

if "documents" not in names:
    client.create_collection(...)

# Or recreate
client.recreate_collection(
    collection_name="documents",
    vectors_config=VectorParams(size=384, distance=Distance.COSINE)
)
```

### Collection Not Found

**Error**: `NotFoundException: Collection 'docs' not found`

**Fix**:
```python
# List available collections
collections = client.get_collections()
print([c.name for c in collections.collections])

# Check exact name (case-sensitive)
try:
    info = client.get_collection("documents")
except Exception as e:
    print(f"Collection not found: {e}")
```

### Vector Dimension Mismatch

**Error**: `ValueError: Vector dimension mismatch. Expected 384, got 768`

**Fix**:
```python
# Check collection config
info = client.get_collection("documents")
print(f"Expected dimension: {info.config.params.vectors.size}")

# Recreate with correct dimension
client.recreate_collection(
    collection_name="documents",
    vectors_config=VectorParams(size=768, distance=Distance.COSINE)  # Match your embeddings
)
```

## Search Issues

### Empty Search Results

**Problem**: Search returns empty results.

**Solutions**:

1. **Verify data exists**:
```python
info = client.get_collection("documents")
print(f"Points: {info.points_count}")

# Scroll to check data
points, _ = client.scroll(
    collection_name="documents",
    limit=10,
    with_payload=True
)
print(points)
```

2. **Check vector format**:
```python
# Must be list of floats
query_vector = embedding.tolist()  # Convert numpy to list

# Check dimensions
print(f"Query dimension: {len(query_vector)}")
```

3. **Verify filter conditions**:
```python
# Test without filter first
results = client.search(
    collection_name="documents",
    query_vector=query,
    limit=10
    # No filter
)

# Then add filter incrementally
```

### Slow Search Performance

**Problem**: Search takes too long.

**Solutions**:

1. **Create payload indexes**:
```python
# Index fields used in filters
client.create_payload_index(
    collection_name="documents",
    field_name="category",
    field_schema="keyword"
)
```

2. **Enable quantization**:
```python
client.update_collection(
    collection_name="documents",
    quantization_config=ScalarQuantization(
        scalar=ScalarQuantizationConfig(type=ScalarType.INT8)
    )
)
```

3. **Tune HNSW parameters**:
```python
# Faster search (less accurate)
client.update_collection(
    collection_name="documents",
    hnsw_config=HnswConfigDiff(ef_construct=64, m=8)
)

# Use ef search parameter
results = client.search(
    collection_name="documents",
    query_vector=query,
    search_params={"hnsw_ef": 64},  # Lower = faster
    limit=10
)
```

4. **Use gRPC**:
```python
client = QdrantClient(
    host="localhost",
    port=6333,
    grpc_port=6334,
    prefer_grpc=True
)
```

### Inconsistent Results

**Problem**: Same query returns different results.

**Solutions**:

1. **Wait for indexing**:
```python
client.upsert(
    collection_name="documents",
    points=points,
    wait=True  # Wait for index update
)
```

2. **Check replication consistency**:
```python
# Strong consistency read
results = client.search(
    collection_name="documents",
    query_vector=query,
    consistency="all"  # Read from all replicas
)
```

## Upsert Issues

### Batch Upsert Fails

**Error**: `PayloadError: Payload too large`

**Fix**:
```python
# Split into smaller batches
def batch_upsert(client, collection, points, batch_size=100):
    for i in range(0, len(points), batch_size):
        batch = points[i:i + batch_size]
        client.upsert(
            collection_name=collection,
            points=batch,
            wait=True
        )

batch_upsert(client, "documents", large_points_list)
```

### Invalid Point ID

**Error**: `ValueError: Invalid point ID`

**Fix**:
```python
# Valid ID types: int or UUID string
from uuid import uuid4

# Integer ID
PointStruct(id=123, vector=vec, payload={})

# UUID string
PointStruct(id=str(uuid4()), vector=vec, payload={})

# NOT valid
PointStruct(id="custom-string-123", ...)  # Use UUID format
```

### Payload Validation Errors

**Error**: `ValidationError: Invalid payload`

**Fix**:
```python
# Ensure JSON-serializable payload
import json

payload = {
    "title": "Document",
    "count": 42,
    "tags": ["a", "b"],
    "nested": {"key": "value"}
}

# Validate before upsert
json.dumps(payload)  # Should not raise

# Avoid non-serializable types
# NOT valid: datetime, numpy arrays, custom objects
payload = {
    "timestamp": datetime.now().isoformat(),  # Convert to string
    "vector": embedding.tolist()  # Convert numpy to list
}
```

## Memory Issues

### Out of Memory

**Error**: `MemoryError` or container killed

**Solutions**:

1. **Enable on-disk storage**:
```python
client.create_collection(
    collection_name="large_collection",
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    on_disk_payload=True,  # Store payloads on disk
    hnsw_config=HnswConfigDiff(on_disk=True)  # Store HNSW on disk
)
```

2. **Use quantization**:
```python
# 4x memory reduction
client.update_collection(
    collection_name="large_collection",
    quantization_config=ScalarQuantization(
        scalar=ScalarQuantizationConfig(
            type=ScalarType.INT8,
            always_ram=False  # Keep on disk
        )
    )
)
```

3. **Increase Docker memory**:
```bash
docker run -m 8g -p 6333:6333 qdrant/qdrant
```

4. **Configure Qdrant storage**:
```yaml
# config.yaml
storage:
  performance:
    max_search_threads: 2
  optimizers:
    memmap_threshold_kb: 20000
```

### High Memory Usage During Indexing

**Fix**:
```python
# Increase indexing threshold for bulk loads
client.update_collection(
    collection_name="documents",
    optimizer_config={
        "indexing_threshold": 50000  # Delay indexing
    }
)

# Bulk insert
client.upsert(collection_name="documents", points=all_points, wait=False)

# Then optimize
client.update_collection(
    collection_name="documents",
    optimizer_config={
        "indexing_threshold": 10000  # Resume normal indexing
    }
)
```

## Cluster Issues

### Node Not Joining Cluster

**Problem**: New node fails to join cluster.

**Fix**:
```bash
# Check network connectivity
docker exec qdrant-node-2 ping qdrant-node-1

# Verify bootstrap URL
docker logs qdrant-node-2 | grep bootstrap

# Check Raft state
curl http://localhost:6333/cluster
```

### Split Brain

**Problem**: Cluster has inconsistent state.

**Fix**:
```bash
# Force leader election
curl -X POST http://localhost:6333/cluster/recover

# Or restart minority nodes
docker restart qdrant-node-2 qdrant-node-3
```

### Replication Lag

**Problem**: Replicas fall behind.

**Fix**:
```python
# Check collection status
info = client.get_collection("documents")
print(f"Status: {info.status}")

# Use strong consistency for critical writes
client.upsert(
    collection_name="documents",
    points=points,
    ordering=WriteOrdering.STRONG
)
```

## Performance Tuning

### Benchmark Configuration

```python
import time
import numpy as np

def benchmark_search(client, collection, n_queries=100, dimension=384):
    # Generate random queries
    queries = [np.random.rand(dimension).tolist() for _ in range(n_queries)]

    # Warmup
    for q in queries[:10]:
        client.search(collection_name=collection, query_vector=q, limit=10)

    # Benchmark
    start = time.perf_counter()
    for q in queries:
        client.search(collection_name=collection, query_vector=q, limit=10)
    elapsed = time.perf_counter() - start

    print(f"QPS: {n_queries / elapsed:.2f}")
    print(f"Latency: {elapsed / n_queries * 1000:.2f}ms")

benchmark_search(client, "documents")
```

### Optimal HNSW Parameters

```python
# High recall (slower)
client.create_collection(
    collection_name="high_recall",
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    hnsw_config=HnswConfigDiff(
        m=32,              # More connections
        ef_construct=200   # Higher build quality
    )
)

# High speed (lower recall)
client.create_collection(
    collection_name="high_speed",
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    hnsw_config=HnswConfigDiff(
        m=8,               # Fewer connections
        ef_construct=64    # Lower build quality
    )
)

# Balanced
client.create_collection(
    collection_name="balanced",
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    hnsw_config=HnswConfigDiff(
        m=16,              # Default
        ef_construct=100   # Default
    )
)
```

## Debugging Tips

### Enable Verbose Logging

```python
import logging

logging.basicConfig(level=logging.DEBUG)
logging.getLogger("qdrant_client").setLevel(logging.DEBUG)
```

### Check Server Logs

```bash
# Docker logs
docker logs -f qdrant

# With timestamps
docker logs --timestamps qdrant

# Last 100 lines
docker logs --tail 100 qdrant
```

### Inspect Collection State

```python
# Collection info
info = client.get_collection("documents")
print(f"Status: {info.status}")
print(f"Points: {info.points_count}")
print(f"Segments: {len(info.segments)}")
print(f"Config: {info.config}")

# Sample points
points, _ = client.scroll(
    collection_name="documents",
    limit=5,
    with_payload=True,
    with_vectors=True
)
for p in points:
    print(f"ID: {p.id}, Payload: {p.payload}")
```

### Test Connection

```python
def test_connection(host="localhost", port=6333):
    try:
        client = QdrantClient(host=host, port=port, timeout=5)
        collections = client.get_collections()
        print(f"Connected! Collections: {len(collections.collections)}")
        return True
    except Exception as e:
        print(f"Connection failed: {e}")
        return False

test_connection()
```

## Getting Help

1. **Documentation**: https://qdrant.tech/documentation/
2. **GitHub Issues**: https://github.com/qdrant/qdrant/issues
3. **Discord**: https://discord.gg/qdrant
4. **Stack Overflow**: Tag `qdrant`

### Reporting Issues

Include:
- Qdrant version: `curl http://localhost:6333/`
- Python client version: `pip show qdrant-client`
- Full error traceback
- Minimal reproducible code
- Collection configuration
