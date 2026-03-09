# Qdrant Advanced Usage Guide

## Distributed Deployment

### Cluster Setup

Qdrant uses Raft consensus for distributed coordination.

```yaml
# docker-compose.yml for 3-node cluster
version: '3.8'
services:
  qdrant-node-1:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
      - "6334:6334"
      - "6335:6335"
    volumes:
      - ./node1_storage:/qdrant/storage
    environment:
      - QDRANT__CLUSTER__ENABLED=true
      - QDRANT__CLUSTER__P2P__PORT=6335
      - QDRANT__SERVICE__HTTP_PORT=6333
      - QDRANT__SERVICE__GRPC_PORT=6334

  qdrant-node-2:
    image: qdrant/qdrant:latest
    ports:
      - "6343:6333"
      - "6344:6334"
      - "6345:6335"
    volumes:
      - ./node2_storage:/qdrant/storage
    environment:
      - QDRANT__CLUSTER__ENABLED=true
      - QDRANT__CLUSTER__P2P__PORT=6335
      - QDRANT__CLUSTER__BOOTSTRAP=http://qdrant-node-1:6335
    depends_on:
      - qdrant-node-1

  qdrant-node-3:
    image: qdrant/qdrant:latest
    ports:
      - "6353:6333"
      - "6354:6334"
      - "6355:6335"
    volumes:
      - ./node3_storage:/qdrant/storage
    environment:
      - QDRANT__CLUSTER__ENABLED=true
      - QDRANT__CLUSTER__P2P__PORT=6335
      - QDRANT__CLUSTER__BOOTSTRAP=http://qdrant-node-1:6335
    depends_on:
      - qdrant-node-1
```

### Sharding Configuration

```python
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, ShardingMethod

client = QdrantClient(host="localhost", port=6333)

# Create sharded collection
client.create_collection(
    collection_name="large_collection",
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    shard_number=6,  # Number of shards
    replication_factor=2,  # Replicas per shard
    write_consistency_factor=1  # Required acks for write
)

# Check cluster status
cluster_info = client.get_cluster_info()
print(f"Peers: {cluster_info.peers}")
print(f"Raft state: {cluster_info.raft_info}")
```

### Replication and Consistency

```python
from qdrant_client.models import WriteOrdering

# Strong consistency write
client.upsert(
    collection_name="critical_data",
    points=points,
    ordering=WriteOrdering.STRONG  # Wait for all replicas
)

# Eventual consistency (faster)
client.upsert(
    collection_name="logs",
    points=points,
    ordering=WriteOrdering.WEAK  # Return after primary ack
)

# Read from specific shard
results = client.search(
    collection_name="documents",
    query_vector=query,
    consistency="majority"  # Read from majority of replicas
)
```

## Hybrid Search

### Dense + Sparse Vectors

Combine semantic (dense) and keyword (sparse) search:

```python
from qdrant_client.models import (
    VectorParams, SparseVectorParams, SparseIndexParams,
    Distance, PointStruct, SparseVector, Prefetch, Query
)

# Create hybrid collection
client.create_collection(
    collection_name="hybrid",
    vectors_config={
        "dense": VectorParams(size=384, distance=Distance.COSINE)
    },
    sparse_vectors_config={
        "sparse": SparseVectorParams(
            index=SparseIndexParams(on_disk=False)
        )
    }
)

# Insert with both vector types
def encode_sparse(text: str) -> SparseVector:
    """Simple BM25-like sparse encoding"""
    from collections import Counter
    tokens = text.lower().split()
    counts = Counter(tokens)
    # Map tokens to indices (use vocabulary in production)
    indices = [hash(t) % 30000 for t in counts.keys()]
    values = list(counts.values())
    return SparseVector(indices=indices, values=values)

client.upsert(
    collection_name="hybrid",
    points=[
        PointStruct(
            id=1,
            vector={
                "dense": dense_encoder.encode("Python programming").tolist(),
                "sparse": encode_sparse("Python programming language code")
            },
            payload={"text": "Python programming language code"}
        )
    ]
)

# Hybrid search with Reciprocal Rank Fusion (RRF)
from qdrant_client.models import FusionQuery

results = client.query_points(
    collection_name="hybrid",
    prefetch=[
        Prefetch(query=dense_query, using="dense", limit=20),
        Prefetch(query=sparse_query, using="sparse", limit=20)
    ],
    query=FusionQuery(fusion="rrf"),  # Combine results
    limit=10
)
```

### Multi-Stage Search

```python
from qdrant_client.models import Prefetch, Query

# Two-stage retrieval: coarse then fine
results = client.query_points(
    collection_name="documents",
    prefetch=[
        Prefetch(
            query=query_vector,
            limit=100,  # Broad first stage
            params={"quantization": {"rescore": False}}  # Fast, approximate
        )
    ],
    query=Query(nearest=query_vector),
    limit=10,
    params={"quantization": {"rescore": True}}  # Accurate reranking
)
```

## Recommendations

### Item-to-Item Recommendations

```python
# Find similar items
recommendations = client.recommend(
    collection_name="products",
    positive=[1, 2, 3],  # IDs user liked
    negative=[4],         # IDs user disliked
    limit=10
)

# With filtering
recommendations = client.recommend(
    collection_name="products",
    positive=[1, 2],
    query_filter={
        "must": [
            {"key": "category", "match": {"value": "electronics"}},
            {"key": "in_stock", "match": {"value": True}}
        ]
    },
    limit=10
)
```

### Lookup from Another Collection

```python
from qdrant_client.models import RecommendStrategy, LookupLocation

# Recommend using vectors from another collection
results = client.recommend(
    collection_name="products",
    positive=[
        LookupLocation(
            collection_name="user_history",
            id="user_123"
        )
    ],
    strategy=RecommendStrategy.AVERAGE_VECTOR,
    limit=10
)
```

## Advanced Filtering

### Nested Payload Filtering

```python
from qdrant_client.models import Filter, FieldCondition, MatchValue, NestedCondition

# Filter on nested objects
results = client.search(
    collection_name="documents",
    query_vector=query,
    query_filter=Filter(
        must=[
            NestedCondition(
                key="metadata",
                filter=Filter(
                    must=[
                        FieldCondition(
                            key="author.name",
                            match=MatchValue(value="John")
                        )
                    ]
                )
            )
        ]
    ),
    limit=10
)
```

### Geo Filtering

```python
from qdrant_client.models import FieldCondition, GeoRadius, GeoPoint

# Find within radius
results = client.search(
    collection_name="locations",
    query_vector=query,
    query_filter=Filter(
        must=[
            FieldCondition(
                key="location",
                geo_radius=GeoRadius(
                    center=GeoPoint(lat=40.7128, lon=-74.0060),
                    radius=5000  # meters
                )
            )
        ]
    ),
    limit=10
)

# Geo bounding box
from qdrant_client.models import GeoBoundingBox

results = client.search(
    collection_name="locations",
    query_vector=query,
    query_filter=Filter(
        must=[
            FieldCondition(
                key="location",
                geo_bounding_box=GeoBoundingBox(
                    top_left=GeoPoint(lat=40.8, lon=-74.1),
                    bottom_right=GeoPoint(lat=40.6, lon=-73.9)
                )
            )
        ]
    ),
    limit=10
)
```

### Full-Text Search

```python
from qdrant_client.models import TextIndexParams, TokenizerType

# Create text index
client.create_payload_index(
    collection_name="documents",
    field_name="content",
    field_schema=TextIndexParams(
        type="text",
        tokenizer=TokenizerType.WORD,
        min_token_len=2,
        max_token_len=15,
        lowercase=True
    )
)

# Full-text filter
from qdrant_client.models import MatchText

results = client.search(
    collection_name="documents",
    query_vector=query,
    query_filter=Filter(
        must=[
            FieldCondition(
                key="content",
                match=MatchText(text="machine learning")
            )
        ]
    ),
    limit=10
)
```

## Quantization Strategies

### Scalar Quantization (INT8)

```python
from qdrant_client.models import ScalarQuantization, ScalarQuantizationConfig, ScalarType

# ~4x memory reduction, minimal accuracy loss
client.create_collection(
    collection_name="scalar_quantized",
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    quantization_config=ScalarQuantization(
        scalar=ScalarQuantizationConfig(
            type=ScalarType.INT8,
            quantile=0.99,       # Clip extreme values
            always_ram=True     # Keep quantized vectors in RAM
        )
    )
)
```

### Product Quantization

```python
from qdrant_client.models import ProductQuantization, ProductQuantizationConfig, CompressionRatio

# ~16x memory reduction, some accuracy loss
client.create_collection(
    collection_name="product_quantized",
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    quantization_config=ProductQuantization(
        product=ProductQuantizationConfig(
            compression=CompressionRatio.X16,
            always_ram=True
        )
    )
)
```

### Binary Quantization

```python
from qdrant_client.models import BinaryQuantization, BinaryQuantizationConfig

# ~32x memory reduction, requires oversampling
client.create_collection(
    collection_name="binary_quantized",
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    quantization_config=BinaryQuantization(
        binary=BinaryQuantizationConfig(always_ram=True)
    )
)

# Search with oversampling
results = client.search(
    collection_name="binary_quantized",
    query_vector=query,
    search_params={
        "quantization": {
            "rescore": True,
            "oversampling": 2.0  # Retrieve 2x candidates, rescore
        }
    },
    limit=10
)
```

## Snapshots and Backups

### Create Snapshot

```python
# Create collection snapshot
snapshot_info = client.create_snapshot(collection_name="documents")
print(f"Snapshot: {snapshot_info.name}")

# List snapshots
snapshots = client.list_snapshots(collection_name="documents")
for s in snapshots:
    print(f"{s.name}: {s.size} bytes")

# Full storage snapshot
full_snapshot = client.create_full_snapshot()
```

### Restore from Snapshot

```python
# Download snapshot
client.download_snapshot(
    collection_name="documents",
    snapshot_name="documents-2024-01-01.snapshot",
    target_path="./backup/"
)

# Restore (via REST API)
import requests

response = requests.put(
    "http://localhost:6333/collections/documents/snapshots/recover",
    json={"location": "file:///backup/documents-2024-01-01.snapshot"}
)
```

## Collection Aliases

```python
# Create alias
client.update_collection_aliases(
    change_aliases_operations=[
        {"create_alias": {"alias_name": "production", "collection_name": "documents_v2"}}
    ]
)

# Blue-green deployment
# 1. Create new collection with updates
client.create_collection(collection_name="documents_v3", ...)

# 2. Populate new collection
client.upsert(collection_name="documents_v3", points=new_points)

# 3. Atomic switch
client.update_collection_aliases(
    change_aliases_operations=[
        {"delete_alias": {"alias_name": "production"}},
        {"create_alias": {"alias_name": "production", "collection_name": "documents_v3"}}
    ]
)

# Search via alias
results = client.search(collection_name="production", query_vector=query, limit=10)
```

## Scroll and Iteration

### Scroll Through All Points

```python
# Paginated iteration
offset = None
all_points = []

while True:
    results, offset = client.scroll(
        collection_name="documents",
        limit=100,
        offset=offset,
        with_payload=True,
        with_vectors=False
    )
    all_points.extend(results)

    if offset is None:
        break

print(f"Total points: {len(all_points)}")
```

### Filtered Scroll

```python
# Scroll with filter
results, _ = client.scroll(
    collection_name="documents",
    scroll_filter=Filter(
        must=[
            FieldCondition(key="status", match=MatchValue(value="active"))
        ]
    ),
    limit=1000
)
```

## Async Client

```python
import asyncio
from qdrant_client import AsyncQdrantClient

async def main():
    client = AsyncQdrantClient(host="localhost", port=6333)

    # Async operations
    await client.create_collection(
        collection_name="async_docs",
        vectors_config=VectorParams(size=384, distance=Distance.COSINE)
    )

    await client.upsert(
        collection_name="async_docs",
        points=points
    )

    results = await client.search(
        collection_name="async_docs",
        query_vector=query,
        limit=10
    )

    return results

results = asyncio.run(main())
```

## gRPC Client

```python
from qdrant_client import QdrantClient

# Prefer gRPC for better performance
client = QdrantClient(
    host="localhost",
    port=6333,
    grpc_port=6334,
    prefer_grpc=True  # Use gRPC when available
)

# gRPC-only client
from qdrant_client import QdrantClient

client = QdrantClient(
    host="localhost",
    grpc_port=6334,
    prefer_grpc=True,
    https=False
)
```

## Multitenancy

### Payload-Based Isolation

```python
# Single collection, filter by tenant
client.upsert(
    collection_name="multi_tenant",
    points=[
        PointStruct(
            id=1,
            vector=embedding,
            payload={"tenant_id": "tenant_a", "text": "..."}
        )
    ]
)

# Search within tenant
results = client.search(
    collection_name="multi_tenant",
    query_vector=query,
    query_filter=Filter(
        must=[FieldCondition(key="tenant_id", match=MatchValue(value="tenant_a"))]
    ),
    limit=10
)
```

### Collection-Per-Tenant

```python
# Create tenant collection
def create_tenant_collection(tenant_id: str):
    client.create_collection(
        collection_name=f"tenant_{tenant_id}",
        vectors_config=VectorParams(size=384, distance=Distance.COSINE)
    )

# Search tenant collection
def search_tenant(tenant_id: str, query_vector: list, limit: int = 10):
    return client.search(
        collection_name=f"tenant_{tenant_id}",
        query_vector=query_vector,
        limit=limit
    )
```

## Performance Monitoring

### Collection Statistics

```python
# Collection info
info = client.get_collection("documents")
print(f"Points: {info.points_count}")
print(f"Indexed vectors: {info.indexed_vectors_count}")
print(f"Segments: {len(info.segments)}")
print(f"Status: {info.status}")

# Detailed segment info
for i, segment in enumerate(info.segments):
    print(f"Segment {i}: {segment}")
```

### Telemetry

```python
# Get telemetry data
telemetry = client.get_telemetry()
print(f"Collections: {telemetry.collections}")
print(f"Operations: {telemetry.operations}")
```
