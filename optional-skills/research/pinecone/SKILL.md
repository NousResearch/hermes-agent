---
name: pinecone
description: Managed vector database for persistent semantic retrieval in RAG pipelines, agent memory, and multi-domain knowledge bases. Use when building agents that need sub-second similarity search over embedded documents, conversations, code, or multimodal data. Covers serverless and pod-based indexes, dense/sparse/hybrid search, namespace isolation, metadata filtering, and multimodal retrieval with voyage-multimodal-3.
version: 1.0.0
author: immuhammadfurqan
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [pinecone, vector-database, rag, embeddings, semantic-search, retrieval, memory, hybrid-search, multimodal]
    related_skills: [bioinformatics, drug-discovery]
    category: research
---

# Pinecone — Vector Database for Agent RAG & Memory

## Overview

Pinecone is a managed vector database providing millisecond-latency similarity search at scale. For Hermes agents it serves two key roles:

1. **Long-term memory** — persist and retrieve conversation history, user preferences, and learned knowledge across sessions beyond what fits in context
2. **RAG retrieval** — semantic search over documents, codebases, research papers, or any corpus the agent needs to reason over

Pinecone is cloud-hosted and accessed via API — no local infrastructure required.

**Tested with:** `pinecone>=6.0.0`

## When to Use This Skill

- Agent needs to search a large knowledge base that doesn't fit in context
- Building a RAG pipeline: embed documents → store in Pinecone → retrieve at query time
- Persistent agent memory across sessions (beyond Hermes's built-in memory)
- Semantic similarity search (find similar documents, code snippets, past conversations)
- Hybrid search combining semantic meaning with exact keyword matching
- Multi-tenant knowledge isolation (separate namespaces per user/project/domain)

## Installation

```bash
# Core SDK
uv pip install "pinecone>=6.0.0"

# For hybrid (dense + sparse BM25) search
uv pip install "pinecone[grpc]>=6.0.0" pinecone-text

# Embedding model companions (choose what you need)
uv pip install voyageai openai sentence-transformers
```

## Quick Start

### Initialize and create an index

```python
import os
from pinecone import Pinecone, ServerlessSpec

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

# Create serverless index (recommended default)
if "agent-memory" not in [idx.name for idx in pc.list_indexes()]:
    pc.create_index(
        name="agent-memory",
        dimension=1024,           # Match your embedding model's output dim
        metric="cosine",          # cosine | dotproduct | euclidean
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index("agent-memory")
```

> **Security:** Always load API keys from environment variables. Never hardcode keys in source files or skills.

### Upsert vectors

```python
vectors = [
    {
        "id": "doc_001",
        "values": embedding_1024d,   # list[float]
        "metadata": {
            "source": "conversation",
            "session_id": "sess_abc",
            "text": "User prefers concise responses under 200 words",
            "timestamp": "2026-05-01"
        }
    }
]

index.upsert(vectors=vectors, namespace="user-prefs")
```

### Query with metadata filtering

```python
results = index.query(
    vector=query_embedding,
    top_k=5,
    namespace="user-prefs",
    include_metadata=True,
    filter={"source": {"$eq": "conversation"}}
)

for match in results["matches"]:
    print(f"{match['score']:.4f}  {match['metadata']['text']}")
```

## Core Capabilities

### Index Management

```python
# List all indexes
indexes = [idx.name for idx in pc.list_indexes()]

# Describe index stats
stats = index.describe_index_stats()
print(f"Total vectors: {stats['total_vector_count']}")
for ns, info in stats["namespaces"].items():
    print(f"  {ns}: {info['vector_count']} vectors")

# Delete index
pc.delete_index("my-index")
```

### Batch Upsert (for large datasets)

```python
def batch_upsert(index, vectors, namespace="", batch_size=100):
    """Upsert vectors in batches — Pinecone limit is 100 vectors per request."""
    total = (len(vectors) + batch_size - 1) // batch_size
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i + batch_size]
        index.upsert(vectors=batch, namespace=namespace)
        print(f"Batch {i // batch_size + 1}/{total} upserted")
```

### Namespace Strategy for Agent Workflows

Namespaces isolate data within a single index — ideal for multi-user or multi-project agents:

```python
# Separate by data type
index.upsert(vectors=memory_vecs,   namespace="agent-memory")
index.upsert(vectors=doc_vecs,      namespace="documents")
index.upsert(vectors=code_vecs,     namespace="codebase")

# Separate by project or user
index.upsert(vectors=vecs, namespace=f"project-{project_id}")
index.upsert(vectors=vecs, namespace=f"user-{user_id}")

# Separate by environment
index.upsert(vectors=vecs, namespace="production")
index.upsert(vectors=vecs, namespace="dev")
```

### Metadata Filtering

Supported operators: `$eq`, `$ne`, `$gt`, `$gte`, `$lt`, `$lte`, `$in`, `$nin`

```python
results = index.query(
    vector=query_emb,
    top_k=10,
    filter={
        "source":   {"$in": ["github", "confluence"]},
        "year":     {"$gte": 2024},
        "language": {"$eq": "python"}
    },
    include_metadata=True
)
```

### Hybrid Search (Dense + Sparse BM25)

Critical when exact keywords matter alongside semantic meaning — function names, API endpoints, error codes:

```python
from pinecone_text.sparse import BM25Encoder

# Fit BM25 on your corpus once, then save
bm25 = BM25Encoder()
bm25.fit(corpus_texts)
bm25.dump("bm25_model.json")
bm25 = BM25Encoder().load("bm25_model.json")

def hybrid_query(query_text: str, alpha: float = 0.5, top_k: int = 10):
    """
    alpha=1.0 -> pure dense (semantic)
    alpha=0.0 -> pure sparse (keyword)
    alpha=0.5 -> balanced hybrid
    """
    dense_vec  = dense_model.encode(query_text).tolist()
    sparse_vec = bm25.encode_queries(query_text)

    return index.query(
        vector=[v * alpha for v in dense_vec],
        sparse_vector={
            "indices": sparse_vec["indices"],
            "values":  [v * (1 - alpha) for v in sparse_vec["values"]]
        },
        top_k=top_k,
        include_metadata=True
    )
```

**Note:** Hybrid search requires `metric="dotproduct"` on index creation.

### Multimodal Retrieval (voyage-multimodal-3)

For agents working with both text and images — diagrams, screenshots, documents with figures:

```python
import voyageai
from PIL import Image

voyage = voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"])

result = voyage.multimodal_embed(
    inputs=[[text_description, Image.open("screenshot.png")]],
    model="voyage-multimodal-3",
    input_type="document"
)
embedding = result.embeddings[0]  # 1024-dim, ready to upsert
```

## Agent RAG Pipeline

Complete RAG pipeline for grounding agent responses in a knowledge base:

```python
import os
from pinecone import Pinecone
import voyageai
from openai import OpenAI

pc     = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
voyage = voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"])
oai    = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
index  = pc.Index("agent-knowledge")

def rag_query(question: str, namespace: str = "", top_k: int = 5) -> str:
    """Retrieve context from Pinecone and generate a grounded answer."""
    # 1. Embed the question
    query_emb = voyage.embed(
        [question], model="voyage-large-2", input_type="query"
    ).embeddings[0]

    # 2. Retrieve relevant chunks
    results = index.query(
        vector=query_emb,
        top_k=top_k,
        namespace=namespace,
        include_metadata=True
    )

    # 3. Build context
    context = "\n\n---\n\n".join(
        f"[Score: {m['score']:.3f}]\n{m['metadata'].get('text', '')}"
        for m in results["matches"]
    )

    # 4. Generate grounded answer
    response = oai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content":
                "Answer based only on the provided context. "
                "If the context doesn't contain enough information, say so."},
            {"role": "user", "content":
                f"Context:\n{context}\n\nQuestion: {question}"}
        ]
    )
    return response.choices[0].message.content
```

## Vector Operations Reference

```python
# Fetch by ID
fetched = index.fetch(ids=["doc_001", "doc_002"], namespace="documents")

# Update metadata
index.update(
    id="doc_001",
    set_metadata={"reviewed": True, "quality": "high"},
    namespace="documents"
)

# Delete specific vectors
index.delete(ids=["doc_003"], namespace="documents")

# Clear a namespace entirely
index.delete(delete_all=True, namespace="dev")
```

## Dimension Selection by Embedding Model

| Model | Dimension | Use Case |
|---|---|---|
| voyage-multimodal-3 | 1024 | Text + image (multimodal) |
| voyage-large-2 | 1536 | High-quality text retrieval |
| text-embedding-3-large (OpenAI) | 3072 | General-purpose text |
| text-embedding-3-small (OpenAI) | 1536 | Lighter general-purpose text |
| all-mpnet-base-v2 | 768 | Open-source general text |

**Always match index dimension to your embedding model.** Dimension cannot be changed after index creation — only by recreating the index.

## Troubleshooting

**Dimension mismatch on upsert:**
```python
emb = model.embed("test")
info = pc.describe_index("my-index")
assert len(emb) == info.dimension, f"Model dim {len(emb)} != index dim {info.dimension}"
```

**Low retrieval quality:**
- Use `input_type="query"` for queries, `"document"` for corpus (voyage, cohere)
- Try hybrid search if exact terms are being missed
- Check metadata filters aren't too restrictive

**Slow upsert for large corpora:**
- Switch to `PineconeGRPC` client (~3x faster ingest)
- Always batch in groups of ≤100 vectors

**Rate limits during ingestion:**
- The embedding API (not Pinecone) is usually the bottleneck
- Add exponential backoff on embedding calls

## Resources

- **Pinecone Docs**: https://docs.pinecone.io
- **Python SDK**: https://github.com/pinecone-io/pinecone-python-client
- **Voyage AI**: https://docs.voyageai.com
- **pinecone-text (BM25)**: https://github.com/pinecone-io/pinecone-text
- **Agent Skills Standard**: https://agentskills.io
