---
name: local-memory-stack
description: >
  100% local semantic memory engine for AI agents — BGE-M3 + ChromaDB + GLiNER.
  Zero API keys, zero cloud. TTL auto-archive, graph-guided retrieval, 5-layer quality gate.
version: 1.0.0
author: Kevin (keven221)
license: MIT
platforms: [linux, macos]
metadata:
  hermes:
    tags: [memory, semantic-search, vector-database, chromadb, embeddings, local-llm, mlops]
    category: mlops
    related_skills: [hermes-agent, model-distillation]
---

# Local Memory Stack

100% local semantic memory engine for Hermes Agent. Zero API keys, zero cloud dependency.

## When to Use

Use this skill when you need:
- Persistent semantic memory across conversations
- Local-only memory without sending data to external APIs
- Knowledge graph-guided retrieval for better context recall
- Automatic memory deduplication and quality filtering

**Not needed when**: Using Hermes's built-in memory (MEMORY.md) is sufficient, or when cloud-based memory providers (mem0, supermemory) are preferred.

## Features

| Feature | Description |
|---------|-------------|
| **BGE-M3 Embeddings** | 1024-dim vectors, multilingual, runs on MPS/CUDA/CPU |
| **ChromaDB Vector Store** | HNSW index for fast approximate nearest neighbor search |
| **GLiNER Entity Extraction** | Automatic NER for people, locations, organizations, dates |
| **Graph-Guided Retrieval** | Cluster + hub matching, 3-157× faster on large datasets |
| **Hybrid Search** | BM25 + vector + RRF fusion for best recall |
| **5-Layer Quality Gate** | Filters code artifacts, task status, trivial chat |
| **TTL Auto-Archive** | 30-365 day configurable retention by tag |
| **3-Tier Deduplication** | Embedding recall → keyword overlap → merge/skip |

## Installation

```bash
# Clone and install
git clone https://github.com/keven221/local-memory-stack.git
cd local-memory-stack
python3 -m venv venv && source venv/bin/activate
pip install -e .

# Models download automatically on first run (~3.3GB total)
# - BGE-M3: 2.2GB
# - GLiNER: 1.1GB
```

## Quick Start

### As Hermes Memory Provider

```bash
# Configure Hermes to use local-memory-stack
hermes config set memory.provider local-memory-stack

# Restart Hermes
hermes gateway restart
```

After setup, the memory engine:
1. **Auto-syncs** conversation turns (quality-gated, filters trivial chat)
2. **Prefetches** top-5 relevant memories before each LLM call
3. **Provides** `recall_memory` tool for agent-initiated searches
4. **Extracts** key facts at session end

### Standalone API

```python
from local_memory_stack import MemoryEngine

engine = MemoryEngine()
engine.start()

# Write a memory
engine.write(
    text="User prefers dark theme and concise responses",
    source="conversation",
    tags=["preference", "ui"],
    auto_extract=True,
)

# Search with reranking
results = engine.query_with_rerank(
    text="What theme does the user prefer?",
    top_k=3,
)

for entry in results:
    print(f"[{entry.score:.2f}] {entry.text}")
```

### REST API

```bash
# Start the server
python -m local_memory_stack.server

# Write
curl -X POST http://localhost:8900/write \
  -H "Content-Type: application/json" \
  -d '{"text": "User prefers dark theme", "source": "conversation"}'

# Search
curl "http://localhost:8900/search?query=theme+preference&top_k=3"

# Health check
curl http://localhost:8900/health
```

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    MemoryEngine                         │
├─────────────────────────────────────────────────────────┤
│  Write Path:                                            │
│    User text → Quality Gate → Dedup → GLiNER → ChromaDB │
│                                                         │
│  Read Path:                                             │
│    Query → BGE-M3 embed → HNSW recall → Rerank → Top-K │
│                                                         │
│  Graph Path (optional):                                 │
│    Query → Cluster locate → Hub match → Neighbor expand │
└─────────────────────────────────────────────────────────┘
```

## Performance

| Operation | Latency | Notes |
|-----------|---------|-------|
| Cold start | ~12s | One-time model load |
| Write | ~50ms | With dedup + entity extraction |
| Semantic search | ~17ms | Vector similarity only |
| Reranked search | ~35ms | With BGE-M3 reranker |
| Graph-guided search | ~50ms | Cluster + hub + neighbors |

## Configuration

### Memory Provider Config

```python
# In hermes_memory_provider.py or config
provider = LocalMemoryStackProvider(
    auto_sync=True,           # Auto-sync conversation turns
    prefetch_top_k=5,         # Number of memories to prefetch
    quality_gate=True,        # Enable 5-layer quality filtering
    graph_enabled=True,       # Enable graph-guided retrieval
)
```

### TTL Configuration

```python
# Configure retention by tag
ttl_config = {
    "preference": 365,  # Keep preferences for 1 year
    "project": 180,     # Keep project context for 6 months
    "conversation": 30, # Keep raw conversations for 30 days
}
```

## Pitfalls

### ⚠️ First Run Downloads ~3.3GB Models

BGE-M3 (2.2GB) and GLiNER (1.1GB) download automatically on first run. Ensure stable internet and sufficient disk space.

### ⚠️ MPS/CUDA Memory

On Apple Silicon, BGE-M3 uses ~2GB GPU memory. On CUDA, ensure sufficient VRAM. Falls back to CPU if GPU unavailable.

### ⚠️ Port 8900 Conflict

Default REST API port is 8900. Change with:
```bash
python -m local_memory_stack.server --port 8901
```

### ⚠️ Quality Gate Filters Aggressively

The 5-layer quality gate filters:
- Code artifacts (JSON, YAML, SQL fragments)
- Task status updates ("done", "completed", "TODO")
- Trivial chat ("ok", "thanks", "got it")
- Very short text (< 10 characters)
- Very long text (> 5000 characters)

Disable with `quality_gate=False` if needed.

## Verification

```bash
# 1. Check health
curl http://localhost:8900/health
# Expected: {"status": "ok"}

# 2. Write and search
curl -X POST http://localhost:8900/write \
  -H "Content-Type: application/json" \
  -d '{"text": "Test memory entry", "source": "test"}'

curl "http://localhost:8900/search?query=test&top_k=1"
# Expected: Results with "Test memory entry"

# 3. Check stats
curl http://localhost:8900/stats
# Expected: {"total_memories": N, ...}
```

## Links

- **GitHub**: https://github.com/keven221/local-memory-stack
- **Issues**: https://github.com/keven221/local-memory-stack/issues
- **Hermes Agent**: https://github.com/NousResearch/hermes-agent
