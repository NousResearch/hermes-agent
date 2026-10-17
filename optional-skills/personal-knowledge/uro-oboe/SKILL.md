---
name: uro-oboe
description: "Uro-oboe (うろ覚え): Personal episodic memory with 1-bit quantized FAISS binary HNSW for fuzzy recall."
version: "0.2.0"
author: Hermes Agent
license: MIT
tags: [memory, episodic, faiss, sqlite, vector-search, quantization, fuzzy-recall, 1bit, hnsw]
tools:
  - episodic_store
  - episodic_recall_fuzzy
  - episodic_fetch
  - episodic_search_fts
  - episodic_delete
  - episodic_stats
requires:
  - faiss-cpu
  - sentence-transformers
  - numpy
  - sqlite3 (stdlib)
metadata:
  hermes:
    tags: [personal-knowledge, memory, vector-search]
    related_skills: []
---

## When to Use

- Store and recall personal notes, code snippets, research findings, creative ideas, sales notes
- Need fast fuzzy search over large personal knowledge base (10k+ entries)
- Want creative serendipity via controlled noise injection (`noise_ratio`)
- Two-stage retrieval: wide binary recall → precise float32 verification
- Offline/local-first: no external API calls, all data stays in SQLite

# Uro-oboe Skill

**Uro-oboe (うろ覚え)** — A personal episodic memory system for Hermes Agent using:
- **Embedding model**: `all-MiniLM-L6-v2` (384 dimensions)
- **Quantization**: 1-bit sign-only `(vec > 0).astype('uint8')`
- **Index**: `faiss.IndexBinaryHNSW` for fast fuzzy search
- **Storage**: SQLite with FTS5 full-text search, tags, and float32 vectors
- **Creative noise**: `noise_ratio` parameter intentionally mixes low-score results
- **Soft deletion**: Logical deletion flag (`is_deleted`) preserves FAISS index integrity

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────┐
│  episodic_store │────▶│  SQLite + FAISS  │◀───│episodic_fetch│
└─────────────────┘     └──────────────────┘     └─────────────┘
                              │
                              ▼
                       ┌──────────────────┐
                       │episodic_recall_fu│
                       │     (1-bit HNSW) │
                       └──────────────────┘
                              │
                              ▼
                       ┌──────────────────┐
                       │episodic_search_ft│
                       │   (FTS5 exact)   │
                       └──────────────────┘
```

## Two-Stage Retrieval

1. **Wide recall** (`episodic_recall_fuzzy`): Binary HNSW index → candidate IDs + scores
2. **Verification** (`episodic_fetch`): Full float32 vectors / FTS5 text from SQLite → precise ranking

## Tools

### episodic_store
Store a new episodic memory.

```json
{
  "text": "string (required) - The memory content",
  "tags": ["string"] - Optional tags for filtering",
  "metadata": {} - Optional JSON metadata",
  "auto_tag": false - Automatically generate tags from content"
}
```

### episodic_recall_fuzzy
Fuzzy recall using 1-bit quantized binary HNSW index.

```json
{
  "query": "string (required) - Search query",
  "k": 10 - Number of candidates to return",
  "noise_ratio": 0.1 - Fraction of low-score results to mix in (0.0-0.5)",
  "tags": ["string"] - Optional tag filter",
  "min_score": 0.0 - Minimum similarity threshold"
}
```

### episodic_fetch
Fetch full memory records by IDs.

```json
{
  "ids": ["integer"] - List of memory IDs to fetch"
}
```

### episodic_search_fts
Full-text search using SQLite FTS5 (exact keyword/phrase matching).

```json
{
  "query": "string (required) - Search query (FTS5 syntax supported)",
  "k": 10 - Maximum results to return"
}
```

### episodic_delete
Soft-delete memories by IDs (logical deletion, excluded from searches).

```json
{
  "ids": ["integer"] - List of memory IDs to delete"
}
```

### episodic_stats
Get memory system statistics.

```json
{}
```

## Installation

```bash
pip install faiss-cpu sentence-transformers numpy
```

## Usage Example

```python
# Store memories (keyword arguments, not dict)
episodic_store(text="User prefers Japanese responses", tags=["preference", "language"])
episodic_store(text="Project uses pytest with xdist", tags=["dev", "testing"])

# Auto-tagging
episodic_store(text="PHPのLaravelで非同期処理を書く時はキューを使う", auto_tag=True)
# → tags: ["php", "coding"] automatically assigned

# Fuzzy recall with creative noise
results = episodic_recall_fuzzy(query="testing preferences", k=5, noise_ratio=0.2)

# Fetch full records
memories = episodic_fetch(ids=[r["id"] for r in results["results"]])

# Exact keyword search via FTS5
fts_results = episodic_search_fts(query="Ollama", k=3)

# Soft delete
episodic_delete(ids=[1, 2, 3])

# Statistics
stats = episodic_stats()
```

## Tool JSON Schemas

### episodic_store
```json
{
  "name": "episodic_store",
  "description": "Store a new episodic memory with text, optional tags, metadata, and auto-tagging.",
  "parameters": {
    "type": "object",
    "properties": {
      "text": {"type": "string", "description": "The memory content to store"},
      "tags": {"type": "array", "items": {"type": "string"}, "description": "Optional tags for filtering"},
      "metadata": {"type": "object", "description": "Optional JSON metadata"},
      "auto_tag": {"type": "boolean", "description": "Automatically generate tags from content (default: false)"}
    },
    "required": ["text"]
  }
}
```

### episodic_recall_fuzzy
```json
{
  "name": "episodic_recall_fuzzy",
  "description": "Fuzzy recall using 1-bit quantized binary HNSW index with optional noise injection.",
  "parameters": {
    "type": "object",
    "properties": {
      "query": {"type": "string", "description": "Search query text"},
      "k": {"type": "integer", "description": "Number of candidates to return", "default": 10},
      "noise_ratio": {"type": "number", "description": "Fraction of low-score results to mix in (0.0-0.5)", "default": 0.1},
      "tags": {"type": "array", "items": {"type": "string"}, "description": "Optional tag filter"},
      "min_score": {"type": "number", "description": "Minimum similarity threshold (0.0-1.0)", "default": 0.0}
    },
    "required": ["query"]
  }
}
```

### episodic_fetch
```json
{
  "name": "episodic_fetch",
  "description": "Fetch full memory records by IDs including vectors and metadata.",
  "parameters": {
    "type": "object",
    "properties": {
      "ids": {"type": "array", "items": {"type": "integer"}, "description": "List of memory IDs to fetch"}
    },
    "required": ["ids"]
  }
}
```

### episodic_search_fts
```json
{
  "name": "episodic_search_fts",
  "description": "Full-text search using SQLite FTS5. Supports exact phrases, prefixes, AND/OR/NOT operators.",
  "parameters": {
    "type": "object",
    "properties": {
      "query": {"type": "string", "description": "Search query (FTS5 syntax: \"exact phrase\", term*, term1 AND term2, etc.)"},
      "k": {"type": "integer", "description": "Maximum results to return", "default": 10}
    },
    "required": ["query"]
  }
}
```

### episodic_delete
```json
{
  "name": "episodic_delete",
  "description": "Soft-delete memories by IDs (logical deletion, excluded from all searches).",
  "parameters": {
    "type": "object",
    "properties": {
      "ids": {"type": "array", "items": {"type": "integer"}, "description": "List of memory IDs to delete"}
    },
    "required": ["ids"]
  }
}
```

### episodic_stats
```json
{
  "name": "episodic_stats",
  "description": "Get memory system statistics (active/deleted counts, index size, embedding config).",
  "parameters": {
    "type": "object",
    "properties": {}
  }
}
```

## Files

- `tools/episodic_memory.py` - Core implementation
- `tools/__init__.py` - Tool exports
- `tests/test_episodic_memory.py` - Verification tests
- `data/` - Runtime data (SQLite DB, FAISS index, mappings)

## Consistency Guarantees

- **Startup verification**: On initialization, SQLite active count vs FAISS index size vs mapping count are compared; mismatch triggers automatic index rebuild from SQLite (source of truth).
- **Atomic writes**: SQLite INSERT → FAISS add → mapping update → index/mapping save in single transaction-like sequence.
- **Soft deletion**: FAISS index never physically shrinks; deleted IDs filtered at query time. Full cleanup via manual rebuild if needed.

## Tag Filtering Performance

- Tag filtering uses pre-fetched valid ID set (single `SELECT id FROM memories WHERE ...`) instead of N+1 per-candidate queries.
- O(1) set membership check during candidate iteration.