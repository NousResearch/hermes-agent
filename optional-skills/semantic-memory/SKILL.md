---
name: semantic-memory
description: >
  Semantic memory engine for Hermes — persistent, searchable, self-evolving
  memory backed by real vector embeddings (fastembed/bge-small-en-v1.5).
  Upgrades Hermes memory from flat markdown to hybrid BM25 + semantic search
  with temporal decay. Use when: storing facts that need semantic recall,
  searching memory by meaning (not just keywords), or when MEMORY.md is at
  capacity and facts need to be externalized to a queryable store.
requires:
  - pip: fastembed
setup: |
  pip install fastembed
  python ~/.hermes/skills/semantic-memory/scripts/mem stats
  # First run auto-creates ~/.hermes/memory-engine/db/memory.db
---

# Semantic Memory Engine

Persistent vector-backed memory for Hermes. Extends the built-in MEMORY.md
with a SQLite + embedding store that supports semantic search, temporal decay,
and hybrid BM25+cosine retrieval.

## When to use

- **Store a fact semantically**: when a fact is too detailed for MEMORY.md
  (which has a 2,200-char limit) but needs to be recalled later by meaning
- **Semantic search**: when `session_search` keyword search isn't finding
  what you need — this finds by *meaning*, not exact words
- **Auto-capture from sessions**: pipe a session summary through `session_hook.py`
  to extract and store facts automatically

## Quick reference

```bash
# Store a fact
python ~/.hermes/skills/semantic-memory/scripts/mem store "Stripe checkout was fixed by removing consent_collection param"

# Search by meaning (not keywords)
python ~/.hermes/skills/semantic-memory/scripts/mem search "what happened with payments"
# → returns: "Stripe checkout was fixed..." (0.87 cosine similarity)

# Auto-extract facts from a session summary
python ~/.hermes/skills/semantic-memory/scripts/session_hook.py "Fixed auth bug by rotating JWT secret. Deployed to staging."

# Show DB stats
python ~/.hermes/skills/semantic-memory/scripts/mem stats

# Recent facts
python ~/.hermes/skills/semantic-memory/scripts/mem recent 10

# Re-embed all facts (after model upgrade)
python ~/.hermes/skills/semantic-memory/scripts/mem ingest
```

## Options

| Flag | Description |
|------|-------------|
| `--source NAME` | Tag the fact with a source (e.g. `--source hermes`, `--source user`) |
| `--confidence 0.9` | Confidence score 0–1 (default: 0.9) |
| `--top 5` | Number of results to return on search (default: 5) |

## How it works

```
Your query
    ↓
Hybrid Retriever
    ├── BM25 (lexical match)        — exact keyword relevance
    ├── Cosine similarity (384-dim) — semantic meaning match
    └── Temporal decay (λ=0.05)    — recency weighting
    ↓
Ranked results (combined score)
    ↓
Deep Layer (optional)
    → pushes activated facts to Surface Buffer proactively
```

## Storage location

All data lives in `~/.hermes/memory-engine/`:
```
~/.hermes/memory-engine/
├── db/
│   └── memory.db      ← SQLite with embeddings (auto-created on first use)
└── scripts/           ← symlinked from this skill
```

## Integration with built-in memory

This skill **complements** (does not replace) Hermes's built-in MEMORY.md:

- **MEMORY.md** → short-lived, high-priority facts injected into every system
  prompt (keep under 2,200 chars, only what's needed every turn)
- **semantic-memory** → long-term archive, searchable by meaning, unlimited
  capacity, queried on demand when you need historical context

Workflow: when MEMORY.md approaches capacity, externalize older facts here
with `mem store`, then remove them from MEMORY.md.

## Performance (benchmarked on Apple Silicon)

| Operation | p50 | p99 |
|-----------|-----|-----|
| DB Read (10 rows) | 0.013ms | 0.020ms |
| Vector Search (384-dim, top-10) | 0.395ms | 0.422ms |
| Full Pipeline (search+rank) | 0.873ms | 0.918ms |

All operations under 1ms p99.
