# Neural Memory Plugin

Local semantic memory with knowledge graph, spreading activation, and auto-connections.

## Features

- **Semantic search** via vector embeddings (hash, tfidf, or sentence-transformers)
- **Knowledge graph** with automatic connection discovery between related memories
- **Spreading activation** for exploring connected ideas beyond direct similarity
- **Fully offline** — no API keys, no cloud, everything stored in local SQLite

## Configuration

```yaml
# config.yaml
memory:
  provider: neural
  neural:
    db_path: ~/.neural_memory/memory.db
    embedding_backend: auto  # auto|hash|tfidf|sentence-transformers
```

Or via environment variables:
- `NEURAL_MEMORY_DB_PATH` — SQLite database path
- `NEURAL_EMBEDDING_BACKEND` — Embedding backend selection

## Tools

| Tool | Description |
|------|-------------|
| `neural_remember` | Store a memory (auto-embedded, auto-connected) |
| `neural_recall` | Semantic search over stored memories |
| `neural_think` | Spreading activation — explore connected ideas |
| `neural_graph` | Knowledge graph statistics |

## Embedding Backends

- **hash** — Fast, no dependencies, deterministic
- **tfidf** — Trained on seen corpus, no external deps
- **sentence-transformers** — Best quality, requires `sentence-transformers` + PyTorch
- **auto** — Picks best available (sentence-transformers > tfidf > hash)

## Dependencies

Core: `sqlite3` (stdlib), `numpy` (for vector ops)

Optional for better embeddings:
```bash
pip install sentence-transformers
```
