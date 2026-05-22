# qdrant_local memory provider

Local hybrid memory provider for Hermes Agent.

Current status: additive local provider implemented through Qdrant projection phase.

- Creates a profile-scoped SQLite registry under `$HERMES_HOME/memory/qdrant_local/`.
- SQLite registry/ledger is the source of truth; Qdrant is a disposable rebuildable projection.
- Indexes session turns idempotently into `documents`, `chunks`, `chunks_fts`, `ingest_events`, and `audit_events`.
- Blocks secret-like content before chunk storage and records an `ingest_blocked` audit event.
- Supports `shadow`, `fts_only`, and `vector` modes.
- `shadow` mode indexes locally but does not inject retrieved memory into prompts.
- `fts_only` mode returns bounded cited recall from SQLite FTS/BM25.
- `vector` mode can rebuild a local Qdrant collection from SQLite chunks and hydrate final results from SQLite.
- Uses deterministic local hash embeddings as the dependency-free baseline; this can be replaced later by a heavier local embedding backend behind the same ledger/projection boundary.
- Exposes no model tools by default. Explicit tools are opt-in with `enable_tools: true`.

Config sketch:

```yaml
memory:
  provider: qdrant_local
  qdrant_local:
    mode: shadow  # off | shadow | fts_only | vector
    storage_path: memory/qdrant_local
    collection: hermes_qdrant_local
    embedding_dimensions: 384
    enable_tools: false
```

Opt-in tools when `enable_tools: true`:

- `vector_memory_status` — status only, no stored content.
- `vector_memory_rebuild` — rebuild Qdrant local projection from SQLite ledger.
- `vector_memory_search` — bounded local private memory search; results are hydrated from SQLite and exclude vectors.

Optional dependency:

- `qdrant-client==1.18.0` is declared as `hermes-agent[qdrant-local]` and in `tools.lazy_deps` as `memory.qdrant_local`.

Privacy and safety invariants:

- Do not store secrets in provider config.
- Secrets, Telegram session files, `.env`, OAuth tokens and credentials must never be indexed.
- Retrieval must enforce scope/ACL before backend search and again after SQLite hydration.
- Keep this provider disabled/inert unless explicitly selected in Hermes memory config.
