# memory-zvec — Zvec Vector Memory Plugin

**Drop-in replacement for `memory-lancedb`**. Same 6 tool schemas, same
prefetch/sync_turn/on_session_end hooks, same Ollama bge-m3:latest embedding —
but backed by Zvec v0.5.0+ with **native hybrid query** (MultiQuery + RRFReRanker).

## Benefits Over memory-lancedb

| Aspect | memory-lancedb (old) | memory-zvec (new) |
|--------|---------------------|-------------------|
| Hybrid fusion | app-layer RRF (30 hand-computed lines) | **Zvec native MultiQuery + RRFReRanker** (3 lines) |
| Scalar filter | `.where()` inside the ANN pipeline | **index-level pre-filter** (every filterable field has InvertIndexParam) |
| FTS engine | Tantivy (bolted on) | **RocksDB native** (`FtsIndexParam`) |
| Insert performance | — | **~13000 docs/s** (1024-dim) |
| Search modes | 3 (hybrid/vector/keyword) | 3 (hybrid/vector/keyword) — **kept identical** |
| Tool schemas | 5 (add/search/list/delete/stats) | 5 — **exactly identical** |
| Migration | — | direct migration; vectors are not re-embedded |

## Install

**Prerequisites:**

1. Zvec installed: `pip install zvec` (or added to the venv)
2. Ollama running: `ollama pull bge-m3:latest`
3. A Hermes agent (any profile)

**Install the plugin:**

```bash
cp -r ~/.hermes/plugins/memory-zvec /path/to/your/hermes/plugins/memory-zvec
```

## Migration: Switch from memory-lancedb

### 1. Run the migration script

```bash
python3 ~/.hermes/profiles/chip_expert/skills/hardware/vector-db-migration/references/migrate-lancedb-to-zvec.py
```

This migrates all 105 LanceDB memories into Zvec in full (vectors already exist; no re-embedding).

### 2. Edit config.yaml

Add to `$HERMES_HOME/config.yaml`:

```yaml
plugins:
  memory-zvec:
    base_url: http://localhost:11434
    embedding_model: bge-m3:latest
    vector_dim: 1024
    zvec_dir: $HERMES_HOME/记忆数据库/zvec_memory
    collection_name: memories
    batch_size: 32
    search_top_k: 5
    min_content_len: 50
    vector_weight: 0.7
    fts_weight: 0.3
    enable_hnsw_optimize: true
```

Then change `memory.provider` from `memory-lancedb` to `memory-zvec`.

### 3. Restart Hermes

```bash
# restart the agent
hermes stop && hermes start
# or reload the config (restart required)
```

### 4. Verify

After startup, say "view memory stats" or "search memory" to confirm the plugin is active. `backend: zvec` in the `vec_memory_stats` output confirms the switch succeeded.

### 5. Rollback

The Zvec and LanceDB data directories coexist. To roll back you only need to:

1. Restore `memory.provider: memory-lancedb` in `config.yaml`
2. Restart Hermes

## Configuration

| Key | Default | Description |
|-----|---------|-------------|
| `base_url` | `http://localhost:11434` | Ollama server URL |
| `embedding_model` | `bge-m3:latest` | Embedding model (Ollama) |
| `vector_dim` | `1024` | Vector dimension (bge-m3 = 1024) |
| `zvec_dir` | `$HERMES_HOME/记忆数据库/zvec_memory` | Zvec collection path |
| `collection_name` | `memories` | Collection name |
| `batch_size` | `32` | Max texts per embedding batch |
| `search_top_k` | `5` | Default top-k results |
| `min_content_len` | `50` | Skip content shorter than this |
| `vector_weight` | `0.7` | Hybrid branch weight (vector) |
| `fts_weight` | `0.3` | Hybrid branch weight (FTS) |
| `enable_hnsw_optimize` | `true` | Call optimize() after batch writes |

## Tool Schemas (Identical to memory-lancedb)

| Tool | Purpose |
|------|---------|
| `vec_memory_add(content, role, session_id, metadata)` | Store a fact |
| `vec_memory_search(query, mode, top_k, session_id, after_timestamp, before_timestamp)` | Semantic/hybrid/keyword search |
| `vec_memory_list(limit, session_id, after_timestamp, before_timestamp)` | Browse memories |
| `vec_memory_delete(memory_ids)` | Delete by IDs |
| `vec_memory_stats()` | Show store stats |

## Known Differences (vs memory-lancedb)

1. **filter syntax**: Zvec uses a single `=` (SQL style), the same as LanceDB — no difference
2. **HNSW parameter name**: `m=16` (lowercase), not `M=16`
3. **stats attribute**: `doc_count`, not `num_rows`
4. **RRF class name**: `RrfReRanker` (lowercase `r`), not `RRFReRanker`
5. **FTS flush on destroy**: Zvec v0.5.0 has a known spurious RocksDB flush log when deleting a collection (no production impact)

## Related Skills

- **`hardware/zvec`** — general Zvec reference manual (schema design, hybrid search patterns, troubleshooting)
- **`hardware/vector-db-migration`** — migration playbook (includes the `migrate-lancedb-to-zvec.py` script)
