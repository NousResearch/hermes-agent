# Hermes Knowledge Retrieval (RAG) Subsystem

Long-term semantic memory for Hermes Agent. Hermes depends on the
`KnowledgeProvider` abstraction only — AnythingLLM, Qdrant, pgvector, Chroma,
Weaviate and the built-in local store are interchangeable behind it.

```
                    Hermes Agent
                          │
          ┌───────────────┴───────────────┐
   Planning / Reasoning            Knowledge Service      (cache, retry,
                                          │                timeout, rerank,
                          ┌───────────────┴──────────┐     merge, logging)
                    KnowledgeProvider (ABC)          │
                          │                          │
        AnythingLLM / local / Qdrant / Weaviate / Chroma / pgvector
```

No reasoning happens in this package. No provider protocol details leak out of it.

## Layout

| File | Role |
|---|---|
| `types.py` | `Document`, `Chunk`, `Citation`, `SearchResult`, `HealthStatus`, `SyncReport` |
| `provider.py` | `KnowledgeProvider` ABC — the only contract Hermes depends on |
| `service.py` | `KnowledgeService` — provider choice, merge, rerank, cache, retry, timeout, logging |
| `embeddings.py` | Stdlib chunking + hashed embeddings (crc32-stable, persistable) |
| `cache.py` | TTL + LRU retrieval cache |
| `config.py` | `knowledge:` config resolution (config.yaml → env → offline default) |
| `sync.py` | Incremental delta sync + checksum manifest |
| `worker.py` | `knowledge-sync-worker`: continuous inotify-driven vault → index sync |
| `providers/` | `local`, `anythingllm`, `qdrant`, `weaviate`, `chroma`, `pgvector` + registry |
| `deploy/` | systemd unit |

## Hermes tools

| Tool | Purpose |
|---|---|
| `knowledge_search` | `query`, `limit`, `workspace`, `filters`, `mode` → `answer`, `sources`, `chunks`, `confidence`, `provider`, `elapsedTime` |
| `knowledge_sync` | Index a path or every configured source (incremental) |
| `knowledge_health` | Provider health, cache stats, index freshness |

Modes: `search` (chunks), `retrieve` (chunks + answer), `similar` (find related documents).

The reasoning loop is steered by `KNOWLEDGE_RETRIEVAL_GUIDANCE`
(`agent/prompt_builder.py`), injected by `agent/system_prompt.py` **only** when
`knowledge_search` is in the schema — prompts stay byte-identical for users who
never enable it. It instructs the model to decide whether external knowledge is
needed, retrieve before answering, and cite every source as `[n]`.

## Configuration

`~/.hermes/config.yaml` (defaults in `hermes_cli/config_defaults.py`):

```yaml
knowledge:
  enabled: true
  provider: local            # or anythingllm | qdrant | weaviate | chroma | pgvector
  fallback_providers: []     # tried in order when the primary errors or finds nothing
  workspace: default
  top_k: 5
  cache_ttl: 300.0
  timeout: 30.0
  retries: 2
  min_score: 0.05
  sync_sources:
    - {type: obsidian, path: /opt/secondbrain, workspace: default}
  provider_options:
    anythingllm: {base_url: http://localhost:3001}
```

Secrets go in `~/.hermes/.env` only: `ANYTHINGLLM_API_KEY`.

## Switching backends

No Hermes code changes. Either:

```bash
hermes config set knowledge.provider anythingllm
```

or, for a brand-new backend:

```python
from packages.knowledge import register_provider, KnowledgeProvider

class MyProvider(KnowledgeProvider):
    name = "mine"
    ...  # search/retrieve/index/update/delete/health

register_provider("mine", MyProvider)
```

## Continuous synchronization

```
Obsidian ──LiveSync/Git──▶ vault on disk ──▶ knowledge-sync-worker ──▶ provider ──▶ Hermes
```

* **Filesystem events**, not periodic full scans (watchdog/inotify; polling fallback).
* **Debounced** — editors write several times per save; events coalesce per path.
* **Incremental** — SHA-256 per file in the manifest. Unchanged → skip,
  changed → `update()`, removed → `delete()`. The vector DB is never rebuilt.
* **Self-healing** — periodic reconciliation catches changes made while down.
* **Observable** — JSON state file plus `GET /health` on port 8787.

Watches `*.md`, `*.txt`, `*.pdf`; ignores `.obsidian/`, `.git/`, `node_modules/`,
`dist/`, `build/`.

Run it:

```bash
python -m packages.knowledge /opt/secondbrain           # foreground
python -m packages.knowledge --once                     # reconcile and exit (cron)
sudo systemctl enable --now knowledge-sync              # service
curl -fsS http://127.0.0.1:8787/health
journalctl -u knowledge-sync -f
```

The unit is at `deploy/knowledge-sync.service` (restart=always, survives reboot,
journal logging, health endpoint, `ProtectHome=read-only` with the vault
read-only and only `~/.hermes` writable).

An optional `enrich=` callback fires once per *new* note (summarize, tag,
link) — it never mutates the original file.

## Tests

```bash
venv/bin/python -m unittest tests.knowledge.test_knowledge tests.knowledge.test_worker
```

59 tests: search, retrieval, citations, sync deltas, provider swap, cache hit
and invalidation, timeout, retry, fallback, graceful failure, AnythingLLM wire
mapping, worker create/update/delete lifecycle, debounce, ignore rules,
health endpoint and enrichment isolation.
