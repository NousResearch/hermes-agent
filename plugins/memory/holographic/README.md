# Holographic Memory Provider

Local SQLite fact store with FTS5 search, trust scoring, entity resolution, and HRR-based compositional retrieval.

## Requirements

None — uses SQLite (always available). NumPy optional for HRR algebra.

## Setup

```bash
hermes memory setup    # select "holographic"
```

Or manually:
```bash
hermes config set memory.provider holographic
```

## Config

Config in `config.yaml` under `plugins.hermes-memory-store`:

| Key | Default | Description |
|-----|---------|-------------|
| `db_path` | `$HERMES_HOME/memory_store.db` | SQLite database path |
| `auto_extract` | `false` | Auto-extract facts at session end |
| `default_trust` | `0.5` | Default trust score for new facts |
| `hrr_dim` | `1024` | HRR vector dimensions |

## Tools

| Tool | Description |
|------|-------------|
| `fact_store` | 9 actions: add, search, probe, related, reason, contradict, update, remove, list |
| `fact_feedback` | Rate facts as helpful/unhelpful (trains trust scores) |

## Retrieval hardening (R2/R3, all local, zero LLM)

* Entity aliases: attach short names/acronyms to entities; queries expand automatically.
* Thai queries fall back to character-bigram matching when token search runs short.
* Score ties break by recency (newer first); trust still dominates via score.
* Empty FTS results fall back to a bounded HRR vector scan instead of nothing.
* `contradict` also catches entity-less `subject = value` conflicts (same value never conflicts).

## Safety (automatic paths only)

* `auto_extract` and the memory-write mirror refuse secret-like
  (API keys, bearer/JWT, private keys, `PASSWORD=...`) and injected-instruction
  content. Explicit `fact_store add` calls are left untouched.
* `prefetch` filters secret/instruction rows before injection (fail-closed).
* Memory content is always treated as DATA, never as instructions.
