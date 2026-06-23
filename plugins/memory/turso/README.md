# Turso Memory Provider

Local-first semantic memory for Hermes using a SQLite-compatible Turso-style store, with optional Turso Cloud sync.

## Setup

```bash
hermes memory setup
# select "turso"
```

Or manually:

```bash
hermes config set memory.provider turso
```

## Config

Behavioral settings are stored in `$HERMES_HOME/turso.json`.

| Key | Default | Description |
|-----|---------|-------------|
| `db_path` | `$HERMES_HOME/turso-memory.db` | Local SQLite-compatible database path |
| `sync_enabled` | `false` | Enable best-effort Turso Cloud sync hooks |
| `top_k` | `6` | Number of memories returned by automatic prefetch |
| `min_similarity` | `0.15` | Minimum retrieval score for automatic prefetch |
| `auto_capture` | `false` | Store completed turns as searchable conversation memories |

Secrets belong in `$HERMES_HOME/.env`:

| Env Var | Description |
|---------|-------------|
| `TURSO_DATABASE_URL` | Optional remote Turso database URL |
| `TURSO_AUTH_TOKEN` | Optional Turso auth token |

## Tools

| Tool | Description |
|------|-------------|
| `turso_memory_search` | Search long-term memory |
| `turso_memory_add` | Store a durable fact, preference, decision, or note |
| `turso_memory_update` | Update a memory by ID |
| `turso_memory_delete` | Delete a memory by ID |
| `turso_memory_sync` | Report or trigger best-effort sync status |

Use the built-in `memory` tool for compact, always-on curated facts. Use Turso memory for deeper searchable recall.

