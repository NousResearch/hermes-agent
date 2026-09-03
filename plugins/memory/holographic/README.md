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
| `db_path_template` | *(empty)* | Optional scoped path template with sanitized `{profile}`, `{platform}`, `{chat}`, and `{user}` placeholders. When set, this takes precedence over `db_path`. |
| `memory_mode` | `hybrid` | `hybrid` automatically prefetches relevant facts; `tools` disables prefetch while keeping `fact_store` and `fact_feedback` available. |
| `auto_extract` | `false` | Auto-extract facts at session end |
| `default_trust` | `0.5` | Default trust score for new facts |
| `hrr_dim` | `1024` | HRR vector dimensions |

For privacy-safe per-chat storage, for example:

```yaml
plugins:
  hermes-memory-store:
    db_path_template: $HERMES_HOME/holographic/{profile}/{platform}/{chat}.db
    memory_mode: tools
    auto_extract: false
```

`db_path` retains its existing behavior when `db_path_template` is unset.

## Tools

| Tool | Description |
|------|-------------|
| `fact_store` | 9 actions: add, search, probe, related, reason, contradict, update, remove, list |
| `fact_feedback` | Rate facts as helpful/unhelpful (trains trust scores) |
