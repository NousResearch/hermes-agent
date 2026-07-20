# Holographic Memory Provider

Local SQLite fact store with FTS5 search, trust scoring, entity resolution, and HRR-based compositional retrieval.

## Requirements

SQLite is always available. The setup wizard installs NumPy for HRR algebra;
without NumPy the provider falls back to FTS5 and token-overlap retrieval.

## Setup

```bash
hermes memory setup    # select "holographic"
```

Or manually:
```bash
uv pip install --python "$(command -v python)" numpy==2.4.3  # enables HRR algebra
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
