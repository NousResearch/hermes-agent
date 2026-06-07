# Holographic Memory Provider

Local SQLite fact store with FTS5 search, trust scoring, entity resolution, and HRR-based compositional retrieval.

> **New**: This provider now supports an optional **Organic Memory Architecture** — biology-inspired modules for salience filtering, silent engrams, and memory consolidation. See [ORGANIC_MEMORY.md](../../../ORGANIC_MEMORY.md) for full details.

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

### Organic Memory (optional, all default-off)

| Key | Default | Description |
|-----|---------|-------------|
| `salience_enabled` | `false` | Enable input salience filtering |
| `salience_min_threshold` | `0.2` | Minimum salience score to store |
| `silent_engram_enabled` | `false` | Enable graceful forgetting (strength decay, not deletion) |
| `silent_engram_half_life_hours` | `720` | Strength half-life in hours (30 days) |
| `consolidation_enabled` | `false` | Enable sleep-like memory consolidation |

## Tools

| Tool | Description |
|------|-------------|
| `fact_store` | 13 actions: add, search, probe, related, reason, contradict, update, remove, list, **recall**, **schemas**, **consolidate**, **health** |
| `fact_feedback` | Rate facts as helpful/unhelpful (trains trust scores) |
