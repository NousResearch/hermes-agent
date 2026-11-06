# hermes-lancedb-memory-migration

Migrate Hermes Agent's session history from FTS5 (built-in SQLite full-text search) to LanceDB vector memory with Ollama embeddings.

## Overview

This skill migrates a Hermes profile's historical sessions from the default FTS5 session search to a semantic vector memory system powered by:

- **Ollama** + **bge-m3:567m** — Local embedding model (no API needed)
- **LanceDB** — Vector database with HNSW index for fast ANN search

## When to Use

- Migrating an existing profile's session history to semantic vector search
- Setting up a new sub-agent profile with LanceDB-backed memory
- Enabling cross-session memory retrieval via natural language queries

## What This Does

1. **Migration script** — Reads sessions from `state.db`, filters low-value conversations, re-embeds with Ollama, writes to LanceDB
2. **Config change** — Switches `memory.provider` from default to `lancedb-embed`
3. **Plugin setup** (optional) — Configures Ollama URL, model name, LanceDB path
4. **Verification** — Full system check: config, data, Ollama model, vector search, gateway restart

## Prerequisites

- **Ollama** running with `bge-m3:567m` model loaded
- **lancedb** pip package installed in Hermes venv
- Hermes profile with existing sessions in `state.db`

```bash
# Check Ollama
bash scripts/check_ollama.sh

# Install lancedb
uv pip install --python ~/.hermes/venv/bin/python lancedb
```

## Quick Start

### 1. Create migration script

Copy `scripts/migrate_template.py` from this repo to your Hermes scripts directory:

```bash
cp scripts/migrate_template.py ~/.hermes/scripts/migrate_<profile>_sessions_to_lancedb.py
```

Edit the top of the script:

```python
PROFILE = "your_profile_name"       # e.g., "chip_expert"
OLLAMA_HOST = "http://localhost:11434"  # adjust if needed
OLLAMA_MODEL = "bge-m3:567m"       # must match your Ollama model name
MIN_ASST_CHARS = 200               # skip sessions with less assistant content
```

Also update `SHORT_PATTERNS` to match your language environment.

### 2. Update config.yaml

```bash
hermes config set memory.provider lancedb-embed --profile <profile>
```

Or edit `~/.hermes/profiles/<profile>/config.yaml` directly:

```yaml
memory:
  memory_enabled: true
  user_profile_enabled: true
  memory_char_limit: 2200
  user_char_limit: 1375
  provider: lancedb-embed
```

### 3. Run migration

```bash
# Dry-run first
~/.hermes/venv/bin/python3 ~/.hermes/scripts/migrate_<profile>_sessions_to_lancedb.py --dry-run

# Execute
~/.hermes/venv/bin/python3 ~/.hermes/scripts/migrate_<profile>_sessions_to_lancedb.py
```

### 4. Verify

```bash
# Check config
grep "provider:" ~/.hermes/profiles/<profile>/config.yaml | grep memory

# Check LanceDB
~/.hermes/venv/bin/python3 -c "
import lancedb
db = lancedb.connect('~/.hermes/profiles/<profile>/lance_memory')
table = db.open_table('memories')
print(f'Rows: {table.count_rows()}')
"

# Restart gateway if needed
systemctl --user restart hermes-gateway-<profile>.service
```

## Architecture

```
Session History (state.db)
        ↓  [migration script]
LanceDB Vector DB  ←  Ollama bge-m3:567m
        ↓  [lancedb-embed plugin]
Hermes Agent (vec_memory_* tools)
```

## Data Locations

| Item | Path |
|------|------|
| Migration script | `~/.hermes/scripts/migrate_<profile>_sessions_to_lancedb.py` |
| Profile config | `~/.hermes/profiles/<profile>/config.yaml` |
| LanceDB data | `~/.hermes/profiles/<profile>/lance_memory/memories.lance` |
| Original sessions | `~/.hermes/profiles/<profile>/state.db` |
| Ollama service | `localhost:11434` |

## Common Issues

### Ollama model name mismatch

```
WARNING: Model 'bge-m3:567m' not loaded
```

Check actual model names:
```bash
bash scripts/check_ollama.sh && ollama list | grep "bge-m3:567m"
```

### Sub-agent memory disabled by default

Sub-agents (`delegate_task`) have `skip_memory=True` by default. To enable:
```python
# In tools/delegate_tool.py
# Change skip_memory=True → skip_memory=False
# Add "memory" to DEFAULT_TOOLSETS
```

### Embedding timeout for large sessions

Script sets 300s timeout. If still timing out, the session may be too large — consider increasing or splitting the session.

## License

MIT
