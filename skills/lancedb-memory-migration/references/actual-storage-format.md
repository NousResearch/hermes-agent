# Actual Ollama Embed Storage Format

**Date:** 2026-05-16
**Finding:** `ollama_embed.db` is a **SQLite file** (not a LanceDB `.lance` directory).

## File Location

```
~/.hermes/ollama_embed.db                  # default agent
~/.hermes/profiles/<name>/ollama_embed.db   # 子 agent
```

Not at `~/.hermes/profiles/<name>/lance_memory/` — that path does not exist.

## Schema

### `memories` table

| Column | Type | Description |
|--------|------|-------------|
| `id` | TEXT | UUID |
| `content` | TEXT | Actual stored text content |
| `role` | TEXT | `turn` / `session_migrated` / etc. |
| `session_id` | TEXT | Source session ID (e.g. `20260510_190500_214051`) |
| `vector` | BLOB | Raw float32 array stored as binary blob |
| `created_at` | REAL | Unix timestamp (e.g. `1778411439.376243`) |
| `metadata` | TEXT | JSON string, currently always `{}` |

### `sessions` table

| Column | Type | Description |
|--------|------|-------------|
| `session_id` | TEXT | Session identifier |
| `started_at` | REAL | Unix timestamp of session start |
| `ended_at` | REAL | Unix timestamp of session end (nullable) |

## How to Inspect

```bash
file ~/.hermes/ollama_embed.db
# → SQLite 3.x database

sqlite3 ~/.hermes/ollama_embed.db ".tables"
# → memories sessions

sqlite3 ~/.hermes/ollama_embed.db ".schema memories"
sqlite3 ~/.hermes/ollama_embed.db ".schema sessions"

# Inspect one row
sqlite3 ~/.hermes/ollama_embed.db "SELECT id, session_id, created_at, length(content) FROM memories LIMIT 3;"

# Time range
sqlite3 ~/.hermes/ollama_embed.db "SELECT MIN(created_at), MAX(created_at) FROM memories;"
```

## Why This Matters

The migration skill template assumes LanceDB uses `.lance` directory format. The actual Ollama embed storage is simpler:
- **Storage engine:** SQLite (not LanceDB `.lance` files)
- **Vector storage:** BLOB column (not ANN index — search is NOT powered by this table)
- **Search:** Powered by Hermes `vec_memory_search` tool, which queries the Hermes-managed LanceDB at `~/.hermes/lance_memory/` separately

> The `ollama_embed.db` SQLite is a **per-agent content store** used by the Ollama embed plugin. The **vector memory search** uses a separate LanceDB at `~/.hermes/lance_memory/` (default agent) or `~/.hermes/profiles/<name>/lance_memory/` (子 agent). These are two different systems.

## Probe Script

```python
#!/usr/bin/env python3
"""Inspect ollama_embed.db schema and content stats."""
import os
import sqlite3, json, datetime

DB = os.path.expanduser("~/.hermes/ollama_embed.db")  # 按需调整

conn = sqlite3.connect(DB)
cur = conn.cursor()

# Tables
cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
print("Tables:", [r[0] for r in cur.fetchall()])

# memories stats
cur.execute("SELECT COUNT(*) FROM memories")
total = cur.fetchone()[0]
cur.execute("SELECT COUNT(*) FROM memories WHERE session_id IS NOT NULL AND session_id != ''")
with_sid = cur.fetchone()[0]
cur.execute("SELECT MIN(created_at), MAX(created_at) FROM memories")
min_t, max_t = cur.fetchone()
cur.execute("SELECT COUNT(DISTINCT session_id) FROM memories WHERE session_id IS NOT NULL")
distinct = cur.fetchone()[0]
print(f"\nmemories: {total} rows | session_id: {with_sid}/{total} ({with_sid/total*100:.0f}%) | distinct: {distinct}")
if min_t:
    print(f"  Time range: {datetime.datetime.fromtimestamp(min_t)} ~ {datetime.datetime.fromtimestamp(max_t)}")

# sessions stats
cur.execute("SELECT COUNT(*) FROM sessions")
print(f"sessions: {cur.fetchone()[0]} rows")

# Metadata check
cur.execute("SELECT COUNT(*) FROM memories WHERE metadata IS NOT NULL AND metadata != '{}'")
meta_populated = cur.fetchone()[0]
print(f"metadata populated: {meta_populated}/{total}")

# Sample row
cur.execute("SELECT id, session_id, role, length(content), created_at, metadata FROM memories LIMIT 1")
row = cur.fetchone()
if row:
    print(f"\nSample row:")
    print(f"  id: {row[0]}")
    print(f"  session_id: {row[1]}")
    print(f"  role: {row[2]}")
    print(f"  content length: {row[3]}")
    print(f"  created_at: {datetime.datetime.fromtimestamp(row[4]) if row[4] else None}")
    print(f"  metadata: {row[5]}")
```
