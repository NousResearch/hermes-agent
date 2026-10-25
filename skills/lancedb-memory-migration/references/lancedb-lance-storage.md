# LanceDB `.lance` Vector Memory — Storage Reference

**Date:** 2026-05-16
**Finding:** `vec_memory_*` tools use LanceDB `.lance` format, NOT `ollama_embed.db`.

> **Do NOT confuse with `ollama_embed.db`** — that is a separate SQLite store used only by the Ollama embed plugin. The actual memory that `vec_memory_add` / `vec_memory_search` / `vec_memory_list` tools read/write is at `lance_memory/memories.lance/`.

## File Location

```
~/.hermes/lance_memory/memories.lance/     # default profile (195 items as of 2026-05-16)
~/.hermes/profiles/<name>/lance_memory/       # sub-agent profiles
```

## Schema (confirmed via inspection 2026-05-16)

```python
{
    "id":           "string",                          # UUID
    "content":      "string",                          # stored text
    "role":         "string",                          # turn | session_migrated | session_end
    "session_id":   "string",                          # e.g. "20260513_223106_8757f640"
    "vector":       "fixed_size_list<float32>[1024]",  # bge-m3:567m 1024-dim
    "created_at":   "double",                          # Unix timestamp
    "metadata":     "string",                          # JSON string (currently always "{}")
}
```

## How to Inspect (Correct System)

```bash
# 1. List physical files
ls ~/.hermes/lance_memory/memories.lance/
# → data/  _latest  _transactions

# 2. Python inspection
~/.hermes/venv/bin/python -c "
import lancedb, datetime

db = lancedb.connect('/home/ktao/.hermes/lance_memory')
tbl = db.open_table('memories')

# Total count
print('Total rows:', tbl.count_rows())

# Schema
print('Schema:', tbl.schema)

# Latest 10 entries
df = tbl.to_pandas()
df['ts'] = df['created_at'].apply(lambda x: datetime.datetime.fromtimestamp(x))
df_sorted = df.sort_values('ts', ascending=False)
for _, row in df_sorted.head(10).iterrows():
    print(f'{row[\"ts\"].strftime(\"%Y-%m-%d %H:%M\")}  {row[\"session_id\"][:22]}  {row[\"role\"]}')
"
```

## Date Distribution (default profile, 2026-05-16)

| Date | Count | Notes |
|------|-------|-------|
| 2026-05-16 | 11 | Today (partial) |
| 2026-05-15 | 2 | Very low usage |
| 2026-05-14 | 8 | Only 00:00-01:00 + 23:00 |
| 2026-05-13 | 71 | Peak day |
| 2026-05-12 | 43 | High activity |
| 2026-05-11 | 35 | High activity |
| 2026-05-10 | 25 | First day |

**Total: 195 items**

## Date Distribution by Hour (2026-05-13 — peak day)

```
00时: 7   09时: 2   14时: 2   15时: 3   16时: 4
17时: 4   18时: 17  19时: 4   20时: 1   21时: 6   22时: 21
```

## Key Insight — Usage Dips Are Real

If memory counts drop after a certain date, it means **actual usage dropped** — NOT that the system broke.

Signs of real usage drop (NOT a bug):
- `gateway.log` inbound message count drops in parallel
- `agent.log` shows fewer "Memory provider registered" events
- Sub-agent profiles show near-zero messages on same dates

Signs of actual write failure (actual bug):
- `agent.log` shows `WARNING.*session_end batch store failed`
- `agent.log` shows `Connection refused` to Ollama
- Memory counts for old dates suddenly decrease

## Probe Script

```python
#!/usr/bin/env python3
"""Inspect LanceDB .lance vector memory stats — the ACTUAL memory system."""
import lancedb, datetime

LANCE_DIR = "~/.hermes/lance_memory"  # change per profile

import os
LANCE_DIR = os.path.expanduser(LANCE_DIR)

db = lancedb.connect(LANCE_DIR)
tbl = db.open_table('memories')

df = tbl.to_pandas()
df['ts'] = df['created_at'].apply(lambda x: datetime.datetime.fromtimestamp(x))

print(f"Total rows: {len(df)}")
print(f"Distinct sessions: {df['session_id'].nunique()}")
print(f"Date range: {df['ts'].min().date()} ~ {df['ts'].max().date()}")
print()

print("=== Date distribution ===")
df['date'] = df['ts'].dt.date
for d, c in df['date'].value_counts().sort_index(ascending=False).items():
    print(f"  {d}: {c}")
print()

print("=== Role distribution ===")
print(df['role'].value_counts())
print()

print("=== Latest 5 ===")
df_sorted = df.sort_values('ts', ascending=False)
for _, row in df_sorted.head(5).iterrows():
    print(f"  {row['ts'].strftime('%Y-%m-%d %H:%M')}  {row['session_id'][:22]}  {row['role']}")
    print(f"    {row['content'][:100].replace(chr(10), ' ')}...")
```