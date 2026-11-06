# lancedb-embed Plugin — Internal Notes

**Date:** 2026-05-16
**File:** `~/.hermes/plugins/memory/lancedb-embed/__init__.py` (645 lines)

## Architecture

```
hermes_state (SQLite: state.db)
    → messages[].timestamp  ← per-message Unix timestamp
    ↓
on_session_end(messages)  ← called by memory_manager.on_session_end()
    ↓ pairs built from messages
    ↓ embedding via Ollama HTTP /api/embed
    ↓ LanceDB table.add(rows)  ← 2026-05-16 fix: now uses per-message timestamp
```

## Two Write Paths

| Path | Trigger | `created_at` source |
|------|---------|---------------------|
| `_tool_add()` | `vec_memory_add` tool | `datetime.now().timestamp()` (one row at a time) |
| `on_session_end()` | session cleanup | `messages[i].get("timestamp")` from hermes_state (fixed 2026-05-16) |

### Bug Fixed 2026-05-16

**Before:** All session_end rows in the same batch got identical `created_at` (write-time moment), making time-ordered search useless within a session.

**After:** Each row reads `messages[i].get("timestamp")`. Falls back to `datetime.now().timestamp()` only if timestamp is absent or invalid.

## Schema (LanceDB `.lance`)

| Field | Type | Source |
|-------|------|--------|
| `id` | string | `uuid.uuid4()` |
| `content` | string | `[user]\n...\n[assistant]\n...` |
| `role` | string | `turn` or `session_end` |
| `session_id` | string | `self._session_id` |
| `vector` | list[float32][1024] | Ollama `bge-m3:567m` |
| `created_at` | double | Unix timestamp (from message or now) |
| `metadata` | string | `{}` (always empty) |

## Probe — Verify Plugin is Active

```bash
strings ~/.hermes/logs/agent.log | grep "lancedb-embed.*registered" | tail -5
# Expected: "INFO ... Memory provider 'lancedb-embed' registered (5 tools)"

# Check session_end writes
strings ~/.hermes/logs/agent.log | grep "session_end batch store" | tail -5
# Should show "stored N session-end memories", not "failed"

# Check for the timestamp fix in plugin
grep -n "messages\[i\].get.*timestamp\|timestamp.*messages\[i\]" \
  ~/.hermes/plugins/memory/lancedb-embed/__init__.py
```

## Probe — Memory Date Distribution

```bash
~/.hermes/venv/bin/python -c "
import lancedb, datetime
db = lancedb.connect('/home/ktao/.hermes/lance_memory')
tbl = db.open_table('memories')
df = tbl.to_pandas()
df['ts'] = df['created_at'].apply(lambda x: datetime.datetime.fromtimestamp(x))
df['date'] = df['ts'].dt.date
print(f'Total: {len(df)} | Sessions: {df[\"session_id\"].nunique()}')
print()
for d, c in df['date'].value_counts().sort_index(ascending=False).items():
    print(f'  {d}: {c}')
"
```