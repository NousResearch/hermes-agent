---
name: memory-strategy
description: Use when managing Hermes Agent long-term memory — choosing between DB-backed (load_from_db) vs file-backed (load_from_disk), understanding session_reset DB recovery, and aligning with upstream v0.13.0 memory architecture. Covers memo.txt vs memory.md vs memories table trade-offs.
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [memory, hermes-state, session-recovery, upgrade, sqlite]
    related_skills: [hermes-backup, hermes-session-auto-recovery]
---

# Memory Strategy — DB-Backed vs File-Backed

## Overview

Hermes Agent has two independent memory subsystems:

1. **Session memory** — what was said in this session (transient, gone after session ends)
2. **Long-term memory** — what persists across sessions

This skill is about long-term memory. There are **two competing implementations**:

| | **Local DB-Backed** (MacFS custom) | **Official File-Backed** (v0.13.0) |
|--|-------------------------------------|-------------------------------------|
| Storage | SQLite `memories` table | `~/.hermes/memories/MEMORY.md` / `USER.md` |
| Read method | `load_from_db()` | `load_from_disk()` |
| Eviction | `memory_evict_for_section()` — LRU by access_count | Manual or line-count based |
| Crash safety | Transaction-protected, atomic | Append-only but no atomicity |
| Freshness | Always current (DB read each turn) | Cached snapshot, refreshed on `refresh()` |
| Source | Local merge commit | `origin/main` v0.13.0 |

**Current status**: MacFS Hermes runs the **DB-backed** version (merged as local commit `329ad1436`).

---

## Architecture

```
memory_tool.py
├── class MemoryStore
│   ├── load_from_db()       ← local: reads from SQLite memories table
│   ├── save_to_disk()       ← persists to MEMORY.md/USER.md (backup export)
│   ├── add()                ← writes to DB
│   ├── memory_evict_for_section()  ← LRU eviction by access_count
│   └── format_for_system_prompt()
│
hermes_state.py
├── SessionDB class
│   ├── memory_upsert()      ← insert/update memory row
│   ├── memory_delete()      ← soft-delete (is_active=0)
│   ├── memory_get_active()  ← read active memories
│   ├── memory_touch()       ← increment access_count
│   ├── memory_evict_for_section()
│   └── memory_total_chars()
│
├── CREATE TABLE memories (   ← the DB table
│     id, section, category, key, value, chars,
│     access_count, last_accessed, is_active,
│     created_at, updated_at
│   )
```

The two systems are **not connected**. `save_to_disk()` exists as a manual export path but is not called by the agent automatically.

---

## memo.txt — The Real Permanent Backup

`memo.txt` (at `~/.hermes/memo.txt`) is the **highest-priority knowledge store** and is **not managed by the memory system at all**.

| | memo.txt | memories table | MEMORY.md / USER.md |
|--|----------|----------------|---------------------|
| Storage | Plain text file | SQLite | Plain text files |
|淘汰 | Never (超长期备份) | Yes (auto-eviction) | Yes (manual) |
| 内容 | Credentials, device IPs, SSH keys, permanent decisions | Runtime facts, user preferences | Legacy memory entries |
| 读取 | File read at startup | `memory_get_active()` | `load_from_disk()` |
| 失效 | Never | `is_active=0` | File deleted |

**Rule**: `memo.txt` is the source of truth for anything that must survive total DB loss. Always update `memo.txt` when making permanent changes to device configs, credentials, or system architecture.

---

## Session Reset Recovery — The session_reset Fix

### The Problem

When Hermes Gateway is killed with SIGTERM (fire-and-forget, no graceful drain):
- Session gets `end_reason = 'session_reset'`
- Default query: `WHERE ended_at IS NULL AND end_reason IS NULL` → **misses the session**
- Result: session disappears from DB, user sees "who are you?"

### The Fix

Local `hermes_state.py` adds `get_recent_active_session()` with fallback:

```sql
WHERE source = ? AND ended_at IS NULL
  AND (end_reason IS NULL OR end_reason = 'session_reset')
```

Official v0.13.0 uses `resume_pending` flag + freshness gate in `gateway/run.py` — covers **graceful drain** but NOT fire-and-forget crash.

### Fix Survival After Upgrade

```
Current branch: local main (ahead of origin/main by 1 commit)
Merge commit: 329ad1436 "Merge hermes-agent v0.13.0 — preserve session_reset DB recovery"
```

The fix is preserved as a local merge strategy. When upgrading to a newer upstream:
1. Fetch + merge as usual
2. The `get_recent_active_session()` method lives in `hermes_state.py` — different file from upstream's crash-recovery logic
3. Conflicts unlikely; if clean, fix survives automatically

To verify after upgrade:
```bash
grep -n "session_reset" ~/.hermes/hermes-agent/hermes_state.py
# Should show: end_reason = 'session_reset' in get_recent_active_session()
```

---

## When to Use Which Memory System

### Use DB-backed (current default) when:
- You want automatic LRU eviction by access_count
- You need transactional atomicity (no partial writes)
- You want to query memories by category or access patterns
- Crash safety matters

### Use memo.txt for:
- SSH keys, credentials, device IPs
- Architecture decisions that never change
- Emergency fallback if DB corrupts
- Anything you must never lose

### Export from DB to file (manual):
```python
# In a Python tool or REPL:
from tools.memory_tool import MemoryStore
from hermes_state import SessionDB

db = SessionDB()
ms = MemoryStore(session_db=db)
for section in ("memory", "user"):
    rows = db.memory_get_active(section)
    for row in rows:
        print(row["value"])  # Manual export to MEMORY.md
```

---

## Upgrade Implications

### v0.13.0 → next upstream update

Before upgrading:
1. Run full backup: `tar czf ~/HermesBackup/Hermes备份/hermes-full-$(date +%Y%m%d_%H%M%S).tar.gz -C /Users/macbookpro .hermes`
2. Verify fix: `grep session_reset ~/.hermes/hermes-agent/hermes_state.py`

After merge:
- Check `hermes_state.py` for `get_recent_active_session()`
- Check `tools/memory_tool.py` for `load_from_db()` (not `load_from_disk()`)
- Restart gateway: `hermes gateway restart`
- Test: send a message, crash-recover, verify session continuity

### If upstream adopts DB-backed memory
If future upstream officially adopts the `memories` table + `load_from_db` approach:
1. The local skill becomes a reference (no longer custom)
2. memo.txt remains the permanent backup regardless
3. `get_recent_active_session()` may conflict — check for duplicate method names

---

## Common Pitfalls

1. **Confusing memory.md with memories table**. `memory_tool.py` can write to both, but the DB path is the active one. MEMORY.md is legacy.

2. **`is_active=0` not `DELETE`**. Memory deletions are soft. Hard deletes bypass the system and leave orphans.

3. **Upgrading without checking fix**. `git fetch && git merge` — always verify the fix survived.

4. **memo.txt not backed up**. DB can corrupt. memo.txt is the fallback. Include it in every full backup.

5. **`access_count` drift**. Every `format_for_system_prompt()` call touches all entries. Heavy use of memory retrieval can inflate counts and prevent eviction of truly stale entries.

---

## Verification Checklist

- [ ] `grep session_reset ~/.hermes/hermes-agent/hermes_state.py` → fix present
- [ ] `grep load_from_db ~/.hermes/hermes-agent/tools/memory_tool.py` → DB path active
- [ ] `ls ~/.hermes/memo.txt` → exists and contains critical data
- [ ] `sqlite3 ~/.hermes/state.db "SELECT COUNT(*) FROM memories WHERE is_active=1"` → table populated
- [ ] Gateway restart test: send message, `kill -TERM <gateway_pid>`, verify session continuity
- [ ] Backup verified: `tar -tzf ~/HermesBackup/Hermes备份/hermes-full-latest.tar.gz | grep memo.txt`
