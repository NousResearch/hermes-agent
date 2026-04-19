# Runtime Snapshot Engine

**Self-versioning state snapshots for Hermes Agent — content-addressed, WAL-protected, branch-safe.**

## Why Not Git?

Git is version control for **code**. Hermes needs version control for **runtime state**.

| Problem | Git | Snapshot Engine |
|---------|-----|-----------------|
| Runtime performance | Too slow (index, packfiles, locking) | Milliseconds |
| Storage | Full copies per commit | Content-addressed dedup |
| Designed for | Source code | Application state |
| Crash recovery | Manual | Automatic WAL replay |
| Branching for experiments | Git branches are for code | State branches for agent state |

Git commits during runtime are like rebuilding the engine while driving — technically possible, but slow, fragile, and destructive to performance.

---

## What Gets Snapshotted

Only Hermes internal state — never the dev repo:

```
~/.hermes/
├── state.db                  # Session history, messages (~50MB)
├── config.yaml               # Agent configuration
├── auth.json                 # Provider credentials
├── cron/jobs.json            # Cron job definitions
├── gateway_state.json        # Gateway state
├── channel_directory.json    # Channel configuration
├── processes.json            # Background processes
│
├── .snapshots/               # ← Snapshot store
│   ├── objects/              # Content-addressed blob store
│   ├── snapshots/            # Snapshot manifests
│   ├── branches/             # Branch references
│   ├── history.db            # Snapshot index (SQLite)
│   ├── HEAD                  # Current snapshot ID
│   └── BRANCH_HEAD           # Current branch name
```

Excluded: `hermes-agent/` (git-managed), `venv/`, `sessions/`, `logs/`, `cache/`, `skills/`.

---

## Architecture

### Content-Addressed Storage

Every file is SHA-256 hashed. Identical files are stored **once**:

```
objects/
  ab/
    cdef1234567890...   # SHA256[:2] = directory, SHA256[2:] = filename
```

**Result:** state.db (50MB) × 100 snapshots ≠ 5GB. More like ~150-300MB total,
because deduplication works across all snapshots.

### Write-Ahead Log (WAL)

Between debounced snapshots, many changes happen. Without WAL, these would be lost on crash.

The WAL logs every state mutation:

```
Memory write     → wal_append("config.yaml", data)
Tool execution   → wal_append("state.db", data)
Agent step       → wal_append(...)
...
Snapshot triggers → wal_flush (entries assigned to snapshot)
```

**Crash Recovery:**
```
Hermes starts after crash
  → Check WAL
  → Unflushed entries found?
  → WAL Replay: restore files from objects/
  → State is recovered
```

### Branching

Branches are copies of agent state — like git branches, but for runtime state instead of code.

**Use case — Safe Hermes Update:**
```
/snapshot branch pre-update         # Branch from current HEAD
/snapshot branch switch pre-update  # Switch to branch
# ... install update ...
# ... test ...
# All good?
/snapshot branch switch main        # Back to main (state restored)
# OOPS, broken?
/snapshot branch switch pre-update  # Instant rollback to pre-update state
```

Branches are **protected**:
- `main` cannot be deleted
- Active branch cannot be deleted
- Branch snapshots are **never** deleted by auto-prune

### Debouncing

Not every change creates a snapshot. The engine debounces:

| Trigger | Debounce | Why |
|---------|----------|-----|
| Memory write | 15 seconds | Rare but valuable |
| Tool result | 30 seconds | Tool batches can be 10+ calls/sec |
| Manual | Immediate | User explicitly requested |

Debounce also checks state hash — if nothing changed since last snapshot, it's skipped entirely.

### Auto-Pruning

Snapshots grow fast. Auto-prune keeps the store clean:

**Retention strategy (default):**
- Last 100 snapshots: ALWAYS keep
- 1 per hour for the last 24 hours
- 1 per day for the last **3 days**
- Branch snapshots: NEVER delete
- WAL entries: 72 hours

---

## CLI Reference

```
/snapshot                          List recent snapshots (all branches)
/snapshot create [label]           Create a snapshot
/snapshot rewind <id>              Restore state from snapshot
/snapshot diff <a> <b>             Compare two snapshots
/snapshot prune                    Delete old snapshots (3-day retention)
/snapshot head                     Show current HEAD + branch

/snapshot branch                   List all branches
/snapshot branch <name>            Create new branch (from HEAD)
/snapshot branch switch <name>     Switch to branch
/snapshot branch delete <name>     Delete branch (protected)

/snapshot wal                      Show WAL status (unflushed entries)
/snapshot wal replay               Replay WAL (crash recovery)
/snapshot list <branch>            Filter snapshots by branch
```

Alias: `/snap` works too.

---

## Programmatic API

```python
from tools.snapshot_engine import (
    SnapshotEngine, auto_snapshot, safe_run,
    wal_append, wal_replay,
    create_branch, switch_branch,
)

engine = SnapshotEngine()

# Manual snapshot
snap_id = engine.snapshot(label="before-upgrade")

# Restore
engine.restore(snap_id)

# Debounced auto-snapshot (in hooks)
auto_snapshot(trigger="memory_write")

# Safe execution (transaction)
result = safe_run(lambda: risky_operation(), label="config-change")
# → Creates snapshot before, restores on exception

# WAL
engine.wal_append("config.yaml", b"new: value\n")
engine.wal_replay()  # After crash

# Branching
engine.create_branch("test-v2")
engine.switch_branch("test-v2")
engine.switch_branch("main")  # Auto-snapshot + restore
```

---

## Use Cases

### 1. Safe Hermes Update

```
/snapshot branch pre-v2
/snapshot branch switch pre-v2
git pull && pip install -e .
# ... test ...
# All good?
/snapshot branch switch main
# Broken?
/snapshot branch switch pre-v2
```

### 2. Config Experimentation

```
/snapshot create "before config change"
# ... edit config.yaml ...
# ... test ...
# Don't like it?
/snapshot rewind 20260412-143000-before-config-change
```

### 3. Crash Recovery

```
# Hermes crashes. On restart:
# → WAL replay runs automatically
# → Unflushed entries are restored
# → No data lost

# Manual check:
/snapshot wal
/snapshot wal replay
```

### 4. Debugging

```
# "Since when did X break?"
/snapshot list
# → All snapshots with timestamps
/snapshot diff 20260412-100000 20260412-140000
# → Shows what changed
```

### 5. Risky Operations

```python
from tools.snapshot_engine import safe_run

result = safe_run(
    lambda: dangerous_config_migration(),
    label="config-migration"
)
# → Creates snapshot, restores on exception
```

---

## Storage Overhead

| What | Size |
|------|------|
| First snapshot | ~70MB (state.db + config + auth + cron) |
| Subsequent snapshot (deduped) | ~1-5MB (only changed files) |
| 100 snapshots | ~150-300MB total |
| WAL entry | ~1-50KB per entry |
| Branch ref | <100 bytes |

Content-addressed deduplication stores identical files only once. Pruning is important — the default 3-day retention keeps everything manageable.

---

## File Structure Detail

```
~/.hermes/.snapshots/
├── HEAD                           # Current snapshot ID
├── BRANCH_HEAD                    # Current branch name ("main")
├── history.db                     # SQLite index
│   ├── snapshots                  # {id, timestamp, label, trigger, branch, ...}
│   ├── snapshot_files             # {snapshot_id, rel_path, sha256, size}
│   └── wal_entries                # {id, timestamp, rel_path, sha256, branch, snapshot_id}
├── objects/                       # Content-addressed blobs
│   ├── ab/
│   │   └── cdef1234567890...      # SHA256[:2] = dir, SHA256[2:] = file
│   └── 98/
│       └── 76abcdef12345678...
├── snapshots/                     # Snapshot manifests
│   ├── 20260412-143000/
│   │   ├── manifest.json          # {rel_path: sha256, ...}
│   │   └── meta.json              # {id, timestamp, label, trigger, branch, ...}
│   └── 20260412-150000-before-upgrade/
│       ├── manifest.json
│       └── meta.json
└── branches/                      # Branch references
    ├── main                       # → snapshot ID
    └── pre-update-v2              # → snapshot ID
```

---

## SQLite Safe Copy

state.db is NOT copied with raw file I/O. SQLite in WAL mode produces inconsistent data with raw copies.

Instead, `sqlite3.Connection.backup()` is used:

```python
conn = sqlite3.connect("file:state.db?mode=ro", uri=True)
backup = sqlite3.connect("snapshot_copy.db")
conn.backup(backup)
```

This is the official SQLite API for consistent backups, even during write access.

---

## Performance

| Operation | Latency |
|-----------|---------|
| Create snapshot | ~100-500ms (depends on state.db size) |
| Restore | ~50-200ms |
| WAL append | <5ms |
| WAL replay | ~50-200ms per file |
| Prune (100 snapshots) | ~100-300ms |
| Debounce check | <1ms |

All hooks are debounced and try/except wrapped. Even if the snapshot engine completely fails, Hermes continues unaffected.

---

## Run Agent Integration

Hooks are already wired into `run_agent.py` (surgical, zero-risk):

```python
# After memory write (~line 6733)
auto_snapshot(trigger="memory_write")

# After tool execution batch (~line 7320 and ~6972)
auto_snapshot(trigger="tool_result")
```

All hooks are debounced — no performance impact.

---

## Tests

41 tests covering all features:

```
pytest tests/test_snapshot_engine.py -v
```

**Test coverage:**
- Snapshot create/restore roundtrip
- Content-addressed deduplication
- SQLite safe copy (including WAL mode)
- Diff between snapshots
- Prune logic with branch protection
- WAL append/flush/replay/dedup/prune
- Branch CRUD/switch/protection/track
- Full lifecycle integration

---

## Summary

The Runtime Snapshot Engine makes Hermes a **self-healing system**:

- **Automatic snapshots** on state changes
- **WAL** for zero-loss between snapshots
- **Branching** for risk-free experiments
- **Instant rollback** to any previous state
- **Crash recovery** via WAL replay
- **Zero performance impact** through debouncing
- **Minimal storage** via content-addressed dedup

Not "git in the background." An embedded state-versioning system designed for agent runtime.
