# Runtime Snapshot Engine — Implementation Plan

**Branch:** `feature/runtime-snapshots`
**Status:** Planning
**Architecture:** Content-addressed storage, event-triggered, debounced, git-free

---

## Problem

Hermes has no way to snapshot its own runtime state. The existing `checkpoint_manager.py` snapshots working directories (project files) via shadow git repos — but the internal state of `~/.hermes` itself (sessions, config, auth, memory, cron jobs) has zero versioning. A bad config change, corrupted state.db, or rogue memory write can brick the agent with no recovery path.

## Solution: Self-Versioning Runtime Snapshots

A lightweight, content-addressed snapshot engine embedded in Hermes that captures internal state on-the-fly, enabling instant rollback and time-travel debugging.

---

## Architecture

```
~/.hermes/
├── state.db                    # Session DB (50MB)
├── config.yaml                 # User config
├── auth.json                   # Provider credentials
├── cron/jobs.json              # Cron job definitions
├── gateway_state.json          # Gateway state
├── channel_directory.json      # Channel config
├── processes.json              # Background processes
│
├── .snapshots/                 # Snapshot store (NEW)
│   ├── objects/                # Content-addressed blob store
│   │   ├── ab/
│   │   │   └── cdef1234...    # SHA256[:2] / SHA256[2:]
│   │   └── 98/
│   │       └── 76abcd...
│   ├── snapshots/              # Snapshot manifests
│   │   ├── 20260412-141500/
│   │   │   ├── manifest.json   # {rel_path: sha256}
│   │   │   └── meta.json       # {timestamp, label, trigger, size}
│   │   └── 20260412-142000/
│   ├── HEAD                    # Current snapshot ID
│   └── history.db              # Snapshot index (SQLite, fast queries)
```

### Content-Addressed Storage
- Files hashed (SHA-256)
- Only stored once — deduplication is automatic
- Same file across 100 snapshots = 1 copy on disk
- Hardlink-friendly for instant restores

### What Gets Snapshotted (INCLUDE)
| Path | Why | Typical Size |
|------|-----|--------------|
| `state.db` | Session history, messages | ~50MB |
| `config.yaml` | Agent configuration | ~18KB |
| `auth.json` | Provider credentials | ~16KB |
| `cron/jobs.json` | Cron job definitions | ~3KB |
| `gateway_state.json` | Gateway state | ~1KB |
| `channel_directory.json` | Channel config | ~28KB |
| `processes.json` | Background processes | tiny |

### What Gets EXCLUDED
| Path | Why |
|------|-----|
| `hermes-agent/` | Dev repo — git handles this |
| `venv/`, `.venv/` | Dependencies |
| `checkpoints/` | Existing checkpoint system |
| `sessions/` | Too large (246MB), state.db covers it |
| `logs/` | Not valuable for rollback |
| `cache/` | Ephemeral |
| `skills/` | Managed separately |
| `node_modules/` | Dependencies |
| `*.db-wal`, `*.db-shm` | SQLite WAL — inconsistent if copied live |
| `.snapshots/` | Self-reference |

---

## Files to Create

### 1. `tools/snapshot_engine.py` (~300 lines)
Core engine — no external deps beyond stdlib.

```
class SnapshotEngine:
    __init__(hermes_home: Path)
    snapshot(label=None, trigger="manual") -> str    # Create snapshot
    restore(snapshot_id: str) -> None                 # Restore state
    list_snapshots(limit=50) -> list[dict]            # List with metadata
    diff(snap_id_a, snap_id_b) -> dict                # Diff two snapshots
    prune(keep_last=100, keep_hourly=24, keep_daily=30) -> int
    get_head() -> Optional[str]
    compute_state_hash() -> str                       # For change detection

# Standalone functions (importable without class)
def auto_snapshot(debounce_seconds=30, label=None, trigger="auto") -> Optional[str]
def safe_run(fn, label="safe-run") -> Any             # Transaction wrapper
```

Key details:
- `EXCLUDE` set matches what we DON'T want
- SQLite WAL handling: skip `*.db-wal` / `*.db-shm`, snapshot `state.db` directly (SQLite handles this fine with WAL mode if we're quick — or we can use `VACUUM INTO` for a clean copy)
- Debounce via global `_last_snapshot_time` + `_last_state_hash`
- `history.db` for fast queries without scanning filesystem

### 2. `tools/snapshot_engine.py` — SQLite Safe Copy
For `state.db` specifically, use SQLite's `VACUUM INTO` or `sqlite3.Connection.backup()` for a consistent copy — NOT raw file copy while WAL is active.

```python
def _safe_copy_db(src: Path, dst: Path):
    """Copy SQLite DB safely, handling WAL mode."""
    import sqlite3
    conn = sqlite3.connect(f"file:{src}?mode=ro", uri=True)
    backup = sqlite3.connect(str(dst))
    conn.backup(backup)
    backup.close()
    conn.close()
```

### 3. CLI Integration (minimal changes)
Add to `hermes_cli/commands.py`:
```python
CommandDef("snapshot", "Create or list runtime snapshots", "Session",
           aliases=("snap",), args_hint="[list|create|rewind <id>|diff <a> <b>|prune]"),
```

Add handler in `cli.py` `process_command()`.

### 4. Hook Points in `run_agent.py` (minimal, surgical)
- **After memory write** (line ~6727): call `auto_snapshot(trigger="memory_write")`
- **After tool execution** (end of `_execute_tool_calls`): call `auto_snapshot(trigger="tool_result")`
- **After agent step** (end of main loop iteration): call `auto_snapshot(trigger="agent_step")`
- All calls are debounced — no performance impact

### 5. Gateway Integration
- Add `/snapshot` to `GATEWAY_KNOWN_COMMANDS`
- Handler in `gateway/run.py`

### 6. Tests (`tests/test_snapshot_engine.py`)
- Hash computation
- Snapshot creation + restore roundtrip
- Deduplication (same file = 1 object)
- Debounce behavior
- SQLite safe copy
- Prune logic
- Diff computation
- Exclusion filtering

---

## Implementation Order

1. **`tools/snapshot_engine.py`** — Core engine (create, restore, list, prune, diff)
2. **`tests/test_snapshot_engine.py`** — Test everything
3. **CLI commands** — `hermes_cli/commands.py` + `cli.py` handler
4. **Hook points** — `run_agent.py` surgical inserts (3 locations)
5. **Gateway** — `/snapshot` command
6. **Safe-run wrapper** — Transaction support for agent operations

---

## Design Decisions

### Why NOT extend `checkpoint_manager.py`?
- Checkpoint manager = filesystem snapshots of WORKING DIR (shadow git repos)
- Snapshot engine = runtime state snapshots of `~/.hermes` internal state
- Different domains, different concerns, different performance profiles
- Keeping them separate avoids coupling and complexity

### Why content-addressed + not git?
- Git is too slow for runtime writes (locking, index, packfiles)
- Content-addressed gives us dedup for free
- No `.git/` inside `~/.hermes/` to collide with dev repo
- Atomic enough for our use case

### Why SQLite safe copy for state.db?
- Raw file copy during WAL mode = corrupted DB
- `sqlite3.Connection.backup()` is the official safe method
- Takes ~100ms for 50MB — fast enough for debounced snapshots

### Why debounce at all?
- Agent can make 10+ tool calls per second
- Without debounce: 10 snapshots/second = disk thrashing
- 30s debounce with change detection = ~1 snapshot per meaningful state change
- Memory writes get slightly lower debounce (15s) since they're rarer and more valuable

---

## Estimated Size Impact

Initial snapshot: ~70MB (state.db + config + auth + cron)
Deduped subsequent snapshots: ~1-5MB (only state.db changes)
100 snapshots: ~150-300MB total (mostly deduped)
Pruned to 100 kept: auto-managed

---

## Crew Assignments

| Role | Responsibility |
|------|---------------|
| **Architect** | Plan, review, approve | (me)
| **Engine Dev** | `tools/snapshot_engine.py` core | (me)
| **Test Writer** | `tests/test_snapshot_engine.py` | (me)
| **CLI Integrator** | commands.py + cli.py changes | (me)
| **Runtime Hook Installer** | run_agent.py surgical hooks | (me)
| **QA** | End-to-end verification | (me)

Solo build — but structured like a crew for clean handoffs.
