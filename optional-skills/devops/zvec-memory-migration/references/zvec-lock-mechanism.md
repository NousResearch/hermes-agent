# Zvec Lock Mechanism — Experimental Verification + Lock Governance Evolution (updated 2026-09-15)

> The first half of this document records the experimental conclusions on zvec 0.5.x/0.6.0 lock semantics (still valid); the second half records the lock governance design and real incidents of the memory-zvec fork (v1.1.0→v1.3.1) — the old "release on shutdown + gc retry" approach is deprecated; do not follow the old documentation.

## Lock Types and Basic Semantics (experimental conclusions)

A file-level LOCK located at `<collection_path>/LOCK`.

| Operation | Result |
|------|------|
| `zvec.open()` | Acquires the rw lock. If already held → `RuntimeError: Can't lock read-write collection` |
| `zvec.open(option=read_only)` | **zvec ≥ 0.5.1 fully mutually exclusive**: it still needs to acquire the lock; if already held → `Can't lock read-only collection` (read-only is **not** a lock-bypass trick; verified the same on 0.6.0) |
| `zvec.create_and_open()` | Same as open, acquires the rw lock. The path must not exist |
| Collections have no `close()`/`release()` | The only release = the Python object's references dropping to zero + `gc.collect()`, or process exit (fd closed, lock falls off) |

**GC release experiment**: `del coll1; gc.collect()` releases immediately; no sleep needed.

**Opening the same path twice within one process**: also collides with the lock (the lock is judged globally by path, with no distinction between inside and outside the process).
**Never call `zvec.open()` again on a path your own process already holds a handle to** — you will collide with your own lock, and after backoff failure a good handle may be overwritten and lost (actually introduced in the first v1.3.0 release; fixed: handle reuse).

## Lock Governance Evolution (memory-zvec fork)

### Incident Timeline (real incidents)

| Date | Event |
|------|------|
| 2026-09-10 | In-tree patch optimize_fix (hot-path HNSW amortized optimization) |
| 2026-09-11 | In-tree patch for the shutdown race (thread tracking + atexit draining) — but the draining was not hooked into initialize, so the race was not truly fixed |
| 2026-09-14 14:07 | <profile> session initialization collided with the previous session's closing batch write, **the entire session's memory silently failed** (journal signature: same-second WARNING "still locked" + ERROR "failed to open (read-only)") |
| 2026-09-14 23:00 | Plugin moved out of the hermes-agent tree → `~/.hermes/plugins/memory-zvec` (user-level, immune to updates); 6 configs point to `memory.provider: memory-zvec` |
| 2026-09-14 late night | Discovered that the dashboard (root profile) web UI **embeds profile chats** (`web_server_chat.py` supports `profile=`, with HERMES_HOME pointed at the sub-profile) and held the <profile> lock for **35 hours** (fd timestamp as hard evidence) — from this, idle release was born |
| 2026-09-15 | 8 adversarial-review fixes (v1.3.1); profile symlinks changed to physical copies |

### v1.1.0 — Root Fix for the Back-to-Back Session Race

`initialize()` runs `_drain_pending_writes(timeout=15s)` before opening the collection
(waits for this process's in-flight write threads; tunable via `HERMES_MEMORY_ZVEC_INIT_DRAIN_S`);
backoff retries of 0.5/1/2/4s against external lock holders; only after all retries fail does it
degrade to read-only (with a warning).

### v1.2.x — Idle Auto-Release + Lazy Reopen

- A module-level watchdog (period `min(5s, idle/3)`) releases the lock for providers that are
  "open but idle for over 30s with no in-flight write threads"
  (`HERMES_MEMORY_ZVEC_IDLE_RELEASE_S`, 0 = disabled).
- All public entry points (prefetch/queue_prefetch/sync_turn/on_session_end/
  handle_tool_call) lazily reopen via `_ensure_open()`; measured reopen+stats ≈ 188ms.
- Only a previous open that **failed** is subject to the 30s retry rate limit; a reopen after idle release executes immediately.

### v1.3.0 — Handle Shared Across Sessions (new sessions with zero wait)

- `shutdown()` became pure bookkeeping and **does not close the handle**: the collection handle is shared across sessions; in-process writes
  were already serialized by `_insert_lock`; measured new-session initialize 136ms→0ms (no longer waits for the old batch write).
- Real release only goes through the idle watchdog's `_release_collection()` or process exit.
- initialize reuses an already-open handle (switching only when the target zvec_dir has changed).

### v1.3.1 — Adversarial-Review Fixes

- `_read_only` is reset at the start of every open ladder (old bug: a single degradation permanently skipped writes).
- Implemented `on_session_switch` rebinding `_session_id` (old bug: after /new, writes went to the first session's id forever); the on_session_end batch write **snapshots the sid at entry** (prevents cross-talk from switch races).
- Read paths capture local references (prevents NoneType from the idle-release race).
- Lazy reopen uses a **short ladder** (drain 2s + backoff 0.5/1/2, worst case ~5.5s instead of 22.5s).
- Cross-process black-window write skips gained throttled warnings (1 per 5min; the end-of-session batch write catches up on the gap).
- prefetch cache capped at 128; with handle reuse, /new skips the Ollama warmup.

## Current Lock Semantics (quick reference)

| Scenario | Behavior |
|------|------|
| New session /new (including one with a batch write in progress) | ~0ms, no lock wait (handle sharing) |
| Write in progress | Lock held (embedding + flush, a few hundred ms per entry), queued in-process via `_insert_lock` |
| Last write completes | Watchdog auto-releases after ~30s |
| Next access | Lazy reopen within ~200ms |
| Cross-process contention | Resolved automatically by drain+backoff; the 30s idle guarantee ensures rotation; black-window skipped writes get throttled warnings, with a full catch-up write at session end |

## Operations Notes

- **Lock-holder diagnosis**: `fuser <collection>/LOCK` → for the PID, check
  `ls -l /proc/<pid>/fd | grep 记忆` (the fd timestamp = when the lock was taken) +
  `tr '\0' ' ' < /proc/<pid>/cmdline` (identity).
- **Release methods**: wait 30s for automatic release; or restart the lock holder's systemd unit
  (restarting the dashboard is low-risk and does not affect the gateway).
- **Journal signatures**: release = INFO `idle Ns — releasing collection lock`;
  black window = WARNING `collection unavailable — per-turn writes skipped`;
  degradation = WARNING `still locked after backoff retries`.
- **Health-checking a locked database** (do not force it open): scan the tails of the RocksDB `LOG`/`LOG.old` for
  corruption/checksum entries, verify the `CURRENT→MANIFEST-*` chain, and check the segment count (≈2 is normal).

## Verification Script (lock semantics self-check)

```python
import zvec, gc
path = "/path/to/memories"
coll1 = zvec.open(path)
try:
    coll2 = zvec.open(path)          # expected to fail: a second open in the same process collides with our own lock
except RuntimeError:
    print("self-collision as expected")
del coll1; gc.collect()
coll3 = zvec.open(path)              # reopenable after GC
print("reopen OK:", coll3.stats.doc_count)
```

Provider-level functional self-test (does not touch production databases; uses a one-off /tmp collection):

```bash
HERMES_HOME=<profile home> ~/.hermes/hermes-agent/venv/bin/python -c "
import sys; sys.path.insert(0, '/home/<user>/.hermes/hermes-agent')
from plugins.memory import load_memory_provider
p = load_memory_provider('memory-zvec', register_skills=False)  # the loaded module name should be _hermes_user_memory.memory-zvec__source_*
import json; p.initialize('t'); print(json.loads(p.handle_tool_call('vec_memory_stats', {})))"
```
