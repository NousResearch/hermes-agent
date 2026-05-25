# Kanban FIFO Race Fix — Detailed Implementation Plan

> Task: t_0f17c083 | Parent investigation: t_6e9bdebe, t_cc791874
> Verified against codebase at /home/c/Desktop/agenda/hermes-agent (commit ~May 25 2026)

## Summary

Fix two race conditions in the kanban→TUI FIFO notification bridge:
1. **Writer ENXIO silent swallow**: `os.open(fifo, O_WRONLY|O_NONBLOCK)` raises `ENXIO` when no reader is connected. The bare `except Exception: pass` at kanban_db.py:2098-2099 loses the notification with zero logging.
2. **Reader not eager/global**: The FIFO reader is per-session (started at `_init_session`), not at gateway startup. If the gateway restarts or no session exists, writers get ENXIO. The reader thread can also die from unhandled exceptions.

**Approach**: Replace per-session FIFO readers with a single eager global reader that feeds a thread-safe queue. Each session's existing notification poller drains the queue. This preserves the zero-polling design while eliminating both races.

---

## Verified Code Locations (Current)

### Writer: hermes_cli/kanban_db.py:2083-2099

```python
_NOTIFY_KINDS = {"completed", "blocked", "gave_up", "crashed", "timed_out"}
if kind in _NOTIFY_KINDS:
    try:
        _fifo_path = os.path.expanduser("~/.hermes/tui_kanban.fifo")
        if os.path.exists(_fifo_path):
            # O_NONBLOCK: open(2) on a FIFO blocks until a reader
            # connects; in CI (and any headless context) there is no
            # reader, so the write op deadlocks the caller.  Non-blocking
            # open + a zero-second select/poll lets us bail gracefully.
            _fd = os.open(_fifo_path, os.O_WRONLY | os.O_NONBLOCK)
            try:
                _fifo = os.fdopen(_fd, "w", encoding="utf-8")
                _fifo.write(json.dumps({"task_id": task_id, "kind": kind}) + "\n")
                _fifo.close()
            except Exception:
                os.close(_fd)
    except Exception:
        pass  # Non-fatal — don't break the event write
```

**Logger in module**: `_log = logging.getLogger(__name__)` (line 92). `import logging` at line 83.

### Reader: tui_gateway/server.py:196-283

```python
def _start_kanban_fifo_reader(sid: str, session: dict) -> None:
    def _fifo_reader() -> None:
        _FMT_KINDS = {"completed", "blocked", "gave_up", "crashed", "timed_out"}
        _cursors: dict[tuple, int] = {}
        while not session.get("_finalized"):
            try:
                with open(_KANBAN_FIFO_PATH, "r", encoding="utf-8") as _fifo:
                    for _line in _fifo:
                        # ... subscription matching + dispatch ...
            except (OSError, IOError):
                pass

    _t = threading.Thread(target=_fifo_reader, daemon=True, name="kanban-fifo")
    _t.start()
```

**Call sites**: line 735 (`_create_session` → session reset), line 2290 (`_init_session` → new session).

### Notification poller: tui_gateway/server.py:3364-3462

```python
def _notification_poller_loop(stop_event, sid, session):
    while not stop_event.is_set() and not session.get("_finalized"):
        try:
            evt = process_registry.completion_queue.get(timeout=0.5)
        except Exception:
            continue
        # ... dispatch process event ...
```

**`queue` already imported** at line 8.

### FIFO path & creation: tui_gateway/server.py:137-160

```python
_KANBAN_FIFO_PATH = os.path.expanduser("~/.hermes/tui_kanban.fifo")
try:
    os.mkfifo(_KANBAN_FIFO_PATH, 0o600)
except FileExistsError:
    pass
except Exception:
    pass

import atexit as _atexit

def _cleanup_kanban_fifo() -> None:
    try:
        if os.path.exists(_KANBAN_FIFO_PATH):
            os.unlink(_KANBAN_FIFO_PATH)
    except Exception:
        pass

_atexit.register(_cleanup_kanban_fifo)
```

---

## Change 1: Fix Writer ENXIO + Add Debug Logging

**File**: `hermes_cli/kanban_db.py`
**Lines**: 2083-2099 → 2083-2105 (+6 lines)

### Exact diff

Replace lines 2083-2099:

```python
    _NOTIFY_KINDS = {"completed", "blocked", "gave_up", "crashed", "timed_out"}
    if kind in _NOTIFY_KINDS:
        try:
            _fifo_path = os.path.expanduser("~/.hermes/tui_kanban.fifo")
            if os.path.exists(_fifo_path):
                # O_NONBLOCK: open(2) on a FIFO blocks until a reader
                # connects; in CI (and any headless context) there is no
                # reader, so the write op deadlocks the caller.  Non-blocking
                # open + a zero-second select/poll lets us bail gracefully.
                _fd = os.open(_fifo_path, os.O_WRONLY | os.O_NONBLOCK)
                try:
                    _fifo = os.fdopen(_fd, "w", encoding="utf-8")
                    _fifo.write(json.dumps({"task_id": task_id, "kind": kind}) + "\n")
                    _fifo.close()
                except Exception:
                    os.close(_fd)
        except Exception:
            pass  # Non-fatal — don't break the event write
```

With:

```python
    _NOTIFY_KINDS = {"completed", "blocked", "gave_up", "crashed", "timed_out"}
    if kind in _NOTIFY_KINDS:
        try:
            _fifo_path = os.path.expanduser("~/.hermes/tui_kanban.fifo")
            if os.path.exists(_fifo_path):
                # O_NONBLOCK: open(2) on a FIFO blocks until a reader
                # connects; in CI (and any headless context) there is no
                # reader, so the write op deadlocks the caller.  Non-blocking
                # open lets us bail gracefully when no reader is connected.
                _fd = os.open(_fifo_path, os.O_WRONLY | os.O_NONBLOCK)
                try:
                    _fifo = os.fdopen(_fd, "w", encoding="utf-8")
                    _fifo.write(json.dumps({"task_id": task_id, "kind": kind}) + "\n")
                    _fifo.close()
                except Exception:
                    os.close(_fd)
        except OSError as _e:
            import errno
            if _e.errno == errno.ENXIO:
                _log.debug("kanban_fifo_no_reader: no TUI listening on %s", _fifo_path)
            else:
                _log.debug("kanban_fifo_open_error: %s", _e)
        except Exception as _e:
            _log.debug("kanban_fifo_write_error: %s", _e)
```

### Rationale
- `errno.ENXIO` (errno 6) is the specific error when no reader is connected to a FIFO. Log at debug level so operators can trace notification delivery.
- Other OS errors and unexpected exceptions are also logged at debug (not warning — these are expected in headless/CI contexts).
- The misleading comment about "zero-second select/poll" is corrected (no such call exists).
- Uses existing `_log` logger (line 92).

---

## Change 2: Add Global FIFO Reader + Queue

**File**: `tui_gateway/server.py`
**Lines**: 137-160 (module level) + 196-283 (reader function) + 3364-3384 (poller loop)

### Step 2a: Add module-level queue and global reader state

**Location**: After line 138 (`_KANBAN_FIFO_PATH = ...`)
**Add** (lines 138a-138d):

```python
# Global queue decouples FIFO reader from session dispatch.
# The reader thread is eager (starts at import) and immortal.
_kanban_fifo_queue: queue.Queue = queue.Queue(maxsize=10000)
_kanban_global_reader_thread: threading.Thread | None = None
```

### Step 2b: Add global reader function

**Location**: After line 160 (`_atexit.register(_cleanup_kanban_fifo)`)
**Add** (lines 161-200):

```python
def _start_global_kanban_reader() -> threading.Thread:
    """Start the global kanban FIFO reader thread (idempotent, eager).

    Reads JSON lines from the FIFO and pushes parsed dicts into
    _kanban_fifo_queue.  Each session's notification poller drains
    the queue for matching subscriptions.  This ensures the FIFO
    always has a reader — eliminating ENXIO races — and notifications
    survive gateway restart (they queue until a session starts).
    """
    global _kanban_global_reader_thread
    if _kanban_global_reader_thread is not None and _kanban_global_reader_thread.is_alive():
        return _kanban_global_reader_thread

    def _reader() -> None:
        while True:
            try:
                with open(_KANBAN_FIFO_PATH, "r", encoding="utf-8") as _fifo:
                    for _line in _fifo:
                        _line = _line.strip()
                        if not _line:
                            continue
                        try:
                            _data = json.loads(_line)
                            _kanban_fifo_queue.put(_data, block=False)
                        except Exception:
                            pass
            except (OSError, IOError):
                # FIFO removed or TUI shutting down — recreate if missing
                # so future writers can connect.
                if not os.path.exists(_KANBAN_FIFO_PATH):
                    try:
                        os.mkfifo(_KANBAN_FIFO_PATH, 0o600)
                    except Exception:
                        pass
                time.sleep(0.5)
            except Exception:
                logger.exception("kanban_fifo_reader_error")
                time.sleep(1.0)

    _t = threading.Thread(target=_reader, daemon=True, name="kanban-fifo-global")
    _t.start()
    _kanban_global_reader_thread = _t
    return _t


# Eager start: the FIFO always has a reader as long as the gateway process lives.
_start_global_kanban_reader()
```

### Step 2c: Extract subscription matching into shared function

**Location**: Replace `_start_kanban_fifo_reader` (lines 196-283) with two functions.

**New `_dispatch_kanban_notification`** (lines 196-255):

```python
def _dispatch_kanban_notification(sid: str, session: dict, data: dict) -> None:
    """Query DB for CLI subscriptions matching ``data["task_id"]`` and
    inject a formatted notification into the agent session.
    """
    _task_id = data.get("task_id", "")
    if not _task_id:
        return

    _FMT_KINDS = {"completed", "blocked", "gave_up", "crashed", "timed_out"}

    try:
        from hermes_cli import kanban_db as _kb
        _conn = _kb.connect()
        try:
            _subs = _kb.list_notify_subs(_conn)
            for _sub in _subs:
                if _sub.get("platform") != "cli":
                    continue
                if _sub["task_id"] != _task_id:
                    continue

                _cursors = session.setdefault("_kanban_cursors", {})
                _ckey = (_task_id, _sub.get("chat_id", sid))
                _last = _cursors.get(_ckey, _sub.get("last_event_id", 0))
                _, _events = _kb.unseen_events_for_sub(
                    _conn, task_id=_task_id,
                    platform="cli", chat_id=_sub["chat_id"],
                    thread_id=_sub.get("thread_id") or "",
                    kinds=_FMT_KINDS,
                )
                if _events:
                    _max_id = max(
                        _last,
                        max(
                            e.id if hasattr(e, "id") else e["id"]
                            for e in _events
                        ),
                    )
                    _cursors[_ckey] = _max_id
                for _ev in _events:
                    _msg = _format_kanban_notification(_ev, _sub)
                    if _msg:
                        _emit("status.update", sid, {"kind": "process", "text": _msg})
                        with session["history_lock"]:
                            if session.get("running"):
                                continue  # defer until idle
                            session["running"] = True
                        _rid = f"__kanban__{int(time.time() * 1000)}"
                        try:
                            _emit("message.start", sid)
                            _run_prompt_submit(_rid, sid, session, _msg)
                        except Exception:
                            with session["history_lock"]:
                                session["running"] = False
        finally:
            _conn.close()
    except Exception:
        pass  # Non-fatal — don't break the poller loop
```

**New `_start_kanban_fifo_reader`** (lines 257-265):

```python
def _start_kanban_fifo_reader(sid: str, session: dict) -> threading.Thread:
    """Ensure the global kanban FIFO reader is running.

    Per-session FIFO readers are replaced by a single global reader
    that feeds a shared queue.  Each session's notification poller
    drains the queue for matching subscriptions.  This function is
    kept for backward compatibility at existing call sites.
    """
    return _start_global_kanban_reader()
```

### Step 2d: Integrate kanban queue into notification poller

**Location**: `tui_gateway/server.py:3364-3384` (`_notification_poller_loop`)
**Replace** lines 3380-3384:

```python
    while not stop_event.is_set() and not session.get("_finalized"):
        try:
            evt = process_registry.completion_queue.get(timeout=0.5)
        except Exception:
            continue
```

With:

```python
    while not stop_event.is_set() and not session.get("_finalized"):
        try:
            evt = process_registry.completion_queue.get(timeout=0.5)
        except Exception:
            # No process event — check kanban queue while we're awake.
            try:
                _kanban_evt = _kanban_fifo_queue.get_nowait()
                _dispatch_kanban_notification(sid, session, _kanban_evt)
            except queue.Empty:
                pass
            continue
```

### Rationale
- **Eager global reader**: Starts at module import time (line ~200), so the FIFO always has a reader. Writers never see ENXIO in normal operation.
- **Queue decoupling**: Events survive gateway restart (they queue in memory). When a session starts, its poller drains the queue.
- **Bounded queue**: `maxsize=10000` prevents unbounded memory growth. If the queue fills, old events are dropped (acceptable — better than blocking the writer).
- **Immortal reader**: Catch-all `except Exception` + sleep prevents thread death. FIFO recreation on missing handles stale inodes.
- **Reuse existing poller**: The notification poller already wakes every 0.5s. Adding a `get_nowait()` check is ~zero overhead.
- **Per-session cursors**: Stored in `session["_kanban_cursors"]` so each session tracks its own subscription state (same as before, just moved from local dict to session dict).

---

## Change 3: Update Call Sites to Store Thread Reference

**File**: `tui_gateway/server.py`
**Lines**: 735, 2290

### Line 735 (inside `_create_session`)

**Current**:
```python
            _start_kanban_fifo_reader(sid, _sessions[sid])
```

**New**:
```python
            _sessions[sid]["_kanban_fifo_thread"] = _start_kanban_fifo_reader(sid, _sessions[sid])
```

### Line 2290 (inside `_init_session`)

**Current**:
```python
    _start_kanban_fifo_reader(sid, _sessions[sid])
```

**New**:
```python
    _sessions[sid]["_kanban_fifo_thread"] = _start_kanban_fifo_reader(sid, _sessions[sid])
```

### Rationale
- Stores the global reader thread reference in the session for observability and health checking.
- Backward compatible: `_start_kanban_fifo_reader` still returns a thread.

---

## Estimated Lines Changed

| File | Lines Changed | Nature |
|------|--------------|--------|
| `hermes_cli/kanban_db.py` | +6 | ENXIO catch + debug logging |
| `tui_gateway/server.py` | +65 / -55 | Global reader, shared dispatch, poller integration |
| **Total** | **~+70 net** | |

---

## Edge Cases Accounted For

| Scenario | Handling |
|----------|----------|
| **No active TUI session** | Global reader runs eagerly at import. Events queue in `_kanban_fifo_queue`. When a session starts, its poller drains the queue. |
| **Gateway restart** | New process → new global reader started at import. Old queue is gone (in-memory), but new events queue fresh. Writer ENXIO logging makes any gap visible. |
| **Multiple TUI sessions** | Each session's poller competes for queue events (same semantics as current competing FIFO readers). First poller to wake gets the event. |
| **FIFO unlinked externally** | Global reader catches `OSError`, recreates FIFO via `os.mkfifo`, sleeps 0.5s, retries. |
| **Queue full (10000 items)** | `put(block=False)` raises `queue.Full`. Event is dropped. This is a safety valve — prevents unbounded memory growth if no session drains the queue for a long time. |
| **Reader thread dies unexpectedly** | The global reader has a broad `except Exception` at the top level with `logger.exception()` and 1s sleep. Thread never exits. |
| **Session finalized while processing** | `_dispatch_kanban_notification` checks `session.get("running")` under `history_lock` (same as current). The poller loop checks `session.get("_finalized")` before each iteration. |
| **DB query fails during dispatch** | Caught by inner `except Exception: pass` in `_dispatch_kanban_notification`. Poller continues. |
| **Windows (no FIFO support)** | Module-level `os.mkfifo` already catches `Exception: pass` (line 146-147). Global reader's `open()` will fail, caught by `OSError`, sleeps 0.5s in a loop. No CPU burn. |

---

## Testing Strategy

### Existing tests (must still pass)

```bash
cd /home/c/Desktop/agenda/hermes-agent
pytest tests/tui_gateway/test_fifo_notification_bridge.py -v
```

Expected: all 18 existing tests pass (6 writer + 14 format + 2 lifecycle + 2 e2e).

**Why they pass**: The external behavior is unchanged — events written to FIFO are still delivered. The global reader + queue is an internal refactoring.

### New tests to add

**File**: `tests/tui_gateway/test_fifo_notification_bridge.py`

#### Test A: ENXIO logged when no reader (writer side)

```python
def test_enxio_no_reader_logged(self, kanban_home, tmp_path, monkeypatch, caplog):
    """When FIFO exists but no reader is connected, O_NONBLOCK open raises ENXIO.

    _append_event should catch this gracefully, log at debug, and not raise.
    """
    import logging
    caplog.set_level(logging.DEBUG, logger="hermes_cli.kanban_db")

    fifo = tmp_path / "tui_kanban.fifo"
    os.mkfifo(str(fifo), 0o600)

    _real_expand = os.path.expanduser
    def _fake_expand(p):
        if "tui_kanban.fifo" in p:
            return str(fifo)
        return _real_expand(p)
    monkeypatch.setattr(os.path, "expanduser", _fake_expand)

    # No reader thread started — open(O_WRONLY|O_NONBLOCK) will raise ENXIO.
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="enxio-test", assignee="worker")
        kb.complete_task(conn, tid, summary="done")

    assert "kanban_fifo_no_reader" in caplog.text
```

#### Test B: Global reader survives exception and keeps running

```python
def test_global_reader_survives_exception(self, tmp_path, monkeypatch):
    """The global FIFO reader thread should survive unexpected exceptions."""
    fifo = tmp_path / "tui_kanban.fifo"
    os.mkfifo(str(fifo), 0o600)

    with patch.dict("sys.modules", {
        "hermes_constants": MagicMock(
            get_hermes_home=MagicMock(return_value=str(tmp_path))
        ),
        "hermes_cli.env_loader": MagicMock(),
        "hermes_cli.banner": MagicMock(),
        "hermes_state": MagicMock(),
    }):
        import importlib
        import tui_gateway.server as srv
        importlib.reload(srv)
        srv._KANBAN_FIFO_PATH = str(fifo)

        # Force the global reader to start (or verify it's already running)
        _t = srv._start_global_kanban_reader()
        assert _t.is_alive()

        # Write a valid event — should be queued.
        with open(str(fifo), "w") as f:
            f.write('{"task_id": "t_test", "kind": "completed"}\n')

        time.sleep(0.3)
        assert not srv._kanban_fifo_queue.empty()

        srv._sessions.clear()
        importlib.reload(srv)
```

#### Test C: Event queued when no session, drained when session starts

```python
def test_event_queued_then_drained(self, tmp_path, monkeypatch):
    """Events should queue when no session exists and be drained when a session starts."""
    fifo = tmp_path / "tui_kanban.fifo"
    os.mkfifo(str(fifo), 0o600)

    with patch.dict("sys.modules", {
        "hermes_constants": MagicMock(
            get_hermes_home=MagicMock(return_value=str(tmp_path))
        ),
        "hermes_cli.env_loader": MagicMock(),
        "hermes_cli.banner": MagicMock(),
        "hermes_state": MagicMock(),
    }):
        import importlib
        import tui_gateway.server as srv
        importlib.reload(srv)
        srv._KANBAN_FIFO_PATH = str(fifo)

        _t = srv._start_global_kanban_reader()
        assert _t.is_alive()

        # Write event while no session exists.
        with open(str(fifo), "w") as f:
            f.write('{"task_id": "t_queued", "kind": "completed"}\n')

        time.sleep(0.3)
        assert srv._kanban_fifo_queue.qsize() == 1

        # Simulate session poller draining the queue.
        _evt = srv._kanban_fifo_queue.get_nowait()
        assert _evt["task_id"] == "t_queued"

        srv._sessions.clear()
        importlib.reload(srv)
```

#### Test D: FIFO recreated after unlink

```python
def test_fifo_recreated_after_unlink(self, tmp_path, monkeypatch):
    """If the FIFO is unlinked, the global reader should recreate it."""
    fifo = tmp_path / "tui_kanban.fifo"
    os.mkfifo(str(fifo), 0o600)

    with patch.dict("sys.modules", {
        "hermes_constants": MagicMock(
            get_hermes_home=MagicMock(return_value=str(tmp_path))
        ),
        "hermes_cli.env_loader": MagicMock(),
        "hermes_cli.banner": MagicMock(),
        "hermes_state": MagicMock(),
    }):
        import importlib
        import tui_gateway.server as srv
        importlib.reload(srv)
        srv._KANBAN_FIFO_PATH = str(fifo)

        _t = srv._start_global_kanban_reader()
        assert _t.is_alive()

        # Unlink the FIFO.
        os.unlink(str(fifo))
        assert not os.path.exists(str(fifo))

        # Wait for reader to detect and recreate.
        time.sleep(1.0)
        assert os.path.exists(str(fifo)), "FIFO should be recreated by global reader"

        srv._sessions.clear()
        importlib.reload(srv)
```

---

## Implementation Order

1. **Change 1** (writer ENXIO) — smallest change, can be done independently
2. **Change 2a+2b** (global reader + queue) — core fix
3. **Change 2c** (shared dispatch function) — refactor existing logic
4. **Change 2d** (poller integration) — wire queue into existing poller
5. **Change 3** (call sites) — store thread reference
6. **Run existing tests** — verify no regressions
7. **Add new tests** — verify fixes

---

## Rollback Plan

If issues arise:
1. Revert `kanban_db.py` changes (restore bare `except Exception: pass`).
2. Revert `tui_gateway/server.py` changes — restore original `_start_kanban_fifo_reader` with per-session reader thread.
3. The call sites (lines 735, 2290) don't need changes since they call the same function name.

No DB schema changes, no config changes, no external dependencies. Pure code revert.

---

## Notes

- The `_fifo_reader` local `_cursors` dict is replaced by `session.setdefault("_kanban_cursors", {})`. This is equivalent because the old `_cursors` was per-session anyway (each session had its own reader thread).
- The `queue` module is already imported at line 8 of `tui_gateway/server.py`.
- The `logger` variable is available in `tui_gateway/server.py` (used throughout the module).
- No changes to `_format_kanban_notification` or `_cleanup_kanban_fifo`.
- The atexit cleanup still unlinks the FIFO on process exit (line 159).
