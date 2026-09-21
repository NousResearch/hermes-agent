"""Direct behavioural tests for the gateway's embedded kanban dispatcher
watcher (``_kanban_dispatcher_watcher``).

Prior coverage tested the mixin structurally (methods exist, MRO resolves,
singleton lock is exclusive) and the auto-decompose config helper in
isolation, but the watcher's three core runtime behaviours had no direct
tests:

1. **Corrupt-board quarantine** — a board DB that fails to open as SQLite
   is fingerprinted and quarantined; dispatch is skipped for that board
   on subsequent ticks while the fingerprint is unchanged.
2. **Health telemetry (bad_ticks / HEALTH_WINDOW)** — when the ready queue
   is non-empty (spawnable work waiting) but the dispatcher spawns 0 workers
   for ``HEALTH_WINDOW`` (6) consecutive ticks, the watcher emits a
   ``kanban dispatcher stuck`` warning.
3. **Multi-board dispatch** — the watcher enumerates every board on disk
   each tick and calls ``dispatch_once`` once per board.

These tests drive the real coroutine, controlling the tick loop via a
sleep patch and a ``_running`` flag. ``dispatch_once`` / ``has_spawnable_*``
are mocked where needed to exercise the specific branch; the quarantine
test uses a real corrupt-bytes board DB so the watcher's
``_is_corrupt_board_db_error`` classifier runs against the genuine
``sqlite3.DatabaseError`` / ``KanbanDbCorruptError`` the production path
raises.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from unittest.mock import patch

import pytest

from hermes_cli import kanban_db as kb


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an initialised default kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    # Ensure the env-var escape hatch doesn't disable dispatch in tests.
    monkeypatch.delenv("HERMES_KANBAN_DISPATCH_IN_GATEWAY", raising=False)
    kb.init_db()
    return home


def _make_runner():
    """Build a bare GatewayRunner with just the state the dispatcher watcher reads."""
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner._running = True
    runner._kanban_dispatcher_lock_handle = None
    return runner


async def _drive_watcher(runner, *, target_ticks: int, tick_counter=None):
    """Run ``_kanban_dispatcher_watcher`` until ``target_ticks`` ticks occur.

    A tick is counted by spying on ``reap_worker_zombies`` (called once at
    the top of each tick before per-board work). After ``target_ticks`` ticks
    we flip ``runner._running = False`` so the loop exits cleanly on its
    next between-tick check.

    ``asyncio.sleep`` is patched in the ``gateway.kanban_watchers`` module
    (where the watcher's module-level ``asyncio`` reference resolves) to
    yield immediately so the test doesn't wait on real seconds.
    """
    import gateway.kanban_watchers as kw

    _orig_sleep = asyncio.sleep
    state = {"ticks": 0}

    if tick_counter is None:
        tick_counter = state

    _real_reap = kb.reap_worker_zombies

    def _counting_reap():
        tick_counter["ticks"] += 1
        if tick_counter["ticks"] >= target_ticks:
            runner._running = False
        return _real_reap()

    async def _fast_sleep(_d):
        await _orig_sleep(0)

    with patch.object(kb, "reap_worker_zombies", _counting_reap), \
         patch.object(kw.asyncio, "sleep", side_effect=_fast_sleep):
        await asyncio.wait_for(
            runner._kanban_dispatcher_watcher(),
            timeout=30.0,
        )


# ---------------------------------------------------------------------------
# 1. Corrupt-board quarantine
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_corrupt_board_db_is_quarantined_and_dispatch_skipped(
    kanban_home, monkeypatch, caplog,
):
    """A board whose DB file is not a valid SQLite database is quarantined.

    The watcher's ``_tick_once_for_board`` opens the board DB via
    ``_kb.connect``; for a corrupt file, ``connect`` raises a
    ``KanbanDbCorruptError`` (or ``sqlite3.DatabaseError``). The watcher
    catches it via ``_is_corrupt_board_db_error``, records the fingerprint
    in ``disabled_corrupt_boards``, logs an error, and returns ``None``
    WITHOUT calling ``dispatch_once``. On a subsequent tick the same
    fingerprint is skipped silently (no retry) until the quarantine timer
    expires.
    """
    kb.create_board("corrupt-board")
    db_path = kb.kanban_db_path("corrupt-board")
    # Overwrite with bytes that are not a valid SQLite database. The header
    # validation in connect() classifies this as corrupt.
    db_path.write_bytes(b"\x00" * 2048)
    # Clear the per-process healthy-DB cache so connect() re-validates.
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))

    dispatch_calls: list[str] = []

    def _spy_dispatch_once(conn, **kwargs):
        dispatch_calls.append(kwargs.get("board", "<unknown>"))
        return kb.DispatchResult()

    monkeypatch.setattr(kb, "dispatch_once", _spy_dispatch_once)
    # Keep the health path quiet so it doesn't open the corrupt board.
    monkeypatch.setattr(kb, "has_spawnable_ready", lambda conn: False)
    monkeypatch.setattr(kb, "has_spawnable_review", lambda conn: False)

    runner = _make_runner()
    with caplog.at_level(logging.ERROR, logger="gateway.run"):
        await _drive_watcher(runner, target_ticks=3)

    # dispatch_once must never have been called for the corrupt board.
    assert "corrupt-board" not in dispatch_calls, (
        f"dispatch_once was called for the quarantined board: {dispatch_calls}"
    )
    # An error was logged about the corrupt board being quarantined.
    corrupt_logs = [
        r for r in caplog.records
        if "corrupt-board" in r.getMessage() and "not a valid" in r.getMessage()
    ]
    assert corrupt_logs, (
        "expected an error log marking the board as quarantined; got: "
        + "\n".join(r.getMessage() for r in caplog.records)
    )


@pytest.mark.asyncio
async def test_corrupt_board_quarantine_skips_retries_while_fingerprint_unchanged(
    kanban_home, monkeypatch, caplog,
):
    """Once quarantined, a board with an unchanged fingerprint is not retried
    by the dispatch path.

    The watcher's ``_tick_once_for_board`` records ``(fingerprint,
    disabled_at)`` and skips dispatch for
    ``CORRUPT_BOARD_RETRY_AFTER_SECONDS`` (300s). On subsequent ticks within
    that window, the fingerprint check returns early — no ``connect``, no
    ``dispatch_once``, and no repeated quarantine error log. The board is
    only retried after the timer expires or the file changes.

    (The separate ``_ready_nonempty`` health probe does open every board
    every tick, but it catches exceptions and does not use the quarantine
    map — that's by design and out of scope for this dispatch-path test.)
    """
    kb.create_board("corrupt-board")
    db_path = kb.kanban_db_path("corrupt-board")
    db_path.write_bytes(b"\x00NOTSQLITE\x00" * 100)
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))

    dispatch_calls: list[str] = []

    def _spy_dispatch_once(conn, **kwargs):
        dispatch_calls.append(kwargs.get("board", "<unknown>"))
        return kb.DispatchResult()

    monkeypatch.setattr(kb, "dispatch_once", _spy_dispatch_once)
    monkeypatch.setattr(kb, "has_spawnable_ready", lambda conn: False)
    monkeypatch.setattr(kb, "has_spawnable_review", lambda conn: False)

    runner = _make_runner()
    with caplog.at_level(logging.ERROR, logger="gateway.run"):
        await _drive_watcher(runner, target_ticks=3)

    # dispatch_once is never called for the quarantined board.
    assert "corrupt-board" not in dispatch_calls
    # The quarantine error log fires exactly once (first tick records the
    # fingerprint). Subsequent ticks skip before connect, so no repeat
    # error log. This proves the fingerprint-based skip is working: if the
    # board were re-opened every tick, the error would repeat.
    quarantine_logs = [
        r for r in caplog.records
        if "corrupt-board" in r.getMessage()
        and "not a valid" in r.getMessage()
    ]
    assert len(quarantine_logs) == 1, (
        f"expected exactly one quarantine error log (first tick only); "
        f"got {len(quarantine_logs)}. Repeated logs mean the fingerprint "
        f"skip is not working and the board is re-opened every tick. "
        f"Logs: {[r.getMessage() for r in quarantine_logs]}"
    )


# ---------------------------------------------------------------------------
# 2. Health telemetry: bad_ticks / HEALTH_WINDOW
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_health_warning_fires_after_health_window_consecutive_stuck_ticks(
    kanban_home, monkeypatch, caplog,
):
    """When the ready queue has spawnable work but the dispatcher spawns 0
    workers for ``HEALTH_WINDOW`` (6) consecutive ticks, the watcher emits a
    ``kanban dispatcher stuck`` warning.

    We mock ``has_spawnable_ready`` to return True (spawnable work waiting)
    and ``dispatch_once`` to return an empty ``DispatchResult`` (nothing
    spawned). After 6 stuck ticks the warning must fire. The warning is
    rate-limited to once per 300s, so we assert exactly one warning.
    """
    monkeypatch.setattr(kb, "has_spawnable_ready", lambda conn: True)
    monkeypatch.setattr(kb, "has_spawnable_review", lambda conn: False)

    def _empty_dispatch(conn, **kwargs):
        return kb.DispatchResult()  # spawned=[] → nothing spawned

    monkeypatch.setattr(kb, "dispatch_once", _empty_dispatch)

    runner = _make_runner()
    with caplog.at_level(logging.WARNING, logger="gateway.run"):
        # HEALTH_WINDOW = 6; drive enough ticks to cross the threshold.
        await _drive_watcher(runner, target_ticks=8)

    stuck_warnings = [
        r for r in caplog.records
        if "kanban dispatcher stuck" in r.getMessage()
    ]
    assert stuck_warnings, (
        "expected a 'kanban dispatcher stuck' warning after "
        "HEALTH_WINDOW consecutive stuck ticks; got: "
        + "\n".join(r.getMessage() for r in caplog.records)
    )
    # The warning message includes the consecutive-tick count (>= 6).
    msg = stuck_warnings[0].getMessage()
    assert "6 consecutive ticks" in msg or "6 consecutive" in msg, (
        f"warning should report >= 6 consecutive ticks; got: {msg}"
    )
    # Rate-limited to once per 300s — within our short run we expect one.
    assert len(stuck_warnings) == 1, (
        f"expected exactly one rate-limited warning; got {len(stuck_warnings)}"
    )


@pytest.mark.asyncio
async def test_health_warning_does_not_fire_when_workers_spawned(
    kanban_home, monkeypatch, caplog,
):
    """If the dispatcher spawns at least one worker on every tick, the
    ``bad_ticks`` counter resets and the stuck warning never fires.

    This is the negative control: spawnable work is waiting AND workers
    are spawned → not stuck. ``bad_ticks`` resets to 0 on any spawn.
    """
    monkeypatch.setattr(kb, "has_spawnable_ready", lambda conn: True)
    monkeypatch.setattr(kb, "has_spawnable_review", lambda conn: False)

    def _spawning_dispatch(conn, **kwargs):
        res = kb.DispatchResult()
        res.spawned = [("t1", "alice", "/tmp/ws")]
        return res

    monkeypatch.setattr(kb, "dispatch_once", _spawning_dispatch)

    runner = _make_runner()
    with caplog.at_level(logging.WARNING, logger="gateway.run"):
        await _drive_watcher(runner, target_ticks=8)

    stuck_warnings = [
        r for r in caplog.records
        if "kanban dispatcher stuck" in r.getMessage()
    ]
    assert not stuck_warnings, (
        "stuck warning must NOT fire when workers are spawned each tick; "
        "got: " + "\n".join(r.getMessage() for r in stuck_warnings)
    )


@pytest.mark.asyncio
async def test_health_warning_does_not_fire_when_ready_queue_empty(
    kanban_home, monkeypatch, caplog,
):
    """An empty ready queue is 'correctly idle', not stuck. ``bad_ticks`` only
    increments when ``ready_pending`` is True AND nothing spawned.
    """
    monkeypatch.setattr(kb, "has_spawnable_ready", lambda conn: False)
    monkeypatch.setattr(kb, "has_spawnable_review", lambda conn: False)

    def _empty_dispatch(conn, **kwargs):
        return kb.DispatchResult()

    monkeypatch.setattr(kb, "dispatch_once", _empty_dispatch)

    runner = _make_runner()
    with caplog.at_level(logging.WARNING, logger="gateway.run"):
        await _drive_watcher(runner, target_ticks=8)

    stuck_warnings = [
        r for r in caplog.records
        if "kanban dispatcher stuck" in r.getMessage()
    ]
    assert not stuck_warnings


# ---------------------------------------------------------------------------
# 3. Multi-board dispatch
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_dispatcher_dispatches_once_per_board_across_multiple_boards(
    kanban_home, monkeypatch,
):
    """The watcher enumerates every board on disk each tick and calls
    ``dispatch_once`` once per board.

    We create two real boards (default + a second), then spy on
    ``dispatch_once`` to record the board slug each call. After one tick,
    both boards must have been dispatched.
    """
    kb.create_board("proj-alpha")

    dispatched_boards: list[str] = []

    def _spy_dispatch_once(conn, **kwargs):
        board = kwargs.get("board", "<missing>")
        dispatched_boards.append(board)
        return kb.DispatchResult()

    monkeypatch.setattr(kb, "dispatch_once", _spy_dispatch_once)
    # Keep health telemetry quiet so it doesn't open extra connections.
    monkeypatch.setattr(kb, "has_spawnable_ready", lambda conn: False)
    monkeypatch.setattr(kb, "has_spawnable_review", lambda conn: False)

    runner = _make_runner()
    await _drive_watcher(runner, target_ticks=2)

    assert "default" in dispatched_boards, (
        f"default board was not dispatched; got {dispatched_boards}"
    )
    assert "proj-alpha" in dispatched_boards, (
        f"second board was not dispatched; got {dispatched_boards}"
    )


@pytest.mark.asyncio
async def test_dispatcher_dispatches_each_board_every_tick(
    kanban_home, monkeypatch,
):
    """Multi-board dispatch is not a one-shot discovery — every tick fans out
    to every board. After N ticks with 2 boards, each board has been
    dispatched at least N times.
    """
    kb.create_board("board-b")

    dispatched_boards: list[str] = []

    def _spy_dispatch_once(conn, **kwargs):
        board = kwargs.get("board", "<missing>")
        dispatched_boards.append(board)
        return kb.DispatchResult()

    monkeypatch.setattr(kb, "dispatch_once", _spy_dispatch_once)
    monkeypatch.setattr(kb, "has_spawnable_ready", lambda conn: False)
    monkeypatch.setattr(kb, "has_spawnable_review", lambda conn: False)

    runner = _make_runner()
    await _drive_watcher(runner, target_ticks=3)

    # Both boards must appear in every tick.
    boards = set(dispatched_boards)
    assert {"default", "board-b"} <= boards, (
        f"expected both boards dispatched; got {boards}"
    )
    # Each board dispatched at least once per tick (>= 3 ticks → >= 3 each).
    default_count = sum(1 for b in dispatched_boards if b == "default")
    boardb_count = sum(1 for b in dispatched_boards if b == "board-b")
    assert default_count >= 3, (
        f"default dispatched {default_count} times; expected >= 3 (once per tick)"
    )
    assert boardb_count >= 3, (
        f"board-b dispatched {boardb_count} times; expected >= 3 (once per tick)"
    )
