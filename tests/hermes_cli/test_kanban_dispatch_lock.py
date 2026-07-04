"""Tests for the kanban dispatcher single-writer lock (issue #35240).

A ``hermes gateway run --replace`` / ``gateway restart`` from a shell on a
systemd/launchd host can leave an orphan dispatcher that escapes the
service cgroup, survives ``systemctl restart``, and becomes a second
long-lived writer on the same ``kanban.db`` — the documented root cause of
multi-writer SQLite WAL corruption. ``dispatch_once`` now wraps each tick in
a non-blocking, board-scoped dispatch lock so two dispatchers can never run
a reclaim/spawn/write tick concurrently. The losing dispatcher returns an
empty ``DispatchResult`` with ``skipped_locked=True`` and does no DB writes.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    db_path = kb.kanban_db_path(board="default")
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
    kb.init_db()
    return home


@pytest.fixture
def conn(kanban_home):
    with kb.connect() as c:
        yield c


def test_uncontended_tick_runs_and_is_not_skipped(conn):
    """With no other holder, a tick runs normally and skipped_locked is False."""
    kb.create_task(conn, title="t", assignee="w")
    result = kb.dispatch_once(conn)
    assert result.skipped_locked is False


def test_held_lock_skips_the_tick_without_writes(conn):
    """While another holder owns the board lock, dispatch_once must skip and
    must NOT invoke spawn_fn (no DB writes happen on a skipped tick)."""
    kb.create_task(conn, title="t", assignee="w")
    db_path = kb.kanban_db_path(board="default")

    spawn_calls: list = []

    def spy_spawn(task, workspace_path, board=None):
        spawn_calls.append(getattr(task, "id", task))
        return 999999

    # Hold the lock, then attempt a contended tick.
    with kb._dispatch_tick_lock(db_path) as held:
        assert held is True  # we genuinely acquired it
        result = kb.dispatch_once(conn, spawn_fn=spy_spawn)

    assert result.skipped_locked is True
    assert result.spawned == []
    assert spawn_calls == [], "spawn_fn must not run while the tick is locked out"


def test_lock_releases_so_next_tick_runs(conn):
    """After the holder releases, the next tick is no longer skipped."""
    kb.create_task(conn, title="t", assignee="w")
    db_path = kb.kanban_db_path(board="default")

    with kb._dispatch_tick_lock(db_path) as held:
        assert held is True
        assert kb.dispatch_once(conn).skipped_locked is True

    # Lock released — a fresh tick proceeds.
    assert kb.dispatch_once(conn).skipped_locked is False


def test_lock_is_board_scoped(conn):
    """Holding board A's dispatch lock must not block a tick on board B —
    distinct boards have distinct DB files and tick independently."""
    db_default = kb.kanban_db_path(board="default")
    db_other = db_default.with_name("other-board-kanban.db")

    # Two different lock files → both acquirable simultaneously.
    with kb._dispatch_tick_lock(db_default) as held_a:
        assert held_a is True
        with kb._dispatch_tick_lock(db_other) as held_b:
            assert held_b is True, "a lock on a different board must be independent"


def test_reentrant_same_path_lock_is_exclusive(conn):
    """A second acquisition of the SAME board's lock from a sibling context
    must report not-held (the flock is exclusive within the host)."""
    db_path = kb.kanban_db_path(board="default")
    with kb._dispatch_tick_lock(db_path) as held_a:
        assert held_a is True
        with kb._dispatch_tick_lock(db_path) as held_b:
            assert held_b is False, "same-board lock must be exclusive"


def test_held_lock_refreshes_mtime_for_external_watchdogs(conn, tmp_path):
    """Each successful acquire must advance the lock file's mtime so external
    watchdogs (e.g. ``kanban_dispatcher_watchdog.py``) can detect a stuck
    dispatcher by mtime-freeze instead of false-positive every TTL.

    Regression guard for the post-#50331 follow-up: the original implementation
    opened the lock with ``"a+b"`` and never wrote to it, so mtime was set only
    on first open and froze forever — making every long-lived dispatcher look
    stale to a naive mtime-based watchdog. The fix refreshes mtime inside the
    critical section, after flock succeeds.
    """
    import os
    import time

    db_path = kb.kanban_db_path(board="default")
    lock_path = db_path.with_name(db_path.name + ".dispatch.lock")

    mtimes: list[float] = []
    # Five sequential acquire/release cycles must yield five distinct mtimes.
    # Sleep > filesystem mtime resolution (1s on most filesystems) — but
    # ext4/APFS sub-second resolution supports distinguishing 0.2s gaps.
    for _ in range(5):
        with kb._dispatch_tick_lock(db_path) as held:
            assert held is True
            assert lock_path.exists(), "lock file must be created on acquire"
            mtimes.append(lock_path.stat().st_mtime)
            time.sleep(0.2)

    assert len(set(mtimes)) == 5, (
        f"lock mtime must advance on every acquire; got {mtimes}"
    )
    assert mtimes == sorted(mtimes), "mtimes must be monotonically increasing"

    # Sanity: a non-blocking loser does NOT refresh mtime (it didn't acquire).
    stale_mtime = lock_path.stat().st_mtime
    time.sleep(0.2)
    with kb._dispatch_tick_lock(db_path) as held_a:
        assert held_a is True
        first_holder_mtime = lock_path.stat().st_mtime
        with kb._dispatch_tick_lock(db_path) as held_b:
            assert held_b is False, "loser must not acquire"
            assert lock_path.stat().st_mtime == first_holder_mtime, (
                "loser must not touch the mtime while holder still holds"
            )
    # After release, the next acquire refreshes again.
    time.sleep(0.2)
    with kb._dispatch_tick_lock(db_path) as held_c:
        assert held_c is True
        assert lock_path.stat().st_mtime > first_holder_mtime, (
            "subsequent acquire must refresh mtime"
        )


def test_utime_failure_does_not_break_acquire(conn, monkeypatch):
    """If ``os.utime`` fails (e.g. read-only FS), acquire must still yield True
    so dispatch continues. Best-effort heartbeat, not a hard requirement."""
    db_path = kb.kanban_db_path(board="default")

    def boom(*_args, **_kwargs):
        raise OSError("simulated read-only filesystem")

    monkeypatch.setattr("hermes_cli.kanban_db.os.utime", boom)
    # Must not raise, must yield True.
    with kb._dispatch_tick_lock(db_path) as held:
        assert held is True
