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
