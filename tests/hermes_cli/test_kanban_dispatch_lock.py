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

import builtins

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


def _hide_module(monkeypatch, missing: str) -> None:
    """Make ``import <missing>`` raise, as it does on a platform without it.

    Patches ``builtins.__import__`` rather than poking ``sys.modules``: the
    locking primitives are imported *inside* ``_dispatch_tick_lock``, so the
    import statement runs on every call and a ``sys.modules[missing] = None``
    would raise ``ImportError`` only on some Python versions. Intercepting the
    import hook reproduces ``ModuleNotFoundError`` exactly, on every version.
    """
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == missing:
            raise ModuleNotFoundError(f"No module named {missing!r}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)


def test_missing_locking_primitive_degrades_to_a_no_op(conn, monkeypatch):
    """A platform with no fcntl/msvcrt must still dispatch (#89653).

    Regression: the handler around the import caught ``(BlockingIOError,
    OSError)`` and the enclosing block caught ``OSError``, but a missing module
    raises ``ModuleNotFoundError`` -> ``ImportError`` -> ``Exception``, which is
    neither. The documented degradation to a no-op was therefore unreachable and
    the tick raised instead, so a platform that merely *cannot enforce*
    single-writer could not dispatch at all.

    The lock's own docstring makes this the intended contract: single-writer
    enforcement is best-effort, and the orphan-dispatcher scenario it defends
    against (#35240) is specific to POSIX service managers.
    """
    db_path = kb.kanban_db_path(board="default")
    _hide_module(monkeypatch, "msvcrt" if kb._IS_WINDOWS else "fcntl")

    with kb._dispatch_tick_lock(db_path) as held:
        assert held is True, (
            "a platform with no locking primitive cannot enforce single-writer, "
            "so the guard must degrade to a no-op and let the tick proceed "
            "rather than raising"
        )


def test_missing_locking_primitive_also_survives_release(conn, monkeypatch):
    """The no-op must survive *exiting* the context manager, not just entering.

    The release path re-imports the same module under a handler that also
    excluded ``ImportError``, so fixing only the acquire path moves the crash
    from ``__enter__`` to ``__exit__`` instead of removing it.

    The test above does also exercise the exit (leaving its ``with`` block runs
    the same ``finally``), but it surfaces a release crash as a teardown error
    on a test named for the acquire contract. This one names the release path
    directly and asserts after the block has closed, so the failure says which
    half regressed.
    """
    db_path = kb.kanban_db_path(board="default")
    _hide_module(monkeypatch, "msvcrt" if kb._IS_WINDOWS else "fcntl")

    seen = None

    with kb._dispatch_tick_lock(db_path) as held:
        seen = held

    assert seen is True, "the degraded tick must report that it may proceed"


def test_missing_primitive_is_not_confused_with_a_busy_lock(conn, monkeypatch):
    """The two failure modes resolve OPPOSITE ways, so pin them together.

    A *busy* lock (another dispatcher owns it) must yield ``False`` - that is
    the whole point of the guard. A *missing primitive* must yield ``True``.
    Collapsing ImportError into the busy tuple would silently disable dispatch
    forever on such a platform, which is the failure this pairing guards
    against.
    """
    db_path = kb.kanban_db_path(board="default")

    with kb._dispatch_tick_lock(db_path) as first:
        assert first is True, "an uncontended lock must be acquired"

        with kb._dispatch_tick_lock(db_path) as second:
            assert second is False, (
                "a lock already held by this process must NOT be reported as "
                "acquired - a busy lock skips the tick"
            )

    _hide_module(monkeypatch, "msvcrt" if kb._IS_WINDOWS else "fcntl")

    with kb._dispatch_tick_lock(db_path) as degraded:
        assert degraded is True, (
            "a missing primitive is permanent, so it must resolve the opposite "
            "way from a busy lock"
        )
