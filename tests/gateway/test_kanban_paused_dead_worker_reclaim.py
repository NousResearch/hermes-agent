"""Dead workers are still reclaimed while `hermes pause` (ESTOP) holds kanban dispatch.

The ESTOP gate skipped the whole dispatcher tick, and ``dispatch_once`` owns every reclaim sweep,
so a worker that died mid-pause (an OOM-killed worker scope; or the worker that engaged the pause
itself) kept its card ``running`` for as long as the pause lasted -- 11.7 h in the field, until an
operator reset it by hand. Paused ticks now run only the non-signalling dead-worker sweeps; nothing
is promoted, spawned, or killed.
"""

from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path

import pytest

from agent import estop
from gateway import kanban_watchers_dispatcher as kwd
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    db_path = kb.kanban_db_path(board="default")
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
    kb.init_db()
    estop._logged_components.clear()
    return home


def _dispatcher(reconcile_orphans: bool = True) -> kwd._KanbanDispatcher:
    settings = kwd._DispatcherSettings(60.0, None, None, 2, 14400, reconcile_orphans, None, None)
    return kwd._KanbanDispatcher(kb, settings)


def _running_task(conn, title: str, pid: int) -> str:
    tid = kb.create_task(conn, title=title, assignee="w")
    kb.claim_task(conn, tid)
    kbd._set_worker_pid(conn, tid, pid)
    # Past the launch grace window, like a worker that ran for a while before it died.
    conn.execute("UPDATE tasks SET started_at = started_at - 9999 WHERE id=?", (tid,))
    conn.execute("UPDATE task_runs SET started_at = started_at - 9999 WHERE task_id=?", (tid,))
    conn.commit()
    return tid


def _status(conn, tid: str) -> str:
    return conn.execute("SELECT status FROM tasks WHERE id=?", (tid,)).fetchone()["status"]


@pytest.mark.platforms("posix")
def test_paused_sweep_releases_dead_worker_and_leaves_live_one(kanban_home):
    dead = subprocess.Popen(["true"])
    dead.wait()
    live = subprocess.Popen(["sleep", "30"])
    try:
        with kbc.connect() as conn:
            dead_tid = _running_task(conn, "oom-killed mid-pause", dead.pid)
            live_tid = _running_task(conn, "still working", live.pid)
            waiting_tid = kb.create_task(conn, title="queued", assignee="w")
            kbd._record_worker_exit(dead.pid, 9)  # SIGKILL, as the OOM killer delivers

        estop.engage(reason="update window")
        released = dict(_dispatcher().paused_reclaim())

        assert released == {"default": [dead_tid]}
        with kbc.connect() as conn:
            assert _status(conn, dead_tid) == "ready"
            assert _status(conn, live_tid) == "running", "a live worker must never be touched"
            assert _status(conn, waiting_tid) == "ready"
            run = conn.execute(
                "SELECT outcome, ended_at FROM task_runs WHERE task_id=? ORDER BY id DESC LIMIT 1",
                (dead_tid,),
            ).fetchone()
            assert run["outcome"] == "crashed" and run["ended_at"] is not None
            # Paused means nothing new starts: the released card is not re-claimed this tick.
            assert conn.execute(
                "SELECT COUNT(*) FROM task_runs WHERE task_id=?", (dead_tid,)
            ).fetchone()[0] == 1
    finally:
        live.kill()
        live.wait()


@pytest.mark.platforms("posix")
def test_paused_sweep_skips_when_board_tick_lock_is_held(kanban_home):
    dead = subprocess.Popen(["true"])
    dead.wait()
    with kbc.connect() as conn:
        tid = _running_task(conn, "dead", dead.pid)

    with kbc._dispatch_tick_lock(kb.kanban_db_path(board="default")) as held:
        assert held
        # Another dispatcher holds the board's single-writer lock: the paused sweep writes nothing.
        assert _dispatcher().paused_reclaim_for_board("default") == []

    with kbc.connect() as conn:
        assert _status(conn, tid) == "running"
    # Once the lock is free the next paused tick releases it.
    assert _dispatcher().paused_reclaim_for_board("default") == [tid]


def test_paused_watcher_tick_sweeps_but_never_dispatches(kanban_home, monkeypatch):
    """The gateway loop's ESTOP branch calls the dead-worker sweep and nothing else."""
    from gateway.kanban_watchers import GatewayKanbanWatchersMixin

    calls: list[str] = []

    class _FakeDispatcher:
        def __init__(self, *_a, **_k):
            pass

        def paused_reclaim(self):
            calls.append("paused_reclaim")
            runner._running = False
            return []

        def tick_once(self):
            calls.append("tick_once")
            return []

        def auto_decompose_tick(self, _n):
            calls.append("auto_decompose")
            return 0

        def ready_nonempty(self):
            return False

    class _Runner(GatewayKanbanWatchersMixin):
        _running = True

        def _kanban_dispatcher_boot(self):
            return (lambda: {}, kb, {})

        async def _sleep_between_ticks(self, interval):
            return None

        def _release_kanban_dispatcher_lock(self):
            return None

    import gateway.kanban_watchers as kw

    async def _no_sleep(_s):
        return None

    monkeypatch.setattr(kw, "_KanbanDispatcher", _FakeDispatcher)
    monkeypatch.setattr(kw.asyncio, "sleep", _no_sleep)
    estop.engage(reason="test")
    runner = _Runner()
    asyncio.run(asyncio.wait_for(runner._kanban_dispatcher_watcher(), timeout=10))

    assert calls == ["paused_reclaim"]
