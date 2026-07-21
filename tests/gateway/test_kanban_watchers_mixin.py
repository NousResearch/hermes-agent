"""Tests for the extracted GatewayKanbanWatchersMixin (god-file Phase 3).

The kanban watcher loops were lifted out of gateway/run.py into a mixin that
GatewayRunner inherits. These tests confirm the mixin exposes the methods and
that GatewayRunner picks them up via the MRO (behavior-neutral relocation).
"""

from __future__ import annotations

import asyncio
import inspect
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import threading
import time

import pytest

from gateway.kanban_watchers import GatewayKanbanWatchersMixin

KANBAN_METHODS = [
    "_kanban_notifier_watcher",
    "_kanban_dispatcher_watcher",
    "_kanban_advance",
    "_kanban_unsub",
    "_kanban_rewind",
    "_deliver_kanban_artifacts",
    "_kanban_db_to_thread",
    "_drain_kanban_db_operations",
]


def test_mixin_defines_kanban_methods():
    for m in KANBAN_METHODS:
        assert hasattr(GatewayKanbanWatchersMixin, m), f"mixin missing {m}"


def test_gateway_runner_inherits_mixin():
    # Import here so a heavy gateway import only happens if the first test passed.
    from gateway.run import GatewayRunner

    assert issubclass(GatewayRunner, GatewayKanbanWatchersMixin)
    # Each kanban method resolves to the mixin's implementation via the MRO.
    for m in KANBAN_METHODS:
        owner = next(c for c in GatewayRunner.__mro__ if m in c.__dict__)
        assert owner is GatewayKanbanWatchersMixin, (
            f"{m} resolved to {owner.__name__}, expected the mixin"
        )


def test_watcher_loops_are_coroutines():
    # The two long-running watchers are async loops.
    assert inspect.iscoroutinefunction(GatewayKanbanWatchersMixin._kanban_notifier_watcher)
    assert inspect.iscoroutinefunction(GatewayKanbanWatchersMixin._kanban_dispatcher_watcher)


def test_singleton_dispatcher_lock_is_exclusive(tmp_path):
    """Only one holder of the dispatcher lock at a time — the backstop that
    stops concurrent dispatchers double reclaiming and corrupting shared
    kanban SQLite index pages under wal_autocheckpoint=0."""
    import os

    from gateway.kanban_watchers import _acquire_singleton_lock, _release_singleton_lock

    lock = tmp_path / "kanban" / ".dispatcher.lock"

    h1, st1 = _acquire_singleton_lock(lock)
    assert st1 == "held" and h1 is not None

    # A second acquire while the first is held must be refused, not granted.
    h2, st2 = _acquire_singleton_lock(lock)
    assert st2 == "contended" and h2 is None

    # Releasing the first lets a fresh acquire succeed (lock is reusable).
    _release_singleton_lock(h1)
    h3, st3 = _acquire_singleton_lock(lock)
    assert st3 == "held" and h3 is not None
    _release_singleton_lock(h3)


def test_kanban_shutdown_drain_waits_for_inflight_db_operation():
    runner = GatewayKanbanWatchersMixin()
    runner._kanban_db_draining = False
    runner._kanban_db_operations = set()
    entered = threading.Event()
    release = threading.Event()

    def db_operation():
        entered.set()
        assert release.wait(timeout=2.0)
        return "committed"

    async def exercise():
        operation = asyncio.create_task(runner._kanban_db_to_thread(db_operation))
        assert await asyncio.to_thread(entered.wait, 1.0)

        drain = asyncio.create_task(runner._drain_kanban_db_operations(1.0))
        await asyncio.sleep(0)
        assert not drain.done()
        release.set()

        assert await drain is True
        assert await operation == "committed"
        assert runner._kanban_db_operations == set()

        # Once shutdown closes admission, no fresh SQLite work may start.
        with pytest.raises(asyncio.CancelledError):
            await runner._kanban_db_to_thread(lambda: None)

    asyncio.run(exercise())


def test_kanban_shutdown_drain_is_bounded():
    runner = GatewayKanbanWatchersMixin()
    runner._kanban_db_draining = False
    runner._kanban_db_operations = set()
    release = threading.Event()

    async def exercise():
        operation = asyncio.create_task(
            runner._kanban_db_to_thread(release.wait, 1.0)
        )
        await asyncio.sleep(0.01)
        assert await runner._drain_kanban_db_operations(0.01) is False
        release.set()
        await operation

    asyncio.run(exercise())


def test_zero_second_agent_drain_still_waits_for_inflight_kanban_commit():
    runner = GatewayKanbanWatchersMixin()
    runner._kanban_db_draining = False
    runner._kanban_db_operations = set()
    entered = threading.Event()
    release = threading.Event()

    def db_operation():
        entered.set()
        assert release.wait(timeout=1.0)

    async def exercise():
        operation = asyncio.create_task(runner._kanban_db_to_thread(db_operation))
        assert await asyncio.to_thread(entered.wait, 1.0)

        drain = asyncio.create_task(runner._drain_kanban_db_operations(0.0))
        await asyncio.sleep(0)
        assert not drain.done()
        release.set()

        assert await drain is True
        await operation

    asyncio.run(exercise())


@pytest.mark.skipif(os.name == "nt", reason="POSIX SIGTERM fault injection")
def test_sigterm_during_kanban_write_drains_before_subprocess_exit(
    tmp_path, monkeypatch
):
    """A gateway-style SIGTERM must not tear down an in-flight SQLite commit."""
    from hermes_cli import kanban_db as kb

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="before shutdown")
        claimed = kb.claim_task(conn, task_id)
        assert claimed is not None
        run_id = claimed.current_run_id
        assert run_id is not None

    entered = tmp_path / "entered"
    release = tmp_path / "release"
    drained = tmp_path / "drained"
    script = textwrap.dedent(
        """
        import asyncio
        import os
        from pathlib import Path
        import signal
        import time

        from gateway.kanban_watchers import GatewayKanbanWatchersMixin
        from hermes_cli import kanban_db as kb

        entered = Path(os.environ["TEST_ENTERED"])
        release = Path(os.environ["TEST_RELEASE"])
        drained = Path(os.environ["TEST_DRAINED"])
        task_id = os.environ["TEST_TASK_ID"]

        async def main():
            runner = GatewayKanbanWatchersMixin()
            runner._kanban_db_draining = False
            runner._kanban_db_operations = set()
            stopped = asyncio.Event()

            def write():
                with kb.connect() as conn:
                    with kb.write_txn(conn):
                        conn.execute(
                            "UPDATE tasks SET status = 'done' WHERE id = ? "
                            "AND status = 'running'",
                            (task_id,),
                        )
                        entered.write_text("1", encoding="utf-8")
                        deadline = time.monotonic() + 5
                        while not release.exists() and time.monotonic() < deadline:
                            time.sleep(0.01)
                        if not release.exists():
                            raise TimeoutError("test never released DB operation")
                        run_id = kb._end_run(
                            conn,
                            task_id,
                            outcome="completed",
                            status="done",
                            summary="committed during shutdown",
                        )
                        kb._append_event(
                            conn,
                            task_id,
                            "completed",
                            {"fault_injection": True},
                            run_id=run_id,
                        )

            async def shutdown():
                ok = await runner._drain_kanban_db_operations(4.0)
                drained.write_text(str(ok), encoding="utf-8")
                stopped.set()

            loop = asyncio.get_running_loop()
            loop.add_signal_handler(
                signal.SIGTERM, lambda: asyncio.create_task(shutdown())
            )
            operation = asyncio.create_task(runner._kanban_db_to_thread(write))
            await stopped.wait()
            await operation

        asyncio.run(main())
        """
    )
    env = os.environ.copy()
    env.update(
        {
            "HERMES_HOME": str(home),
            "TEST_ENTERED": str(entered),
            "TEST_RELEASE": str(release),
            "TEST_DRAINED": str(drained),
            "TEST_TASK_ID": task_id,
        }
    )
    proc = subprocess.Popen(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[2],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 5
        while not entered.exists() and proc.poll() is None and time.monotonic() < deadline:
            time.sleep(0.01)
        assert entered.exists(), proc.communicate(timeout=1)

        proc.terminate()
        time.sleep(0.1)
        assert proc.poll() is None, "subprocess exited before its DB write drained"
        release.write_text("1", encoding="utf-8")
        stdout, stderr = proc.communicate(timeout=5)
        assert proc.returncode == 0, (stdout, stderr)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=2)

    assert drained.read_text(encoding="utf-8") == "True"
    with kb.connect() as conn:
        task = kb.get_task(conn, task_id)
        assert task is not None
        assert task.status == "done"
        assert task.current_run_id is None
        run = conn.execute(
            "SELECT status, outcome, ended_at FROM task_runs WHERE id = ?",
            (run_id,),
        ).fetchone()
        assert (run["status"], run["outcome"]) == ("done", "completed")
        assert run["ended_at"] is not None
        terminal_events = conn.execute(
            "SELECT kind FROM task_events WHERE task_id = ? AND kind = 'completed'",
            (task_id,),
        ).fetchall()
        assert len(terminal_events) == 1
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
        assert conn.execute("PRAGMA quick_check").fetchone()[0] == "ok"
