"""Cross-process release events wake the existing dispatcher, without changing blockers."""

import asyncio
import os
from pathlib import Path

import pytest

from gateway.kanban_watchers import GatewayKanbanWatchersMixin
from gateway.kanban_watchers_dispatcher import (
    _KanbanDispatcher,
    _resolve_dispatcher_settings,
)
from hermes_cli import (
    kanban_db as kb,
    kanban_db_connect as kbc,
    kanban_db_dispatch as kbd,
)


@pytest.mark.parametrize("release", ["complete", "block", "paused"])
def test_release_wakes_existing_dispatcher_and_preserves_pause(
    tmp_path, monkeypatch, release
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(kbd, "_profile_exists_fn", lambda: lambda name: True)
    monkeypatch.setattr(kbd, "_memory_pressure_level", lambda: "normal")
    config = {
        "kanban": {
            "dispatch_interval_seconds": 60,
            "max_in_progress": 1,
            "max_in_progress_per_profile": 1,
            "auto_decompose": False,
        }
    }
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: config)
    kb.init_db()
    watcher = GatewayKanbanWatchersMixin()
    watcher._running = True
    with kbc.connect_closing() as conn:
        task = kb.create_task(conn, title="Releasing", assignee="builder")
        claimed = kb.claim_task(conn, task, claimer="test")
        kbd._set_worker_pid(conn, task, os.getpid())
        successor = kb.create_task(
            conn,
            title="Next",
            assignee="builder",
            parents=[] if release == "block" else [task],
        )
    sleeps, spawned = [], []

    def spawn(task, workspace, **kwargs):
        spawned.append(task.id)
        watcher._running = False
        return os.getpid()

    async def release_during_sleep(delay):
        sleeps.append(delay)
        # First sleep is gateway startup; the second is between dispatch ticks.
        if len(sleeps) == 2:
            with kbc.connect_closing() as conn:
                if release == "block":
                    kb.block_task(
                        conn,
                        task,
                        reason="awaiting review from approving owner",
                        kind="needs_input",
                        expected_run_id=claimed.current_run_id,
                    )
                else:
                    kb.complete_task(
                        conn,
                        task,
                        result="Finished implementation",
                        expected_run_id=claimed.current_run_id,
                    )
            if release == "paused":
                (tmp_path / "ESTOP").touch()
        if len(sleeps) >= 3:
            watcher._running = False

    monkeypatch.setattr(kbd, "_default_spawn", spawn)
    monkeypatch.setattr(asyncio, "sleep", release_during_sleep)

    asyncio.run(watcher._kanban_dispatcher_watcher())

    assert spawned == ([] if release == "paused" else [successor])
    assert len(sleeps) == (3 if release == "paused" else 2)
    with kbc.connect_closing() as conn:
        assert kb.get_task(conn, task).status == (
            "blocked" if release == "block" else "done"
        )
        assert kb.get_task(conn, successor).status == (
            "ready" if release == "paused" else "running"
        )


def test_nonrelease_events_do_not_shorten_wait_and_pinned_boards_are_deduplicated(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    pinned = kb.kanban_db_path()
    monkeypatch.setenv("HERMES_KANBAN_DB", str(pinned))
    dispatcher = _KanbanDispatcher(kb, _resolve_dispatcher_settings({}, kb))
    monkeypatch.setattr(dispatcher, "_board_slugs", lambda: ["default", "other"])
    watcher = GatewayKanbanWatchersMixin()
    watcher._running = True
    assert not dispatcher.released_tasks_changed()
    sleeps = []

    async def comment_during_sleep(delay):
        sleeps.append(delay)
        if len(sleeps) == 1:
            with kbc.connect_closing() as conn:
                task = kb.create_task(conn, title="New task")
                kb.add_comment(conn, task, "tester", "still working")

    monkeypatch.setattr(asyncio, "sleep", comment_during_sleep)

    asyncio.run(
        watcher._sleep_between_ticks(3, wake_check=dispatcher.released_tasks_changed)
    )

    assert len(sleeps) == 3
    # A real release through either slug is observed exactly once.
    with kbc.connect_closing() as conn:
        task = kb.create_task(conn, title="Release")
        kb.complete_task(conn, task, result="Done")
    assert dispatcher.released_tasks_changed()
    assert not dispatcher.released_tasks_changed()
