"""Cross-board concurrency tests for ``kanban.global_max_in_progress``."""

from __future__ import annotations

import os
import time
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
    monkeypatch.setattr("hermes_cli.profiles.profile_exists", lambda _profile: True)
    kb._INITIALIZED_PATHS.clear()
    kb.init_db(board="default")
    kb.create_board("other", name="Other")
    return home


def _mark_running(conn, task_id: str) -> None:
    now = int(time.time())
    conn.execute(
        "UPDATE tasks SET status = 'running', claim_lock = ?, claim_expires = ?, "
        "worker_pid = ?, started_at = ?, last_heartbeat_at = ? WHERE id = ?",
        (f"lock-{task_id}", now + 3600, os.getpid(), now, now, task_id),
    )
    conn.commit()


def _spawn_stub(calls: list[str]):
    def spawn(task, _workspace_path, board=None):
        calls.append(f"{board}:{task.id}")
        return os.getpid()

    return spawn


def test_global_cap_counts_running_tasks_on_other_boards(kanban_home):
    with kb.connect(board="default") as default_conn:
        running_id = kb.create_task(default_conn, title="already running", assignee="worker")
        _mark_running(default_conn, running_id)

    calls: list[str] = []
    with kb.connect(board="other") as other_conn:
        ready_id = kb.create_task(other_conn, title="must wait", assignee="worker")
        result = kb.dispatch_once(
            other_conn,
            board="other",
            global_max_in_progress=1,
            spawn_fn=_spawn_stub(calls),
        )
        assert result.spawned == []
        assert calls == []
        assert kb.get_task(other_conn, ready_id).status == "ready"


def test_global_cap_only_fills_remaining_cross_board_slots(kanban_home):
    with kb.connect(board="default") as default_conn:
        running_id = kb.create_task(default_conn, title="already running", assignee="worker")
        _mark_running(default_conn, running_id)

    calls: list[str] = []
    with kb.connect(board="other") as other_conn:
        task_ids = [
            kb.create_task(other_conn, title=f"ready {n}", assignee="worker")
            for n in range(3)
        ]
        result = kb.dispatch_once(
            other_conn,
            board="other",
            global_max_in_progress=2,
            spawn_fn=_spawn_stub(calls),
        )
        assert len(result.spawned) == 1
        assert len(calls) == 1
        statuses = [kb.get_task(other_conn, task_id).status for task_id in task_ids]
        assert statuses.count("running") == 1
        assert statuses.count("ready") == 2


def test_global_cap_serializes_dispatch_across_boards(kanban_home):
    calls: list[str] = []
    with kb.connect(board="other") as other_conn:
        kb.create_task(other_conn, title="must not race", assignee="worker")
        with kb._global_dispatch_tick_lock() as held:
            assert held is True
            result = kb.dispatch_once(
                other_conn,
                board="other",
                global_max_in_progress=2,
                spawn_fn=_spawn_stub(calls),
            )

    assert result.skipped_locked is True
    assert calls == []


def test_global_cap_composes_with_stricter_board_cap(kanban_home):
    calls: list[str] = []
    with kb.connect(board="other") as other_conn:
        first = kb.create_task(other_conn, title="local running", assignee="worker")
        _mark_running(other_conn, first)
        kb.create_task(other_conn, title="must wait locally", assignee="worker")
        result = kb.dispatch_once(
            other_conn,
            board="other",
            max_in_progress=1,
            global_max_in_progress=4,
            spawn_fn=_spawn_stub(calls),
        )

    assert result.spawned == []
    assert calls == []


def test_review_only_dispatch_respects_remaining_global_slot(kanban_home):
    with kb.connect(board="default") as default_conn:
        running_id = kb.create_task(
            default_conn, title="already running", assignee="worker"
        )
        _mark_running(default_conn, running_id)

    calls: list[str] = []
    with kb.connect(board="other") as other_conn:
        review_ids = [
            kb.create_task(other_conn, title=f"review {n}", assignee="worker")
            for n in range(2)
        ]
        other_conn.executemany(
            "UPDATE tasks SET status = 'review' WHERE id = ?",
            [(task_id,) for task_id in review_ids],
        )
        other_conn.commit()

        result = kb.dispatch_once(
            other_conn,
            board="other",
            global_max_in_progress=2,
            spawn_fn=_spawn_stub(calls),
        )
        statuses = [kb.get_task(other_conn, task_id).status for task_id in review_ids]

    assert len(result.spawned) == 1
    assert len(calls) == 1
    assert statuses.count("running") == 1
    assert statuses.count("review") == 1


def test_dry_run_reports_only_global_headroom(kanban_home):
    with kb.connect(board="default") as conn:
        for index in range(3):
            kb.create_task(conn, title=f"ready {index}", assignee="worker")

        result = kb.dispatch_once(
            conn,
            board="default",
            dry_run=True,
            global_max_in_progress=1,
        )
        running = conn.execute(
            "SELECT COUNT(*) FROM tasks WHERE status = 'running'"
        ).fetchone()[0]

    assert len(result.spawned) == 1
    assert running == 0


def test_running_count_deduplicates_board_aliases(kanban_home, monkeypatch):
    with kb.connect(board="default") as conn:
        running_id = kb.create_task(conn, title="already running", assignee="worker")
        _mark_running(conn, running_id)
        shared_path = kb.kanban_db_path(board="default")
        monkeypatch.setattr(
            kb,
            "list_boards",
            lambda include_archived=False: [
                {"slug": "default"},
                {"slug": "alias"},
            ],
        )
        monkeypatch.setattr(kb, "kanban_db_path", lambda board=None: shared_path)

        count = kb._count_running_across_boards(
            current_conn=conn,
            current_board="default",
        )

    assert count == 1


def test_run_daemon_passes_global_cap(kanban_home, monkeypatch):
    import threading

    stop = threading.Event()
    captured = {}

    def fake_dispatch_once(_conn, **kwargs):
        captured.update(kwargs)
        stop.set()
        return kb.DispatchResult()

    monkeypatch.setattr(kb, "dispatch_once", fake_dispatch_once)
    kb.run_daemon(
        interval=0,
        global_max_in_progress=2,
        stop_event=stop,
    )

    assert captured["global_max_in_progress"] == 2
