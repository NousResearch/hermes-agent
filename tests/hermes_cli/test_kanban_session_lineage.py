from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


class FakeSessionDB:
    def __init__(self, records: dict[str, dict]):
        self.records = records

    def get_session(self, session_id: str):
        return self.records.get(session_id)


def test_session_lineage_rejects_cross_task_workspace_and_title_mismatch(
    kanban_home,
):
    conn = kb.connect()
    try:
        tid = kb.create_task(
            conn,
            title="Build correct project",
            assignee="worker",
            workspace_kind="dir",
            workspace_path=str(kanban_home / "project-a"),
            session_id="foreign-session",
        )
        session_db = FakeSessionDB(
            {
                "foreign-session": {
                    "id": "foreign-session",
                    "title": "Unrelated compressed session",
                    "kanban_task_id": "other-task",
                    "kanban_board": "default",
                    "workspace_path": str(kanban_home / "project-b"),
                }
            }
        )

        verdict = kb.validate_task_session_lineage(
            conn, tid, session_db=session_db, board="default"
        )

        assert verdict.ok is False
        assert "foreign-session" in verdict.reason
        assert "other-task" in verdict.reason
        assert "workspace" in verdict.reason
    finally:
        conn.close()


def test_dispatch_blocks_session_lineage_mismatch_before_spawn(
    kanban_home, all_assignees_spawnable
):
    spawned = []
    conn = kb.connect()
    try:
        tid = kb.create_task(
            conn,
            title="lineage guarded card",
            assignee="worker",
            workspace_kind="dir",
            workspace_path=str(kanban_home / "project-a"),
            session_id="foreign-session",
        )
        session_db = FakeSessionDB(
            {
                "foreign-session": {
                    "id": "foreign-session",
                    "title": "another task",
                    "kanban_task_id": "other-task",
                    "kanban_board": "default",
                    "workspace_path": str(kanban_home / "project-b"),
                }
            }
        )

        res = kb.dispatch_once(
            conn,
            spawn_fn=lambda task, workspace: spawned.append(task.id) or 43210,
            session_db=session_db,
            board="default",
        )

        assert spawned == []
        assert tid in res.spawn_blocked_session_lineage
        task = kb.get_task(conn, tid)
        assert task.status == "blocked"
        assert task.consecutive_failures == 0
        assert task.last_failure_error
        assert "session lineage" in task.last_failure_error
        assert "foreign-session" in task.last_failure_error
        run = kb.latest_run(conn, tid)
        assert run is not None
        assert run.outcome == "spawn_blocked_session_lineage"
    finally:
        conn.close()


def test_worker_running_claim_requires_current_run_live_process_and_running_status(
    kanban_home, all_assignees_spawnable
):
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="running claim", assignee="worker")
        kb.dispatch_once(conn, spawn_fn=lambda task, workspace: 24680)

        live = kb.worker_running_claim(
            conn, tid, process_alive=lambda pid: pid == 24680, now=1_700_000_000
        )
        assert live.running is True
        assert live.reason == "running"

        dead = kb.worker_running_claim(
            conn, tid, process_alive=lambda pid: False, now=1_700_000_000
        )
        assert dead.running is False
        assert "process" in dead.reason

        kb.block_task(conn, tid, "operator stop", kind="needs_input")
        blocked = kb.worker_running_claim(
            conn, tid, process_alive=lambda pid: True, now=1_700_000_000
        )
        assert blocked.running is False
        assert "status" in blocked.reason
    finally:
        conn.close()
