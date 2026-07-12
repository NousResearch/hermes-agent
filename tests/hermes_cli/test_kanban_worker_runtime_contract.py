"""Regression coverage for the dispatcher-owned Kanban worker runtime contract."""
from __future__ import annotations

import os
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
    kb._worker_registry.clear()
    kb._recent_worker_exits.clear()
    yield home
    kb._worker_registry.clear()


def test_registered_worker_nonzero_exit_is_retained_and_redacted(kanban_home, monkeypatch):
    class Proc:
        pid = 42420
        returncode = 7

        def poll(self):
            return self.returncode

    monkeypatch.setattr(kb, "_pid_alive", lambda _pid: False)
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="runtime", assignee="worker")
        claimed = kb.claim_task(conn, task_id)
        assert claimed is not None
        kb._register_worker_process(
            Proc(), claimed, board="default", log_path="/tmp/worker.log",
            route={"profile": "worker", "provider": "safe", "model": "safe-model"},
        )
        kb._set_worker_pid(conn, task_id, Proc.pid)

        assert kb.reap_worker_zombies() == [Proc.pid]
        assert kb.detect_crashed_workers(conn) == [task_id]

        run = kb.list_runs(conn, task_id)[0]
        assert run.metadata["exit"]["kind"] == "nonzero_exit"
        assert run.metadata["exit"]["code"] == 7
        assert "api_key" not in str(run.metadata).lower()


def test_unknown_dead_pid_is_blocked_not_blindly_retried(kanban_home, monkeypatch):
    monkeypatch.setattr(kb, "_pid_alive", lambda _pid: False)
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="unknown", assignee="worker")
        claimed = kb.claim_task(conn, task_id)
        assert claimed is not None
        kb._set_worker_pid(conn, task_id, 98765)

        assert kb.detect_crashed_workers(conn) == [task_id]
        task = kb.get_task(conn, task_id)
        assert task.status == "blocked"
        assert "unknown" in (task.last_failure_error or "")
        run = kb.list_runs(conn, task_id)[0]
        assert run.outcome == "unknown_exit"
        assert run.metadata["exit"]["kind"] == "unknown"


def test_preflight_rejects_missing_workspace_before_spawn(kanban_home):
    task = kb.Task(
        id="t_preflight", title="preflight", body=None, assignee="missing",
        status="running", priority=0, created_by=None, created_at=0,
        started_at=None, completed_at=None, workspace_kind="dir",
        workspace_path=None, claim_lock="host:1", claim_expires=None, tenant=None,
    )
    result = kb._preflight_worker_spawn(task, "/does/not/exist", board="default")
    assert result["ok"] is False
    assert result["code"] == "workspace_invalid"
