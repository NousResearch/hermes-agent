"""Regression coverage for the dispatcher-owned Kanban worker runtime contract."""
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
    kb._worker_registry.clear()
    kb._worker_exit_context.clear()
    kb._recent_worker_exits.clear()
    yield home
    kb._worker_registry.clear()
    kb._worker_exit_context.clear()
    kb._recent_worker_exits.clear()


def _task(task_id="t_runtime"):
    return kb.Task(
        id=task_id, title="runtime", body=None, assignee="worker", status="running",
        priority=0, created_by=None, created_at=0, started_at=None, completed_at=None,
        workspace_kind="dir", workspace_path=None, claim_lock="host:1",
        claim_expires=None, tenant=None,
    )


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
    result = kb._preflight_worker_spawn(_task(), "/does/not/exist", board="default")
    assert result["ok"] is False
    assert result["code"] == "workspace_invalid"


def test_reclaim_never_signals_pid_without_live_gateway_ownership(kanban_home, monkeypatch):
    """A host-local lock and PID are not proof that this gateway owns it."""
    calls = []
    monkeypatch.setattr(kb, "_claimer_id", lambda: "host:dispatcher")

    result = kb._terminate_reclaimed_worker(
        42420, "host:old-dispatcher", signal_fn=lambda *args: calls.append(args),
    )

    assert calls == []
    assert result["termination_attempted"] is False
    assert result["ownership_verified"] is False
    assert result["ownership_reason"] == "missing_registry_record"


def test_timeout_never_signals_persisted_pid_without_live_gateway_ownership(
    kanban_home, monkeypatch,
):
    """Timeout cleanup cannot kill PID 424242 after a dispatcher restart."""
    calls = []
    monkeypatch.setattr(kb, "_claimer_id", lambda: "host:dispatcher")
    monkeypatch.setattr(kb.time, "time", lambda: 10_000)
    monkeypatch.setattr(kb.os, "killpg", lambda *args: calls.append(args))
    with kb.connect() as conn:
        task_id = kb.create_task(
            conn, title="expired", assignee="worker", max_runtime_seconds=1,
        )
        claimed = kb.claim_task(conn, task_id, claimer="host:old-dispatcher")
        assert claimed is not None
        kb._set_worker_pid(conn, task_id, 424242)
        conn.execute(
            "UPDATE task_runs SET started_at = ? WHERE id = ?",
            (1, claimed.current_run_id),
        )

        assert kb.enforce_max_runtime(conn) == []

        assert calls == []
        assert kb.get_task(conn, task_id).status == "running"


def test_reclaim_never_signals_on_process_group_mismatch(kanban_home, monkeypatch):
    class Proc:
        pid = 42420

        def poll(self):
            return None

    monkeypatch.setattr(kb, "_claimer_id", lambda: "host:dispatcher")
    monkeypatch.setattr(kb.os, "getpgid", lambda _pid: 999)
    kb._register_worker_process(Proc(), _task("t_owned"), board="default", log_path="/tmp/worker.log")
    kb._worker_registry[42420].process_group = 111
    calls = []

    result = kb._terminate_reclaimed_worker(
        42420, "host:old-dispatcher", signal_fn=lambda *args: calls.append(args),
    )

    assert calls == []
    assert result["termination_attempted"] is False
    assert result["ownership_verified"] is False
    assert result["ownership_reason"] == "process_group_mismatch"


def test_execution_route_is_persisted_without_runtime_credentials(kanban_home):
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="route", assignee="worker")
        task = kb.claim_task(conn, task_id)
        assert task is not None
        task.execution_route = {
            "profile": "worker", "provider": "openai", "model": "gpt-test",
            "credential_ref": "credential_pool", "api_key": "must-not-persist",
        }
        kb._persist_execution_route(conn, task)

        assert kb._latest_execution_route(conn, task_id) == {
            "profile": "worker", "provider": "openai", "model": "gpt-test",
            "credential_ref": "credential_pool",
        }
        assert "must-not-persist" not in str(kb.list_events(conn, task_id))


def test_exit_context_is_bounded_and_releases_exited_popen_references(kanban_home, monkeypatch):
    class Proc:
        def __init__(self, pid):
            self.pid = pid

        def poll(self):
            return 1

    monkeypatch.setattr(kb, "_RECENT_WORKER_EXITS_MAX", 8)
    task = _task("t_bounded")
    for pid in range(1000, 1032):
        kb._register_worker_process(Proc(pid), task, board="default", log_path="/tmp/worker.log")
    kb._poll_registered_workers()

    assert len(kb._recent_worker_exits) <= 8
    assert len(kb._worker_exit_context) <= 8
