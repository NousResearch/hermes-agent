"""Regression tests for merging worker-tree ownership with exit envelopes."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

from hermes_cli import kanban_db as kb
from hermes_cli.kanban_worker_outcomes import (
    FailureClass,
    WorkerExitEnvelope,
    classify_failure,
    worker_exit_envelope_path,
    write_exit_envelope,
)


def _running_task(conn):
    task_id = kb.create_task(conn, title="worker lifecycle", assignee="default")
    claimed = kb.claim_task(conn, task_id)
    assert claimed is not None
    assert claimed.current_run_id is not None
    return task_id, int(claimed.current_run_id)


def _mark_dead_worker(conn, task_id: str, envelope_path: Path, pid: int) -> None:
    conn.execute(
        "UPDATE tasks SET worker_pid=?, worker_start_time=?, "
        "worker_owner_kind=?, worker_owner_id=?, worker_exit_envelope=?, "
        "started_at=? WHERE id=?",
        (
            pid,
            123,
            "process_group",
            str(pid),
            str(envelope_path),
            int(time.time()) - 30,
            task_id,
        ),
    )
    conn.commit()


def _owned_tree_absent(monkeypatch):
    monkeypatch.setattr(kb, "_pid_alive", lambda _pid: False)
    monkeypatch.setattr(
        kb,
        "_classify_worker_exit",
        lambda _pid: ("nonzero_exit", 7),
    )
    monkeypatch.setattr(
        kb,
        "_terminate_reclaimed_worker",
        lambda *_args, **_kwargs: {
            "tree_cleanup": True,
            "already_absent": True,
            "survived_cleanup": False,
            "identity_verified": True,
            "pid_reused": False,
            "graceful_cleanup": False,
            "forced_cleanup": False,
            "forced_action": None,
        },
    )


def test_zero_exit_precedes_normal_transcript_failure_markers():
    """Normal output words must not turn a successful wrapper into failure."""
    result = classify_failure(
        error="workspace canary write_file review diff completed",
        exit_code=0,
    )
    assert result.failure_class is FailureClass.SUCCESS
    assert result.retry_scope == "none"


def test_detect_crash_releases_provider_transport_without_task_budget(tmp_path, monkeypatch):
    logs = tmp_path / "logs"
    monkeypatch.setattr(kb, "worker_logs_dir", lambda board=None: logs)
    monkeypatch.setattr(kb, "read_worker_log", lambda *args, **kwargs: "")
    conn = kb.connect(tmp_path / "kanban.db")
    try:
        task_id, run_id = _running_task(conn)
        envelope_path = worker_exit_envelope_path(logs, task_id, run_id)
        write_exit_envelope(
            envelope_path,
            WorkerExitEnvelope(
                task_id=task_id,
                run_id=run_id,
                pid=701,
                exit_code=7,
                provider="openai",
                model="worker-model",
                failure_class=FailureClass.PROVIDER_TRANSPORT,
                redacted_error="provider connection reset",
            ),
        )
        _mark_dead_worker(conn, task_id, envelope_path, 701)
        _owned_tree_absent(monkeypatch)

        assert kb.detect_crashed_workers(conn) == []
        task = conn.execute(
            "SELECT status, worker_pid, consecutive_failures FROM tasks WHERE id=?",
            (task_id,),
        ).fetchone()
        assert task["status"] == "ready"
        assert task["worker_pid"] is None
        assert task["consecutive_failures"] == 0
        run = conn.execute(
            "SELECT outcome, error FROM task_runs WHERE id=?", (run_id,)
        ).fetchone()
        assert run["outcome"] == FailureClass.PROVIDER_TRANSPORT.value
        assert run["error"] == "provider connection reset"
        event = conn.execute(
            "SELECT kind, payload FROM task_events WHERE task_id=? "
            "ORDER BY id DESC LIMIT 1",
            (task_id,),
        ).fetchone()
        assert event["kind"] == FailureClass.PROVIDER_TRANSPORT.value
        assert json.loads(event["payload"])["failure_class"] == "provider_transport"
    finally:
        conn.close()


def test_detect_crash_rejects_stale_envelope_and_is_idempotent(tmp_path, monkeypatch):
    logs = tmp_path / "logs"
    monkeypatch.setattr(kb, "worker_logs_dir", lambda board=None: logs)
    monkeypatch.setattr(kb, "read_worker_log", lambda *args, **kwargs: "")
    conn = kb.connect(tmp_path / "kanban.db")
    try:
        task_id, run_id = _running_task(conn)
        envelope_path = worker_exit_envelope_path(logs, task_id, run_id)
        write_exit_envelope(
            envelope_path,
            WorkerExitEnvelope(
                task_id=task_id,
                run_id=run_id - 1,
                pid=999,
                exit_code=7,
                failure_class=FailureClass.PROVIDER_AUTH,
                redacted_error="stale provider auth receipt",
            ),
        )
        _mark_dead_worker(conn, task_id, envelope_path, 702)
        _owned_tree_absent(monkeypatch)

        assert kb.detect_crashed_workers(conn) == [task_id]
        task = conn.execute(
            "SELECT status, consecutive_failures, last_failure_error "
            "FROM tasks WHERE id=?",
            (task_id,),
        ).fetchone()
        assert task["status"] == "ready"
        assert task["consecutive_failures"] == 1
        assert "no terminal exit envelope" in task["last_failure_error"]
        run = conn.execute(
            "SELECT outcome FROM task_runs WHERE id=?", (run_id,)
        ).fetchone()
        assert run["outcome"] == "worker_exit_unknown"
        event = conn.execute(
            "SELECT kind, payload FROM task_events WHERE task_id=? "
            "ORDER BY id DESC LIMIT 1",
            (task_id,),
        ).fetchone()
        assert event["kind"] == "worker_exit_unknown"
        assert json.loads(event["payload"])["failure_class"] == "worker_exit_unknown"

        assert kb.detect_crashed_workers(conn) == []
        assert conn.execute(
            "SELECT COUNT(*) FROM task_events WHERE task_id=? "
            "AND kind='worker_exit_unknown'",
            (task_id,),
        ).fetchone()[0] == 1
    finally:
        conn.close()


def test_default_spawn_wraps_command_and_passes_run_scoped_receipt(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    logs = tmp_path / "logs"
    captured = {}

    def fake_spawn(argv, **kwargs):
        captured["argv"] = list(argv)
        captured["kwargs"] = kwargs
        return SimpleNamespace(pid=703)

    monkeypatch.setattr(kb, "spawn_worker_process", fake_spawn)
    monkeypatch.setattr(kb, "_resolve_hermes_argv", lambda: ["hermes"])
    monkeypatch.setattr(kb, "_resolve_worker_profile_route", lambda _: ("profile-model", "profile-provider"))
    monkeypatch.setattr(kb, "_resolve_worker_cli_toolsets", lambda _: None)
    monkeypatch.setattr(kb, "worker_logs_dir", lambda board=None: logs)
    monkeypatch.setattr(kb, "kanban_db_path", lambda board=None: tmp_path / "kanban.db")
    monkeypatch.setattr(kb, "workspaces_root", lambda board=None: tmp_path / "workspaces")
    monkeypatch.setattr(kb, "_retag_legacy_worker_sessions", lambda _root: None)
    monkeypatch.setattr(kb, "_rotate_worker_log", lambda *args: None)
    monkeypatch.setattr(kb, "worker_log_rotation_config", lambda: (1024, 1))

    task = SimpleNamespace(
        id="t_spawn_envelope",
        title="spawn envelope",
        body="",
        assignee="default",
        tenant=None,
        session_id="session-1",
        model_override=None,
        provider_override=None,
        reasoning_effort=None,
        skills=[],
        goal_mode=False,
        goal_max_turns=None,
        review_auto=False,
        branch_name=None,
        current_run_id=19,
        claim_lock="host:worker",
        max_runtime_seconds=None,
    )

    pid = kb._default_spawn(
        task,
        str(workspace),
        board="test-board",
        _write_exit_envelope=True,
        _detach_worker=True,
    )
    assert pid == 703
    assert captured["argv"][:4] == [
        sys.executable,
        "-m",
        "hermes_cli.kanban_worker_outcomes",
        "--",
    ]
    assert captured["argv"][4:] == ["hermes", "-p", "default", "--cli", "--accept-hooks", "chat", "-q", "work kanban task t_spawn_envelope"]
    assert captured["kwargs"]["envelope_path"] == worker_exit_envelope_path(logs, task.id, 19)
    assert captured["kwargs"]["kill_on_parent_exit"] is False
    assert captured["kwargs"]["env"]["HERMES_KANBAN_EXIT_ENVELOPE"] == str(captured["kwargs"]["envelope_path"])
    assert captured["kwargs"]["env"]["HERMES_KANBAN_RUN_ID"] == "19"
    assert captured["kwargs"]["env"]["HERMES_KANBAN_PROVIDER"] == "profile-provider"
    assert captured["kwargs"]["env"]["HERMES_KANBAN_MODEL"] == "profile-model"
    captured["kwargs"]["stdout"].close()
