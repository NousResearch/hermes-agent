"""Regression specs for active-task/workspace restart recovery."""

import json
import logging
import subprocess
import time
from unittest.mock import patch

from gateway.active_task import (
    ActiveTaskStore,
    build_active_task_recovery_note,
)
from gateway.run import GatewayRunner
from tools.process_registry import ProcessRegistry, ProcessSession


def test_active_task_store_roundtrips_workspace_snapshot(tmp_path):
    store_path = tmp_path / "session_active_tasks.json"
    repo_path = tmp_path / "project"
    repo_path.mkdir()

    store = ActiveTaskStore(store_path)
    store.upsert(
        session_key="agent:main:discord:thread:thread-parent:thread-1",
        session_id="sid",
        platform="discord",
        chat_id="thread-parent",
        thread_id="thread-1",
        repo_path=str(repo_path),
        branch="main",
        head="abc123",
        command="python tools/refill.py",
        task_summary="Signal Room refill batch",
        status="active",
        process_session_id="proc_active",
        pid=12345,
        latest_log_path=str(repo_path / "runtime/refill.log"),
        latest_summary_path=str(repo_path / "runtime/refill.json"),
        resume_reason="restart_timeout",
    )

    reloaded = ActiveTaskStore(store_path).get(
        "agent:main:discord:thread:thread-parent:thread-1"
    )

    assert reloaded is not None
    assert reloaded.session_id == "sid"
    assert reloaded.platform == "discord"
    assert reloaded.chat_id == "thread-parent"
    assert reloaded.thread_id == "thread-1"
    assert reloaded.repo_path == str(repo_path)
    assert reloaded.branch == "main"
    assert reloaded.command == "python tools/refill.py"
    assert reloaded.task_summary == "Signal Room refill batch"
    assert reloaded.status == "active"
    assert reloaded.process_session_id == "proc_active"
    assert reloaded.pid == 12345
    assert reloaded.latest_log_path.endswith("runtime/refill.log")
    assert reloaded.latest_summary_path.endswith("runtime/refill.json")
    assert reloaded.resume_reason == "restart_timeout"


def test_gateway_workspace_resolver_prefers_active_task_cwd(tmp_path, monkeypatch):
    active_repo = tmp_path / "active-project"
    systemd_cwd = tmp_path / "gateway-checkout"
    terminal_cwd = tmp_path / "terminal-cwd"
    active_repo.mkdir()
    systemd_cwd.mkdir()
    terminal_cwd.mkdir()

    runner = object.__new__(GatewayRunner)
    runner.active_task_store = ActiveTaskStore(tmp_path / "active_tasks.json")
    runner.active_task_store.upsert(
        session_key="agent:main:discord:thread:thread-parent:thread-1",
        repo_path=str(active_repo),
        branch="main",
        command="python tools/refill.py",
        status="active",
    )
    monkeypatch.setenv("TERMINAL_CWD", str(terminal_cwd))

    resolved = runner._resolve_agent_working_directory(
        "agent:main:discord:thread:thread-parent:thread-1",
        fallback_cwd=str(systemd_cwd),
    )

    assert resolved == str(active_repo)
    reloaded = runner.active_task_store.get(
        "agent:main:discord:thread:thread-parent:thread-1"
    )
    assert reloaded.command == "python tools/refill.py"
    assert reloaded.process_session_id is None


def test_gateway_workspace_resolver_falls_back_to_terminal_cwd(tmp_path, monkeypatch):
    systemd_cwd = tmp_path / "gateway-checkout"
    terminal_cwd = tmp_path / "terminal-cwd"
    systemd_cwd.mkdir()
    terminal_cwd.mkdir()

    runner = object.__new__(GatewayRunner)
    runner.active_task_store = ActiveTaskStore(tmp_path / "active_tasks.json")
    monkeypatch.setenv("TERMINAL_CWD", str(terminal_cwd))

    resolved = runner._resolve_agent_working_directory(
        "agent:main:discord:thread:thread-parent:thread-1",
        fallback_cwd=str(systemd_cwd),
    )

    assert resolved == str(terminal_cwd)


def test_gateway_workspace_resolver_records_foreground_session_snapshot(tmp_path, monkeypatch):
    repo_path = tmp_path / "project"
    repo_path.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo_path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo_path, check=True)
    subprocess.run(["git", "config", "user.name", "test"], cwd=repo_path, check=True)
    (repo_path / "README.md").write_text("# Project\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=repo_path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=repo_path, check=True)
    expected_head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    runner = object.__new__(GatewayRunner)
    runner.active_task_store = ActiveTaskStore(tmp_path / "active_tasks.json")
    monkeypatch.delenv("TERMINAL_CWD", raising=False)

    resolved = runner._resolve_agent_working_directory(
        "agent:main:discord:thread:thread-parent:thread-1",
        fallback_cwd=str(repo_path),
    )

    reloaded = runner.active_task_store.get(
        "agent:main:discord:thread:thread-parent:thread-1"
    )
    assert resolved == str(repo_path)
    assert reloaded is not None
    assert reloaded.repo_path == str(repo_path)
    assert reloaded.branch
    assert reloaded.head == expected_head
    assert reloaded.mode == "foreground_session"
    assert reloaded.command is None
    assert reloaded.pid is None
    assert reloaded.process_session_id is None
    assert "SENSITIVE_SENTINEL_DO_NOT_PERSIST" not in json.dumps(reloaded.to_dict())


def test_foreground_session_snapshot_replaces_stale_same_session_metadata(tmp_path):
    repo_path = tmp_path / "project"
    repo_path.mkdir()
    store_path = tmp_path / "active_tasks.json"
    session_key = "agent:main:discord:thread:thread-parent:thread-1"
    runner = object.__new__(GatewayRunner)
    runner.active_task_store = ActiveTaskStore(store_path)
    runner.active_task_store.upsert(
        session_key=session_key,
        session_id="old-session-id",
        platform="discord",
        chat_id="thread-parent",
        thread_id="thread-1",
        repo_path=str(tmp_path / "old-project"),
        branch="old-branch",
        head="old-head",
        command="python stale_task.py",
        task_summary="old user task text",
        status="active",
        process_session_id="proc_old",
        pid=9876,
        latest_log_path=str(tmp_path / "old.log"),
        latest_summary_path=str(tmp_path / "old.json"),
        resume_reason="shutdown_timeout",
    )

    runner._record_foreground_session_workspace(session_key, str(repo_path))

    raw = json.loads(store_path.read_text(encoding="utf-8"))[session_key]
    forbidden_fields = {
        "session_id",
        "platform",
        "chat_id",
        "thread_id",
        "command",
        "task_summary",
        "pid",
        "process_session_id",
        "latest_log_path",
        "latest_summary_path",
        "resume_reason",
    }
    assert set(raw) == {
        "session_key",
        "repo_path",
        "branch",
        "head",
        "mode",
        "status",
        "updated_at",
    }
    assert forbidden_fields.isdisjoint(raw)
    assert raw["session_key"] == session_key
    assert raw["repo_path"] == str(repo_path)
    assert raw["mode"] == "foreground_session"
    assert raw["status"] == "active"
    assert "old user task text" not in json.dumps(raw)


def test_gateway_workspace_resolver_does_not_record_foreground_without_session_key(tmp_path, monkeypatch):
    repo_path = tmp_path / "project"
    repo_path.mkdir()
    store_path = tmp_path / "active_tasks.json"
    runner = object.__new__(GatewayRunner)
    runner.active_task_store = ActiveTaskStore(store_path)
    monkeypatch.delenv("TERMINAL_CWD", raising=False)

    resolved = runner._resolve_agent_working_directory("", fallback_cwd=str(repo_path))

    assert resolved == str(repo_path)
    assert not store_path.exists()


def test_resume_recovery_logs_when_active_task_store_file_is_absent(tmp_path, caplog):
    runner = object.__new__(GatewayRunner)
    runner.active_task_store = ActiveTaskStore(tmp_path / "missing_active_tasks.json")

    with caplog.at_level(logging.INFO, logger="gateway.run"):
        note = runner._build_resume_recovery_note(
            "agent:main:discord:thread:thread-parent:thread-1",
            "restart_timeout",
        )

    assert "Active workspace/process state: unknown" in note
    assert "active-task store file is absent" in caplog.text


def test_resume_recovery_logs_when_active_task_store_file_is_malformed(tmp_path, caplog):
    store_path = tmp_path / "active_tasks.json"
    store_path.write_text("{not json", encoding="utf-8")
    runner = object.__new__(GatewayRunner)
    runner.active_task_store = ActiveTaskStore(store_path)

    with caplog.at_level(logging.WARNING, logger="gateway.active_task"):
        note = runner._build_resume_recovery_note(
            "agent:main:discord:thread:thread-parent:thread-1",
            "restart_timeout",
        )

    assert "Active workspace/process state: unknown" in note
    assert "failed to parse active-task store" in caplog.text


def test_resume_pending_note_includes_active_task_facts(tmp_path):
    repo_path = tmp_path / "project"
    repo_path.mkdir()
    record = ActiveTaskStore(tmp_path / "active_tasks.json").upsert(
        session_key="agent:main:discord:thread:thread-parent:thread-1",
        repo_path=str(repo_path),
        branch="main",
        head="abc123",
        command="python tools/refill.py",
        task_summary="Signal Room refill batch",
        status="detached",
        process_session_id="proc_active",
        latest_summary_path=str(repo_path / "runtime/refill.json"),
        resume_reason="restart_timeout",
    )

    note = build_active_task_recovery_note(record, "restart_timeout")

    assert "Previous active task: Signal Room refill batch" in note
    assert f"Previous repo path: {repo_path}" in note
    assert "Previous branch: main" in note
    assert "Previous HEAD: abc123" in note
    assert "Previous command: python tools/refill.py" in note
    assert "Process status: detached" in note
    assert "Process/session id: proc_active" in note
    assert "Latest summary path:" in note
    assert "Next safe recovery check:" in note


def test_resume_pending_note_without_active_task_reports_unknown():
    note = build_active_task_recovery_note(None, "restart_timeout")

    assert "Active workspace/process state: unknown" in note
    assert "Do not silently default to the gateway working directory" in note
    assert "Next safe recovery check:" in note


def test_process_checkpoint_flushes_after_notification_metadata_changes(tmp_path):
    registry = ProcessRegistry()
    session = ProcessSession(
        id="proc_active",
        command="python tools/refill.py",
        task_id="agent:main:discord:thread:thread-parent:thread-1",
        session_key="agent:main:discord:thread:thread-parent:thread-1",
        pid=12345,
        pid_scope="host",
        cwd="/tmp/project",
        started_at=time.time(),
    )
    registry._running[session.id] = session
    checkpoint = tmp_path / "processes.json"

    with patch("tools.process_registry.CHECKPOINT_PATH", checkpoint):
        registry._write_checkpoint()
        registry.update_session_metadata(
            session.id,
            notify_on_complete=True,
            watcher_platform="discord",
            watcher_chat_id="thread-parent",
            watcher_thread_id="thread-1",
            watcher_interval=5,
        )

        data = json.loads(checkpoint.read_text())

    assert data[0]["notify_on_complete"] is True
    assert data[0]["watcher_platform"] == "discord"
    assert data[0]["watcher_interval"] == 5
