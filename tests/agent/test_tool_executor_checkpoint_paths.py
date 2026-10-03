"""Behavioral coverage for checkpoint path resolution across terminal backends."""

import json
import os
import threading
from types import SimpleNamespace

import pytest

from agent.tool_executor import (
    _ToolCallRef,
    _begin_tool_execution,
    _ensure_file_checkpoint,
    _ensure_terminal_checkpoint,
)
from agent.turn_explainers import TurnExplainersMixin
from tools.checkpoint_manager import CheckpointManager
from tools.terminal_tool import _active_environments, _env_lock


@pytest.fixture
def manager(tmp_path, monkeypatch):
    monkeypatch.setattr("tools.checkpoint_manager.CHECKPOINT_BASE", tmp_path / "checkpoints")
    return CheckpointManager(enabled=True)


@pytest.fixture
def container_task_id(monkeypatch):
    class FakeDockerEnvironment:  # class-name hint is how file_tools_paths classifies live envs
        pass

    task_id = "container-checkpoint-task"
    with _env_lock:
        monkeypatch.setitem(_active_environments, task_id, FakeDockerEnvironment())
    yield task_id
    with _env_lock:
        _active_environments.pop(task_id, None)


def test_relative_file_checkpoint_uses_task_workspace(tmp_path, monkeypatch):
    """Checkpoint lookup must use the same cwd as a relative file mutation."""
    process_cwd = tmp_path / "opt" / "hermes"
    workspace_cwd = tmp_path / "opt" / "data" / "workspace"
    process_cwd.mkdir(parents=True)
    workspace_cwd.mkdir(parents=True)

    # Both directories contain content so checkpointing the wrong one would
    # still succeed and remain observable as the regression did in Docker.
    (process_cwd / "pyproject.toml").write_text("[project]\nname = 'hermes'\n")
    (workspace_cwd / "pyproject.toml").write_text("[project]\nname = 'workspace'\n")
    (workspace_cwd / "existing.txt").write_text("before\n")

    monkeypatch.chdir(process_cwd)
    monkeypatch.setenv("TERMINAL_CWD", str(workspace_cwd))
    monkeypatch.setattr(
        "tools.checkpoint_manager.CHECKPOINT_BASE",
        tmp_path / "checkpoints",
    )

    manager = CheckpointManager(enabled=True)
    agent = SimpleNamespace(_checkpoint_mgr=manager)

    _ensure_file_checkpoint(
        agent,
        "write_file",
        {"path": "test_permissions2.txt"},
        "gateway-session",
    )

    assert manager.list_checkpoints(str(workspace_cwd))
    assert manager.list_checkpoints(str(process_cwd)) == []


def test_container_backend_task_leaves_host_store_untouched(tmp_path, manager, container_task_id):
    """A container-backed task's paths are container paths: no host snapshot from the file or
    destructive-terminal hooks and no host ledger entry, even when an unrelated host tree shares
    the spelling. A local task on the same manager still gets its checkpoint (control)."""
    host_file = tmp_path / "workspace" / "project" / "a.txt"
    host_file.parent.mkdir(parents=True)
    host_file.write_text("host content\n", encoding="utf-8")
    path = str(host_file)
    agent = SimpleNamespace(
        _checkpoint_mgr=manager, _turn_failed_file_mutations={}, _turn_file_mutation_paths=set(),
        quiet_mode=True, tool_progress_callback=None, tool_start_callback=None, _touch_activity=lambda *_: None,
    )

    _ensure_file_checkpoint(agent, "write_file", {"path": path}, container_task_id)
    ref = _ToolCallRef("terminal", {"command": f"rm -rf {host_file.parent}"}, container_task_id, "call-1", [])
    _begin_tool_execution(agent, ref, None)
    TurnExplainersMixin._record_file_mutation_result(
        agent, "write_file", {"path": path, "content": "x"},
        json.dumps({"bytes_written": 1, "resolved_path": path}), False, task_id=container_task_id,
    )
    assert manager.list_checkpoints(str(host_file.parent)) == []
    assert not (tmp_path / "checkpoints").exists()

    _ensure_file_checkpoint(agent, "write_file", {"path": path}, "local-task")
    assert len(manager.list_checkpoints(str(host_file.parent))) == 1


@pytest.mark.asyncio
async def test_container_session_refuses_host_rollback_on_every_surface(tmp_path, monkeypatch, manager, capsys):
    """A host checkpoint left by an earlier local session is never restored or diffed from a
    container-backed session: CLI /rollback + /diff session, gateway /rollback + /diff session,
    and the rollback.restore RPC all answer with the backend reason instead."""
    from gateway import run as gateway_run
    from gateway.config import Platform
    from gateway.platforms.event import MessageEvent
    from gateway.session import SessionSource
    from hermes_cli.cli_commands_mixin import CLICommandsMixin
    from tui_gateway import server

    host_dir = tmp_path / "workspace" / "project"
    host_dir.mkdir(parents=True)
    (host_dir / "a.txt").write_text("before\n", encoding="utf-8")
    manager.ensure_checkpoint(str(host_dir), "earlier local session")
    monkeypatch.setenv("TERMINAL_ENV", "docker")  # the configured backend, as the product bridges it
    monkeypatch.setenv("TERMINAL_CWD", str(host_dir))

    def refuse(*_args, **_kwargs):
        raise AssertionError("host checkpoint operation reached from a container-backed session")

    monkeypatch.setattr(manager, "restore", refuse)
    monkeypatch.setattr(manager, "diff", refuse)
    monkeypatch.setattr(manager, "session_diff", refuse)

    cli = SimpleNamespace(
        _checkpoint_manager=lambda _lines: manager,
        _resolve_checkpoint_ref=lambda ref, cps: cps[int(ref) - 1]["hash"],
        _rollback_restore=refuse, _rollback_diff=refuse,
    )
    for command in ("/rollback 1 --all", "/rollback diff 1"):
        CLICommandsMixin._handle_rollback_command(cli, command)
        assert "docker" in "".join(capsys.readouterr())
    CLICommandsMixin._print_session_diff(cli, str(host_dir), False)
    assert "docker" in "".join(capsys.readouterr())

    runner = object.__new__(gateway_run.GatewayRunner)
    runner._checkpoint_manager = lambda: manager
    source = SessionSource(platform=Platform.TELEGRAM, user_id="u", chat_id="c", user_name="t", chat_type="dm")
    for text in ("/rollback 1", "/rollback 1 --all"):
        assert "docker" in await runner._handle_rollback_command(MessageEvent(text=text, source=source))
    assert "docker" in await runner._handle_diff_command(MessageEvent(text="/diff session", source=source))
    listing = await runner._handle_rollback_command(MessageEvent(text="/rollback", source=source))
    assert "docker" in listing and "earlier local session" in listing  # listing stays visible

    session = {
        "agent": SimpleNamespace(_checkpoint_mgr=manager), "cwd": str(host_dir), "running": False,
        "session_key": "container-key", "history": [], "history_lock": threading.Lock(), "history_version": 0,
    }
    monkeypatch.setitem(server._sessions, "container-sid", session)
    resp = server.handle_request(
        {"id": "1", "method": "rollback.restore", "params": {"session_id": "container-sid", "hash": "1"}}
    )
    assert resp["result"]["success"] is False and "docker" in resp["result"]["error"]

    monkeypatch.setenv("TERMINAL_ENV", "local")  # control: a local session still dispatches
    monkeypatch.setattr(manager, "restore", lambda *a, **k: {"success": True, "restored_to": "x", "reason": "r"})
    resp = server.handle_request(
        {"id": "2", "method": "rollback.restore", "params": {"session_id": "container-sid", "hash": "1"}}
    )
    assert resp["result"]["success"] is True
    assert (host_dir / "a.txt").read_text(encoding="utf-8") == "before\n"
    assert os.environ["TERMINAL_ENV"] == "local"
def _make_terminal_agent(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "tools.checkpoint_manager.CHECKPOINT_BASE",
        tmp_path / "checkpoints",
    )
    manager = CheckpointManager(enabled=True)
    return SimpleNamespace(_checkpoint_mgr=manager), manager


def test_terminal_checkpoint_fires_for_bypassing_command(tmp_path, monkeypatch):
    """A command the destructive-string classifier can't recognize must still
    establish a checkpoint for its working dir (#69171)."""
    work_dir = tmp_path / "project"
    work_dir.mkdir()
    (work_dir / "victim.txt").write_text("keep me\n")

    agent, manager = _make_terminal_agent(tmp_path, monkeypatch)

    # Absolute-path rm — bypasses the _is_destructive_command regex.
    _ensure_terminal_checkpoint(
        agent, {"command": "/bin/rm victim.txt", "workdir": str(work_dir)}
    )

    assert manager.list_checkpoints(str(work_dir))


def test_terminal_checkpoint_fires_for_innocuous_command(tmp_path, monkeypatch):
    """Correctness no longer depends on the command string: even a read-only
    command checkpoints the working dir once per turn (#69171)."""
    work_dir = tmp_path / "project"
    work_dir.mkdir()
    (work_dir / "a.txt").write_text("x\n")

    agent, manager = _make_terminal_agent(tmp_path, monkeypatch)

    _ensure_terminal_checkpoint(agent, {"command": "ls -la", "workdir": str(work_dir)})

    assert manager.list_checkpoints(str(work_dir))


def test_terminal_checkpoint_dedups_within_turn(tmp_path, monkeypatch):
    """ensure_checkpoint is per-turn idempotent, so repeated terminal calls in
    the same turn don't stack extra snapshots for the same dir (#69171)."""
    work_dir = tmp_path / "project"
    work_dir.mkdir()
    (work_dir / "a.txt").write_text("x\n")

    agent, manager = _make_terminal_agent(tmp_path, monkeypatch)

    _ensure_terminal_checkpoint(agent, {"command": "ls", "workdir": str(work_dir)})
    before = len(manager.list_checkpoints(str(work_dir)))
    _ensure_terminal_checkpoint(agent, {"command": "cat a.txt", "workdir": str(work_dir)})
    after = len(manager.list_checkpoints(str(work_dir)))

    assert before == after == 1

    # A fresh turn resets the dedup set and allows a new snapshot.
    manager.new_turn()
    (work_dir / "a.txt").write_text("y\n")
    _ensure_terminal_checkpoint(agent, {"command": "echo hi", "workdir": str(work_dir)})
    assert len(manager.list_checkpoints(str(work_dir))) == 2
