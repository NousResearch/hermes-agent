"""Behavioral coverage for checkpoint path resolution across terminal backends."""

import json
import os
from types import SimpleNamespace

import pytest

from agent.tool_executor import _ToolCallRef, _begin_tool_execution, _ensure_file_checkpoint
from agent.turn_explainers import TurnExplainersMixin
from tools.checkpoint_manager import CheckpointManager
from tools.terminal_tool import _active_environments, _env_lock


@pytest.fixture
def manager(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "tools.checkpoint_manager.CHECKPOINT_BASE", tmp_path / "checkpoints"
    )
    return CheckpointManager(enabled=True)


@pytest.fixture
def container_task_id(monkeypatch):
    class FakeDockerEnvironment:
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


def _assert_container_skipped(manager):
    assert manager.list_all_checkpoints() == []
    assert manager.unsupported_backend == "docker"
    assert "docker" in manager.unsupported_backend_reason()


def test_container_backend_file_checkpoint_is_not_taken_on_missing_host_path(
    manager, container_task_id,
):
    agent = SimpleNamespace(_checkpoint_mgr=manager)
    _ensure_file_checkpoint(
        agent, "write_file", {"path": "/workspace/project/a.txt"}, container_task_id
    )
    _assert_container_skipped(manager)


@pytest.mark.skipif(os.name == "nt", reason="POSIX absolute container path")
def test_container_backend_file_checkpoint_does_not_snapshot_a_colliding_host_tree(
    tmp_path, manager, container_task_id,
):
    host_dir = tmp_path / "workspace" / "project"
    host_dir.mkdir(parents=True)
    (host_dir / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    (host_dir / "a.txt").write_text("host content\n", encoding="utf-8")
    _ensure_file_checkpoint(
        SimpleNamespace(_checkpoint_mgr=manager),
        "write_file",
        {"path": str(host_dir / "a.txt")},
        container_task_id,
    )
    assert manager.list_checkpoints(str(host_dir)) == []
    _assert_container_skipped(manager)


def test_container_backend_destructive_terminal_checkpoint_is_not_taken(
    tmp_path, monkeypatch, manager, container_task_id,
):
    # Pin the host cwd the branch would snapshot on main to a small tree, not the test process cwd.
    host_cwd = tmp_path / "host-cwd"
    host_cwd.mkdir()
    (host_cwd / "keep.txt").write_text("host", encoding="utf-8")
    monkeypatch.setenv("TERMINAL_CWD", str(host_cwd))
    agent = SimpleNamespace(
        quiet_mode=True,
        tool_progress_callback=None,
        tool_start_callback=None,
        _checkpoint_mgr=manager,
        _touch_activity=lambda *_: None,
    )
    ref = _ToolCallRef(
        "terminal",
        {"command": "rm -f /workspace/project/a.txt"},
        container_task_id,
        "call-1",
        [],
    )
    _begin_tool_execution(agent, ref, None)
    _assert_container_skipped(manager)


def test_container_backend_post_write_ledger_is_not_recorded(
    tmp_path, manager, container_task_id,
):
    # A real host file at the container path's spelling: on main the ledger loop hashes it
    # on the host and persists an entry, which is exactly the false attribution to prevent.
    host_file = tmp_path / "workspace" / "project" / "a.txt"
    host_file.parent.mkdir(parents=True)
    host_file.write_text("after\n", encoding="utf-8")
    path = str(host_file)
    agent = SimpleNamespace(
        _turn_failed_file_mutations={},
        _turn_file_mutation_paths=set(),
        _checkpoint_mgr=manager,
    )
    TurnExplainersMixin._record_file_mutation_result(
        agent,
        "write_file",
        {"path": path, "content": "after\n"},
        json.dumps({"bytes_written": 6, "resolved_path": path}),
        False,
        task_id=container_task_id,
    )
    assert not (tmp_path / "checkpoints" / "store" / "ledgers").exists()
    _assert_container_skipped(manager)


def test_local_backend_behaviour_unchanged(tmp_path, monkeypatch, manager):
    workspace = tmp_path / "local-project"
    workspace.mkdir()
    (workspace / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    (workspace / "existing.txt").write_text("before\n", encoding="utf-8")
    monkeypatch.setenv("TERMINAL_ENV", "local")
    monkeypatch.setenv("TERMINAL_CWD", str(workspace))
    _ensure_file_checkpoint(
        SimpleNamespace(_checkpoint_mgr=manager),
        "write_file",
        {"path": "new.txt"},
        "local-checkpoint-task",
    )
    assert manager.list_checkpoints(str(workspace))


def test_container_session_rollback_restore_is_refused(tmp_path, monkeypatch, manager, container_task_id, capsys):
    """A host checkpoint that predates the container session must not be restored from it."""
    from hermes_cli.cli_commands_mixin import CLICommandsMixin

    host_dir = tmp_path / "workspace" / "project"
    host_dir.mkdir(parents=True)
    (host_dir / "a.txt").write_text("before\n", encoding="utf-8")
    monkeypatch.setenv("TERMINAL_ENV", "docker")  # the configured backend, as the product bridges it
    monkeypatch.setenv("TERMINAL_CWD", str(host_dir))
    manager.ensure_checkpoint(str(host_dir), "earlier local session")
    assert manager.list_checkpoints(str(host_dir))
    # Fresh session: no mutation has been observed, so only the backend classification can refuse.

    def refuse(*_args, **_kwargs):
        raise AssertionError("host restore reached from a container-backed session")

    cli = SimpleNamespace(
        _checkpoint_manager=lambda _lines: manager,
        _resolve_checkpoint_ref=lambda ref, cps: cps[int(ref) - 1]["hash"],
        _rollback_restore=refuse,
        _rollback_diff=refuse,
    )
    for command in ("/rollback 1 --all", "/rollback diff 1"):
        CLICommandsMixin._handle_rollback_command(cli, command)
        assert "docker" in capsys.readouterr().out
    assert len(manager.list_checkpoints(str(host_dir))) == 1  # no pre-rollback snapshot either


def test_container_session_rpc_restore_is_refused(tmp_path, monkeypatch, manager, container_task_id):
    import threading

    from tui_gateway import server

    host_dir = tmp_path / "workspace" / "project"
    host_dir.mkdir(parents=True)
    (host_dir / "a.txt").write_text("before\n", encoding="utf-8")
    monkeypatch.setenv("TERMINAL_ENV", "docker")
    manager.ensure_checkpoint(str(host_dir), "earlier local session")
    session = {
        "agent": SimpleNamespace(_checkpoint_mgr=manager), "cwd": str(host_dir), "running": False,
        "session_key": "container-key", "history": [], "history_lock": threading.Lock(), "history_version": 0,
    }
    monkeypatch.setitem(server._sessions, "container-sid", session)
    resp = server.handle_request(
        {"id": "1", "method": "rollback.restore", "params": {"session_id": "container-sid", "hash": "1"}}
    )
    assert resp["result"]["success"] is False
    assert "docker" in resp["result"]["error"]
    assert len(manager.list_checkpoints(str(host_dir))) == 1


def test_local_session_rollback_restore_still_dispatches(tmp_path, monkeypatch, manager, capsys):
    """The other direction: a local session with a host checkpoint restores as before."""
    import threading

    from hermes_cli.cli_commands_mixin import CLICommandsMixin
    from tui_gateway import server

    host_dir = tmp_path / "local-project"
    host_dir.mkdir()
    (host_dir / "a.txt").write_text("before", encoding="utf-8")
    monkeypatch.setenv("TERMINAL_ENV", "local")
    monkeypatch.setenv("TERMINAL_CWD", str(host_dir))
    manager.ensure_checkpoint(str(host_dir), "earlier local session")

    calls = []
    cli = SimpleNamespace(
        _checkpoint_manager=lambda _lines: manager,
        _resolve_checkpoint_ref=lambda ref, cps: cps[int(ref) - 1]["hash"],
        _rollback_restore=lambda *args: calls.append(args),
    )
    CLICommandsMixin._handle_rollback_command(cli, "/rollback 1 --all")
    assert len(calls) == 1 and "Checkpoints are not taken" not in capsys.readouterr().out

    session = {
        "agent": SimpleNamespace(_checkpoint_mgr=manager), "cwd": str(host_dir), "running": False,
        "session_key": "local-key", "history": [], "history_lock": threading.Lock(), "history_version": 0,
    }
    monkeypatch.setitem(server._sessions, "local-sid", session)
    resp = server.handle_request(
        {"id": "1", "method": "rollback.restore", "params": {"session_id": "local-sid", "hash": "1"}}
    )
    assert resp["result"]["success"] is True

    # A persistent docker container of the launch profile already occupies the "default" registry
    # slot; this local session must still be classified by its own key, not by that cached env.
    class FakeDockerEnvironment:
        pass

    with _env_lock:
        monkeypatch.setitem(_active_environments, "default", FakeDockerEnvironment())
    resp = server.handle_request(
        {"id": "2", "method": "rollback.restore", "params": {"session_id": "local-sid", "hash": "1"}}
    )
    assert resp["result"]["success"] is True
