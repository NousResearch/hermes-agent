"""Behavioral coverage for file-tool checkpoint path resolution."""

from types import SimpleNamespace

import json

from agent.tool_executor import _ensure_file_checkpoint
from agent.turn_explainers import TurnExplainersMixin
from tools.checkpoint_manager import CheckpointManager


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


def test_container_file_mutations_never_touch_host_checkpoint_state(tmp_path, monkeypatch):
    class _Manager:
        enabled = True

        def ensure_checkpoint(self, *_args, **_kwargs):
            raise AssertionError("container path reached host checkpoint store")

        def get_working_dir_for_path(self, *_args, **_kwargs):
            raise AssertionError("container path reached host checkpoint root lookup")

        def record_agent_write(self, *_args, **_kwargs):
            raise AssertionError("container path reached host write ledger")

    monkeypatch.setenv("TERMINAL_ENV", "docker")
    agent = SimpleNamespace(
        _checkpoint_mgr=_Manager(),
        _turn_failed_file_mutations={},
        _turn_file_mutation_paths=set(),
    )

    _ensure_file_checkpoint(agent, "write_file", {"path": "/workspace/app.py"}, "container-session")
    TurnExplainersMixin._record_file_mutation_result(
        agent,
        "write_file",
        {"path": "/workspace/app.py", "content": "updated"},
        json.dumps({"bytes_written": 7}),
        False,
        task_id="container-session",
    )

    assert agent._turn_file_mutation_paths == {"/workspace/app.py"}
