"""Behavioral coverage for file-tool checkpoint path resolution."""

from types import SimpleNamespace

from agent.tool_executor import _begin_tool_execution, _ensure_file_checkpoint
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


def test_begin_tool_execution_records_kanban_progress(monkeypatch):
    recorded = []
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_worker")
    monkeypatch.setattr(
        "tools.kanban_tools.record_current_worker_tool_progress",
        lambda name, args: recorded.append((name, args)),
    )
    agent = SimpleNamespace(
        quiet_mode=True,
        tool_progress_mode="off",
        tool_progress_callback=None,
        tool_start_callback=None,
        _checkpoint_mgr=SimpleNamespace(enabled=False),
        _current_tool=None,
        _touch_activity=lambda _label: None,
    )

    _begin_tool_execution(
        agent,
        function_name="read_file",
        function_args={"path": "same.txt"},
        effective_task_id="task",
        tool_call_id="call-1",
        display_index=1,
    )

    assert recorded == [("read_file", {"path": "same.txt"})]
