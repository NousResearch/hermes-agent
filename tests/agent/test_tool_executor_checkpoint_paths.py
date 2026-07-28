"""Behavioral coverage for file-tool checkpoint path resolution."""

from types import SimpleNamespace

from agent.tool_executor import _ensure_file_checkpoint, _ensure_terminal_checkpoint
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
