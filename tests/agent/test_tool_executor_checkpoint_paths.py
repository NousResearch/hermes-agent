"""Behavioral coverage for file-tool checkpoint path resolution."""

from types import SimpleNamespace

from agent.tool_executor import _ensure_file_checkpoint
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


def test_named_local_target_checkpoint_uses_target_cwd(tmp_path, monkeypatch):
    import tools.execution_targets as targets_mod

    target_cwd = tmp_path / "named-local"
    target_cwd.mkdir()
    (target_cwd / "pyproject.toml").write_text("[project]\nname = 'target'\n")
    (target_cwd / "existing.txt").write_text("before\n")
    monkeypatch.setattr(
        targets_mod,
        "_load_merged_config",
        lambda: {
            "terminal": {
                "default_target": "alpha",
                "targets": {
                    "alpha": {"backend": "local", "cwd": str(target_cwd)},
                },
            },
        },
    )
    monkeypatch.setattr(
        "tools.checkpoint_manager.CHECKPOINT_BASE",
        tmp_path / "checkpoints",
    )
    manager = CheckpointManager(enabled=True)
    agent = SimpleNamespace(_checkpoint_mgr=manager)

    _ensure_file_checkpoint(
        agent,
        "write_file",
        {"path": "existing.txt", "target": "alpha"},
        "gateway-session",
    )

    assert manager.list_checkpoints(str(target_cwd))


def test_remote_target_skips_host_checkpoint(monkeypatch):
    import tools.execution_targets as targets_mod

    monkeypatch.setattr(
        targets_mod,
        "_load_merged_config",
        lambda: {
            "terminal": {
                "default_target": "devbox",
                "targets": {
                    "devbox": {
                        "backend": "ssh",
                        "ssh_host": "example.invalid",
                        "cwd": "/workspace/project",
                    },
                },
            },
        },
    )

    class FailIfCheckpointed:
        def get_working_dir_for_path(self, path):
            raise AssertionError(f"host checkpoint attempted for remote path {path}")

    agent = SimpleNamespace(_checkpoint_mgr=FailIfCheckpointed())
    _ensure_file_checkpoint(
        agent,
        "write_file",
        {"path": "remote.txt", "target": "devbox"},
        "gateway-session",
    )
