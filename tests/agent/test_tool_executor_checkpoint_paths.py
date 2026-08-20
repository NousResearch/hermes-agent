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


def test_v4a_patch_checkpoint_uses_paths_embedded_in_patch(tmp_path, monkeypatch):
    """V4A patches must snapshot files whose paths live in the patch body."""
    workspace_cwd = tmp_path / "workspace"
    workspace_cwd.mkdir()
    (workspace_cwd / "pyproject.toml").write_text("[project]\nname = 'workspace'\n")
    (workspace_cwd / "existing.txt").write_text("before\n")

    monkeypatch.setenv("TERMINAL_CWD", str(workspace_cwd))
    monkeypatch.setattr(
        "tools.checkpoint_manager.CHECKPOINT_BASE",
        tmp_path / "checkpoints",
    )

    manager = CheckpointManager(enabled=True)
    agent = SimpleNamespace(_checkpoint_mgr=manager)

    _ensure_file_checkpoint(
        agent,
        "patch",
        {
            "mode": "patch",
            "patch": """*** Begin Patch
*** Update File: existing.txt
@@
-before
+after
*** End Patch""",
        },
        "gateway-session",
    )

    checkpoints = manager.list_checkpoints(str(workspace_cwd))
    assert checkpoints

    (workspace_cwd / "existing.txt").write_text("after\n")
    result = manager.restore(str(workspace_cwd), checkpoints[0]["hash"])

    assert result["success"] is True
    assert (workspace_cwd / "existing.txt").read_text() == "before\n"


def test_v4a_patch_checkpoint_rejects_traversal_before_snapshot(tmp_path, monkeypatch):
    """Rejected V4A paths must not copy an out-of-workspace project."""
    workspace_cwd = tmp_path / "workspace"
    outside_cwd = tmp_path / "outside"
    workspace_cwd.mkdir()
    outside_cwd.mkdir()
    (workspace_cwd / "pyproject.toml").write_text("[project]\nname = 'workspace'\n")
    (outside_cwd / "pyproject.toml").write_text("[project]\nname = 'outside'\n")
    (outside_cwd / "secret.txt").write_text("do not snapshot\n")

    monkeypatch.setenv("TERMINAL_CWD", str(workspace_cwd))
    monkeypatch.setattr(
        "tools.checkpoint_manager.CHECKPOINT_BASE",
        tmp_path / "checkpoints",
    )

    manager = CheckpointManager(enabled=True)
    agent = SimpleNamespace(_checkpoint_mgr=manager)

    _ensure_file_checkpoint(
        agent,
        "patch",
        {
            "mode": "patch",
            "patch": """*** Begin Patch
*** Update File: ../outside/secret.txt
@@
-do not snapshot
+changed
*** End Patch""",
        },
        "gateway-session",
    )

    assert manager.list_checkpoints(str(outside_cwd)) == []


def test_v4a_patch_checkpoint_ignores_headers_after_end_marker(tmp_path, monkeypatch):
    """Checkpoint targets must match the paths the V4A parser will apply."""
    workspace_cwd = tmp_path / "workspace"
    outside_cwd = tmp_path / "outside"
    workspace_cwd.mkdir()
    outside_cwd.mkdir()
    (workspace_cwd / "pyproject.toml").write_text("[project]\nname = 'workspace'\n")
    (workspace_cwd / "existing.txt").write_text("before\n")
    (outside_cwd / "pyproject.toml").write_text("[project]\nname = 'outside'\n")
    (outside_cwd / "secret.txt").write_text("do not snapshot\n")

    monkeypatch.setenv("TERMINAL_CWD", str(workspace_cwd))
    monkeypatch.setattr(
        "tools.checkpoint_manager.CHECKPOINT_BASE",
        tmp_path / "checkpoints",
    )

    manager = CheckpointManager(enabled=True)
    agent = SimpleNamespace(_checkpoint_mgr=manager)

    _ensure_file_checkpoint(
        agent,
        "patch",
        {
            "mode": "patch",
            "patch": f"""*** Begin Patch
*** Update File: existing.txt
@@
-before
+after
*** End Patch
*** Update File: {outside_cwd / 'secret.txt'}""",
        },
        "gateway-session",
    )

    assert manager.list_checkpoints(str(workspace_cwd))
    assert manager.list_checkpoints(str(outside_cwd)) == []
