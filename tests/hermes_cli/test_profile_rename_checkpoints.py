"""Profile rename must preserve profile-local checkpoint history."""

from pathlib import Path
from unittest.mock import patch

import pytest

from hermes_cli.profiles import create_profile, rename_profile
from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from tools.checkpoint_manager import CheckpointManager


@pytest.fixture()
def profile_env(tmp_path, monkeypatch):
    """Isolate profile paths and the process-level Hermes root."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    default_home = tmp_path / ".hermes"
    default_home.mkdir(exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(default_home))
    return default_home


def test_rename_preserves_profile_local_checkpoint_history(profile_env):
    """A moved profile keeps rollback history under the moved workspace path.

    Checkpoint refs, project metadata, and the safe-restore ledger are keyed by the
    absolute workdir path. Renaming ``profiles/old`` to ``profiles/new`` moves the
    checkpoint store itself, but it also changes every profile-local workdir key.
    """
    old_dir = create_profile("oldname", no_alias=True)
    workdir = old_dir / "workspace"
    workdir.mkdir()
    (workdir / "pyproject.toml").write_text("[project]\nname = 'rename-checkpoint'\n", encoding="utf-8")
    tracked = workdir / "note.txt"
    tracked.write_text("before\n", encoding="utf-8")

    token = set_hermes_home_override(old_dir)
    try:
        manager = CheckpointManager(enabled=True, max_snapshots=5)
        assert manager.ensure_checkpoint(str(workdir), "before profile rename") is True
        checkpoint_hash = manager.list_checkpoints(str(workdir))[0]["hash"]
        tracked.write_text("after\n", encoding="utf-8")
        manager.record_agent_write(str(tracked))
    finally:
        reset_hermes_home_override(token)

    with patch("hermes_cli.profiles.check_alias_collision", return_value="skip"), \
         patch("hermes_cli.profiles._live_default_multiplexer", return_value=False):
        new_dir = rename_profile("oldname", "newname")

    new_workdir = new_dir / "workspace"
    new_tracked = new_workdir / "note.txt"
    token = set_hermes_home_override(new_dir)
    try:
        manager = CheckpointManager(enabled=True, max_snapshots=5)
        checkpoints = manager.list_checkpoints(str(new_workdir))
        assert [entry["hash"] for entry in checkpoints] == [checkpoint_hash]

        project_paths = {entry["workdir"] for entry in manager.list_all_checkpoints()}
        assert str(new_workdir.resolve()) in project_paths
        assert str(workdir.resolve()) not in project_paths

        plan = manager.safe_restore_plan(str(new_workdir), checkpoint_hash)
        assert plan["success"] is True
        assert plan["restore"] == ["note.txt"]
        assert plan["skipped"] == []

        restored = manager.restore(str(new_workdir), checkpoint_hash, safe=True)
        assert restored["success"] is True
        assert new_tracked.read_text(encoding="utf-8") == "before\n"
    finally:
        reset_hermes_home_override(token)
