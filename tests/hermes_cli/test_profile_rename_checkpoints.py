"""Profile rename must preserve profile-local checkpoint history (#112973)."""

from pathlib import Path
from unittest.mock import patch

import pytest

from hermes_cli.profile_identity import migrate_profile_identity
from hermes_cli.profiles import create_profile, rename_profile
from hermes_constants import reset_hermes_home_override, set_hermes_home_override
import tools.checkpoint_manager as cm
from tools.checkpoint_manager import CheckpointManager
from tools import checkpoint_maintenance as maintenance


@pytest.fixture()
def profile_env(tmp_path, monkeypatch):
    """Isolate profile paths and the process-level Hermes root."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    default_home = tmp_path / ".hermes"
    default_home.mkdir(exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(default_home))
    return default_home


def test_rename_preserves_profile_local_checkpoint_history(profile_env, tmp_path):
    """A moved profile keeps rollback history under the moved workspace path.

    Checkpoint refs, project metadata, and the safe-restore ledger are keyed by the
    absolute workdir path. Renaming ``profiles/old`` to ``profiles/new`` moves the
    checkpoint store itself, but it also changes every profile-local workdir key.
    """
    old_dir = create_profile("oldname", no_alias=True)
    workdir = old_dir / "project"
    workdir.mkdir()
    outside = tmp_path / "outside-project"  # control: not under the profile dir, same store
    outside.mkdir()
    (outside / "keep.txt").write_text("v1\n", encoding="utf-8")
    (workdir / "pyproject.toml").write_text("[project]\nname = 'rename-checkpoint'\n", encoding="utf-8")
    tracked = workdir / "note.txt"
    tracked.write_text("before\n", encoding="utf-8")

    token = set_hermes_home_override(old_dir)
    try:
        manager = CheckpointManager(enabled=True, max_snapshots=5)
        assert manager.ensure_checkpoint(str(workdir), "before profile rename") is True
        checkpoint_hash = manager.list_checkpoints(str(workdir))[0]["hash"]
        assert manager.ensure_checkpoint(str(outside), "outside profile") is True
        outside_hash = manager.list_checkpoints(str(outside))[0]["hash"]
        tracked.write_text("after\n", encoding="utf-8")
        manager.record_agent_write(str(tracked))
    finally:
        reset_hermes_home_override(token)

    with patch("hermes_cli.profiles.check_alias_collision", return_value="skip"), \
         patch("hermes_cli.profiles._live_default_multiplexer", return_value=False):
        new_dir = rename_profile("oldname", "newname")

    new_workdir = new_dir / "project"
    new_tracked = new_workdir / "note.txt"
    token = set_hermes_home_override(new_dir)
    try:
        manager = CheckpointManager(enabled=True, max_snapshots=5)
        checkpoints = manager.list_checkpoints(str(new_workdir))
        assert [entry["hash"] for entry in checkpoints] == [checkpoint_hash]
        assert [entry["hash"] for entry in manager.list_checkpoints(str(outside))] == [outside_hash]

        project_paths = {entry["workdir"] for entry in manager.list_all_checkpoints()}
        assert str(new_workdir.resolve()) in project_paths
        assert str(workdir.resolve()) not in project_paths

        plan = manager._safe_restore_plan(str(new_workdir), checkpoint_hash)
        assert plan["success"] is True
        assert plan["restore"] == ["note.txt"]
        assert plan["skipped"] == []

        restored = manager.restore(str(new_workdir), checkpoint_hash, safe=True)
        assert restored["success"] is True
        assert new_tracked.read_text(encoding="utf-8") == "before\n"
    finally:
        reset_hermes_home_override(token)


def test_retry_after_partial_rekey_keeps_checkpoints_taken_under_new_name(profile_env):
    """``migrate-identity`` after a mid-way failure must not rewind history to the old tip.

    Between the failed rename and the retry the user keeps working under the new profile, so the
    new ref already carries checkpoints (and ledger entries) the old identity never saw.
    """
    old_dir = create_profile("oldname", no_alias=True)
    workdir = old_dir / "project"
    workdir.mkdir()
    (workdir / "pyproject.toml").write_text("[project]\nname = 'retry'\n", encoding="utf-8")
    note = workdir / "note.txt"
    note.write_text("v1\n", encoding="utf-8")

    token = set_hermes_home_override(old_dir)
    try:
        assert CheckpointManager(enabled=True, max_snapshots=5).ensure_checkpoint(str(workdir), "c1") is True
    finally:
        reset_hermes_home_override(token)

    with patch("hermes_cli.profiles.check_alias_collision", return_value="skip"), \
         patch("hermes_cli.profiles._live_default_multiplexer", return_value=False), \
         patch.object(maintenance, "_delete_ref", return_value=False):  # old ref survives: partial rekey
        new_dir = rename_profile("oldname", "newname")

    new_workdir = new_dir / "project"
    new_note = new_workdir / "note.txt"
    token = set_hermes_home_override(new_dir)
    try:
        new_note.write_text("v2\n", encoding="utf-8")
        manager = CheckpointManager(enabled=True, max_snapshots=5)
        assert manager.ensure_checkpoint(str(new_workdir), "c2") is True
        manager.record_agent_write(str(new_note))
        before = [entry["hash"] for entry in manager.list_checkpoints(str(new_workdir))]
        assert len(before) == 2
        store = cm._store_path(new_dir / "checkpoints")
        ledger_before = cm._load_ledger(store, cm._project_hash(str(new_workdir)))

        assert migrate_profile_identity("oldname", "newname") is True

        manager = CheckpointManager(enabled=True, max_snapshots=5)
        assert [entry["hash"] for entry in manager.list_checkpoints(str(new_workdir))] == before
        assert cm._load_ledger(store, cm._project_hash(str(new_workdir))) == ledger_before
        assert cm._list_project_refs(store, str(new_workdir)) == [cm._ref_name(cm._project_hash(str(new_workdir)))]
    finally:
        reset_hermes_home_override(token)
