"""Regression tests for index locks wedging ``hermes update`` (#63038)."""

from __future__ import annotations

import os
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from hermes_cli import gitlock, update_cmd, update_contract, update_receipt
from hermes_cli import main as hermes_main


pytestmark = pytest.mark.windows_only


@pytest.fixture
def isolated_update(tmp_path, monkeypatch):
    """Contain every updater boundary even if the index guard is bypassed."""
    root = tmp_path / "checkout"
    root.mkdir()
    home = tmp_path / "hermes-home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv(hermes_main._UPDATE_REEXEC_ENV, raising=False)
    monkeypatch.setattr(update_receipt, "_current", None)
    monkeypatch.setattr(hermes_main, "PROJECT_ROOT", root)
    monkeypatch.setattr("hermes_cli.config.is_managed", lambda: False)
    admission = Mock(return_value=None)
    monkeypatch.setattr(update_contract, "evaluate_update_admission", admission)
    monkeypatch.setattr(hermes_main, "_resolve_update_branch", lambda args: "main")
    monkeypatch.setattr(hermes_main, "_install_hangup_protection", lambda **kwargs: None)
    monkeypatch.setattr(hermes_main, "_finalize_update_output", lambda state: None)
    monkeypatch.setattr("hermes_cli.update_handoff.wait_for_shim_parent_exit", lambda: None)
    update_lock = Mock()
    update_lock.acquire.return_value = True
    monkeypatch.setattr("hermes_cli.update_lock.UpdateLock", lambda: update_lock)
    backup = Mock(side_effect=AssertionError("backup boundary reached"))
    monkeypatch.setattr(hermes_main, "_run_pre_update_backup", backup)
    gateway = Mock(side_effect=AssertionError("gateway boundary reached"))
    monkeypatch.setattr(hermes_main, "_pause_windows_gateways_for_update", gateway)
    monkeypatch.setattr(hermes_main, "_resume_windows_gateways_after_update", gateway)
    monkeypatch.setattr(update_cmd, "_clear_windows_venv_holders_or_exit", gateway)
    git = Mock(side_effect=AssertionError("Git boundary reached"))
    monkeypatch.setattr(update_cmd, "_prepare_git_command", git)
    monkeypatch.setattr(update_cmd, "_git_run", git)
    locks = Mock(side_effect=AssertionError("stale-lock cleanup reached"))
    packs = Mock(side_effect=AssertionError("tmp-pack cleanup reached"))
    monkeypatch.setattr(gitlock, "clear_stale_git_locks", locks)
    monkeypatch.setattr(gitlock, "clear_stale_tmp_packs", packs)
    monkeypatch.setattr(
        update_cmd, "_resolve_update_options",
        lambda args, gateway_mode: SimpleNamespace(gw_input_fn=None, assume_yes=True),
    )
    monkeypatch.setattr(
        update_cmd, "_begin_update_receipt_and_plan",
        lambda args: update_receipt.begin_update_receipt(),
    )
    return SimpleNamespace(
        root=root, home=home, admission=admission, update_lock=update_lock,
        backup=backup, gateway=gateway, git=git, locks=locks, packs=packs,
    )


def _git_dir(root, linked_worktree):
    if linked_worktree:
        git_dir = root.parent / "actual-git-dir"
        git_dir.mkdir()
        (root / ".git").write_text("gitdir: ../actual-git-dir\n", encoding="utf-8")
    else:
        git_dir = root / ".git"
        git_dir.mkdir()
    return git_dir


@pytest.mark.parametrize("age_seconds", [0, 7200], ids=["fresh", "old"])
@pytest.mark.parametrize("linked_worktree", [False, True], ids=["normal-git", "linked-git"])
def test_update_index_lock_aborts_without_deleting(
    tmp_path, capsys, age_seconds, linked_worktree
):
    project_root = tmp_path / "worktree"
    project_root.mkdir()
    git_dir = _git_dir(project_root, linked_worktree)

    lock = git_dir / "index.lock"
    lock.write_bytes(b"owned by another Git process\n")
    original = lock.read_bytes()
    if age_seconds:
        old_time = time.time() - age_seconds
        os.utime(lock, (old_time, old_time))

    with pytest.raises(SystemExit) as exc_info:
        update_cmd._abort_if_update_index_locked(project_root)

    assert exc_info.value.code == 2
    assert lock.read_bytes() == original
    output = capsys.readouterr().out
    assert f"Git index lock exists: {lock}" in output
    assert "Remove-Item -LiteralPath" in output
    assert str(lock) in output


@pytest.mark.parametrize("force,force_venv", [(False, False), (True, False), (False, True), (True, True)])
def test_update_aborts_before_backup_or_git_mutation_and_finalizes_receipt(
    isolated_update, force, force_venv
):
    git_dir = _git_dir(isolated_update.root, False)
    lock = git_dir / "index.lock"
    lock.write_bytes(b"in use\n")

    with pytest.raises(SystemExit) as exc_info:
        hermes_main.cmd_update(SimpleNamespace(force=force, force_venv=force_venv))

    assert exc_info.value.code == 2
    assert lock.read_bytes() == b"in use\n"
    isolated_update.backup.assert_not_called()
    isolated_update.gateway.assert_not_called()
    isolated_update.git.assert_not_called()
    isolated_update.locks.assert_not_called()
    isolated_update.packs.assert_not_called()
    isolated_update.update_lock.release.assert_called_once()
    isolated_update.admission.assert_called_once_with(isolated_update.root)
    receipt = update_receipt.read_latest_receipt()
    assert receipt is not None
    assert receipt["outcome"] == "refused"
    assert receipt["exit_code"] == 2
    assert receipt["finished_at"] is not None
    assert update_receipt._current is None


@pytest.mark.parametrize("age_seconds", [0, 7200], ids=["fresh", "old"])
@pytest.mark.parametrize("linked_worktree", [False, True], ids=["normal-git", "linked-git"])
@pytest.mark.parametrize("force", [False, True], ids=["default", "force"])
def test_check_preserves_index_lock_before_cleanup(
    isolated_update, capsys, age_seconds, linked_worktree, force
):
    git_dir = _git_dir(isolated_update.root, linked_worktree)
    lock = git_dir / "index.lock"
    lock.write_bytes(b"owned by another Git process\n")
    if age_seconds:
        old_time = time.time() - age_seconds
        os.utime(lock, (old_time, old_time))

    with pytest.raises(SystemExit) as exc_info:
        hermes_main.cmd_update(SimpleNamespace(check=True, force=force, branch=None))

    assert exc_info.value.code == 2
    assert lock.read_bytes() == b"owned by another Git process\n"
    assert f"Git index lock exists: {lock}" in capsys.readouterr().out
    assert isolated_update.admission.call_count == 2
    isolated_update.locks.assert_not_called()
    isolated_update.packs.assert_not_called()
    isolated_update.git.assert_not_called()
    isolated_update.backup.assert_not_called()
    isolated_update.gateway.assert_not_called()


def test_check_without_index_lock_keeps_stale_cleanup(isolated_update, monkeypatch):
    _git_dir(isolated_update.root, False)
    isolated_update.locks.side_effect = None
    isolated_update.locks.return_value = []
    isolated_update.packs.side_effect = None
    isolated_update.packs.return_value = []
    boundary = Mock(side_effect=AssertionError("shallow probe reached"))
    monkeypatch.setattr(update_cmd, "_is_shallow_checkout", boundary)

    with pytest.raises(AssertionError, match="shallow probe reached"):
        hermes_main.cmd_update(SimpleNamespace(check=True, branch=None))

    isolated_update.locks.assert_called_once_with(isolated_update.root)
    isolated_update.packs.assert_called_once_with(isolated_update.root)
    isolated_update.git.assert_not_called()
    isolated_update.backup.assert_not_called()
    isolated_update.gateway.assert_not_called()


def test_update_without_index_lock_reaches_backup(isolated_update):
    _git_dir(isolated_update.root, False)

    with pytest.raises(AssertionError, match="backup boundary reached"):
        hermes_main.cmd_update(SimpleNamespace())

    isolated_update.backup.assert_called_once()
    isolated_update.gateway.assert_not_called()
    isolated_update.git.assert_not_called()
