"""Regression tests for index locks wedging ``hermes update`` (#63038)."""

from __future__ import annotations

import os
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from hermes_cli import main as hermes_main
from hermes_cli import update_cmd


pytestmark = pytest.mark.windows_only


@pytest.mark.parametrize("age_seconds", [0, 7200], ids=["fresh", "old"])
@pytest.mark.parametrize("linked_worktree", [False, True], ids=["normal-git", "linked-git"])
def test_update_index_lock_aborts_without_deleting(
    tmp_path, capsys, age_seconds, linked_worktree
):
    if linked_worktree:
        git_dir = tmp_path / "actual-git-dir"
        project_root = tmp_path / "worktree"
        git_dir.mkdir()
        project_root.mkdir()
        (project_root / ".git").write_text(
            "gitdir: ../actual-git-dir\n", encoding="utf-8"
        )
    else:
        project_root = tmp_path
        git_dir = project_root / ".git"
        git_dir.mkdir()

    lock = git_dir / "index.lock"
    lock.touch()
    if age_seconds:
        old_time = time.time() - age_seconds
        os.utime(lock, (old_time, old_time))

    with pytest.raises(SystemExit) as exc_info:
        update_cmd._abort_if_update_index_locked(project_root)

    assert exc_info.value.code == 2
    assert lock.exists()
    output = capsys.readouterr().out
    assert f"Git index lock exists: {lock}" in output
    assert "Remove-Item -LiteralPath" in output
    assert str(lock) in output


def test_update_aborts_before_backup_or_git_mutation(tmp_path, monkeypatch):
    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    lock = git_dir / "index.lock"
    lock.touch()
    backup = Mock()
    git_run = Mock()
    monkeypatch.setattr(hermes_main, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(hermes_main, "_run_pre_update_backup", backup)
    monkeypatch.setattr(update_cmd.subprocess, "run", git_run)

    with pytest.raises(SystemExit) as exc_info:
        update_cmd._cmd_update_impl(SimpleNamespace(), gateway_mode=False)

    assert exc_info.value.code == 2
    backup.assert_not_called()
    git_run.assert_not_called()
