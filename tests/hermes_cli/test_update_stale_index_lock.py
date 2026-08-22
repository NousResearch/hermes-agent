"""Regression tests for index locks wedging ``hermes update`` (#63038)."""

from __future__ import annotations

import os
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from hermes_cli import main as hermes_main


@pytest.mark.parametrize("age_seconds", [0, 7200])
def test_update_index_lock_aborts_without_deleting(
    tmp_path, capsys, monkeypatch, age_seconds
):
    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    lock = git_dir / "index.lock"
    lock.touch()
    if age_seconds:
        old_time = time.time() - age_seconds
        os.utime(lock, (old_time, old_time))
    monkeypatch.setattr(hermes_main, "_is_windows", lambda: False)

    with pytest.raises(SystemExit) as exc_info:
        hermes_main._abort_if_update_index_locked(tmp_path)

    assert exc_info.value.code == 2
    assert lock.exists()
    output = capsys.readouterr().out
    assert f"Git index lock exists: {lock}" in output
    assert "rm -f -- " in output
    assert str(lock) in output


def test_missing_update_index_lock_is_noop(tmp_path):
    (tmp_path / ".git").mkdir()
    assert hermes_main._abort_if_update_index_locked(tmp_path) is None


def test_linked_worktree_index_lock_aborts_without_deleting(
    tmp_path, capsys, monkeypatch
):
    git_dir = tmp_path / "actual-git-dir"
    git_dir.mkdir()
    lock = git_dir / "index.lock"
    lock.touch()
    worktree = tmp_path / "worktree"
    worktree.mkdir()
    (worktree / ".git").write_text("gitdir: ../actual-git-dir\n", encoding="utf-8")
    monkeypatch.setattr(hermes_main, "_is_windows", lambda: False)

    with pytest.raises(SystemExit) as exc_info:
        hermes_main._abort_if_update_index_locked(worktree)

    assert exc_info.value.code == 2
    assert lock.exists()
    assert str(lock) in capsys.readouterr().out


def test_windows_index_lock_recovery_uses_powershell(tmp_path, capsys, monkeypatch):
    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    lock = git_dir / "index.lock"
    lock.touch()
    monkeypatch.setattr(hermes_main, "_is_windows", lambda: True)

    with pytest.raises(SystemExit):
        hermes_main._abort_if_update_index_locked(tmp_path)

    assert lock.exists()
    assert "Remove-Item -LiteralPath" in capsys.readouterr().out


def test_update_aborts_before_backup_or_git_mutation(tmp_path, monkeypatch):
    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    (git_dir / "index.lock").touch()
    backup = Mock()
    git_run = Mock()
    monkeypatch.setattr(hermes_main, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(hermes_main, "_is_windows", lambda: False)
    monkeypatch.setattr(hermes_main, "_run_pre_update_backup", backup)
    monkeypatch.setattr(hermes_main.subprocess, "run", git_run)

    with pytest.raises(SystemExit) as exc_info:
        hermes_main._cmd_update_impl(SimpleNamespace(), gateway_mode=False)

    assert exc_info.value.code == 2
    backup.assert_not_called()
    git_run.assert_not_called()
