"""Tests for the ``hermes update`` hard-reset safety guard.

Background — auto-updater hard-reset incident (2026-06-08): a non-interactive
``hermes update`` hit a diverged history (``git pull --ff-only`` failed) and
fell back to ``git reset --hard origin/main``. Uncommitted work is autostashed,
but a hard reset ALSO moves the branch ref past local commits that were never
pushed — silently orphaning them (recoverable only via reflog). A local
``main`` commit was lost this way while a human/agent worked in the same
checkout.

These tests cover the two helpers the guard relies on, exercised against REAL
git repositories in ``tmp_path`` (no mocking of git itself), so they prove the
guard reads actual on-disk repo state correctly. The end-to-end refusal
behavior in ``cmd_update`` is covered in ``test_update_autostash.py``.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from hermes_cli import main as hermes_main

pytestmark = pytest.mark.skipif(
    shutil.which("git") is None, reason="git not available"
)


def _git(cwd: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=True
    )


def _init_repo(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    _git(path, "init", "-q", "-b", "main")
    _git(path, "config", "user.email", "t@example.com")
    _git(path, "config", "user.name", "t")
    _git(path, "commit", "-q", "--allow-empty", "-m", "init")


# ---------------------------------------------------------------------------
# _detect_git_operation_in_progress
# ---------------------------------------------------------------------------

def test_detect_returns_none_on_clean_repo(tmp_path):
    _init_repo(tmp_path)
    assert hermes_main._detect_git_operation_in_progress(tmp_path) is None


def test_detect_flags_merge_in_progress(tmp_path):
    _init_repo(tmp_path)
    (tmp_path / ".git" / "MERGE_HEAD").write_text("deadbeef\n")
    assert hermes_main._detect_git_operation_in_progress(tmp_path) == "a merge"


def test_detect_flags_cherry_pick_in_progress(tmp_path):
    _init_repo(tmp_path)
    (tmp_path / ".git" / "CHERRY_PICK_HEAD").write_text("deadbeef\n")
    assert hermes_main._detect_git_operation_in_progress(tmp_path) == "a cherry-pick"


def test_detect_flags_rebase_in_progress(tmp_path):
    _init_repo(tmp_path)
    (tmp_path / ".git" / "rebase-merge").mkdir()
    assert hermes_main._detect_git_operation_in_progress(tmp_path) == "a rebase"


def test_detect_flags_index_lock(tmp_path):
    _init_repo(tmp_path)
    (tmp_path / ".git" / "index.lock").write_text("")
    label = hermes_main._detect_git_operation_in_progress(tmp_path)
    assert label is not None and "index.lock" in label


def test_detect_resolves_gitdir_file_for_worktrees(tmp_path):
    """A linked worktree has a ``.git`` *file* pointing at the real git dir;
    the detector must follow it rather than treat the file as a repo dir."""
    real_git = tmp_path / "realgit"
    real_git.mkdir()
    (real_git / "MERGE_HEAD").write_text("deadbeef\n")
    worktree = tmp_path / "wt"
    worktree.mkdir()
    (worktree / ".git").write_text(f"gitdir: {real_git}\n")
    assert hermes_main._detect_git_operation_in_progress(worktree) == "a merge"


# ---------------------------------------------------------------------------
# _count_local_ahead_commits
# ---------------------------------------------------------------------------

def _clone_with_remote(tmp_path: Path) -> Path:
    """Create a bare origin + working clone with main tracking origin/main."""
    origin = tmp_path / "origin.git"
    origin.mkdir()
    _git(origin, "init", "-q", "--bare", "-b", "main")

    work = tmp_path / "work"
    _init_repo(work)
    _git(work, "remote", "add", "origin", str(origin))
    _git(work, "push", "-q", "-u", "origin", "main")
    return work


def test_ahead_count_zero_when_in_sync(tmp_path):
    work = _clone_with_remote(tmp_path)
    assert hermes_main._count_local_ahead_commits(["git"], work, "main") == 0


def test_ahead_count_reflects_unpushed_local_commits(tmp_path):
    work = _clone_with_remote(tmp_path)
    _git(work, "commit", "-q", "--allow-empty", "-m", "local 1")
    _git(work, "commit", "-q", "--allow-empty", "-m", "local 2")
    assert hermes_main._count_local_ahead_commits(["git"], work, "main") == 2


def test_ahead_count_none_when_origin_ref_missing(tmp_path):
    """No origin/<branch> ref → rev-list errors → guard treats as unknown."""
    work = tmp_path / "noremote"
    _init_repo(work)
    assert hermes_main._count_local_ahead_commits(["git"], work, "main") is None
