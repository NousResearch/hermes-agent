"""Regression tests for custom kanban branches already checked out elsewhere."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from hermes_cli import kanban_db_workspace as kbw


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )


def _make_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo.parent, "init", "-b", "main", str(repo))
    (repo / "README.md").write_text("base\n", encoding="utf-8")
    _git(repo, "add", "README.md")
    _git(
        repo,
        "-c",
        "user.name=Test User",
        "-c",
        "user.email=test@example.com",
        "-c",
        "commit.gpgsign=false",
        "commit",
        "-m",
        "init",
    )
    return repo


def test_existing_custom_branch_reports_the_worktree_holding_it(tmp_path: Path):
    repo = _make_repo(tmp_path)
    occupied = repo / ".worktrees" / "existing"
    occupied.parent.mkdir()
    _git(repo, "worktree", "add", "-b", "feat/x", str(occupied), "HEAD")

    target = repo / ".worktrees" / "requested"
    with pytest.raises(RuntimeError) as exc_info:
        kbw._ensure_git_worktree(repo, target, "feat/x")

    message = str(exc_info.value)
    assert "branch feat/x is already checked out" in message
    assert str(occupied) in message
    assert "release that card's worktree or set a different branch_name" in message
    assert not target.exists()
    assert _git(occupied, "branch", "--show-current").stdout.strip() == "feat/x"


def test_available_custom_branch_is_materialized(tmp_path: Path):
    repo = _make_repo(tmp_path)
    _git(repo, "branch", "feat/new")
    target = repo / ".worktrees" / "requested"

    kbw._ensure_git_worktree(repo, target, "feat/new")

    assert target.is_dir()
    assert _git(target, "branch", "--show-current").stdout.strip() == "feat/new"
