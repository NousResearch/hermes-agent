"""Worktree paths returned by Git must address the actual checkout."""
import subprocess

import pytest

from hermes_cli import web_git


@pytest.mark.platforms("linux", "macos")
def test_worktree_listing_preserves_quoted_and_whitespace_paths(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()

    def git(*args):
        subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)

    git("init")
    git("config", "user.email", "test@example.invalid")
    git("config", "user.name", "Test")
    git("commit", "--allow-empty", "-m", "baseline")
    worktree = tmp_path / "work\n tree "
    git("worktree", "add", "-b", "feature", str(worktree))
    rows = web_git.worktree_list(str(root))
    assert [r["path"] for r in rows] == [str(root), str(worktree)]
    assert rows[1]["branch"] == "feature"
