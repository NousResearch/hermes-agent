"""Gate tests for hermes_cli/worktree_safety.py — is_branch_unmerged().

Spec (must match exactly):
  is_branch_unmerged(repo: Path, branch: str, base: str = "main") -> bool
  - True  iff `branch` has at least one commit NOT reachable from `base`.
  - False if every commit of `branch` is an ancestor of (or equal to) `base`.
  - branch == base -> False.
  - Unknown branch OR unknown base -> raise ValueError naming the missing ref.
  - Must not mutate the repository (read-only git commands only).
"""
import subprocess
from pathlib import Path

import pytest

from hermes_cli.worktree_safety import is_branch_unmerged


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True, capture_output=True, text=True,
    ).stdout.strip()


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    r = tmp_path / "r"
    r.mkdir()
    _git(r, "init", "-b", "main")
    _git(r, "config", "user.email", "t@t")
    _git(r, "config", "user.name", "t")
    (r / "a.txt").write_text("a\n")
    _git(r, "add", "-A")
    _git(r, "commit", "-m", "A")
    return r


def test_unmerged_branch_true(repo: Path):
    _git(repo, "checkout", "-b", "feat")
    (repo / "b.txt").write_text("b\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", "B")
    _git(repo, "checkout", "main")
    assert is_branch_unmerged(repo, "feat") is True


def test_merged_branch_false(repo: Path):
    _git(repo, "checkout", "-b", "feat")
    (repo / "b.txt").write_text("b\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", "B")
    _git(repo, "checkout", "main")
    _git(repo, "merge", "--no-ff", "feat", "-m", "merge")
    assert is_branch_unmerged(repo, "feat") is False


def test_branch_equals_base(repo: Path):
    assert is_branch_unmerged(repo, "main") is False


def test_branch_at_older_ancestor_false(repo: Path):
    _git(repo, "branch", "old")  # points at A
    (repo / "c.txt").write_text("c\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", "C")  # main moves ahead
    assert is_branch_unmerged(repo, "old") is False


def test_unknown_branch_raises(repo: Path):
    with pytest.raises(ValueError, match="nope"):
        is_branch_unmerged(repo, "nope")


def test_unknown_base_raises(repo: Path):
    with pytest.raises(ValueError, match="devel"):
        is_branch_unmerged(repo, "main", base="devel")


def test_readonly(repo: Path):
    _git(repo, "checkout", "-b", "feat")
    (repo / "b.txt").write_text("b\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", "B")
    _git(repo, "checkout", "main")
    before = _git(repo, "rev-parse", "main") + _git(repo, "rev-parse", "feat")
    is_branch_unmerged(repo, "feat")
    after = _git(repo, "rev-parse", "main") + _git(repo, "rev-parse", "feat")
    assert before == after
