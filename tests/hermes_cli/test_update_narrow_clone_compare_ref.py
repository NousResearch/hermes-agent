"""Regression for #125112: the updater must resolve its compare ref on narrow clones.

A tag-pinned ``git clone --depth 1 --single-branch --branch <tag>`` configures
``remote.origin.fetch`` as a tag-only refspec. Fetching a branch by name then
writes only ``FETCH_HEAD`` — no ``refs/remotes/origin/<branch>`` tracking ref —
so the updater's ``rev-parse origin/<branch>`` failed with "Branch not found"
even though the remote has the branch. The fix fetches by explicit refspec, which
always writes the tracking ref.

Tests run the REAL fetch helpers against a real local ``file://`` origin and a
real narrow clone (git behavior is the thing under test; mocks would prove nothing).
"""

import subprocess
from pathlib import Path

import pytest

from hermes_cli import update_cmd_check

git_cmd = ["git"]


def _run(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args], cwd=root, capture_output=True, text=True, encoding="utf-8", errors="replace",
    )


@pytest.fixture
def narrow_clone(tmp_path: Path) -> Path:
    """Bare origin with ``main`` + one tag, and a tag-pinned --single-branch clone of it."""
    origin = tmp_path / "origin.git"
    seed = tmp_path / "seed"
    _run(tmp_path, "init", "-q", "--bare", str(origin))
    _run(tmp_path, "clone", "-q", f"file://{origin}", str(seed))
    (seed / "f.txt").write_text("a\n")
    _run(seed, "add", "f.txt")
    _run(seed, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "init")
    _run(seed, "tag", "v1")
    assert _run(seed, "push", "-q", "origin", "main", "v1").returncode == 0
    work = tmp_path / "work"
    _run(tmp_path, "clone", "-q", "--depth", "1", "--single-branch", "--branch", "v1", f"file://{origin}", str(work))
    # Prove the clone is the narrow shape from the issue before testing anything.
    assert _run(work, "config", "--get-all", "remote.origin.fetch").stdout.strip() == "+refs/tags/v1:refs/tags/v1"
    return work


def test_branch_name_fetch_leaves_compare_ref_unresolved(narrow_clone: Path) -> None:
    """The pre-fix mechanism: a branch-name fetch succeeds but origin/main never lands."""
    fetch = _run(narrow_clone, "fetch", "-q", "--depth", "1", "origin", "main")
    assert fetch.returncode == 0
    assert _run(narrow_clone, "rev-parse", "--verify", "--quiet", "origin/main").returncode != 0


def test_fetch_compare_branch_creates_tracking_ref(narrow_clone: Path) -> None:
    """The fixed --check path: the refspec fetch makes the compare ref resolvable."""
    fetch_result, compare_branch = update_cmd_check.fetch_compare_branch(
        git_cmd, narrow_clone, "main", ["--depth", "1"],
    )
    assert fetch_result.returncode == 0, fetch_result.stderr
    assert compare_branch == "origin/main"
    assert update_cmd_check.compare_ref_exists(git_cmd, narrow_clone, compare_branch)


def test_fetch_compare_branch_reports_missing_branch(narrow_clone: Path) -> None:
    """A genuinely absent branch still reports missing — no false positive introduced."""
    fetch_result, compare_branch = update_cmd_check.fetch_compare_branch(
        git_cmd, narrow_clone, "no-such-branch", ["--depth", "1"],
    )
    assert fetch_result.returncode != 0
    assert compare_branch == "origin/no-such-branch"
    assert not update_cmd_check.compare_ref_exists(git_cmd, narrow_clone, compare_branch)
