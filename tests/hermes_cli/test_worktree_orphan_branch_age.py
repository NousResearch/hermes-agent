"""Regression: the orphan-branch pruner must not reap work that is only seconds old.

``_prune_orphaned_branches`` deletes local ``hermes/hermes-*`` and ``pr-*`` branches
that have no worktree. A branch is worktree-less for a while before its work is
finished — the ref is created before its tree, and a tree can be reaped on an earlier
pass — so with no age gate the pruner destroys seconds-old in-flight work the moment
it is momentarily orphaned. Age decides, not the name.
"""

import os
import subprocess

import pytest

from hermes_cli import worktree_ops

STALE_DATE = "2026-01-01T00:00:00+00:00"


def _git(repo, *args, env=None):
    merged = dict(os.environ)
    merged.update(env or {})
    subprocess.run(["git", *args], cwd=repo, capture_output=True, check=True, env=merged)


def _branches(repo):
    out = subprocess.run(
        ["git", "branch", "--format=%(refname:short)"],
        cwd=repo, capture_output=True, text=True, check=True,
    ).stdout
    return {b.strip() for b in out.splitlines() if b.strip()}


@pytest.fixture
def repo(tmp_path):
    """One worktree-less branch created now, one whose tip commit is days old."""
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q", "-b", "main")
    _git(root, "config", "user.email", "test@test.com")
    _git(root, "config", "user.name", "Test")
    _git(root, "commit", "-q", "--allow-empty", "-m", "base")
    _git(root, "branch", "hermes/hermes-fresh")
    _git(root, "checkout", "-q", "-b", "hermes/hermes-stale", "main")
    _git(root, "commit", "-q", "--allow-empty", "-m", "stale",
         env={"GIT_AUTHOR_DATE": STALE_DATE, "GIT_COMMITTER_DATE": STALE_DATE})
    _git(root, "checkout", "-q", "main")
    return root


def test_a_branch_younger_than_the_cutoff_is_kept(repo):
    worktree_ops._prune_orphaned_branches(str(repo), max_age_hours=24)

    assert "hermes/hermes-fresh" in _branches(repo)


def test_a_branch_older_than_the_cutoff_is_reaped(repo):
    worktree_ops._prune_orphaned_branches(str(repo), max_age_hours=24)

    assert "hermes/hermes-stale" not in _branches(repo)


def test_the_default_call_keeps_a_branch_created_seconds_ago(repo):
    """No arguments: the startup pass must not reap in-flight work.

    This is the regression itself — on base the branch is gone after this call.
    """
    worktree_ops._prune_orphaned_branches(str(repo))

    assert "hermes/hermes-fresh" in _branches(repo)


def test_the_cutoff_and_not_the_name_decides(repo):
    """With the cutoff set to now, the branch the name rule would also spare goes."""
    worktree_ops._prune_orphaned_branches(str(repo), max_age_hours=0)

    assert _branches(repo) == {"main"}
