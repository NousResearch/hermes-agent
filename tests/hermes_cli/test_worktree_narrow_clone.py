"""Worktrees from ``origin/<branch>`` on a tag-pinned narrow clone (#125686).

Older installers made ``git clone --depth 1 --single-branch --branch <tag>`` checkouts, whose
``remote.origin.fetch`` maps only the tag. A fetch by branch name there writes FETCH_HEAD and never
``origin/<branch>``, so every worktree surface that fetches a base and then names it had nothing to
resolve. Real repos, real git, no mocks.
"""

import subprocess
from pathlib import Path

import pytest

from hermes_cli import web_git, worktree_ops


def _git(cwd, *args) -> str:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True).stdout.strip()


def _commit(repo: Path, name: str, msg: str) -> None:
    (repo / name).write_text(msg)
    _git(repo, "add", name)
    _git(repo, "-c", "user.email=t@t", "-c", "user.name=T", "commit", "-qm", msg)


@pytest.fixture
def narrow_clone(tmp_path):
    """(clone, upstream main sha, upstream feature sha); both branches moved after the clone."""
    up = tmp_path / "upstream"
    up.mkdir()
    _git(up, "init", "-q", "-b", "main")
    _commit(up, "f", "c1")
    _git(up, "tag", "v1")
    _git(up, "branch", "feature")
    clone = tmp_path / "clone"
    _git(tmp_path, "clone", "-q", "--depth", "1", "--single-branch", "--branch", "v1", up.as_uri(), str(clone))
    assert _git(clone, "config", "--get-all", "remote.origin.fetch") == "+refs/tags/v1:refs/tags/v1"
    # origin/feature exists but is stale, as after an earlier explicit-refspec fetch.
    _git(clone, "fetch", "-q", "origin", "+refs/heads/feature:refs/remotes/origin/feature")
    _commit(up, "f", "c2")
    _git(up, "checkout", "-q", "feature")
    _commit(up, "g", "feature c2")
    _git(up, "checkout", "-q", "main")
    return clone, _git(up, "rev-parse", "main"), _git(up, "rev-parse", "feature")


def test_dashboard_worktrees_start_from_the_remote_tip(narrow_clone):
    clone, main_sha, feature_sha = narrow_clone

    new = web_git.worktree_add(str(clone), {"base": "origin/main", "name": "x"})
    assert _git(new["path"], "rev-parse", "HEAD") == main_sha

    converted = web_git.worktree_add(str(clone), {"existingBranch": "origin/feature"})
    assert converted["branch"] == "feature"
    assert _git(converted["path"], "rev-parse", "HEAD") == feature_sha


def test_cli_worktree_base_resolves_to_the_remote_tip(narrow_clone):
    clone, main_sha, _ = narrow_clone

    ref, _label = worktree_ops._resolve_worktree_base(str(clone))

    assert _git(clone, "rev-parse", "--verify", f"{ref}^{{commit}}") == main_sha
