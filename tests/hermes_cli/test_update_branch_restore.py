"""Tests for _restore_feature_branch_after_update.

After a successful ``hermes update`` pull, the checkout must return to the
user's own feature branch (rebased onto the updated target) instead of
stranding the install on the target branch — the 2026-07-19 incident where
an update mid-session silently swapped a running install from a feature
branch to stock main. Uses real throwaway git repos; no network.
"""

from __future__ import annotations

import subprocess

import pytest

from hermes_cli.main import _restore_feature_branch_after_update

GIT = ["git"]


def _git(repo, *argv, check=True):
    return subprocess.run(
        GIT + list(argv), cwd=repo, capture_output=True, text=True, check=check
    )


def _current_branch(repo) -> str:
    return _git(repo, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip()


@pytest.fixture
def repo(tmp_path):
    """A repo on branch main (1 commit) with feature branch (1 extra commit),
    then main advanced by one more commit — the post-`git pull` state, with
    HEAD left on main exactly as the update flow leaves it."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.email", "t@example.com")
    _git(repo, "config", "user.name", "t")
    (repo / "base.txt").write_text("base\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", "base")

    _git(repo, "checkout", "-b", "feature")
    (repo / "feature.txt").write_text("feature\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", "feature work")

    _git(repo, "checkout", "main")
    (repo / "upstream.txt").write_text("upstream\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", "upstream update")
    return repo


def _patch_syntax_guard(monkeypatch, ok=True, failing="hermes_cli/config.py"):
    import hermes_cli.main as main_mod

    monkeypatch.setattr(
        main_mod,
        "_validate_critical_files_syntax",
        lambda root: (True, None, None) if ok else (False, failing, "boom"),
    )


def test_returns_to_feature_branch_rebased(repo, monkeypatch):
    _patch_syntax_guard(monkeypatch, ok=True)
    _restore_feature_branch_after_update(GIT, repo, "feature", "main")

    assert _current_branch(repo) == "feature"
    # Rebased: feature now contains main's new commit
    merge_base = _git(repo, "merge-base", "feature", "main").stdout.strip()
    main_sha = _git(repo, "rev-parse", "main").stdout.strip()
    assert merge_base == main_sha
    assert (repo / "feature.txt").exists()
    assert (repo / "upstream.txt").exists()


def test_noop_when_already_on_target(repo, monkeypatch):
    _patch_syntax_guard(monkeypatch, ok=True)
    _restore_feature_branch_after_update(GIT, repo, "main", "main")
    assert _current_branch(repo) == "main"


def test_noop_for_detached_head_marker(repo, monkeypatch):
    _patch_syntax_guard(monkeypatch, ok=True)
    _restore_feature_branch_after_update(GIT, repo, "HEAD", "main")
    assert _current_branch(repo) == "main"


def test_conflict_falls_back_to_target_and_preserves_branch(repo, monkeypatch):
    _patch_syntax_guard(monkeypatch, ok=True)
    # Make feature and main conflict on the same file
    _git(repo, "checkout", "feature")
    (repo / "base.txt").write_text("feature version\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", "feature edits base")
    feature_sha = _git(repo, "rev-parse", "feature").stdout.strip()

    _git(repo, "checkout", "main")
    (repo / "base.txt").write_text("main version\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", "main edits base")

    _restore_feature_branch_after_update(GIT, repo, "feature", "main")

    # Fell back to main; feature branch untouched (no half-done rebase)
    assert _current_branch(repo) == "main"
    assert _git(repo, "rev-parse", "feature").stdout.strip() == feature_sha
    # No rebase in progress
    assert not (repo / ".git" / "rebase-merge").exists()
    assert not (repo / ".git" / "rebase-apply").exists()


def test_syntax_guard_failure_returns_to_target(repo, monkeypatch):
    _patch_syntax_guard(monkeypatch, ok=False)
    _restore_feature_branch_after_update(GIT, repo, "feature", "main")
    assert _current_branch(repo) == "main"
    # The rebase itself completed — the branch now contains upstream — but
    # HEAD stays on the safe target because the rebased tree can't bootstrap.
    merge_base = _git(repo, "merge-base", "feature", "main").stdout.strip()
    assert merge_base == _git(repo, "rev-parse", "main").stdout.strip()


def test_dirty_tree_autostash(repo, monkeypatch):
    """Stash restore in the update flow may leave uncommitted edits on the
    target; --autostash must carry them through the rebase."""
    _patch_syntax_guard(monkeypatch, ok=True)
    (repo / "local-edit.txt").write_text("uncommitted\n")

    _restore_feature_branch_after_update(GIT, repo, "feature", "main")

    assert _current_branch(repo) == "feature"
    assert (repo / "local-edit.txt").read_text() == "uncommitted\n"
