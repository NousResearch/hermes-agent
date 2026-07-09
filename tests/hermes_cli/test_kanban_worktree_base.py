"""Tests for the kanban dispatch path branching worktrees from the fresh tip.

The interactive ``hermes -w`` path already branches new worktrees from the
freshly-fetched remote tip (``cli._setup_worktree`` → ``_resolve_worktree_base``).
The kanban DISPATCH path (``kanban_db._ensure_git_worktree``) historically
hardcoded ``HEAD`` for the new-branch case, so a dispatched card branched from
the standalone clone's (possibly stale) local ``HEAD`` — rooting the branch on
an old merge base and surfacing later as textual merge conflicts against a
moved ``origin/main``.

These tests exercise the REAL ``kanban_db._ensure_git_worktree`` against a real
local "remote" repo (so ``git fetch`` works offline in the hermetic sandbox),
proving the new-branch worktree includes commits that exist on the remote but
not on the stale local ``HEAD`` — and that it still fails soft to ``HEAD`` when
the remote is unreachable.
"""

import subprocess
from pathlib import Path

import pytest

from hermes_cli import kanban_db


def _run(args, cwd):
    return subprocess.run(args, cwd=cwd, capture_output=True, text=True, timeout=30)


def _commit(repo, name, msg):
    (Path(repo) / name).write_text(msg + "\n")
    _run(["git", "add", "."], repo)
    _run(["git", "commit", "-m", msg], repo)


def _head(repo):
    return _run(["git", "rev-parse", "HEAD"], repo).stdout.strip()


@pytest.fixture(autouse=True)
def _sync_on(monkeypatch):
    """Default every test to worktree_sync ON unless it overrides the flag.

    ``_ensure_git_worktree`` reads ``worktree_sync`` via
    ``hermes_cli.config.load_config``; pin it so the test does not depend on
    the developer's real ~/.hermes/config.yaml.
    """
    monkeypatch.setattr(
        "hermes_cli.config.load_config", lambda: {"worktree_sync": True}
    )


@pytest.fixture
def remote_and_clone(tmp_path):
    """A bare 'remote' + a clone that is intentionally BEHIND the remote.

    Returns (clone_path, remote_head_sha, stale_local_head_sha).
    """
    remote = tmp_path / "remote.git"
    seed = tmp_path / "seed"
    seed.mkdir()
    _run(["git", "init"], seed)
    _run(["git", "config", "user.email", "t@t.com"], seed)
    _run(["git", "config", "user.name", "T"], seed)
    # Pin the seed repo's branch name so push + remote default are 'main'.
    _run(["git", "checkout", "-b", "main"], seed)
    _commit(seed, "README.md", "base commit")
    _run(["git", "init", "--bare", str(remote)], tmp_path)
    _run(["git", "remote", "add", "origin", str(remote)], seed)
    _run(["git", "push", "origin", "main"], seed)
    # Set the bare remote's default branch so a clone gets origin/HEAD ->
    # origin/main and a tracking branch (mirrors a real GitHub remote).
    _run(["git", "symbolic-ref", "HEAD", "refs/heads/main"], remote)

    # Clone it (this clone tracks origin/main).
    clone = tmp_path / "clone"
    _run(["git", "clone", str(remote), str(clone)], tmp_path)
    _run(["git", "config", "user.email", "t@t.com"], clone)
    _run(["git", "config", "user.name", "T"], clone)
    stale_local_head = _head(clone)

    # Advance the REMOTE past the clone (simulating other merges landing on
    # main while this clone sat stale).
    _commit(seed, "feature.txt", "remote-only commit")
    _run(["git", "push", "origin", "main"], seed)
    remote_head = _head(seed)

    assert remote_head != stale_local_head
    return clone, remote_head, stale_local_head


class TestEnsureGitWorktreeSyncBase:
    def test_new_branch_branches_from_remote_tip(self, remote_and_clone):
        """worktree_sync on: a NEW-branch dispatch worktree starts from the
        fetched remote tip, not the stale local HEAD — so it contains the
        remote-only commit."""
        clone, remote_head, stale_local_head = remote_and_clone
        target = clone / ".worktrees" / "card-1"
        kanban_db._ensure_git_worktree(clone, target, "wt/card-1")

        assert target.exists()
        wt_head = _head(target)
        assert wt_head == remote_head, (
            "dispatched worktree should start from the fetched remote tip"
        )
        assert wt_head != stale_local_head
        # And it must contain the remote-only file.
        assert (target / "feature.txt").exists()

    def test_sync_disabled_branches_from_local_head(self, remote_and_clone, monkeypatch):
        """worktree_sync off: opt back into the old behavior — branch from the
        stale local HEAD (no remote-only commit)."""
        monkeypatch.setattr(
            "hermes_cli.config.load_config", lambda: {"worktree_sync": False}
        )
        clone, remote_head, stale_local_head = remote_and_clone
        target = clone / ".worktrees" / "card-off"
        kanban_db._ensure_git_worktree(clone, target, "wt/card-off")

        assert _head(target) == stale_local_head
        assert not (target / "feature.txt").exists()


class TestEnsureGitWorktreeOfflineFallback:
    def test_unreachable_remote_falls_back_to_head(self, remote_and_clone):
        """A fetch hiccup must never hard-fail worktree creation: with the
        remote pointed at a nonexistent path, resolution/fetch can't reach the
        tip, so creation falls back to local HEAD and still succeeds."""
        clone, _remote_head, stale_local_head = remote_and_clone
        # Break the remote so `git fetch` inside _resolve_worktree_base fails.
        _run(["git", "remote", "set-url", "origin",
              str(clone.parent / "does-not-exist.git")], clone)

        target = clone / ".worktrees" / "card-offline"
        # Must not raise.
        kanban_db._ensure_git_worktree(clone, target, "wt/card-offline")

        assert target.exists()
        # Fetch failed, so the base falls back to the stale local HEAD.
        assert _head(target) == stale_local_head

    def test_unusable_base_ref_retries_from_head(self, remote_and_clone, monkeypatch):
        """If base resolution yields a ref that ``git worktree add`` can't use
        (e.g. a partial fetch left it dangling), creation retries once from
        local HEAD rather than hard-failing — mirrors _setup_worktree."""
        clone, _remote_head, stale_local_head = remote_and_clone
        # Force a base ref that does not exist so the first `worktree add` fails.
        monkeypatch.setattr(
            kanban_db, "_resolve_worktree_base",
            lambda root: ("origin/does-not-exist", "bogus"),
        )
        target = clone / ".worktrees" / "card-retry"
        # Must not raise; the HEAD retry succeeds.
        kanban_db._ensure_git_worktree(clone, target, "wt/card-retry")

        assert target.exists()
        assert _head(target) == stale_local_head

    def test_no_remote_repo_still_creates(self, tmp_path):
        """A repo with no remote at all resolves base to HEAD and creates the
        new-branch worktree without error."""
        repo = tmp_path / "no-remote"
        repo.mkdir()
        _run(["git", "init"], repo)
        _run(["git", "config", "user.email", "t@t.com"], repo)
        _run(["git", "config", "user.name", "T"], repo)
        _commit(repo, "README.md", "only commit")
        head = _head(repo)

        target = repo / ".worktrees" / "card-lonely"
        kanban_db._ensure_git_worktree(repo, target, "wt/card-lonely")

        assert target.exists()
        assert _head(target) == head


class TestEnsureGitWorktreeExistingBranch:
    def test_existing_branch_resume_unchanged(self, remote_and_clone):
        """The existing-branch resume path is untouched by the base-freshness
        change: when the branch already exists, the worktree checks it out
        as-is (no new base is picked)."""
        clone, _remote_head, stale_local_head = remote_and_clone
        # Create the branch first (off the stale local HEAD), with a commit that
        # is unique to it so we can prove the worktree checks out THIS branch.
        _run(["git", "branch", "wt/existing", "HEAD"], clone)
        # Put a commit on the branch via a throwaway worktree, then remove it,
        # so the branch tip diverges from both HEAD and the remote tip.
        tmp_wt = clone / ".worktrees" / "seed-existing"
        _run(["git", "-C", str(clone), "worktree", "add", str(tmp_wt), "wt/existing"], clone)
        _commit(tmp_wt, "branch_only.txt", "commit only on wt/existing")
        branch_tip = _head(tmp_wt)
        _run(["git", "-C", str(clone), "worktree", "remove", "--force", str(tmp_wt)], clone)

        assert branch_tip != stale_local_head

        target = clone / ".worktrees" / "card-resume"
        kanban_db._ensure_git_worktree(clone, target, "wt/existing")

        assert target.exists()
        assert _head(target) == branch_tip, (
            "resume path should check out the existing branch tip as-is"
        )
        assert (target / "branch_only.txt").exists()

    def test_already_materialized_worktree_is_noop(self, remote_and_clone):
        """If the target is already a linked worktree of the same repo,
        _ensure_git_worktree returns early without re-adding (idempotent)."""
        clone, remote_head, _stale = remote_and_clone
        target = clone / ".worktrees" / "card-idem"
        kanban_db._ensure_git_worktree(clone, target, "wt/idem")
        first_head = _head(target)
        # Second call with the same target+branch must not raise.
        kanban_db._ensure_git_worktree(clone, target, "wt/idem")
        assert _head(target) == first_head
