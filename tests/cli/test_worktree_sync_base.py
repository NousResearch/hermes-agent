"""Tests for worktree base-ref resolution — branch from the fresh remote tip.

A worktree created off the standalone clone's local ``HEAD`` roots the new
branch on a stale base when that clone lags the remote. ``_resolve_worktree_base``
fetches and branches from the remote tip instead so the worktree starts current.

These tests exercise the REAL ``worktree_ops._resolve_worktree_base`` /
``cli._setup_worktree`` against a real local "remote" repo (so ``git fetch``
works offline in the hermetic sandbox), proving the worktree includes commits
that exist on the remote but not on the stale local HEAD.
"""

import os
import subprocess
import time
from pathlib import Path

import pytest

from hermes_cli import worktree_ops

import cli


def _run(args, cwd):
    return subprocess.run(args, cwd=cwd, capture_output=True, text=True, timeout=30)


def _commit(repo, name, msg):
    (Path(repo) / name).write_text(msg + "\n")
    _run(["git", "add", "."], repo)
    _run(["git", "commit", "-m", msg], repo)


def _head(repo):
    return _run(["git", "rev-parse", "HEAD"], repo).stdout.strip()


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


class TestResolveWorktreeBase:
    def test_resolves_to_fetched_upstream(self, remote_and_clone):
        clone, remote_head, stale_local_head = remote_and_clone
        base_ref, label = worktree_ops._resolve_worktree_base(str(clone))
        # Should resolve to the upstream tracking ref and have fetched it.
        assert base_ref == "origin/main"
        assert "fetched" in label
        # The fetched ref now points at the remote tip, not the stale local HEAD.
        resolved = _run(["git", "rev-parse", base_ref], clone).stdout.strip()
        assert resolved == remote_head
        assert resolved != stale_local_head

    def test_falls_back_to_head_without_remote(self, tmp_path):
        repo = tmp_path / "no-remote"
        repo.mkdir()
        _run(["git", "init"], repo)
        _run(["git", "config", "user.email", "t@t.com"], repo)
        _run(["git", "config", "user.name", "T"], repo)
        _commit(repo, "README.md", "only commit")
        base_ref, label = worktree_ops._resolve_worktree_base(str(repo))
        assert base_ref == "HEAD"
        assert "HEAD" in label


class TestResolveWorktreeBaseStartupCost:
    """The fetch on the -w startup path must be cheap and stall-proof.

    A flaky smart-HTTP connection used to stall ``hermes -w`` for 30-60s
    (30s fetch timeout in step 1, then a SECOND 30s fetch in step 2).
    """

    def test_fresh_fetch_head_skips_the_fetch(self, remote_and_clone, monkeypatch):
        """FETCH_HEAD younger than the freshness window -> no fetch at all."""
        clone, remote_head, _ = remote_and_clone
        # Prime FETCH_HEAD (and the origin/main tracking ref) with a real fetch.
        _run(["git", "fetch", "origin", "main"], clone)

        calls = []
        real_run = subprocess.run

        def spy(args, **kw):
            if isinstance(args, (list, tuple)) and "fetch" in args:
                calls.append(list(args))
            return real_run(args, **kw)

        monkeypatch.setattr(subprocess, "run", spy)
        base_ref, label = worktree_ops._resolve_worktree_base(str(clone))
        assert base_ref == "origin/main"
        assert "fetched" in label and "ago" in label
        assert calls == [], "fresh FETCH_HEAD must skip the network fetch"
        # And the skipped-fetch ref still points at the remote tip.
        resolved = _run(["git", "rev-parse", base_ref], clone).stdout.strip()
        assert resolved == remote_head

    def test_stale_fetch_head_refetches(self, remote_and_clone):
        """FETCH_HEAD older than the window -> a real fetch happens."""
        clone, remote_head, _ = remote_and_clone
        _run(["git", "fetch", "origin", "main"], clone)
        fetch_head = Path(clone) / ".git" / "FETCH_HEAD"
        old = time.time() - 3600
        os.utime(fetch_head, (old, old))
        base_ref, label = worktree_ops._resolve_worktree_base(str(clone))
        assert base_ref == "origin/main"
        assert label == "origin/main (fetched)"

    def test_fetch_timeout_falls_back_to_cached_ref(self, remote_and_clone, monkeypatch):
        """A stalled fetch must yield the locally-cached tracking ref, fast —
        not cascade into a second fetch or blow up."""
        clone, remote_head, stale_local_head = remote_and_clone

        real_run = subprocess.run
        fetches = []

        def stall_fetches(args, **kw):
            if isinstance(args, (list, tuple)) and "fetch" in args:
                fetches.append(list(args))
                raise subprocess.TimeoutExpired(cmd=args, timeout=kw.get("timeout", 5))
            return real_run(args, **kw)

        monkeypatch.setattr(subprocess, "run", stall_fetches)
        start = time.monotonic()
        base_ref, label = worktree_ops._resolve_worktree_base(str(clone))
        elapsed = time.monotonic() - start
        # Cached tracking ref, single fetch attempt, no step-2 cascade.
        assert base_ref == "origin/main"
        assert "cached" in label and "timed out" in label
        assert len(fetches) == 1, "timeout must not cascade into a second fetch"
        assert elapsed < 5, f"fallback path took {elapsed:.1f}s — must be fast"
        # The cached ref is the clone-time origin/main (pre-advance), which is
        # still a valid base — staleness is backstopped by the pre-push gate.
        resolved = _run(["git", "rev-parse", base_ref], clone).stdout.strip()
        assert resolved == stale_local_head

    def test_fetch_timeout_without_cached_ref_falls_back_to_head(
        self, tmp_path, monkeypatch
    ):
        """No usable tracking ref + failed fetch -> HEAD, never a bogus ref."""
        # A repo whose branch claims an upstream that has no tracking ref.
        repo = tmp_path / "broken-upstream"
        repo.mkdir()
        _run(["git", "init"], repo)
        _run(["git", "config", "user.email", "t@t.com"], repo)
        _run(["git", "config", "user.name", "T"], repo)
        _run(["git", "checkout", "-b", "main"], repo)
        _commit(repo, "README.md", "base")
        _run(["git", "remote", "add", "origin", str(tmp_path / "nonexistent.git")], repo)
        _run(
            ["git", "config", "branch.main.remote", "origin"], repo
        )
        _run(
            ["git", "config", "branch.main.merge", "refs/heads/main"], repo
        )
        base_ref, label = worktree_ops._resolve_worktree_base(str(repo))
        assert base_ref == "HEAD"
        assert "HEAD" in label


class TestSetupWorktreeSyncBase:
    def test_sync_true_branches_from_remote_tip(self, remote_and_clone, monkeypatch):
        clone, remote_head, stale_local_head = remote_and_clone
        info = cli._setup_worktree(str(clone), sync_base=True)
        assert info is not None
        # The new worktree's HEAD must be the REMOTE tip, not the stale local one.
        wt_head = _head(info["path"])
        assert wt_head == remote_head, "worktree should start from the fetched remote tip"
        assert wt_head != stale_local_head
        # And it must contain the remote-only file.
        assert (Path(info["path"]) / "feature.txt").exists()

    def test_sync_false_branches_from_local_head(self, remote_and_clone):
        clone, remote_head, stale_local_head = remote_and_clone
        info = cli._setup_worktree(str(clone), sync_base=False)
        assert info is not None
        # Opted out -> branch from the stale local HEAD (old behavior).
        wt_head = _head(info["path"])
        assert wt_head == stale_local_head
        assert not (Path(info["path"]) / "feature.txt").exists()

    def test_default_is_sync_true(self, remote_and_clone):
        """The default path (no sync_base arg) branches from the remote tip."""
        clone, remote_head, _ = remote_and_clone
        info = cli._setup_worktree(str(clone))
        assert info is not None
        assert _head(info["path"]) == remote_head
