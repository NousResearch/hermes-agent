"""Tests for git self-heal: atomic worktree-add failure cleanup + pack maintenance.

Regression for the Aug 2026 `hermes -w` timeout incident: 39 accumulated packs
slowed object lookups until `git worktree add` blew its 30s timeout, and the
timed-out add left a partially-materialized worktree plus a LOCKED admin entry
(lock pid = the live hermes process), poisoning every retry.

Two behaviors:
1. `_cleanup_failed_worktree_add` — removes the partial dir, the admin entry
   (even when LOCKED), and the orphaned branch, so a failed add is atomic.
2. `_maintain_pack_health` — repacks when *.pack count reaches the sprawl
   threshold; no-op below it.
"""

import subprocess
from pathlib import Path

import pytest


def _git(cwd, *args, check=True):
    return subprocess.run(
        ["git", *args], cwd=str(cwd), capture_output=True, text=True, check=check
    )


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q", "-b", "main")
    _git(root, "config", "user.email", "t@t")
    _git(root, "config", "user.name", "t")
    (root / "f.txt").write_text("x\n")
    _git(root, "add", "-A")
    _git(root, "commit", "-qm", "init")
    return root


class TestCleanupFailedWorktreeAdd:
    def _simulate_timed_out_add(self, repo):
        """Reproduce git's post-timeout wreckage: partial dir + LOCKED admin
        entry + branch. Built from a real add then re-locking + damaging it,
        which yields the same on-disk shape as a killed `worktree add`."""
        wt = repo / ".worktrees" / "hermes-dead00"
        _git(repo, "worktree", "add", str(wt), "-b", "hermes/hermes-dead00")
        # Live-pid lock, exactly what `hermes -w` writes before the checkout.
        _git(repo, "worktree", "lock", str(wt), "--reason", "hermes pid=999999")
        # Partial materialization: gut the checkout but keep the dir + .git file.
        for child in wt.iterdir():
            if child.name != ".git":
                child.unlink()
        return wt

    def test_sweeps_dir_admin_entry_and_branch(self, repo):
        from cli import _cleanup_failed_worktree_add

        wt = self._simulate_timed_out_add(repo)
        admin = repo / ".git" / "worktrees" / "hermes-dead00"
        assert admin.exists() and (admin / "locked").exists()

        _cleanup_failed_worktree_add(str(repo), wt, "hermes/hermes-dead00")

        assert not wt.exists(), "partial worktree dir must be removed"
        assert not admin.exists(), "LOCKED admin entry must be removed"
        branches = _git(repo, "branch", "--list", "hermes/hermes-dead00").stdout
        assert branches.strip() == "", "orphaned branch must be deleted"

    def test_retry_succeeds_after_cleanup(self, repo):
        """The whole point: the same worktree name is creatable again."""
        from cli import _cleanup_failed_worktree_add

        wt = self._simulate_timed_out_add(repo)
        _cleanup_failed_worktree_add(str(repo), wt, "hermes/hermes-dead00")

        result = _git(
            repo, "worktree", "add", str(wt), "-b", "hermes/hermes-dead00", check=False
        )
        assert result.returncode == 0, f"retry failed: {result.stderr}"

    def test_noop_when_nothing_exists(self, repo):
        """Fail-soft on an error path where git never created anything."""
        from cli import _cleanup_failed_worktree_add

        _cleanup_failed_worktree_add(
            str(repo), repo / ".worktrees" / "never-existed", "hermes/never-existed"
        )  # must not raise


class TestMaintainPackHealth:
    def _pack_count(self, repo):
        return len(list((repo / ".git" / "objects" / "pack").glob("*.pack")))

    def _make_packs(self, repo, n):
        """Create n distinct packs by committing + repacking incrementally."""
        for i in range(n):
            (repo / f"p{i}.txt").write_text(f"{i}\n")
            _git(repo, "add", "-A")
            _git(repo, "commit", "-qm", f"c{i}")
            # `repack` without -a packs only loose objects → one new pack.
            _git(repo, "repack", "-q")
        return self._pack_count(repo)

    def test_repacks_at_threshold(self, repo, monkeypatch):
        import cli

        made = self._make_packs(repo, 6)
        assert made >= 6
        monkeypatch.setattr(cli, "_PACK_SPRAWL_THRESHOLD", 5)

        cli._maintain_pack_health(str(repo))

        after = self._pack_count(repo)
        assert after <= 2, f"expected consolidation, still {after} packs"

    def test_noop_below_threshold(self, repo, monkeypatch):
        import cli

        made = self._make_packs(repo, 3)
        monkeypatch.setattr(cli, "_PACK_SPRAWL_THRESHOLD", 50)

        cli._maintain_pack_health(str(repo))

        assert self._pack_count(repo) == made, "below threshold must be a no-op"

    def test_fail_soft_on_missing_pack_dir(self, tmp_path):
        from cli import _maintain_pack_health

        _maintain_pack_health(str(tmp_path / "not-a-repo"))  # must not raise
