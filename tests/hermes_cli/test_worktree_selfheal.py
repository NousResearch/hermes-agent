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
        from hermes_cli.worktree_ops import _cleanup_failed_worktree_add

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
        from hermes_cli.worktree_ops import _cleanup_failed_worktree_add

        wt = self._simulate_timed_out_add(repo)
        _cleanup_failed_worktree_add(str(repo), wt, "hermes/hermes-dead00")

        result = _git(
            repo, "worktree", "add", str(wt), "-b", "hermes/hermes-dead00", check=False
        )
        assert result.returncode == 0, f"retry failed: {result.stderr}"

    def test_noop_when_nothing_exists(self, repo):
        """Fail-soft on an error path where git never created anything."""
        from hermes_cli.worktree_ops import _cleanup_failed_worktree_add

        _cleanup_failed_worktree_add(
            str(repo), repo / ".worktrees" / "never-existed", "hermes/never-existed"
        )  # must not raise


class TestMaintainPackHealth:
    def _pack_count(self, repo):
        return len(list((repo / ".git" / "objects" / "pack").glob("*.pack")))

    def _make_packs(self, repo, n):
        """Create exactly n distinct packs via git pack-objects (deterministic
        across git versions — incremental `git repack` consolidates small
        packs on newer CI git builds, which made the count nondeterministic)."""
        pack_dir = repo / ".git" / "objects" / "pack"
        pack_dir.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            (repo / f"p{i}.txt").write_text(f"{i}\n")
            _git(repo, "add", "-A")
            _git(repo, "commit", "-qm", f"c{i}")
            sha = _git(repo, "rev-parse", f"HEAD^{{commit}}").stdout.strip()
            # One pack per commit object: pipe the sha into pack-objects.
            subprocess.run(
                ["git", "pack-objects", "-q", str(pack_dir / f"tpack{i}")],
                input=f"{sha}\n", cwd=str(repo), capture_output=True, text=True, check=True,
            )
        return self._pack_count(repo)

    def test_repacks_at_threshold(self, repo, monkeypatch):
        import cli
        from hermes_cli import worktree_ops

        made = self._make_packs(repo, 12)
        # Behavior contract, not a snapshot: the geometric repack leaves a size progression
        # (plus a cruft pack on newer git), so the exact count varies by git build. What must
        # hold: sprawl went DOWN and lookups now go through one multi-pack-index.
        threshold = 2
        monkeypatch.setattr(worktree_ops, "_PACK_SPRAWL_THRESHOLD", threshold)
        assert made > threshold, f"fixture failed to produce sprawl (made={made})"

        cli._maintain_pack_health(str(repo))

        after = self._pack_count(repo)
        assert after < made, f"pack count must strictly decrease (made={made}, after={after})"
        assert (repo / ".git" / "objects" / "pack" / "multi-pack-index").exists()

    def test_noop_below_threshold(self, repo, monkeypatch):
        import cli
        from hermes_cli import worktree_ops

        made = self._make_packs(repo, 2)
        monkeypatch.setattr(worktree_ops, "_PACK_SPRAWL_THRESHOLD", 50)

        cli._maintain_pack_health(str(repo))

        assert self._pack_count(repo) == made, "below threshold must be a no-op"  # same-count contract

    def test_fail_soft_on_missing_pack_dir(self, tmp_path):
        from cli import _maintain_pack_health

        _maintain_pack_health(str(tmp_path / "not-a-repo"))  # must not raise


class TestRepackStampede:
    """Regression for the Sep 2026 shared-clone incident: every ``hermes -w`` launch started its
    own full repack, and a timed-out repack left ``pack-objects`` running for days."""

    def test_one_repack_per_clone_per_interval(self, repo, monkeypatch):
        from hermes_cli import worktree_ops

        monkeypatch.setattr(worktree_ops, "_PACK_SPRAWL_THRESHOLD", 0)
        runs: list = []
        monkeypatch.setattr(worktree_ops, "_run_bounded_repack", lambda root: runs.append(root))

        for _ in range(3):  # three concurrent-ish launches sharing the clone
            worktree_ops._maintain_pack_health(str(repo))
        assert runs == [str(repo)], "N launches inside the interval must produce exactly one repack"

        # A stale stamp (older than the interval) hands the slot to the next launch.
        monkeypatch.setattr(worktree_ops, "_REPACK_MIN_INTERVAL", 0)
        worktree_ops._maintain_pack_health(str(repo))
        assert len(runs) == 2

    @pytest.mark.linux_only
    def test_timeout_kills_the_whole_repack_tree(self, tmp_path, monkeypatch):
        import os
        import time

        from hermes_cli import worktree_ops

        # A stand-in ``git`` that forks a long-lived grandchild, the way repack forks pack-objects.
        shim_dir = tmp_path / "bin"
        shim_dir.mkdir()
        pidfile = tmp_path / "grandchild.pid"
        shim = shim_dir / "git"
        shim.write_text(f"#!/bin/sh\nsleep 300 &\necho $! > {pidfile}\nwait\n")
        shim.chmod(0o755)
        monkeypatch.setenv("PATH", f"{shim_dir}{os.pathsep}{os.environ['PATH']}")
        monkeypatch.setattr(worktree_ops, "_REPACK_TIMEOUT", 1)

        worktree_ops._run_bounded_repack(str(tmp_path))

        grandchild = int(pidfile.read_text().strip())
        deadline = time.time() + 5
        while time.time() < deadline:
            if not Path(f"/proc/{grandchild}").exists():
                return
            time.sleep(0.05)
        subprocess.run(["kill", "-9", str(grandchild)], check=False)
        pytest.fail("pack-objects stand-in survived the repack timeout")
