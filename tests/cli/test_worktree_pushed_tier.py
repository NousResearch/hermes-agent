"""Pushed-branch tier of the worktree pruners (#A, Aug 2026).

Behavior contract: a clean worktree whose branch head EXACTLY matches what
origin holds (open-PR lane on a single-branch-refspec install) gets its TREE
reaped while its BRANCH ref survives — including surviving the
orphaned-branch pass. Anything not provably on origin stays preserved.

Uses a real local bare repo as `origin` so `git ls-remote`/push are genuine,
no network and no mocks.
"""

import os
import subprocess
import time

import pytest


def _run(args, cwd):
    return subprocess.run(
        args, cwd=cwd, capture_output=True, text=True, encoding="utf-8"
    )


@pytest.fixture
def repo_with_bare_origin(tmp_path):
    """A working clone whose origin is a real local bare repo.

    Mirrors the managed-install fetch config: single-branch refspec for main
    only, so pushed side branches never gain refs/remotes/origin/<branch>
    entries — the exact condition that made the old pruner misread pushed PR
    lanes as 'unpushed'.
    """
    bare = tmp_path / "origin.git"
    bare.mkdir()
    _run(["git", "init", "--bare", "--initial-branch=main"], bare)

    repo = tmp_path / "clone"
    repo.mkdir()
    _run(["git", "init", "--initial-branch=main"], repo)
    _run(["git", "config", "user.email", "test@test.com"], repo)
    _run(["git", "config", "user.name", "Test"], repo)
    (repo / "README.md").write_text("# repo\n")
    _run(["git", "add", "."], repo)
    _run(["git", "commit", "-m", "init"], repo)
    _run(["git", "remote", "add", "origin", str(bare)], repo)
    # Managed-install shape: fetch refspec covers main ONLY.
    _run(
        ["git", "config", "remote.origin.fetch",
         "+refs/heads/main:refs/remotes/origin/main"],
        repo,
    )
    _run(["git", "push", "-u", "origin", "main"], repo)
    _run(["git", "fetch", "origin"], repo)
    return repo


def _age(path, hours=100):
    t = time.time() - hours * 3600
    os.utime(path, (t, t))


def _mk_worktree(repo, name, branch, commit=True, push=False, extra_commit_after_push=False):
    """Create .worktrees/<name> on <branch> with an optional pushed commit."""
    (repo / ".worktrees").mkdir(exist_ok=True)
    p = repo / ".worktrees" / name
    _run(["git", "worktree", "add", str(p), "-b", branch, "HEAD"], repo)
    if commit:
        (p / f"{name}.txt").write_text("payload\n")
        _run(["git", "add", f"{name}.txt"], p)
        _run(["git", "commit", "-m", f"work on {name}"], p)
    if push:
        # Plain push — deliberately NO refs/remotes/* tracking ref appears,
        # because the fetch refspec only covers main.
        result = _run(["git", "push", "origin", branch], p)
        assert result.returncode == 0, result.stderr
    if extra_commit_after_push:
        (p / "later.txt").write_text("post-push work\n")
        _run(["git", "add", "later.txt"], p)
        _run(["git", "commit", "-m", "post-push commit"], p)
    _age(p)
    return p


def _branch_exists(repo, branch):
    return _run(
        ["git", "rev-parse", "--verify", "--quiet", f"refs/heads/{branch}"], repo
    ).returncode == 0


class TestFetchRemoteBranchHeads:
    def test_lists_pushed_branches(self, repo_with_bare_origin):
        import cli
        repo = repo_with_bare_origin
        _mk_worktree(repo, "hermes-a", "hermes/hermes-a", push=True)
        heads = cli._fetch_remote_branch_heads(str(repo))
        assert heads is not None
        assert "main" in heads
        assert "hermes/hermes-a" in heads

    def test_unreachable_remote_returns_none(self, tmp_path):
        import cli
        repo = tmp_path / "r"
        repo.mkdir()
        _run(["git", "init"], repo)
        _run(["git", "remote", "add", "origin", str(tmp_path / "missing.git")], repo)
        assert cli._fetch_remote_branch_heads(str(repo)) is None


class TestBranchPushedExact:
    def test_exact_match_true(self, repo_with_bare_origin):
        import cli
        repo = repo_with_bare_origin
        wt = _mk_worktree(repo, "hermes-x", "hermes/hermes-x", push=True)
        heads = cli._fetch_remote_branch_heads(str(repo))
        assert cli._worktree_branch_pushed_exact(str(wt), heads) is True

    def test_local_ahead_of_push_false(self, repo_with_bare_origin):
        import cli
        repo = repo_with_bare_origin
        wt = _mk_worktree(
            repo, "hermes-y", "hermes/hermes-y",
            push=True, extra_commit_after_push=True,
        )
        heads = cli._fetch_remote_branch_heads(str(repo))
        assert cli._worktree_branch_pushed_exact(str(wt), heads) is False

    def test_never_pushed_false(self, repo_with_bare_origin):
        import cli
        repo = repo_with_bare_origin
        wt = _mk_worktree(repo, "hermes-z", "hermes/hermes-z", push=False)
        heads = cli._fetch_remote_branch_heads(str(repo))
        assert cli._worktree_branch_pushed_exact(str(wt), heads) is False

    def test_none_heads_false(self, repo_with_bare_origin):
        import cli
        repo = repo_with_bare_origin
        wt = _mk_worktree(repo, "hermes-n", "hermes/hermes-n", push=True)
        assert cli._worktree_branch_pushed_exact(str(wt), None) is False
        assert cli._worktree_branch_pushed_exact(str(wt), {}) is False


class TestStartupPrunerPushedTier:
    """cli._prune_stale_worktrees against the real repo fixture."""

    def test_pushed_lane_tree_reaped_branch_kept(self, repo_with_bare_origin):
        import cli
        repo = repo_with_bare_origin
        wt = _mk_worktree(repo, "hermes-pushed", "hermes/hermes-pushed", push=True)

        cli._prune_stale_worktrees(str(repo))

        assert not wt.exists(), (
            "pushed-as-is open-PR lane must be reaped (checkout is redundant)"
        )
        assert _branch_exists(repo, "hermes/hermes-pushed"), (
            "branch ref must survive the reap AND the orphaned-branch pass"
        )

    def test_unpushed_lane_still_preserved(self, repo_with_bare_origin):
        import cli
        repo = repo_with_bare_origin
        wt = _mk_worktree(repo, "hermes-unpushed", "hermes/hermes-unp", push=False)

        cli._prune_stale_worktrees(str(repo))

        assert wt.exists(), "never-pushed unique work must survive, any age"

    def test_ahead_of_pushed_lane_preserved(self, repo_with_bare_origin):
        import cli
        repo = repo_with_bare_origin
        wt = _mk_worktree(
            repo, "hermes-ahead", "hermes/hermes-ahead",
            push=True, extra_commit_after_push=True,
        )

        cli._prune_stale_worktrees(str(repo))

        assert wt.exists(), (
            "local commits beyond the pushed head are unpushed work — preserve"
        )

    def test_pushed_but_dirty_preserved(self, repo_with_bare_origin):
        import cli
        repo = repo_with_bare_origin
        wt = _mk_worktree(repo, "hermes-pdirty", "hermes/hermes-pdirty", push=True)
        (wt / "uncommitted.txt").write_text("in-flight\n")
        _age(wt)

        cli._prune_stale_worktrees(str(repo))

        assert wt.exists(), "dirty guard outranks the pushed tier"

    def test_offline_remote_preserves(self, repo_with_bare_origin, tmp_path):
        import cli
        repo = repo_with_bare_origin
        wt = _mk_worktree(repo, "hermes-off", "hermes/hermes-off", push=True)
        # Sever the remote AFTER pushing: ls-remote now fails -> None ->
        # pushed tier must degrade to preserve.
        _run(["git", "remote", "set-url", "origin", str(tmp_path / "gone.git")], repo)

        cli._prune_stale_worktrees(str(repo))

        assert wt.exists(), "unverifiable push state must fail toward preserve"


class TestAttendedGcPushedTier:
    """worktree_gc audit/reclaim behavior for the pushed tier."""

    def test_audit_verdict_and_reclaim_keeps_branch(self, repo_with_bare_origin):
        import cli  # noqa: F401  (worktree_gc lazily imports cli)
        from hermes_cli import worktree_gc

        repo = repo_with_bare_origin
        wt = _mk_worktree(repo, "salv-lane", "salv/pushed-lane", push=True)

        records = worktree_gc.audit_worktrees(str(repo), with_sizes=False)
        by_name = {r.name: r for r in records}
        assert by_name["salv-lane"].verdict == "reap-keep-branch"
        assert "pushed to origin" in by_name["salv-lane"].reason

        actions = worktree_gc.reclaim_worktrees(str(repo), records=records)
        assert any("salv-lane" in a and "kept" in a for a in actions)
        assert not wt.exists()
        assert _branch_exists(repo, "salv/pushed-lane")

    def test_audit_never_pushed_keeps(self, repo_with_bare_origin):
        import cli  # noqa: F401
        from hermes_cli import worktree_gc

        repo = repo_with_bare_origin
        wt = _mk_worktree(repo, "salv-keep", "salv/never-pushed", push=False)

        records = worktree_gc.audit_worktrees(str(repo), with_sizes=False)
        by_name = {r.name: r for r in records}
        assert by_name["salv-keep"].verdict == "keep"
        worktree_gc.reclaim_worktrees(str(repo), records=records)
        assert wt.exists()


class TestCronWorktreeMaintenance:
    def test_throttle_dispatches_once_per_interval(self, monkeypatch):
        import cron.scheduler as sched

        calls = []
        monkeypatch.setattr(sched, "_last_worktree_maintenance_at", None)
        monkeypatch.setattr(
            sched.threading, "Thread",
            lambda *a, **k: type("T", (), {"start": lambda self: calls.append(k.get("name"))})(),
        )

        sched._maybe_run_worktree_maintenance()
        sched._maybe_run_worktree_maintenance()  # inside interval — no-op

        assert len(calls) == 1

    def test_repo_discovery_requires_worktrees_dir(self, monkeypatch, tmp_path):
        import cron.scheduler as sched

        # A job workdir inside a git repo WITHOUT .worktrees/ is filtered out.
        repo = tmp_path / "jobrepo"
        repo.mkdir()
        _run(["git", "init"], repo)
        monkeypatch.setattr(
            "cron.jobs.load_jobs",
            lambda: [{"workdir": str(repo)}],
        )
        repos = sched._worktree_maintenance_repos()
        assert str(repo) not in repos

        # Adding .worktrees/ makes it eligible.
        (repo / ".worktrees").mkdir()
        repos = sched._worktree_maintenance_repos()
        assert str(repo) in repos

    def test_maintenance_prunes_via_real_pruner(self, repo_with_bare_origin, monkeypatch):
        import cron.scheduler as sched

        repo = repo_with_bare_origin
        wt = _mk_worktree(repo, "hermes-cronreap", "hermes/hermes-cronreap", push=True)

        monkeypatch.setattr(sched, "_last_worktree_maintenance_at", None)
        monkeypatch.setattr(
            sched, "_worktree_maintenance_repos", lambda: [str(repo)]
        )

        sched._maybe_run_worktree_maintenance()
        # The sweep runs on a daemon thread — wait for it (bounded).
        deadline = time.time() + 30
        while wt.exists() and time.time() < deadline:
            time.sleep(0.2)

        assert not wt.exists(), "cron-tick maintenance must run the real pruner"
        assert _branch_exists(repo, "hermes/hermes-cronreap")
