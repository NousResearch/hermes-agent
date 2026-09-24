"""Tests for git worktree isolation (CLI --worktree / -w flag).

Verifies worktree creation, cleanup, .worktreeinclude handling,
.gitignore management, and integration with the CLI.  (#652)
"""

import os
import subprocess
import pytest

from hermes_cli import worktree_ops


@pytest.fixture
def git_repo(tmp_path):
    """Create a temporary git repo for testing."""
    repo = tmp_path / "test-repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"],
        cwd=repo, capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=repo, capture_output=True,
    )
    # Create initial commit (worktrees need at least one commit)
    (repo / "README.md").write_text("# Test Repo\n")
    subprocess.run(["git", "add", "."], cwd=repo, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "Initial commit"],
        cwd=repo, capture_output=True,
    )
    subprocess.run(
        ["git", "remote", "add", "origin", "https://example.com/test-repo.git"],
        cwd=repo, capture_output=True,
    )
    # Add a fake remote ref so cleanup logic sees the initial commit as
    # "pushed" when a remote is configured.
    subprocess.run(
        ["git", "update-ref", "refs/remotes/origin/main", "HEAD"],
        cwd=repo, capture_output=True,
    )
    return repo


@pytest.fixture
def git_repo_no_remote(tmp_path):
    """Create a temporary git repo with no configured remotes."""
    repo = tmp_path / "test-repo-no-remote"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"],
        cwd=repo, capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=repo, capture_output=True,
    )
    (repo / "README.md").write_text("# Test Repo\n")
    subprocess.run(["git", "add", "."], cwd=repo, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "Initial commit"],
        cwd=repo, capture_output=True,
    )
    return repo


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


        # Should not crash — just skip all lines


class TestWorktreeLockReaping:
    """Exercise the REAL cli._prune_stale_worktrees lock/dirty/unpushed logic.

    These import the actual production functions so the behavior contract is
    enforced against the shipped code:

    - live-locked (owning pid running)  -> never reaped, any age
    - dead-locked clean (owning pid gone) -> unlocked + reaped (fixes the
      accumulation bug: `git worktree remove --force` refuses a locked tree)
    - dirty (uncommitted) at >72h        -> preserved
    - unpushed commits at any age        -> preserved
    - clean/unlocked stale               -> reaped (aggressive cleanup intact)
    """

    @staticmethod
    def _age(path, hours):
        import time
        t = time.time() - (hours * 3600)
        os.utime(path, (t, t))

    @staticmethod
    def _mk(cli, repo, name, pid=None, dirty=False, unpushed=False, age_h=100):
        p = repo / ".worktrees" / name
        (repo / ".worktrees").mkdir(exist_ok=True)
        subprocess.run(
            ["git", "worktree", "add", str(p), "-b", f"hermes/{name}", "HEAD"],
            cwd=repo, capture_output=True,
        )
        if pid is not None:
            subprocess.run(
                ["git", "worktree", "lock", "--reason", f"hermes pid={pid}", str(p)],
                cwd=repo, capture_output=True,
            )
        if unpushed:
            (p / "work.txt").write_text("x")
            subprocess.run(["git", "add", "work.txt"], cwd=p, capture_output=True)
            subprocess.run(["git", "commit", "-m", "wip"], cwd=p, capture_output=True)
        if dirty:
            (p / "dirty.txt").write_text("uncommitted")
        TestWorktreeLockReaping._age(p, age_h)
        return p

    def test_live_locked_survives_at_any_age(self, git_repo):
        import cli
        wt = self._mk(cli, git_repo, "hermes-live", pid=os.getpid())
        cli._prune_stale_worktrees(str(git_repo))
        assert wt.exists(), "live-locked worktree (this pid) must never be reaped"

    def test_dead_locked_clean_is_reaped(self, git_repo):
        import cli
        wt = self._mk(cli, git_repo, "hermes-dead", pid=999999)
        # sanity: this is the accumulation bug — remove --force alone can't do it
        assert worktree_ops._worktree_lock_is_live(str(git_repo), str(wt)) == "dead"
        cli._prune_stale_worktrees(str(git_repo))
        assert not wt.exists(), "dead-locked clean worktree should be unlocked + reaped"

    def test_dead_locked_dirty_survives(self, git_repo):
        import cli
        wt = self._mk(cli, git_repo, "hermes-deaddirty", pid=999999, dirty=True)
        cli._prune_stale_worktrees(str(git_repo))
        assert wt.exists(), "dead-locked worktree with uncommitted work must survive"

    def test_dead_locked_unpushed_survives(self, git_repo):
        import cli
        wt = self._mk(cli, git_repo, "hermes-deadunp", pid=999999, unpushed=True)
        cli._prune_stale_worktrees(str(git_repo))
        assert wt.exists(), "dead-locked worktree with unpushed commits must survive"

    def test_unlocked_clean_stale_is_reaped(self, git_repo):
        import cli
        wt = self._mk(cli, git_repo, "hermes-nolock", pid=None)
        cli._prune_stale_worktrees(str(git_repo))
        assert not wt.exists(), "clean unlocked stale worktree should be reaped"

    def test_dirty_survives_over_72h(self, git_repo):
        import cli
        wt = self._mk(cli, git_repo, "hermes-dirty72", pid=None, dirty=True, age_h=100)
        cli._prune_stale_worktrees(str(git_repo))
        assert wt.exists(), "dirty worktree must survive even past the 72h tier"


class TestWorktreeLockPredicate:
    """_worktree_lock_is_live classification (real cli helper)."""

    def _mk_locked(self, repo, name, reason):
        p = repo / ".worktrees" / name
        (repo / ".worktrees").mkdir(exist_ok=True)
        subprocess.run(
            ["git", "worktree", "add", str(p), "-b", f"hermes/{name}", "HEAD"],
            cwd=repo, capture_output=True,
        )
        subprocess.run(
            ["git", "worktree", "lock", "--reason", reason, str(p)],
            cwd=repo, capture_output=True,
        )
        return p

    def test_unlocked_returns_none(self, git_repo):
        p = git_repo / ".worktrees" / "hermes-x"
        (git_repo / ".worktrees").mkdir(exist_ok=True)
        subprocess.run(
            ["git", "worktree", "add", str(p), "-b", "hermes/hermes-x", "HEAD"],
            cwd=git_repo, capture_output=True,
        )
        assert worktree_ops._worktree_lock_is_live(str(git_repo), str(p)) is None


    def test_foreign_lock_reason_returns_dead(self, git_repo):
        p = self._mk_locked(git_repo, "hermes-foreign", "some other tool")
        assert worktree_ops._worktree_lock_is_live(str(git_repo), str(p)) == "dead"

    def test_bad_repo_root_fails_safe_to_live(self, tmp_path):
        # Not a git repo -> git query fails -> must report "live" (never delete)
        assert worktree_ops._worktree_lock_is_live(str(tmp_path), str(tmp_path / "x")) == "live"


class TestWidenedPruner:
    """Behavior contracts for the widened pruner (#all-.worktrees coverage,
    squash-merge escape hatch, kanban exclusion, preserved-work warning).

    Previously only ``hermes-*`` directories were considered, so salvage/
    review/port lanes created with raw ``git worktree add`` accumulated
    forever (real incident: 117 dirs / 26 GB). And squash-merged branches'
    local commits are unreachable from refs/remotes/* forever, so the
    unpushed guard preserved fully-merged scratch trees indefinitely.
    """

    @staticmethod
    def _age(path, hours):
        import time
        t = time.time() - (hours * 3600)
        os.utime(path, (t, t))

    @staticmethod
    def _mk(repo, name, commit=False, dirty=False, age_h=100):
        p = repo / ".worktrees" / name
        (repo / ".worktrees").mkdir(exist_ok=True)
        subprocess.run(
            ["git", "worktree", "add", str(p), "-b", f"wt/{name}", "HEAD"],
            cwd=repo, capture_output=True,
        )
        sha = None
        if commit:
            (p / "work.txt").write_text(f"work for {name}\n")
            subprocess.run(["git", "add", "work.txt"], cwd=p, capture_output=True)
            subprocess.run(["git", "commit", "-m", "wip"], cwd=p, capture_output=True)
            sha = subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=p,
                capture_output=True, text=True,
            ).stdout.strip()
        if dirty:
            (p / "dirty.txt").write_text("uncommitted")
        TestWidenedPruner._age(p, age_h)
        return p, sha

    @staticmethod
    def _merge_upstream(repo, sha):
        """Simulate a squash-merge: land a patch-equivalent commit (different
        SHA) on the branch refs/remotes/origin/main points at."""
        # Distinct committer identity forces a distinct SHA even when the
        # cherry-pick lands in the same second as the original commit.
        subprocess.run(
            ["git", "-c", "user.email=merger@test.com", "-c", "user.name=Merger",
             "cherry-pick", sha],
            cwd=repo, capture_output=True,
        )
        new_head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo,
            capture_output=True, text=True,
        ).stdout.strip()
        assert new_head != sha
        subprocess.run(
            ["git", "update-ref", "refs/remotes/origin/main", new_head],
            cwd=repo, capture_output=True,
        )

    # -- named (non hermes-*) directories are now covered ------------------

    def test_named_clean_stale_tree_is_reaped(self, git_repo):
        import cli
        wt, _ = self._mk(git_repo, "salvage-12345", age_h=80)
        cli._prune_stale_worktrees(str(git_repo))
        assert not wt.exists(), "clean named tree past 72h soft tier should be reaped"

    def test_named_tree_gets_3x_grace(self, git_repo):
        import cli
        wt, _ = self._mk(git_repo, "salvage-fresh", age_h=48)
        cli._prune_stale_worktrees(str(git_repo))
        assert wt.exists(), "named tree under 72h must be kept (3x scratch timeline)"


    # -- squash-merge escape hatch ------------------------------------------

    def test_squash_merged_tree_is_reaped(self, git_repo):
        import cli
        wt, sha = self._mk(git_repo, "hermes-merged", commit=True, age_h=100)
        self._merge_upstream(git_repo, sha)
        assert cli._worktree_has_unpushed_commits(str(wt)), (
            "precondition: commit unreachable from remotes (the leak this fixes)"
        )
        cli._prune_stale_worktrees(str(git_repo))
        assert not wt.exists(), (
            "worktree whose commits are all patch-equivalent upstream is merged "
            "work and should be reaped"
        )


    # -- _worktree_commits_all_merged_upstream unit contracts ----------------


    def test_merged_predicate_fails_safe_without_upstream(self, git_repo_no_remote):
        """No remote: the local trunk is the baseline (a tree at trunk IS merged). No trunk at
        all — detached main checkout, no main/master — leaves nothing to compare against -> False."""
        repo = git_repo_no_remote
        p = repo / ".worktrees" / "hermes-noremote"
        (repo / ".worktrees").mkdir(exist_ok=True)
        subprocess.run(
            ["git", "worktree", "add", str(p), "-b", "wt/noremote", "HEAD"],
            cwd=repo, capture_output=True,
        )
        assert worktree_ops._worktree_commits_all_merged_upstream(str(p)) is True

        trunk = subprocess.run(["git", "branch", "--show-current"], cwd=repo, capture_output=True,
                               text=True).stdout.strip()
        subprocess.run(["git", "checkout", "-q", "--detach"], cwd=repo, capture_output=True)
        subprocess.run(["git", "branch", "-m", trunk, "scratch/not-a-trunk"], cwd=repo, capture_output=True)
        assert worktree_ops._worktree_merge_base_ref(str(p)) is None
        assert worktree_ops._worktree_commits_all_merged_upstream(str(p)) is False
        assert worktree_ops._worktree_has_unpushed_commits(str(p)) is True


    # -- preserved-work warning ----------------------------------------------


class TestMergeVerdictCache:
    """The ``git cherry`` patch-equivalence probe is memoized on disk because it
    dominates ``hermes -w`` startup (~0.2-1.0s per worktree, re-run on every
    launch for every tree preserved as unpushed).

    The invariant that makes caching safe: the verdict is a pure function of the
    ``(base_sha, head_sha)`` range, so a cache entry is only reused while BOTH
    endpoints are unchanged. These tests pin that relationship rather than any
    specific timing or cache contents.
    """

    _age = staticmethod(TestWidenedPruner._age)
    _mk = staticmethod(TestWidenedPruner._mk)
    _merge_upstream = staticmethod(TestWidenedPruner._merge_upstream)

    def test_cache_hit_matches_uncached_verdict(self, git_repo):
        """A cached verdict must equal what the real git call returns."""
        wt, sha = self._mk(git_repo, "hermes-cachehit", commit=True)
        self._merge_upstream(git_repo, sha)

        uncached = worktree_ops._worktree_commits_all_merged_upstream(str(wt))
        cache = {}
        cold = worktree_ops._worktree_commits_all_merged_upstream(str(wt), cache=cache)
        warm = worktree_ops._worktree_commits_all_merged_upstream(str(wt), cache=cache)

        assert cache, "verdict should have been memoized"
        assert uncached is cold is warm is True


    def test_new_commit_invalidates_cached_verdict(self, git_repo):
        """Moving HEAD must not reuse the old entry — the key includes head_sha.

        This is the guard against the cache turning a 'merged, reapable' verdict
        into a stale approval to delete a tree that has since gained real work.
        """
        wt, sha = self._mk(git_repo, "hermes-moves", commit=True)
        self._merge_upstream(git_repo, sha)

        cache = {}
        assert worktree_ops._worktree_commits_all_merged_upstream(str(wt), cache=cache) is True
        key_after_merge = set(cache)

        # New local-only work lands in the worktree.
        (wt / "more.txt").write_text("unmerged work\n")
        subprocess.run(["git", "add", "more.txt"], cwd=wt, capture_output=True)
        subprocess.run(["git", "commit", "-m", "new work"], cwd=wt, capture_output=True)

        assert worktree_ops._worktree_commits_all_merged_upstream(str(wt), cache=cache) is False
        assert set(cache) != key_after_merge, "moved HEAD must produce a new key"


    def test_cache_is_bounded(self, monkeypatch, tmp_path):
        """The cache file must not grow without limit across sessions."""
        from hermes_cli import worktree_ops
        path = tmp_path / "verdicts.json"
        monkeypatch.setattr(worktree_ops, "_worktree_merge_cache_path", lambda: path)
        monkeypatch.setattr(worktree_ops, "_WORKTREE_MERGE_CACHE_MAX", 10)

        worktree_ops._save_worktree_merge_cache({f"sha{i}..sha{i}:20": True for i in range(50)})
        assert len(worktree_ops._load_worktree_merge_cache()) == 10


class TestPruneParallelEquivalence:
    """Classification runs on a thread pool, so verdicts must not depend on
    concurrency: the same mixed board must produce the same survivors whether
    the pool has 1 worker or many."""

    _age = staticmethod(TestWidenedPruner._age)
    _mk = staticmethod(TestWidenedPruner._mk)
    _merge_upstream = staticmethod(TestWidenedPruner._merge_upstream)

    def _board(self, git_repo, tag=""):
        """A board covering every verdict branch: reapable, dirty, unpushed."""
        names = {}
        for i in range(3):
            n = f"hermes-merged{tag}{i}"
            wt, sha = self._mk(git_repo, n, commit=True)
            self._merge_upstream(git_repo, sha)
            names[n] = wt
        for i in range(3):
            n = f"hermes-unpushed{tag}{i}"
            wt, _ = self._mk(git_repo, n, commit=True)
            names[n] = wt
        for i in range(2):
            n = f"hermes-dirty{tag}{i}"
            wt, _ = self._mk(git_repo, n, dirty=True)
            names[n] = wt
        n = f"hermes-fresh{tag}"
        wt, _ = self._mk(git_repo, n, commit=True, age_h=1)
        names[n] = wt
        # Every tree must really exist, otherwise the survivor comparison below
        # is vacuous (a failed `git worktree add` would look like a reap).
        for name, p in names.items():
            assert p.exists(), f"fixture worktree {name} was not created"
        return names

    @staticmethod
    def _kinds(survivors):
        """Reduce survivor names to their kind, dropping the phase tag."""
        out = set()
        for n in survivors:
            for kind in ("merged", "unpushed", "dirty", "fresh"):
                if n.startswith(f"hermes-{kind}"):
                    out.add(kind)
        return out

    def test_single_and_multi_worker_agree(self, git_repo, monkeypatch):
        import cli

        # Phase A — force the serial path (pool sized to 1 worker).
        board = self._board(git_repo, tag="a")
        monkeypatch.setattr(cli.os, "cpu_count", lambda: 1)
        cli._prune_stale_worktrees(str(git_repo))
        serial = self._kinds({n for n, p in board.items() if p.exists()})

        # Phase B — an independent board (distinct branch names so creation
        # can't collide with phase A's leftover refs) run through a real pool.
        try:
            worktree_ops._worktree_merge_cache_path().unlink()
        except Exception:
            pass
        board2 = self._board(git_repo, tag="b")
        monkeypatch.setattr(cli.os, "cpu_count", lambda: 8)
        cli._prune_stale_worktrees(str(git_repo))
        parallel = self._kinds({n for n, p in board2.items() if p.exists()})

        assert parallel == serial, (
            f"pool width changed the outcome: serial={serial} parallel={parallel}"
        )
        # Sanity: the board exercised both outcomes, so the equality above is
        # not comparing two empty (or two identical-because-nothing-ran) sets.
        assert serial == {"dirty", "unpushed", "fresh"}, serial

    def test_pool_failure_falls_back_to_serial(self, git_repo, monkeypatch):
        """A ThreadPoolExecutor failure must not block startup."""
        import cli

        wt, sha = self._mk(git_repo, "hermes-poolfail", commit=True)
        self._merge_upstream(git_repo, sha)

        class _Boom:
            def __init__(self, *a, **kw):
                raise RuntimeError("cannot start thread")

        import concurrent.futures

        monkeypatch.setattr(concurrent.futures, "ThreadPoolExecutor", _Boom)
        monkeypatch.setattr(cli.os, "cpu_count", lambda: 8)

        cli._prune_stale_worktrees(str(git_repo))
        assert not wt.exists(), "serial fallback must still reap the merged tree"


class TestShallowCloneDeepening:
    """Shallow installer clones (`git clone --depth 1`) break the unpushed
    guard: the shallow boundary disconnects an older worktree HEAD from
    origin/*, so `git log HEAD --not --remotes` misreports already-public
    commits as unpushed and every aged worktree is preserved forever
    (real incident: 21 of 25 hermes-* trees stuck on a default install).

    These build a REAL shallow clone over file:// and verify the pruner
    deepens it and reaps the false-positive tree.
    """

    @staticmethod
    def _run(cmd, cwd):
        return subprocess.run(
            cmd, cwd=cwd, capture_output=True, text=True,
        )

    @classmethod
    def _upstream(cls, tmp_path):
        """Upstream repo with one commit (A). Returns its path."""
        up = tmp_path / "upstream"
        up.mkdir()
        cls._run(["git", "init", "-b", "main"], up)
        cls._run(["git", "config", "user.email", "test@test.com"], up)
        cls._run(["git", "config", "user.name", "Test"], up)
        (up / "README.md").write_text("# upstream\n")
        cls._run(["git", "add", "."], up)
        cls._run(["git", "commit", "-m", "A"], up)
        return up

    @classmethod
    def _advance_upstream(cls, up, name):
        (up / f"{name}.txt").write_text(f"{name}\n")
        cls._run(["git", "add", "."], up)
        cls._run(["git", "commit", "-m", name], up)

    @classmethod
    def _shallow_clone(cls, tmp_path, up):
        clone = tmp_path / "shallow-clone"
        subprocess.run(
            ["git", "clone", "--depth", "1", f"file://{up}", str(clone)],
            capture_output=True, text=True,
        )
        cls._run(["git", "config", "user.email", "test@test.com"], clone)
        cls._run(["git", "config", "user.name", "Test"], clone)
        return clone

    @staticmethod
    def _age(path, hours=100):
        import time
        t = time.time() - (hours * 3600)
        os.utime(path, (t, t))

    def _stuck_worktree(self, tmp_path):
        """Build the incident shape: shallow clone at A, worktree at A,
        upstream advances to B, shallow fetch moves origin/main to B.
        Worktree HEAD (A) is now disconnected from origin/main (B)."""
        import cli

        up = self._upstream(tmp_path)
        clone = self._shallow_clone(tmp_path, up)
        assert cli._repo_is_shallow(str(clone)), "fixture must start shallow"

        wt = clone / ".worktrees" / "hermes-shallowstuck"
        (clone / ".worktrees").mkdir()
        self._run(
            ["git", "worktree", "add", str(wt), "-b", "hermes/hermes-shallowstuck", "HEAD"],
            clone,
        )

        self._advance_upstream(up, "B")
        # Same shape as the updater: shallow fetch of the new tip only.
        self._run(["git", "fetch", "--depth", "1", "origin", "main"], clone)
        self._run(
            ["git", "update-ref", "refs/remotes/origin/main", "FETCH_HEAD"], clone,
        )
        self._age(wt)
        return up, clone, wt

    def test_shallow_disconnect_reproduces_false_unpushed(self, tmp_path):
        """Sanity: without deepening, the primitive misreports unpushed."""
        import cli

        _, clone, wt = self._stuck_worktree(tmp_path)
        assert cli._worktree_has_unpushed_commits(str(wt)), (
            "expected the shallow disconnect to look like unpushed commits — "
            "if this stops reproducing, the fixture no longer exercises the bug"
        )

    def test_repo_is_shallow_detection(self, tmp_path, git_repo):
        import cli

        up = self._upstream(tmp_path)
        clone = self._shallow_clone(tmp_path, up)
        assert cli._repo_is_shallow(str(clone)) is True
        assert cli._repo_is_shallow(str(git_repo)) is False
        assert cli._repo_is_shallow(str(tmp_path / "nonexistent")) is False

    def test_deepen_connects_history_and_clears_false_unpushed(self, tmp_path):
        import cli

        _, clone, wt = self._stuck_worktree(tmp_path)
        assert cli._worktree_has_unpushed_commits(str(wt))

        assert worktree_ops._deepen_shallow_repo(str(clone)) is True
        assert not cli._repo_is_shallow(str(clone))
        assert not cli._worktree_has_unpushed_commits(str(wt)), (
            "after deepening, the worktree's HEAD is an ancestor of "
            "origin/main and must no longer count as unpushed"
        )

    def test_pruner_deepens_and_reaps_stuck_worktree(self, tmp_path):
        """E2E: the startup pruner itself unshallows and reaps the tree."""
        import cli

        _, clone, wt = self._stuck_worktree(tmp_path)
        cli._prune_stale_worktrees(str(clone))
        assert not cli._repo_is_shallow(str(clone)), "pruner should deepen"
        assert not wt.exists(), (
            "deepened history proves the tree is merged/public — reap it"
        )

    def test_deepen_offline_fails_soft_and_preserves(self, tmp_path):
        """Unreachable remote: deepen fails, verdicts stay conservative."""
        import cli

        _, clone, wt = self._stuck_worktree(tmp_path)
        # Point origin somewhere that does not exist.
        self._run(
            ["git", "remote", "set-url", "origin", f"file://{tmp_path}/gone"],
            clone,
        )
        assert worktree_ops._deepen_shallow_repo(str(clone), timeout=30) is False
        cli._prune_stale_worktrees(str(clone))
        assert wt.exists(), (
            "offline deepen failure must fall back to preserving the tree"
        )

    def test_deepen_noop_on_full_clone(self, git_repo):
        assert worktree_ops._deepen_shallow_repo(str(git_repo)) is True

    def test_real_unpushed_work_survives_deepening(self, tmp_path):
        """Deepening must not turn genuinely unpushed commits reapable."""
        import cli

        _, clone, wt = self._stuck_worktree(tmp_path)
        (wt / "real-work.txt").write_text("novel\n")
        self._run(["git", "add", "real-work.txt"], wt)
        self._run(["git", "commit", "-m", "real unpushed work"], wt)
        self._age(wt)

        cli._prune_stale_worktrees(str(clone))
        assert wt.exists(), (
            "genuinely unpushed commit must survive even after deepening"
        )


class TestPrMergedEscapeHatch:
    """Rebase-merged PRs whose diff changed during salvage defeat ``git
    cherry`` (patch-id mismatch), so the pruner asks GitHub whether the
    branch's PR is MERGED. These tests stub the ``gh`` binary on PATH — the
    contract is about how the pruner consumes the answer, not about GitHub.

    Contract:
    - gh reports a merged PR + tree is clean  -> reaped
    - gh reports no merged PR                 -> preserved
    - gh missing/failing                      -> preserved (fail safe)
    - dirty tree                              -> never reaped regardless of gh
    """

    _age = staticmethod(TestWorktreeLockReaping._age)

    @staticmethod
    def _mk_diverged(repo, name, age_h=100):
        """Worktree with a commit NOT patch-equivalent to anything upstream."""
        p = repo / ".worktrees" / name
        (repo / ".worktrees").mkdir(exist_ok=True)
        subprocess.run(
            ["git", "worktree", "add", str(p), "-b", f"hermes/{name}", "HEAD"],
            cwd=repo, capture_output=True,
        )
        (p / "salvaged.txt").write_text("diff that was reworked during salvage\n")
        subprocess.run(["git", "add", "salvaged.txt"], cwd=p, capture_output=True)
        subprocess.run(["git", "commit", "-m", "salvaged work"], cwd=p, capture_output=True)
        TestPrMergedEscapeHatch._age(p, age_h)
        return p

    @staticmethod
    def _stub_gh(tmp_path, monkeypatch, stdout='[{"number": 1}]', exit_code=0):
        gh = tmp_path / "bin" / "gh"
        gh.parent.mkdir(parents=True, exist_ok=True)
        gh.write_text(f"#!/bin/sh\nprintf '%s' '{stdout}'\nexit {exit_code}\n")
        gh.chmod(0o755)
        monkeypatch.setenv("PATH", f"{gh.parent}:{os.environ['PATH']}")

    def test_merged_pr_tree_is_reaped(self, git_repo, tmp_path, monkeypatch):
        import cli
        wt = self._mk_diverged(git_repo, "hermes-rebase-merged")
        assert worktree_ops._worktree_commits_all_merged_upstream(str(wt)) is False, (
            "precondition: cherry must NOT consider this merged — the PR "
            "check is the only thing that can reap it"
        )
        self._stub_gh(tmp_path, monkeypatch)
        cli._prune_stale_worktrees(str(git_repo))
        assert not wt.exists(), (
            "clean tree whose branch has a MERGED PR is merged work — reap it"
        )

    def test_no_merged_pr_preserved(self, git_repo, tmp_path, monkeypatch):
        import cli
        wt = self._mk_diverged(git_repo, "hermes-pr-open")
        self._stub_gh(tmp_path, monkeypatch, stdout="[]")
        cli._prune_stale_worktrees(str(git_repo))
        assert wt.exists(), "no merged PR -> still unpushed work, preserve"

    def test_gh_failure_fails_safe(self, git_repo, tmp_path, monkeypatch):
        import cli
        wt = self._mk_diverged(git_repo, "hermes-gh-down")
        self._stub_gh(tmp_path, monkeypatch, stdout="", exit_code=1)
        cli._prune_stale_worktrees(str(git_repo))
        assert wt.exists(), "gh failure must preserve the tree (fail safe)"

    def test_dirty_tree_never_reaped_even_with_merged_pr(
        self, git_repo, tmp_path, monkeypatch
    ):
        import cli
        wt = self._mk_diverged(git_repo, "hermes-dirty-merged")
        (wt / "uncommitted.txt").write_text("in-flight\n")
        self._age(wt, 100)
        self._stub_gh(tmp_path, monkeypatch)
        cli._prune_stale_worktrees(str(git_repo))
        assert wt.exists(), "dirty guard outranks the PR-merged verdict"

    def test_merged_verdict_memoized_by_branch_and_head(
        self, git_repo, tmp_path, monkeypatch
    ):
        wt = self._mk_diverged(git_repo, "hermes-memo")
        self._stub_gh(tmp_path, monkeypatch)
        cache: dict = {}
        assert worktree_ops._worktree_branch_pr_merged(str(wt), cache=cache) is True
        keys = [k for k in cache if k.startswith("pr-merged:")]
        assert len(keys) == 1 and cache[keys[0]] is True
        # Break gh: a cached True verdict must not re-consult it.
        self._stub_gh(tmp_path, monkeypatch, stdout="", exit_code=1)
        assert worktree_ops._worktree_branch_pr_merged(str(wt), cache=cache) is True

    def test_negative_verdict_not_cached(self, git_repo, tmp_path, monkeypatch):
        wt = self._mk_diverged(git_repo, "hermes-nocache-neg")
        self._stub_gh(tmp_path, monkeypatch, stdout="[]")
        cache: dict = {}
        assert worktree_ops._worktree_branch_pr_merged(str(wt), cache=cache) is False
        assert not [k for k in cache if k.startswith("pr-merged:")], (
            "False must not be memoized — the PR can merge later with the "
            "same (branch, head) key"
        )
