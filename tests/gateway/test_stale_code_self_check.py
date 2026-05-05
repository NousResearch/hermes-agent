"""Tests for the gateway stale-code self-check (Issue #17648).

A gateway that survives ``hermes update`` keeps pre-update modules cached
in ``sys.modules``.  Later imports of names added post-update (e.g.
``cfg_get`` from PR #17304) raise ImportError against the stale module
object.

The self-check compares the git HEAD SHA at boot to the current SHA on
disk.  ``hermes update`` always moves HEAD forward via ``git pull``;
agent-driven file edits (Hermes editing ``run_agent.py`` / ``gateway/run.py``
during a self-dev session) never move HEAD — so the SHA signal is free of
the false-positive class that the earlier mtime-based check suffered from.
"""

import os
import time
from pathlib import Path

import pytest

from gateway.run import (
    GatewayRunner,
    _compute_repo_mtime,
    _read_git_head_sha,
    _STALE_CODE_SENTINELS,
    _GIT_SHA_CACHE_TTL_SECS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tmp_repo(tmp_path: Path) -> Path:
    """Create a fake repo with all stale-code sentinel files."""
    for rel in _STALE_CODE_SENTINELS:
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("# test sentinel\n")
    return tmp_path


def _make_git_repo(tmp_path: Path, sha: str = "a" * 40, branch: str = "main") -> Path:
    """Stamp a minimal .git directory so _read_git_head_sha can resolve a SHA.

    We don't run real git — just lay down the files the reader walks
    (.git/HEAD pointing at refs/heads/<branch>, refs/heads/<branch>
    containing the SHA).
    """
    git_dir = tmp_path / ".git"
    git_dir.mkdir(parents=True, exist_ok=True)
    (git_dir / "HEAD").write_text(f"ref: refs/heads/{branch}\n")
    refs_dir = git_dir / "refs" / "heads"
    refs_dir.mkdir(parents=True, exist_ok=True)
    (refs_dir / branch).write_text(f"{sha}\n")
    return tmp_path


def _set_head_sha(repo_root: Path, sha: str, branch: str = "main") -> None:
    """Rewrite the current branch ref to a new SHA (simulates git pull)."""
    (repo_root / ".git" / "refs" / "heads" / branch).write_text(f"{sha}\n")


def _make_runner(
    repo_root: Path,
    *,
    boot_sha: str | None,
    boot_wall: float = None,
    boot_mtime: float = 0.0,
):
    """Bare GatewayRunner with just the stale-check attributes set."""
    if boot_wall is None:
        boot_wall = time.time()
    runner = object.__new__(GatewayRunner)
    runner._repo_root_for_staleness = repo_root
    runner._boot_wall_time = boot_wall
    runner._boot_git_sha = boot_sha
    runner._boot_repo_mtime = boot_mtime
    runner._stale_code_notified = set()
    runner._stale_code_restart_triggered = False
    runner._cached_current_sha = boot_sha
    runner._cached_current_sha_at = boot_wall
    return runner


# ---------------------------------------------------------------------------
# _read_git_head_sha — raw SHA reader
# ---------------------------------------------------------------------------

def test_read_git_head_sha_branch_ref(tmp_path):
    """Resolves ref: refs/heads/<branch> → SHA from refs/heads/<branch>."""
    sha = "b" * 40
    _make_git_repo(tmp_path, sha=sha, branch="main")
    assert _read_git_head_sha(tmp_path) == sha


def test_read_git_head_sha_detached_head(tmp_path):
    """Detached HEAD: .git/HEAD contains the SHA directly."""
    sha = "c" * 40
    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    (git_dir / "HEAD").write_text(f"{sha}\n")
    assert _read_git_head_sha(tmp_path) == sha


def test_read_git_head_sha_packed_refs(tmp_path):
    """Falls back to packed-refs when refs/heads/<branch> is missing."""
    sha = "d" * 40
    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    (git_dir / "HEAD").write_text("ref: refs/heads/main\n")
    # No refs/heads/main file — only packed-refs
    (git_dir / "packed-refs").write_text(
        f"# pack-refs with: peeled fully-peeled sorted\n"
        f"{sha} refs/heads/main\n"
    )
    assert _read_git_head_sha(tmp_path) == sha


def test_read_git_head_sha_worktree_gitdir_file(tmp_path):
    """Worktree: .git is a file with `gitdir: <path>` pointing to the real git dir.

    Real git worktrees store shared refs (refs/heads/*) in the main
    checkout's .git/ and write a ``commondir`` pointer into the
    worktree-gitdir.  The reader must follow commondir to resolve the
    branch ref — this is the layout Hermes dev sessions actually use.
    """
    sha = "e" * 40
    # Main repo layout
    main_repo = tmp_path / "main-repo"
    main_git = main_repo / ".git"
    (main_git / "refs" / "heads").mkdir(parents=True)
    (main_git / "HEAD").write_text("ref: refs/heads/main\n")
    (main_git / "refs" / "heads" / "main").write_text("0" * 40 + "\n")

    # Worktree lives in main-repo/.git/worktrees/<name>/
    worktree_git_dir = main_git / "worktrees" / "feature"
    worktree_git_dir.mkdir(parents=True)
    (worktree_git_dir / "HEAD").write_text("ref: refs/heads/feature\n")
    # commondir points back at the main .git (relative path, "../..")
    (worktree_git_dir / "commondir").write_text("../..\n")
    # Feature branch ref lives in the shared refs/heads
    (main_git / "refs" / "heads" / "feature").write_text(f"{sha}\n")

    # Worktree checkout with .git file pointing at worktree_git_dir
    worktree = tmp_path / "wt"
    worktree.mkdir()
    (worktree / ".git").write_text(f"gitdir: {worktree_git_dir}\n")

    assert _read_git_head_sha(worktree) == sha


def test_read_git_head_sha_worktree_packed_refs_in_common(tmp_path):
    """Worktree + packed-refs in common dir: fallback still resolves."""
    sha = "f" * 40
    main_repo = tmp_path / "main-repo"
    main_git = main_repo / ".git"
    main_git.mkdir(parents=True)
    (main_git / "HEAD").write_text("ref: refs/heads/main\n")
    # packed-refs in the common (main) .git
    (main_git / "packed-refs").write_text(
        f"# pack-refs with: peeled fully-peeled sorted\n"
        f"{sha} refs/heads/feature\n"
    )

    worktree_git_dir = main_git / "worktrees" / "feature"
    worktree_git_dir.mkdir(parents=True)
    (worktree_git_dir / "HEAD").write_text("ref: refs/heads/feature\n")
    (worktree_git_dir / "commondir").write_text("../..\n")

    worktree = tmp_path / "wt"
    worktree.mkdir()
    (worktree / ".git").write_text(f"gitdir: {worktree_git_dir}\n")

    assert _read_git_head_sha(worktree) == sha


def test_read_git_head_sha_no_git_returns_none(tmp_path):
    """No .git dir → None (non-git install, safely disables the check)."""
    assert _read_git_head_sha(tmp_path) is None


def test_read_git_head_sha_malformed_head_returns_none(tmp_path):
    """Empty HEAD file → None (don't loop on corrupt repos)."""
    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    (git_dir / "HEAD").write_text("")
    assert _read_git_head_sha(tmp_path) is None


# ---------------------------------------------------------------------------
# _detect_stale_code — the main regression guard
# ---------------------------------------------------------------------------

def test_detect_stale_code_false_when_sha_unchanged(tmp_path):
    """Boot SHA == current SHA → not stale (no restart)."""
    sha = "a" * 40
    _make_git_repo(tmp_path, sha=sha)
    runner = _make_runner(tmp_path, boot_sha=sha)
    # Force fresh read by expiring the cache
    runner._cached_current_sha_at = 0.0
    assert runner._detect_stale_code() is False


def test_detect_stale_code_true_after_git_pull(tmp_path):
    """Boot SHA != current SHA → stale (hermes update happened)."""
    boot_sha = "a" * 40
    _make_git_repo(tmp_path, sha=boot_sha)
    runner = _make_runner(tmp_path, boot_sha=boot_sha)
    # Simulate git pull moving HEAD forward
    _set_head_sha(tmp_path, "b" * 40)
    runner._cached_current_sha_at = 0.0  # expire cache
    assert runner._detect_stale_code() is True


def test_detect_stale_code_ignores_agent_file_edits(tmp_path):
    """THE CORE REGRESSION: agent edits to source files do NOT trigger restart.

    This is the motivating incident for the SHA-based check.  Under the
    previous mtime-based scheme, any ``patch`` / ``write_file`` call
    against run_agent.py / gateway/run.py / hermes_cli/config.py would
    flip the stale-check to True and force a gateway restart on the
    next message — even though no update actually happened.  SHA
    comparison decouples the two: git HEAD only moves on ``git pull``,
    never on file writes.
    """
    sha = "a" * 40
    _make_git_repo(tmp_path, sha=sha)
    _make_tmp_repo(tmp_path)  # lay down sentinel files too
    runner = _make_runner(tmp_path, boot_sha=sha)

    # Simulate the agent editing run_agent.py and gateway/run.py with
    # mtimes far into the future — exactly the scenario that used to
    # false-positive the old mtime check.
    future = time.time() + 10_000
    for rel in _STALE_CODE_SENTINELS:
        p = tmp_path / rel
        if p.is_file():
            p.write_text("# agent just edited this\n")
            os.utime(p, (future, future))

    # HEAD SHA has NOT moved — check must stay False.
    runner._cached_current_sha_at = 0.0  # expire cache
    assert runner._detect_stale_code() is False


def test_detect_stale_code_false_for_non_git_install(tmp_path):
    """Non-git install (no .git dir) → check disabled, never fires."""
    # No .git dir at all; runner's boot_sha is None
    runner = _make_runner(tmp_path, boot_sha=None)
    # Even if we pretended the current SHA differed, the check should
    # short-circuit on boot_sha=None and return False.
    assert runner._detect_stale_code() is False


def test_detect_stale_code_false_when_no_boot_wall_time(tmp_path):
    """No boot snapshot at all → can't tell → not stale (no restart loop)."""
    runner = _make_runner(tmp_path, boot_sha="a" * 40, boot_wall=0.0)
    assert runner._detect_stale_code() is False


def test_detect_stale_code_handles_disappearing_git_dir(tmp_path):
    """.git vanishes mid-run → current_sha = None → not stale (don't loop)."""
    sha = "a" * 40
    _make_git_repo(tmp_path, sha=sha)
    runner = _make_runner(tmp_path, boot_sha=sha)
    # Nuke the git dir after boot
    import shutil
    shutil.rmtree(tmp_path / ".git")
    runner._cached_current_sha_at = 0.0  # expire cache
    assert runner._detect_stale_code() is False


# ---------------------------------------------------------------------------
# SHA cache
# ---------------------------------------------------------------------------

def test_current_sha_cache_collapses_bursts(tmp_path, monkeypatch):
    """Consecutive calls inside the TTL window reuse the cached SHA."""
    sha = "a" * 40
    _make_git_repo(tmp_path, sha=sha)
    runner = _make_runner(tmp_path, boot_sha=sha)

    read_calls = {"n": 0}
    real_reader = _read_git_head_sha

    def counting_reader(repo_root):
        read_calls["n"] += 1
        return real_reader(repo_root)

    from gateway import run as run_mod
    monkeypatch.setattr(run_mod, "_read_git_head_sha", counting_reader)

    # Force cache expiry so the first call definitely reads
    runner._cached_current_sha_at = 0.0
    runner._current_git_sha_cached()
    first_count = read_calls["n"]

    # Immediate second/third calls should hit cache (no new read)
    runner._current_git_sha_cached()
    runner._current_git_sha_cached()
    assert read_calls["n"] == first_count


def test_current_sha_cache_expires_after_ttl(tmp_path, monkeypatch):
    """After _GIT_SHA_CACHE_TTL_SECS elapses, a fresh read happens."""
    sha = "a" * 40
    _make_git_repo(tmp_path, sha=sha)
    runner = _make_runner(tmp_path, boot_sha=sha)

    read_calls = {"n": 0}
    real_reader = _read_git_head_sha

    def counting_reader(repo_root):
        read_calls["n"] += 1
        return real_reader(repo_root)

    from gateway import run as run_mod
    monkeypatch.setattr(run_mod, "_read_git_head_sha", counting_reader)

    runner._cached_current_sha_at = 0.0
    runner._current_git_sha_cached()
    first = read_calls["n"]

    # Age the cache past the TTL
    runner._cached_current_sha_at = time.time() - (_GIT_SHA_CACHE_TTL_SECS + 1.0)
    runner._current_git_sha_cached()
    assert read_calls["n"] == first + 1


# ---------------------------------------------------------------------------
# _trigger_stale_code_restart — idempotency preserved
# ---------------------------------------------------------------------------

def test_trigger_stale_code_restart_is_idempotent(tmp_path):
    """Calling _trigger_stale_code_restart twice only requests restart once."""
    sha = "a" * 40
    _make_git_repo(tmp_path, sha=sha)
    runner = _make_runner(tmp_path, boot_sha=sha)

    calls = []

    def fake_request_restart(*, detached=False, via_service=False):
        calls.append((detached, via_service))
        return True

    runner.request_restart = fake_request_restart

    runner._trigger_stale_code_restart()
    runner._trigger_stale_code_restart()
    runner._trigger_stale_code_restart()

    assert len(calls) == 1
    assert runner._stale_code_restart_triggered is True


def test_trigger_stale_code_restart_survives_request_failure(tmp_path):
    """If request_restart raises, we swallow and mark as triggered anyway."""
    sha = "a" * 40
    _make_git_repo(tmp_path, sha=sha)
    runner = _make_runner(tmp_path, boot_sha=sha)

    def boom(*, detached=False, via_service=False):
        raise RuntimeError("no event loop")

    runner.request_restart = boom

    # Should not raise
    runner._trigger_stale_code_restart()

    # Marked triggered so we don't retry on every subsequent message
    assert runner._stale_code_restart_triggered is True


# ---------------------------------------------------------------------------
# Class-level defaults — tests that build bare runners via object.__new__
# ---------------------------------------------------------------------------

def test_class_level_defaults_prevent_uninitialized_access():
    """Partial construction via object.__new__ must not crash _detect_stale_code."""
    runner = object.__new__(GatewayRunner)
    # Don't set any instance attrs — class-level defaults should kick in
    runner._repo_root_for_staleness = Path(".")
    # _boot_wall_time / _boot_git_sha fall through to class defaults
    # (0.0 and None respectively)
    assert runner._detect_stale_code() is False
    # _stale_code_restart_triggered falls through to class default (False)
    assert runner._stale_code_restart_triggered is False


# ---------------------------------------------------------------------------
# Legacy mtime reader kept for compatibility — light sanity check only
# ---------------------------------------------------------------------------

def test_compute_repo_mtime_still_returns_newest(tmp_path):
    """_compute_repo_mtime remains available for any legacy callers."""
    repo = _make_tmp_repo(tmp_path)

    baseline = time.time() - 100
    for rel in _STALE_CODE_SENTINELS:
        os.utime(repo / rel, (baseline, baseline))

    newer = time.time()
    os.utime(repo / "hermes_cli/config.py", (newer, newer))

    result = _compute_repo_mtime(repo)
    assert abs(result - newer) < 1.0


def test_compute_repo_mtime_missing_files_returns_zero(tmp_path):
    """Legacy sanity: missing sentinels → 0.0."""
    assert _compute_repo_mtime(tmp_path) == 0.0
