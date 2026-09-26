"""Tests for hermes_cli.gitlock — stale git lock recovery + ancestry probe.

These cover the two failure modes that produced the false "update available"
notification and the hard ``update --check`` failure after a crashed fetch on
a shallow clone:

1. A stale ``.git/shallow.lock`` makes every later ``git fetch`` fail with
   "File exists" unless cleared.
2. On a shallow clone the update check compares tip SHAs, so local
   cherry-picks on top of the remote tip look like "update available" even
   though HEAD already contains the remote tip.

The module's safety rules are also pinned: a *young* lock or a *running git
process* must never be cleared.
"""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

import pytest

from hermes_cli.gitlock import (
    LOCK_NAMES,
    STALE_LOCK_MIN_AGE_SECONDS,
    clear_stale_git_locks,
)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A real, tiny git repo with two commits (no network)."""
    root = tmp_path / "repo"
    root.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=root, check=True)
    (root / "a.txt").write_text("one\n")
    subprocess.run(["git", "add", "a.txt"], cwd=root, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "first"], cwd=root, check=True)
    (root / "b.txt").write_text("two\n")
    subprocess.run(["git", "add", "b.txt"], cwd=root, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "second"], cwd=root, check=True)
    return root


def _touch(path: Path, age_seconds: float) -> None:
    """Create (or truncate) a file and backdate its mtime."""
    path.touch()
    old = time.time() - age_seconds
    os.utime(path, (old, old))


@pytest.fixture
def no_git_running(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin the process guard: pretend no git process is running.

    The removal tests exercise the sweep itself, not the guard.  Without
    this pin they are flaky on CI: the parallel per-file test runner is
    almost always running a real ``git`` subprocess somewhere, the
    ``pgrep -x git`` probe hits, and the sweep (correctly) refuses to
    remove anything — failing the assertion for reasons unrelated to the
    code under test.  The guard's own behavior is pinned separately in
    :func:`test_clear_skips_sweep_while_git_running`.
    """
    import hermes_cli.gitlock as gitlock

    monkeypatch.setattr(gitlock, "_git_proc_running", lambda: False)


def test_clear_removes_stale_shallow_lock(repo: Path, no_git_running: None) -> None:
    _touch(repo / ".git" / "shallow.lock", STALE_LOCK_MIN_AGE_SECONDS + 60)
    removed = clear_stale_git_locks(repo)
    assert str(repo / ".git" / "shallow.lock") in removed
    assert not (repo / ".git" / "shallow.lock").exists()


def test_clear_removes_all_stale_lock_kinds(repo: Path, no_git_running: None) -> None:
    for name in LOCK_NAMES:
        _touch(repo / ".git" / name, STALE_LOCK_MIN_AGE_SECONDS + 60)
    removed = clear_stale_git_locks(repo)
    assert len(removed) == len(LOCK_NAMES)
    for name in LOCK_NAMES:
        assert not (repo / ".git" / name).exists()


def test_clear_skips_sweep_while_git_running(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A running git process must block the sweep even for stale locks."""
    import hermes_cli.gitlock as gitlock

    monkeypatch.setattr(gitlock, "_git_proc_running", lambda: True)
    _touch(repo / ".git" / "shallow.lock", STALE_LOCK_MIN_AGE_SECONDS + 60)
    removed = clear_stale_git_locks(repo)
    assert removed == []
    assert (repo / ".git" / "shallow.lock").exists()


def test_clear_keeps_young_lock(repo: Path, no_git_running: None) -> None:
    _touch(repo / ".git" / "shallow.lock", 1)  # 1 second old — presumably live
    removed = clear_stale_git_locks(repo)
    assert removed == []
    assert (repo / ".git" / "shallow.lock").exists()


def test_clear_noop_on_non_repo(tmp_path: Path) -> None:
    bare = tmp_path / "not-a-repo"
    bare.mkdir()
    assert clear_stale_git_locks(bare) == []


def test_clear_noop_with_no_locks(repo: Path) -> None:
    assert clear_stale_git_locks(repo) == []


# ---- Partial-clone pack-objects fetch crash (#124272) ----
#
# On a partial clone (tree:0 filter) git 2.53/2.54 crashes EVERY fetch: index-pack's
# repack_local_links feeds pack-objects --exclude-promisor-objects-best-effort and
# pack-objects BUG()s (SIGABRT) on a legitimately missing promisor object. The crash is
# deterministic, so the recovery is one retry with the promisor machinery disabled (the
# reporter's verified workaround). These pin the recognizer against look-alike failures
# and the retry contract: retry exactly once, only on this crash, args untouched.

from subprocess import CompletedProcess  # noqa: E402

from hermes_cli.gitlock import (  # noqa: E402
    fetch_with_partial_clone_recovery,
    is_partial_clone_pack_objects_crash,
)

_CRASH_STDERR = (
    "remote: Enumerating objects: 12, done.\n"
    "BUG: builtin/pack-objects.c:4842: should_include_obj should only be called on existing objects\n"
    "error: pack-objects died of signal 6\n"
    "fatal: could not finish pack-objects to repack local links\n"
    "fatal: index-pack failed\n"
)


def test_crash_recognizer_matches_reported_stderr():
    assert is_partial_clone_pack_objects_crash(_CRASH_STDERR)


def test_crash_recognizer_rejects_unrelated_failures():
    assert not is_partial_clone_pack_objects_crash(
        "fatal: Authentication failed for 'https://github.com/example.git'")
    assert not is_partial_clone_pack_objects_crash(
        "error: pack-objects died of signal 6")  # one marker alone is not the crash
    assert not is_partial_clone_pack_objects_crash("")
    assert not is_partial_clone_pack_objects_crash(None)


def test_recovery_retries_once_with_promisor_disabled():
    calls = []

    def runner(git_cmd, args):
        calls.append((list(git_cmd), list(args)))
        if len(calls) == 1:
            return CompletedProcess(git_cmd + args, 1, stdout="", stderr=_CRASH_STDERR)
        return CompletedProcess(git_cmd + args, 0, stdout="", stderr="")

    result = fetch_with_partial_clone_recovery(runner, ["git"], ["fetch", "origin", "main"])

    assert [args for _, args in calls] == [["fetch", "origin", "main"]] * 2
    assert calls[1][0] == ["git", "-c", "remote.origin.promisor="]
    assert result.returncode == 0


def test_recovery_passes_through_success_and_unrelated_failure():
    calls = []

    def failing_runner(git_cmd, args):
        calls.append((list(git_cmd), list(args)))
        return CompletedProcess(git_cmd + args, 1, stdout="", stderr="fatal: Permission denied (publickey)")

    result = fetch_with_partial_clone_recovery(failing_runner, ["git"], ["fetch", "origin", "main"])
    assert result.returncode == 1
    assert calls == [(["git"], ["fetch", "origin", "main"])]

    calls.clear()
    ok_runner = lambda gc, a: (calls.append((list(gc), list(a))),
                               CompletedProcess(gc + a, 0, stdout="", stderr=""))[-1]
    result = fetch_with_partial_clone_recovery(ok_runner, ["git"], ["fetch", "--no-tags", "origin", "abc123"])
    assert result.returncode == 0
    assert calls == [(["git"], ["fetch", "--no-tags", "origin", "abc123"])]
