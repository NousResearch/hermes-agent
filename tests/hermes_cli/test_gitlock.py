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
# On a partial clone git 2.53+ crashes fetches: index-pack's repack_local_links feeds
# pack-objects --exclude-promisor-objects-best-effort the objects outside promisor packs, and
# pack-objects BUG()s (SIGABRT) on the missing objects they lead to. Unmarked packs keep the
# crash coming, so the recovery marks them and retries once. These pin the recognizer against
# look-alike failures and the retry contract: retry exactly once, only on this crash, args untouched.

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

# Git for Windows 2.54 field evidence (#124293): no signal line — the BUG()
# assertion is followed directly by the repack fatal and index-pack failure.
_CRASH_STDERR_WINDOWS = (
    "remote: Enumerating objects: 12, done.\n"
    "BUG: builtin/pack-objects.c:4967: should_include_obj should only be called on existing objects\n"
    "fatal: could not finish pack-objects to repack local links\n"
    "fatal: index-pack failed\n"
)

# git 2.55 builds report the aborted helper as "fetch-pack: invalid index-pack output" and no
# longer print "index-pack failed" (#125138).
_CRASH_STDERR_GIT_255 = (
    "BUG: builtin/pack-objects.c:5004: should_include_obj should only be called on existing objects\n"
    "fatal: fetch-pack: invalid index-pack output\n"
)


def test_crash_recognizer_rejects_unrelated_failures():
    assert not is_partial_clone_pack_objects_crash(
        "fatal: Authentication failed for 'https://github.com/example.git'")
    assert not is_partial_clone_pack_objects_crash(
        "error: pack-objects died of signal 6")  # one marker alone is not the crash
    assert not is_partial_clone_pack_objects_crash(
        "BUG: builtin/pack-objects.c:4842: should_include_obj should only be called on existing objects\n"
        "fatal: index-pack failed\n")  # fingerprint without a terminator line: not this crash
    assert not is_partial_clone_pack_objects_crash(
        "fatal: fetch-pack: invalid index-pack output\n")  # 2.55 wrapper alone is not the crash
    assert not is_partial_clone_pack_objects_crash("")
    assert not is_partial_clone_pack_objects_crash(None)


@pytest.mark.parametrize(
    "crash_stderr",
    [_CRASH_STDERR, _CRASH_STDERR_WINDOWS, _CRASH_STDERR_GIT_255],
    ids=["posix", "windows", "git-255"],
)
def test_recovery_marks_unmarked_packs_and_retries_the_same_fetch(crash_stderr, tmp_path):
    pack_dir = tmp_path / ".git" / "objects" / "pack"
    pack_dir.mkdir(parents=True)
    (pack_dir / "pack-local.pack").write_bytes(b"")
    (pack_dir / "pack-fetched.pack").write_bytes(b"")
    (pack_dir / "pack-fetched.promisor").write_bytes(b"")
    calls = []

    def runner(git_cmd, args):
        calls.append((list(git_cmd), list(args)))
        if len(calls) == 1:
            return CompletedProcess(git_cmd + args, 1, stdout="", stderr=crash_stderr)
        return CompletedProcess(git_cmd + args, 0, stdout="", stderr="")

    result = fetch_with_partial_clone_recovery(runner, ["git"], ["fetch", "origin", "main"], tmp_path)

    assert calls == [(["git"], ["fetch", "origin", "main"])] * 2
    assert (pack_dir / "pack-local.promisor").exists()
    assert result.returncode == 0

# ---- Broken remote-tracking ref recovery (#130509) ----

_MISSING_OBJECT_OID = "aa1a546a2298bce39f7929dedb3b8d1f6cf69d4d"
_MISSING_OBJECT_STDERR = f"error: Could not read {_MISSING_OBJECT_OID}\n"
_TRACKING_REFSPEC = "+refs/heads/main:refs/remotes/origin/main"


def _init_repo(root: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)


def _write_broken_origin_main(root: Path) -> tuple[Path, Path]:
    ref = root / ".git" / "refs" / "remotes" / "origin" / "main"
    reflog = root / ".git" / "logs" / "refs" / "remotes" / "origin" / "main"
    ref.parent.mkdir(parents=True, exist_ok=True)
    reflog.parent.mkdir(parents=True, exist_ok=True)
    ref.write_text(_MISSING_OBJECT_OID + "\n", encoding="ascii")
    reflog.write_text(
        f"{'0' * 40} {_MISSING_OBJECT_OID} Test <test@example.com> 1700000000 +0000\tfetch\n",
        encoding="ascii",
    )
    return ref, reflog


def test_missing_object_recovery_drops_only_broken_target_and_retries(tmp_path):
    _init_repo(tmp_path)
    ref, reflog = _write_broken_origin_main(tmp_path)
    calls = []

    def runner(git_cmd, args):
        calls.append((list(git_cmd), list(args)))
        if len(calls) == 1:
            return CompletedProcess(git_cmd + args, 1, stdout="", stderr=_MISSING_OBJECT_STDERR)
        return CompletedProcess(git_cmd + args, 0, stdout="", stderr="")

    args = ["fetch", "origin", _TRACKING_REFSPEC]
    result = fetch_with_partial_clone_recovery(runner, ["git"], args, tmp_path)

    assert result.returncode == 0
    assert calls == [(["git"], args)] * 2
    assert not ref.exists()
    assert not reflog.exists()  # git update-ref -d removes the broken tracking reflog too


def test_missing_object_recovery_does_not_drop_a_readable_tracking_ref(tmp_path):
    _init_repo(tmp_path)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "--allow-empty", "-q", "-m", "seed"], cwd=tmp_path, check=True)
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=tmp_path, check=True, capture_output=True, text=True,
    ).stdout.strip()
    subprocess.run(
        ["git", "update-ref", "refs/remotes/origin/main", head], cwd=tmp_path, check=True)

    calls = []

    def runner(git_cmd, args):
        calls.append((list(git_cmd), list(args)))
        return CompletedProcess(git_cmd + args, 1, stdout="", stderr=_MISSING_OBJECT_STDERR)

    args = ["fetch", "origin", _TRACKING_REFSPEC]
    result = fetch_with_partial_clone_recovery(runner, ["git"], args, tmp_path)

    assert result.returncode == 1
    assert calls == [(["git"], args)]
    kept = subprocess.run(
        ["git", "rev-parse", "refs/remotes/origin/main"],
        cwd=tmp_path, check=True, capture_output=True, text=True,
    ).stdout.strip()
    assert kept == head


def test_unrelated_fetch_failure_never_drops_even_a_broken_tracking_ref(tmp_path):
    _init_repo(tmp_path)
    ref, _ = _write_broken_origin_main(tmp_path)
    calls = []

    def runner(git_cmd, args):
        calls.append((list(git_cmd), list(args)))
        return CompletedProcess(
            git_cmd + args, 128, stdout="",
            stderr="fatal: Authentication failed for 'https://github.com/example/repo.git'\n",
        )

    args = ["fetch", "origin", _TRACKING_REFSPEC]
    result = fetch_with_partial_clone_recovery(runner, ["git"], args, tmp_path)

    assert result.returncode == 128
    assert calls == [(["git"], args)]
    assert ref.exists()

