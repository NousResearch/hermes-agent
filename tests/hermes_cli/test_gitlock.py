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
import shutil
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


# ---- partial-clone pack growth (#129712, #127711) ----
#
# Every on-demand fetch from a promisor remote writes its own pack, and a commit-graph write over
# commits the graph has not seen lazy-fetches their trees, one pack each. The fold rides
# `git gc --auto` (git's own gc.autoPackLimit decides when it is worth it). Real git against a
# local blobless clone: these pin how the pieces interact, not their command lines.

from hermes_cli.gitlock import settle_partial_clone_maintenance, disable_tree0_auto_maintenance  # noqa: E402

_GIT_ENV = {**os.environ, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"}


def _run_git(*args: str, cwd: Path, env: dict = _GIT_ENV) -> str:
    return subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", *args], cwd=cwd, env=env,
                          check=True, capture_output=True, text=True).stdout.strip()


def _packs(repo: Path) -> list:
    return list((repo / ".git" / "objects" / "pack").glob("pack-*.pack"))


def _blobless_clone(tmp_path: Path, commits: int) -> tuple[Path, Path, Path]:
    """(seed, upstream.git, clone): a blobless clone of an upstream with ``commits`` commits."""
    seed, up, clone = tmp_path / "seed", tmp_path / "up.git", tmp_path / "clone"
    _run_git("init", "-q", "-b", "main", str(seed), cwd=tmp_path)
    for i in range(commits):
        (seed / f"d{i % 3}").mkdir(exist_ok=True)
        (seed / f"d{i % 3}" / "f.txt").write_text(f"v{i}\n", encoding="utf-8")
        _run_git("add", "-A", cwd=seed)
        _run_git("commit", "-qm", f"c{i}", cwd=seed)
    _run_git("clone", "-q", "--bare", str(seed), str(up), cwd=tmp_path)
    _run_git("config", "uploadpack.allowFilter", "true", cwd=up)
    _run_git("config", "uploadpack.allowAnySHA1InWant", "true", cwd=up)
    _run_git("clone", "-q", "--filter=tree:0", "--no-checkout", up.as_uri(), str(clone), cwd=tmp_path)
    return seed, up, clone


@pytest.fixture
def partial_clone(tmp_path: Path) -> Path:
    """Blobless clone that has lazy-fetched one pack per object it was asked for."""
    _seed, _up, clone = _blobless_clone(tmp_path, 3)
    _run_git("config", "gc.autoPackLimit", "2", cwd=clone)  # the wild default (50) cannot be crossed here
    # Else each lazy fetch's detached auto-maintenance races the test and folds the packs itself.
    _run_git("config", "maintenance.auto", "false", cwd=clone)
    for name in ("d0/f.txt", "d1/f.txt", "d2/f.txt"):
        _run_git("cat-file", "-p", _run_git("rev-parse", f"HEAD:{name}", cwd=clone), cwd=clone)
    return clone


def test_gits_own_gc_does_not_lazy_fetch_the_trees_of_commits_a_bloom_graph_has_not_seen(tmp_path: Path) -> None:
    """A gc that folds the packs, left to write a commit-graph, adds one pack per unseen commit."""
    seed, up, clone = _blobless_clone(tmp_path, 12)
    full = tmp_path / "full"
    _run_git("clone", "-q", up.as_uri(), str(full), cwd=tmp_path)
    _run_git("commit-graph", "write", "--reachable", "--changed-paths", cwd=full)
    graph = clone / ".git" / "objects" / "info"
    for entry in (full / ".git" / "objects" / "info").glob("commit-graph*"):
        (shutil.copytree if entry.is_dir() else shutil.copy)(entry, graph / entry.name)
    for i in range(10):
        (seed / "new.txt").write_text(f"n{i}\n", encoding="utf-8")
        _run_git("add", "-A", cwd=seed)
        _run_git("commit", "-qm", f"n{i}", cwd=seed)
    _run_git("push", "-q", str(up), "main", cwd=seed)
    _run_git("-c", "maintenance.auto=false", "fetch", "-q", "origin", cwd=clone)
    _run_git("-c", "maintenance.auto=false", "log", "-p", cwd=clone)
    _run_git("config", "gc.autoPackLimit", "3", cwd=clone)
    assert len(_packs(clone)) > 3

    settle_partial_clone_maintenance(clone)
    _run_git("-c", "gc.autoDetach=false", "gc", "--auto", cwd=clone)

    assert len(_packs(clone)) == 1


def test_maintenance_keys_leave_gits_own_fold_running(tmp_path: Path) -> None:
    """Between updates, git's post-lazy-fetch auto maintenance is what folds packs (git <= 2.53).

    The keys stop the commit-graph write without switching that fold off: ``maintenance.auto=false``
    let the packs pile up until the next ``hermes update``. Compared against a stock-config clone
    because newer git (2.55) does not fold lazy-fetch packs on its own at all.
    """
    def lazy_packs(name: str, keys: bool) -> int:
        (tmp_path / name).mkdir()
        _seed, _up, clone = _blobless_clone(tmp_path / name, 3)
        for key, value in (("gc.autoPackLimit", "2"), ("gc.autoDetach", "false"), ("maintenance.autoDetach", "false")):
            _run_git("config", key, value, cwd=clone)  # git's own auto gc, in the foreground
        if keys:
            # what the first cut persisted: all three keys together, none of the new one
            for key in ("maintenance.auto", "gc.writeCommitGraph", "fetch.writeCommitGraph"):
                _run_git("config", key, "false", cwd=clone)
            disable_tree0_auto_maintenance(clone)
        for path in ("d0/f.txt", "d1/f.txt", "d2/f.txt"):
            _run_git("cat-file", "-p", _run_git("rev-parse", f"HEAD:{path}", cwd=clone), cwd=clone)
        return len(_packs(clone))

    assert lazy_packs("keys", keys=True) <= lazy_packs("stock", keys=False)


def test_an_operators_own_maintenance_auto_false_is_never_erased(repo: Path) -> None:
    _run_git("config", "maintenance.auto", "false", cwd=repo)

    disable_tree0_auto_maintenance(repo)
    disable_tree0_auto_maintenance(repo)

    assert _run_git("config", "--local", "--get", "maintenance.auto", cwd=repo) == "false"


def test_a_checkout_the_first_cut_configured_folds_again(partial_clone: Path) -> None:
    """73c17151f61 persisted ``gc.auto=0`` (beside ``maintenance.auto=false`` and
    ``fetch.writeCommitGraph=false``); left in place, ``gc --auto`` is a no-op and no fold ever runs."""
    for key, value in (("gc.auto", "0"), ("fetch.writeCommitGraph", "false")):
        _run_git("config", key, value, cwd=partial_clone)
    assert len(_packs(partial_clone)) > 2

    settle_partial_clone_maintenance(partial_clone)
    _run_git("-c", "gc.autoDetach=false", "gc", "--auto", cwd=partial_clone)

    assert len(_packs(partial_clone)) == 1
    left = subprocess.run(["git", "config", "--local", "--get-regexp", r"^(maintenance\.auto|gc\.auto)$"],
                          cwd=partial_clone, capture_output=True, text=True).stdout
    assert left == ""


def test_non_partial_checkout_is_left_alone(repo: Path) -> None:
    settle_partial_clone_maintenance(repo)
    keys = subprocess.run(["git", "config", "--local", "--get-regexp", "maintenance|writecommitgraph"], cwd=repo,
                          capture_output=True, text=True).stdout
    assert keys == "", "a full clone keeps git's stock maintenance"


# ---- a killed git's index.lock (#132089) ----


def _status_blocked_on_a_fifo(repo: Path, *argv: str) -> subprocess.Popen:
    """A real ``git status`` that takes ``.git/index.lock`` and then blocks: its untracked scan
    opens a FIFO ``.gitignore`` nobody writes."""
    (repo / "junk").mkdir()
    os.mkfifo(repo / "junk" / ".gitignore")
    (repo / "junk" / "x").touch()
    (repo / "a.txt").touch()  # stat-dirty: status refreshes (and so locks) the index
    proc = subprocess.Popen(["git", *argv, "status", "--porcelain"], cwd=repo,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    deadline = time.monotonic() + 20
    while not (repo / ".git" / "index.lock").exists():
        assert proc.poll() is None and time.monotonic() < deadline, "git status never took index.lock"
        time.sleep(0.05)
    return proc


@pytest.mark.skipif(not Path("/proc/self/fd").is_dir(), reason="ownership proof via /proc (Linux)")
def test_a_killed_gits_index_lock_goes_at_once_and_a_live_ones_stays(repo: Path) -> None:
    """The next update must not die on the lock a killed one left (the age floor kept it for 10
    minutes), and must never take the lock of a git that is still running."""
    from hermes_cli.gitlock import release_dead_index_lock

    lock = repo / ".git" / "index.lock"
    proc = _status_blocked_on_a_fifo(repo)
    try:
        assert release_dead_index_lock(repo) is False
        assert lock.exists()
    finally:
        proc.kill()
        proc.wait()

    assert lock.exists(), "premise: a SIGKILLed status strands index.lock"
    assert release_dead_index_lock(repo) is True
    assert not lock.exists()
