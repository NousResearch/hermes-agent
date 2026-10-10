"""A checkout change costs ONE process one sweep.

When the checkout changes between launches and two hermes processes start
together, both reach the launch-time stale-bytecode guard. Each walks the tree
and deletes every ``__pycache__`` directory; each then recompiles the import set
the other has just deleted, writing ``.pyc`` files back over each other. The
guard is correct. Running it N times concurrently is not, and nothing
serialized it.

**The witness is a COUNT, never a duration.** "The boot got faster" passes under
any mutant on a fast machine and fails spuriously on a loaded one. What these
tests assert is how many times the purge function was invoked, which side logged
which outcome, and whether a file was unlinked — all of them things the test owns
and the production code cannot satisfy any other way.
"""

from __future__ import annotations

import json
import logging
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from hermes_cli import main as hermes_main
from hermes_cli import _bytecode_sweep as sweep


def _make_repo(tmp_path: Path, sha: str = "a" * 40) -> Path:
    """Minimal git checkout layout that _read_git_revision_fingerprint groks."""

    repo = tmp_path / "repo"
    git_dir = repo / ".git"
    (git_dir / "refs" / "heads").mkdir(parents=True)
    (git_dir / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    (git_dir / "refs" / "heads" / "main").write_text(sha + "\n", encoding="utf-8")
    return repo


def _make_pycache(repo: Path, subdir: str = "hermes_cli") -> Path:
    cache = repo / subdir / "__pycache__"
    cache.mkdir(parents=True)
    (cache / "main.cpython-311.pyc").write_bytes(b"stale")
    return cache


def _stale_stamp(repo: Path) -> None:
    """Record a fingerprint that does NOT match the repo — i.e. "checkout changed"."""

    (repo / sweep._BYTECODE_FINGERPRINT_FILE).write_text(
        "git:refs/heads/main:" + "0" * 40, encoding="utf-8"
    )


class _CountingPurge:
    """A stand-in for ``_clear_bytecode_cache`` that counts and can be held.

    The counter lives HERE, in the test, not in the production module — which is
    what makes it unforgeable: a mutant that lets two processes sweep must call
    this object twice, and no status field, log line or lock-file state can
    substitute for the second call.
    """

    def __init__(self, *, removed: int = 4) -> None:
        self.calls = 0
        self.removed = removed
        self.entered = threading.Event()
        self.release = threading.Event()
        self.hold = False
        self._lock = threading.Lock()

    def __call__(self, root):
        with self._lock:
            self.calls += 1
        self.entered.set()
        if self.hold:
            self.release.wait(timeout=8)
        return self.removed


@pytest.fixture
def repo(monkeypatch, tmp_path):
    checkout = _make_repo(tmp_path, sha="b" * 40)
    _make_pycache(checkout)
    _stale_stamp(checkout)
    monkeypatch.setattr(hermes_main, "PROJECT_ROOT", checkout)
    return checkout


def _outcomes(caplog) -> list[str]:
    out = []
    for record in caplog.records:
        message = record.getMessage()
        if "__pycache__" not in message:
            continue
        for token in message.split():
            if token.startswith("outcome="):
                out.append(token.split("=", 1)[1])
    return out


# ---------------------------------------------------------------------------
# The winner
# ---------------------------------------------------------------------------


def test_the_winner_sweeps_restamps_and_releases_its_lock(monkeypatch, repo, caplog):
    purge = _CountingPurge()
    monkeypatch.setattr(hermes_main, "_clear_bytecode_cache", purge)

    with caplog.at_level(logging.INFO, logger=sweep.logger.name):
        sweep._sweep_stale_bytecode_if_checkout_changed()

    assert purge.calls == 1
    assert _outcomes(caplog) == ["swept"]
    # Released, or every later launch on this checkout would wait for a ghost.
    assert not sweep._bytecode_sweep_lock_path().exists()
    recorded = (repo / sweep._BYTECODE_FINGERPRINT_FILE).read_text(
        encoding="utf-8"
    )
    assert recorded.strip().endswith("b" * 40)


def test_an_unchanged_checkout_never_touches_the_lock(monkeypatch, repo):
    """The cheap path stays cheap: the fingerprint check is outside the lock.

    Anti-vacuity. *Mutation:* claim the lock BEFORE comparing fingerprints.
    *Probed field:* the lock file's non-existence on a run where nothing changed
    — every hermes invocation on an unchanged checkout would otherwise create and
    unlink a file in the checkout root, and two of them would serialize on it for
    no reason at all.
    """

    (repo / sweep._BYTECODE_FINGERPRINT_FILE).write_text(
        "git:refs/heads/main:" + "b" * 40, encoding="utf-8"
    )
    created: list[Path] = []
    real_claim = sweep._claim_bytecode_sweep_lock

    def _watching_claim(path):
        created.append(path)
        return real_claim(path)

    monkeypatch.setattr(sweep, "_claim_bytecode_sweep_lock", _watching_claim)

    sweep._sweep_stale_bytecode_if_checkout_changed()

    assert created == []
    assert not sweep._bytecode_sweep_lock_path().exists()


# ---------------------------------------------------------------------------
# The loser
# ---------------------------------------------------------------------------


def test_two_concurrent_entries_run_exactly_one_sweep(monkeypatch, repo, caplog):
    """The stage's whole point, as a count.

    Anti-vacuity. *Mutation:* remove the lock (``lock = lock_path`` — everybody
    wins). *Probed field:* ``purge.calls``, a counter owned by this test's fake,
    which BOTH threads reach under the mutant. It cannot read 1 under two
    unserialized calls, and no log line, outcome string, or lock-file state can
    stand in for the missing second invocation. *Why the race is deterministic:*
    the winner is HELD inside the fake purge until this test releases it, so both
    threads are provably past the fingerprint check and inside the contended
    region before either can finish — the mutant has no scheduling in which only
    one of them sweeps.

    Second, independent witness in the same case: the pair of logged outcomes.
    One ``swept`` and one ``waited_for_winner`` is a different assertion, in a
    different mechanism (the purge line), and a mutant that somehow counted right
    while logging both sides as ``swept`` still dies here.
    """

    purge = _CountingPurge()
    purge.hold = True
    monkeypatch.setattr(hermes_main, "_clear_bytecode_cache", purge)
    monkeypatch.setattr(sweep, "_BYTECODE_SWEEP_LOCK_WAIT_SECONDS", 5.0)

    def _enter():
        sweep._sweep_stale_bytecode_if_checkout_changed()

    with caplog.at_level(logging.INFO, logger=sweep.logger.name):
        threads = [threading.Thread(target=_enter, name=f"sweeper-{i}") for i in range(2)]
        for thread in threads:
            thread.start()
        # The winner is now parked inside the purge, still holding the lock.
        assert purge.entered.wait(timeout=8), "no thread reached the purge"
        # Give the loser time to reach its wait loop while the lock is still held.
        time.sleep(0.3)
        purge.release.set()
        for thread in threads:
            thread.join(timeout=15)
            assert not thread.is_alive()

    assert purge.calls == 1, "the sweep ran more than once"
    assert sorted(_outcomes(caplog)) == ["swept", "waited_for_winner"]
    assert not sweep._bytecode_sweep_lock_path().exists()


def test_a_loser_whose_wait_expires_proceeds_without_sweeping(monkeypatch, repo, caplog):
    """Fail-open, and it must not steal the live winner's lock.

    Anti-vacuity. *Mutation:* sweep anyway once the wait expires (the
    "be safe, purge" instinct). *Probed field:* ``purge.calls == 0`` — the whole
    reason this stage exists is that a SECOND full purge plus recompile is the
    cost being removed, so a mutant that purges on expiry has un-fixed the bug
    while keeping every log line and every lock interaction intact. Paired second
    witness: the lock file is still on disk afterwards, so the expired waiter did
    not unlink a claim that may still be live — a mutant that broke the lock on
    expiry would leave the real winner's release unlinking nothing and the next
    launch waiting on a file this process created.
    """

    purge = _CountingPurge()
    monkeypatch.setattr(hermes_main, "_clear_bytecode_cache", purge)
    monkeypatch.setattr(sweep, "_BYTECODE_SWEEP_LOCK_WAIT_SECONDS", 0.2)
    # A lock held by somebody else, fresh enough to be honoured.
    lock_path = sweep._bytecode_sweep_lock_path()
    lock_path.write_bytes(b"999999\n")

    with caplog.at_level(logging.INFO, logger=sweep.logger.name):
        sweep._sweep_stale_bytecode_if_checkout_changed()

    assert purge.calls == 0
    assert _outcomes(caplog) == ["proceeded_unswept"]
    assert lock_path.exists(), "an expired waiter must not steal a live lock"
    # And it must not have restamped: the winner owns the stamp, and a loser that
    # restamped without sweeping would tell every LATER launch the cache is clean.
    recorded = (repo / sweep._BYTECODE_FINGERPRINT_FILE).read_text(
        encoding="utf-8"
    )
    assert recorded.strip().endswith("0" * 40)


def test_a_stale_lock_is_broken_rather_than_honoured_forever(monkeypatch, repo, caplog):
    """A sweeper killed mid-walk must not brick every future launch's guard.

    Anti-vacuity. *Mutation:* honour any existing lock unconditionally (drop the
    staleness break). *Probed field:* ``purge.calls == 1`` in the presence of a
    pre-existing lock file — the mutant reads 0 because it waits out the bound
    and proceeds unswept, which is exactly the "the guard silently stopped
    existing" failure the guard was built to prevent.
    """

    purge = _CountingPurge()
    monkeypatch.setattr(hermes_main, "_clear_bytecode_cache", purge)
    monkeypatch.setattr(sweep, "_BYTECODE_SWEEP_LOCK_WAIT_SECONDS", 0.2)
    lock_path = sweep._bytecode_sweep_lock_path()
    lock_path.write_bytes(b"1\n")
    ancient = time.time() - (sweep._BYTECODE_SWEEP_LOCK_STALE_SECONDS + 60)
    os.utime(lock_path, (ancient, ancient))

    with caplog.at_level(logging.INFO, logger=sweep.logger.name):
        sweep._sweep_stale_bytecode_if_checkout_changed()

    assert purge.calls == 1
    assert _outcomes(caplog) == ["swept"]
    assert not lock_path.exists()


def test_a_fresh_lock_is_not_treated_as_stale(monkeypatch, repo):
    """The other side of the staleness rule, so the bound discriminates.

    Anti-vacuity. *Mutation:* break EVERY existing lock (staleness threshold of
    0). *Probed field:* ``purge.calls == 0`` against a lock whose mtime is now —
    the two staleness tests together pin the discriminator, and neither
    break-everything nor break-nothing passes both.
    """

    purge = _CountingPurge()
    monkeypatch.setattr(hermes_main, "_clear_bytecode_cache", purge)
    monkeypatch.setattr(sweep, "_BYTECODE_SWEEP_LOCK_WAIT_SECONDS", 0.2)
    sweep._bytecode_sweep_lock_path().write_bytes(b"1\n")

    sweep._sweep_stale_bytecode_if_checkout_changed()

    assert purge.calls == 0


# ---------------------------------------------------------------------------
# A live holder past the age bound
# ---------------------------------------------------------------------------


def _age_past_stale(lock_path: Path) -> None:
    ancient = time.time() - (sweep._BYTECODE_SWEEP_LOCK_STALE_SECONDS + 180)
    os.utime(lock_path, (ancient, ancient))


def _write_lock(lock_path: Path, *, pid: int, create_time, host: str) -> None:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(
        json.dumps({"pid": pid, "create_time": create_time, "host": host}) + "\n",
        encoding="utf-8",
    )


def test_a_live_same_host_holder_keeps_its_lock_past_the_age_bound(tmp_path):
    """A purge that outlasts the age bound must not lose its lock.

    *Mutation:* break on age alone (drop the liveness check). *Probed field:* the
    lock file still exists after a break attempt on a lock naming THIS running
    process — the mutant unlinks it, and the next launch sweeps on top of the
    live winner.
    """

    pytest.importorskip("psutil")
    from hermes_cli.process_identity import _process_create_time

    lock_path = tmp_path / "lock"
    _write_lock(
        lock_path,
        pid=os.getpid(),
        create_time=_process_create_time(),
        host=socket.gethostname(),
    )
    _age_past_stale(lock_path)

    assert sweep._break_stale_bytecode_sweep_lock(lock_path) is False
    assert lock_path.exists()


def test_an_exited_same_host_holder_loses_its_lock_past_the_age_bound(tmp_path):
    """Positive control for the test above: the same payload, a dead holder.

    *Mutation:* never break a same-host lock. *Probed field:* the lock is gone —
    the only variable changed from the live case is which process it names.
    """

    pytest.importorskip("psutil")
    from hermes_cli.process_identity import _process_create_time

    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child_create_time = _process_create_time(child.pid)
    child.wait()

    lock_path = tmp_path / "lock"
    _write_lock(
        lock_path,
        pid=child.pid,
        create_time=child_create_time,
        host=socket.gethostname(),
    )
    _age_past_stale(lock_path)

    assert sweep._break_stale_bytecode_sweep_lock(lock_path) is True
    assert not lock_path.exists()


def test_another_hosts_lock_is_still_broken_by_age(tmp_path):
    """A pid from another machine cannot be checked here, so age decides."""

    lock_path = tmp_path / "lock"
    _write_lock(lock_path, pid=os.getpid(), create_time=None, host="other-host")
    _age_past_stale(lock_path)

    assert sweep._break_stale_bytecode_sweep_lock(lock_path) is True
    assert not lock_path.exists()


def test_a_bare_pid_lock_from_the_old_format_is_still_broken_by_age(tmp_path):
    """A lock written before the payload carried a host falls back to age."""

    lock_path = tmp_path / "lock"
    lock_path.write_bytes(f"{os.getpid()}\n".encode("utf-8"))
    _age_past_stale(lock_path)

    assert sweep._break_stale_bytecode_sweep_lock(lock_path) is True
    assert not lock_path.exists()


def test_a_winner_still_sweeping_past_the_age_bound_is_not_swept_over(
    monkeypatch, repo, caplog
):
    """The reported overlap, end to end: the winner's lock ages past the bound
    while it is still inside the purge, and a later launch arrives.

    *Mutation:* break on age alone. *Probed field:* ``purge.calls == 1`` — the
    mutant breaks the winner's lock, claims it, and calls the purge a second
    time while the first call is still running.
    """

    pytest.importorskip("psutil")
    purge = _CountingPurge()
    purge.hold = True
    monkeypatch.setattr(hermes_main, "_clear_bytecode_cache", purge)
    monkeypatch.setattr(sweep, "_BYTECODE_SWEEP_LOCK_WAIT_SECONDS", 0.2)
    lock_path = sweep._bytecode_sweep_lock_path()

    winner = threading.Thread(target=sweep._sweep_stale_bytecode_if_checkout_changed)
    winner.start()
    try:
        assert purge.entered.wait(timeout=8)
        _age_past_stale(lock_path)
        with caplog.at_level(logging.INFO, logger=sweep.logger.name):
            sweep._sweep_stale_bytecode_if_checkout_changed()
        assert purge.calls == 1
        assert _outcomes(caplog) == ["proceeded_unswept"]
        assert lock_path.exists()
    finally:
        purge.release.set()
        winner.join(timeout=8)
    assert not lock_path.exists()


# ---------------------------------------------------------------------------
# The untouched paths
# ---------------------------------------------------------------------------


def test_a_non_git_install_is_an_unchanged_no_op(monkeypatch, tmp_path):
    """No fingerprint means no sweep and no lock — the ZIP-update path clears
    explicitly, and this guard must not invent a claim it cannot resolve."""

    checkout = tmp_path / "zip-install"
    (checkout / "hermes_cli").mkdir(parents=True)
    monkeypatch.setattr(hermes_main, "PROJECT_ROOT", checkout)
    purge = _CountingPurge()
    monkeypatch.setattr(hermes_main, "_clear_bytecode_cache", purge)

    sweep._sweep_stale_bytecode_if_checkout_changed()

    assert purge.calls == 0
    assert not sweep._bytecode_sweep_lock_path().exists()


def test_a_checkout_that_cannot_hold_a_lock_still_sweeps(monkeypatch, repo):
    """A read-only checkout root, or a share without ``O_EXCL``.

    This is the regression the three-valued claim exists to prevent, and the
    first draft of this stage had it: collapsing "somebody else is sweeping" and
    "nobody here can hold a lock" into one "no lock" answer sent the second case
    down the loser path, which on such a filesystem means NOBODY ever sweeps —
    the stale-bytecode guard silently disabled, which is the whole bug class
    (#6207, #60242) the guard was built to close, reintroduced by an
    optimisation.

    Anti-vacuity. *Mutation:* treat ``unavailable`` as ``contended``. *Probed
    field:* ``purge.calls == 1`` while ``_claim_bytecode_sweep_lock`` is stubbed
    to report ``unavailable`` — the mutant reads 0. And the paired case below
    (``contended`` with an expired wait) reads 0 legitimately, so a mutant that
    made BOTH sweep fails that one instead. Neither collapse passes both.
    """

    purge = _CountingPurge()
    monkeypatch.setattr(hermes_main, "_clear_bytecode_cache", purge)
    monkeypatch.setattr(
        sweep,
        "_claim_bytecode_sweep_lock",
        lambda path: sweep._SWEEP_CLAIM_UNAVAILABLE,
    )
    monkeypatch.setattr(sweep, "_BYTECODE_SWEEP_LOCK_WAIT_SECONDS", 0.2)

    sweep._sweep_stale_bytecode_if_checkout_changed()

    assert purge.calls == 1
    # It holds no lock, so it must not unlink one — a concurrent winner's claim
    # is not this process's to release.
    assert not sweep._bytecode_sweep_lock_path().exists()
    recorded = (repo / sweep._BYTECODE_FINGERPRINT_FILE).read_text(
        encoding="utf-8"
    )
    assert recorded.strip().endswith("b" * 40)


def test_losing_the_reclaim_after_breaking_a_stale_lock_is_still_contended(
    monkeypatch, repo, caplog
):
    """The narrow race inside the staleness break.

    Breaking a stale lock and re-claiming it is two syscalls, and a third process
    can claim in the gap. Reporting ``claimed`` there would be the worst of both
    worlds: this process would sweep concurrently with the real holder AND, on the
    way out, unlink a lock it does not hold — re-creating the double-purge race the
    stage exists to remove, and leaving the real winner's own release unlinking
    nothing.

    Anti-vacuity. *Mutation:* return ``claimed`` from the reclaim's
    ``FileExistsError`` arm. *Probed fields:* the claim token, AND ``purge.calls
    == 0`` end to end. The mutant sweeps, so the count reads 1; and it cannot
    satisfy the token assertion by any other route because the fixture makes every
    ``O_EXCL`` open of the lock path fail.
    """

    lock_path = sweep._bytecode_sweep_lock_path()
    lock_path.write_bytes(b"1\n")
    ancient = time.time() - (sweep._BYTECODE_SWEEP_LOCK_STALE_SECONDS + 60)
    os.utime(lock_path, (ancient, ancient))

    real_open = os.open

    def _always_taken(path, flags, *args, **kwargs):
        if str(path) == str(lock_path):
            raise FileExistsError(17, "claimed in the gap")
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(hermes_main.os, "open", _always_taken)

    assert (
        sweep._claim_bytecode_sweep_lock(lock_path)
        == sweep._SWEEP_CLAIM_CONTENDED
    )

    purge = _CountingPurge()
    monkeypatch.setattr(hermes_main, "_clear_bytecode_cache", purge)
    monkeypatch.setattr(sweep, "_BYTECODE_SWEEP_LOCK_WAIT_SECONDS", 0.2)
    lock_path.write_bytes(b"1\n")
    os.utime(lock_path, (ancient, ancient))

    with caplog.at_level(logging.INFO, logger=sweep.logger.name):
        sweep._sweep_stale_bytecode_if_checkout_changed()

    assert purge.calls == 0
    assert _outcomes(caplog) == ["waited_for_winner"]


def test_a_filesystem_that_refuses_the_open_reports_unavailable_not_contended(
    monkeypatch, repo
):
    """The claim function's OWN three-way discrimination, driven for real.

    The sibling case above stubs ``_claim_bytecode_sweep_lock`` wholesale, so it
    pins the CALLER's handling and says nothing about the function's internals —
    and a mutation that changed ``except OSError: return UNAVAILABLE`` to
    ``return CONTENDED`` survived it. This case drives the real function by making
    ``os.open`` raise a non-``FileExistsError`` ``OSError`` for the lock path
    only, which is what a read-only checkout root does.

    Anti-vacuity. *Probed field:* the returned claim token AND, end to end,
    ``purge.calls == 1`` — the mutant returns ``contended``, the caller then
    waits out a bound on a lock file that does not exist and proceeds unswept, so
    the count reads 0. Neither the token nor the count can be satisfied by the
    other; both are asserted.
    """

    lock_path = sweep._bytecode_sweep_lock_path()
    real_open = os.open

    def _refusing_open(path, flags, *args, **kwargs):
        if str(path) == str(lock_path):
            raise PermissionError(13, "read-only checkout root")
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(hermes_main.os, "open", _refusing_open)

    assert (
        sweep._claim_bytecode_sweep_lock(lock_path)
        == sweep._SWEEP_CLAIM_UNAVAILABLE
    )

    purge = _CountingPurge()
    monkeypatch.setattr(hermes_main, "_clear_bytecode_cache", purge)
    monkeypatch.setattr(sweep, "_BYTECODE_SWEEP_LOCK_WAIT_SECONDS", 0.2)

    sweep._sweep_stale_bytecode_if_checkout_changed()

    assert purge.calls == 1, "an unlockable checkout lost the stale-bytecode guard"


def test_an_unavailable_claim_does_not_release_a_concurrent_winners_lock(
    monkeypatch, repo
):
    """Second witness for the release rule, on the path that has no claim.

    Anti-vacuity. *Mutation:* release unconditionally in the ``finally``.
    *Probed field:* the survival of a lock file this process did not create —
    under the mutant the file is gone, and the real winner's own release then
    unlinks nothing while the next launch waits on a ghost.
    """

    purge = _CountingPurge()
    monkeypatch.setattr(hermes_main, "_clear_bytecode_cache", purge)
    monkeypatch.setattr(
        sweep,
        "_claim_bytecode_sweep_lock",
        lambda path: sweep._SWEEP_CLAIM_UNAVAILABLE,
    )
    lock_path = sweep._bytecode_sweep_lock_path()
    lock_path.write_bytes(b"424242\n")

    sweep._sweep_stale_bytecode_if_checkout_changed()

    assert purge.calls == 1
    assert lock_path.exists(), "a process holding no claim released somebody's lock"


def _untracked(root: Path) -> str:
    import subprocess

    return subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all"],
        capture_output=True, text=True, check=True,
    ).stdout


def test_a_held_sweep_lock_leaves_git_status_clean(monkeypatch, tmp_path):
    """The lock's directory ignores itself, so the root `.gitignore` needs no
    entry for it. *Probed field:* `git status` of a real repository holding the
    lock; the positive control deletes the directory's ignore file and the same
    lock IS untracked litter."""

    import shutil
    import subprocess

    if shutil.which("git") is None:
        pytest.skip("git is not on PATH")
    root = tmp_path / "checkout"
    root.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    monkeypatch.setattr(hermes_main, "PROJECT_ROOT", root)

    lock_path = sweep._bytecode_sweep_lock_path()
    assert sweep._claim_bytecode_sweep_lock(lock_path) == sweep._SWEEP_CLAIM_CLAIMED
    assert lock_path.exists()
    assert _untracked(root) == ""

    (lock_path.parent / ".gitignore").unlink()
    assert ".bytecode-sweep/lock" in _untracked(root)
