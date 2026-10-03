"""Regression for #126784: the orphan-break allowance in ``_acquire_db_flock``.

A contender that times out on a lock file whose recorded holder is provably dead
unlinks the file and retakes it on a fresh inode.  ``_acquire_db_flock`` used to
grant exactly ONE break per call (a single ``broke_lock`` flag).  A second,
concurrent breaker can replace the file again between our ``unlink`` and the inode
check that follows it, so ``same_file`` is False and the loop reopens and retries --
with its allowance already spent.  On the next contention expiry it returned False,
"held by another process", without re-reading the holder record, even though the
record naming the fresh inode is a dead holder too and a legitimate break was still
available.  Callers see ``fts_rebuild_admission`` yield False, so full structural FTS
rebuilds defer indefinitely ("reindexing never finishes", no error).

The second half of the fix makes the holder record durable: ``_rewrite_lock_file``
fsyncs, so a contender can never read a never-durable (stale) record of a live
holder.

Simulation contract -- deterministic and host-independent on purpose (the issue asks
for exactly this shape, and a three-process flock race can only assert the outcome,
never the loop's bookkeeping):

* ``fcntl`` is a stand-in: contended while a foreign open file description holds the
  lock, granted once on the fresh inode this process just retook.
* ``os`` inside ``hermes_state_common`` is proxied so ``os.stat`` reports a
  mismatched inode on the first post-break check (the racer replaced the file) and
  ``os.kill`` reports the recorded holder as gone.  Everything else -- ``open``, the
  record read/write, ``_lock_holder_provably_dead``, the deadline arithmetic -- is
  the production path.
* The break's filesystem effect is simulated: ``unlink`` only counts the break.
  Windows cannot unlink a file this process still holds open (the real unlink sits
  before ``handle.close()``), and the issue's own regression recipe is to inject the
  replacement inode through ``os.stat``.  The path therefore keeps naming a file that
  carries another dead holder record -- the racer's orphaned record.
* The clock is virtual, so nothing here depends on wall-clock timing.

The end-to-end orphaned-fd break against real processes stays covered by
``tests/hermes_state/test_fts_rebuild_admission.py`` (POSIX).
"""

import contextlib
import errno
import json
import os
import sys
import types
from pathlib import Path

import hermes_state_common as hsc

# A pid no live process can own on any host this suite runs on; never signalled for
# real (see ``_RacerOs.kill``).
_DEAD_PID = 2 ** 30 - 7
_DEAD_RECORD = {"pid": _DEAD_PID, "start_ticks": 4242, "acquired_at": 0.0}
_POLL_SECONDS = 0.1
# Three poll intervals: the post-break budget closes deterministically under the fake clock.
_BREAK_BUDGET_SECONDS = 0.25


class _FakeClock:
    """Virtual clock: ``sleep`` is the only thing that advances time."""

    def __init__(self):
        self.now = 1_000.0

    def monotonic(self):
        return self.now

    def time(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class _FakeFcntl(types.ModuleType):
    """``fcntl`` model for the racer scenario.

    Contended until this process breaks the stale file; on the fresh inode the next
    call is granted once, which is where the racer's replacement is detected.  With
    ``grant_after_break=False`` the fresh inode is held forever (permanent contention).
    An unexpected call count is a runaway loop, so it fails loudly instead of hanging.
    """

    LOCK_EX = 2
    LOCK_NB = 4
    LOCK_UN = 8

    def __init__(self, breaks, grant_after_break=True, max_calls=64):
        super().__init__("fcntl")
        self._breaks = breaks
        self._grant_after_break = grant_after_break
        self._max_calls = max_calls
        self._seen_breaks = 0
        self._calls_since_break = 0
        self.calls = 0

    def flock(self, file_descriptor, operation):
        self.calls += 1
        assert self.calls <= self._max_calls, (
            "runaway retry loop: _acquire_db_flock made %d flock calls" % self.calls
        )
        if self._breaks["n"] != self._seen_breaks:
            self._seen_breaks = self._breaks["n"]
            self._calls_since_break = 0
        self._calls_since_break += 1
        if self._grant_after_break and self._seen_breaks and self._calls_since_break == 1:
            return
        raise BlockingIOError(errno.EAGAIN, "resource temporarily unavailable")


class _RacerOs:
    """``hermes_state_common.os`` proxy: the loop sees a racer's file, the rest is real.

    * ``unlink`` counts this process's breaks and leaves the path naming a replacement
      file that carries another orphaned dead-holder record -- the racer's record, i.e.
      the premise for a legitimate second break (see the module docstring for why the
      break is simulated rather than performed).
    * ``stat`` reports a different inode on the selected post-break checks (the racer
      replaced the file after our unlink).
    * ``kill`` reports the recorded holder as gone: probing a sentinel pid for real is
      neither safe nor portable (on Windows ``os.kill(pid, 0)`` terminates a process).
    """

    def __init__(self, lock_path, breaks, mismatched_stat_calls=()):
        self._lock_path = str(lock_path)
        self._breaks = breaks
        self._mismatched = set(mismatched_stat_calls)
        self.stat_calls = 0

    def unlink(self, path):
        assert path == self._lock_path
        self._breaks["n"] += 1

    def stat(self, path, *args, **kwargs):
        result = os.stat(path, *args, **kwargs)
        self.stat_calls += 1
        if self.stat_calls in self._mismatched:
            return os.stat_result(
                (
                    result.st_mode,
                    result.st_ino + 1,
                    result.st_dev,
                    result.st_nlink,
                    result.st_uid,
                    result.st_gid,
                    result.st_size,
                    result.st_atime,
                    result.st_mtime,
                    result.st_ctime,
                )
            )
        return result

    def fstat(self, file_descriptor):
        return os.fstat(file_descriptor)

    def kill(self, pid, sig):
        if pid == _DEAD_PID:
            raise ProcessLookupError(pid)
        raise OSError(errno.EPERM, "not this test's pid")

    def __getattr__(self, name):
        return getattr(os, name)


@contextlib.contextmanager
def _racer_scenario(tmp_path, monkeypatch, *, grant_after_break=True, mismatched_stat_calls=(1,)):
    """The lock file of a dead (orphaned-fd) holder, plus a racer that replaces it."""
    lock_path = Path(tmp_path) / "state.db.fts_rebuild.lock"
    lock_path.write_bytes(json.dumps(_DEAD_RECORD, sort_keys=True).encode("utf-8"))
    breaks = {"n": 0}
    monkeypatch.setattr(hsc, "os", _RacerOs(lock_path, breaks, mismatched_stat_calls))
    clock = _FakeClock()
    monkeypatch.setattr(hsc, "time", clock)
    monkeypatch.setattr(hsc, "_LOCK_BREAK_REACQUIRE_SECONDS", _BREAK_BUDGET_SECONDS)
    monkeypatch.setitem(
        sys.modules, "fcntl", _FakeFcntl(breaks, grant_after_break=grant_after_break)
    )
    handle = lock_path.open("a+b")
    try:
        yield lock_path, handle, breaks, clock
    finally:
        with contextlib.suppress(OSError):
            handle.close()


def test_racer_invalidated_break_does_not_exhaust_the_allowance(tmp_path, monkeypatch):
    """#126784: one break per call is not enough -- a racer can invalidate it.

    The contender breaks the orphaned lock, the racer replaces the file before the
    inode check, and the replacement file also names a dead holder.  The loop must be
    allowed to break again and acquire, instead of reporting a live holder for the
    whole timeout.  The caller budget is one re-acquire window -- the shape the
    recovery exists for: a contender that asked to wait at all keeps the break's
    patience (see ``test_non_blocking_probe_spends_no_extra_wait`` for the
    ``timeout_seconds=0`` probe, which must not).
    """
    with _racer_scenario(tmp_path, monkeypatch) as (lock_path, handle, breaks, clock):
        acquired, handle = hsc._acquire_db_flock(
            str(lock_path), handle, _BREAK_BUDGET_SECONDS, _POLL_SECONDS, "FTS rebuild lock"
        )
        try:
            assert breaks["n"] >= 2, (
                "the loop stopped after a single break: the racer-invalidated break "
                "consumed the whole allowance"
            )
            assert acquired is True, (
                "a dead holder on the replacement inode was reported as held forever"
            )
        finally:
            with contextlib.suppress(OSError):
                handle.close()


def test_non_blocking_probe_spends_no_extra_wait(tmp_path, monkeypatch):
    """``timeout_seconds=0`` means no waiting -- a break must not buy the probe any.

    ``retry_deferred_fts_recovery`` probes with ``timeout_seconds=0`` while holding
    the in-process state.db lock, so the orphan break may still be attempted (it is
    what unblocks a dead holder, and it acquires outright when the fresh inode is
    free) but a racer-invalidated break must defer at once instead of re-arming a
    re-acquire budget the caller never asked for.
    """
    with _racer_scenario(tmp_path, monkeypatch) as (lock_path, handle, breaks, clock):
        started = clock.now
        acquired, handle = hsc._acquire_db_flock(
            str(lock_path), handle, 0.0, _POLL_SECONDS, "FTS rebuild lock"
        )
        try:
            assert breaks["n"] == 1, (
                "the probe must still get its one orphan break, no more: broke %d" % breaks["n"]
            )
            assert clock.now == started, (
                "a timeout_seconds=0 probe blocked %.2fs of post-break waiting"
                % (clock.now - started)
            )
            assert acquired is False, "a contended replacement inode must fail closed at once"
        finally:
            with contextlib.suppress(OSError):
                handle.close()


def test_break_allowance_is_bounded_and_still_fails_closed(tmp_path, monkeypatch):
    """The retries must stay bounded: permanent contention still defers (fail closed)."""
    max_attempts = getattr(hsc, "_MAX_BREAK_ATTEMPTS", 1)
    with _racer_scenario(
        tmp_path, monkeypatch, grant_after_break=False, mismatched_stat_calls=()
    ) as (lock_path, handle, breaks, _clock):
        acquired, handle = hsc._acquire_db_flock(
            str(lock_path), handle, 0.0, _POLL_SECONDS, "FTS rebuild lock"
        )
        try:
            assert breaks["n"] > 1, (
                "the loop never retried a break, so the allowance is one-shot rather "
                "than a bounded number of attempts"
            )
            assert breaks["n"] <= max_attempts, (
                "the loop broke the lock %d times, past its declared bound %d"
                % (breaks["n"], max_attempts)
            )
            assert acquired is False, "an unreleasable holder must fail closed"
        finally:
            with contextlib.suppress(OSError):
                handle.close()


def test_holder_record_write_is_fsynced(tmp_path, monkeypatch):
    """#126784 second half: the record must be durable before contenders can read it."""

    class _FsyncSpy:
        def __init__(self):
            self.calls = []

        def fsync(self, file_descriptor):
            self.calls.append(file_descriptor)

        def __getattr__(self, name):
            return getattr(os, name)

    lock_path = tmp_path / "state.db.fts_rebuild.lock"
    spy = _FsyncSpy()
    monkeypatch.setattr(hsc, "os", spy)
    handle = lock_path.open("a+b")
    try:
        file_descriptor = handle.fileno()
        hsc._write_lock_holder_record(handle)
    finally:
        handle.close()
    assert b'"pid"' in lock_path.read_bytes()
    assert spy.calls, (
        "the holder record was written without fsync: a contender can read a stale, "
        "never-durable record of a live holder"
    )
    assert set(spy.calls) == {file_descriptor}
