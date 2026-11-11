"""A kernel-blocked vnode must not wedge Hermes startup (#113187).

``_iter_darwin_fd_targets`` calls ``proc_pidfdinfo`` for every descriptor of
every process. If any process holds a descriptor whose vnode lookup blocks in
the kernel (e.g. uninterruptible I/O on a dead network share), that call never
returns -- and it used to run synchronously on the SessionDB connect path, so
one stuck unrelated process froze every Hermes start indefinitely.

``_iter_darwin_sidecar_holders`` must therefore bound the whole pass with a
wall-clock deadline (a per-iteration check cannot fire while blocked inside a
single call) and fail open with a warning. libproc is faked throughout, so
these run on every platform.
"""

from __future__ import annotations

import ctypes
import logging
import os
import struct
import threading
import time

import hermes_state_dbfile as dbfile
from hermes_state_dbfile import (
    _DARWIN_FD_INO_OFFSET,
    _DARWIN_FD_PATH_OFFSET,
    _DARWIN_FD_DEV_OFFSET,
    _DARWIN_FD_RECORD_SIZE,
    _iter_darwin_sidecar_holders,
)

# Guard so a regression (scan blocks again) fails instead of hanging the suite.
HANG_GUARD = 30.0


class _HungLibproc:
    """One pid, one fd, whose vnode lookup blocks until the test releases it."""

    def __init__(self, release):
        self.release = release

    def proc_pidinfo(self, pid, flavor, arg, buf, size):
        payload = struct.pack("<iI", 7, 1)
        ctypes.memmove(buf, payload, len(payload))
        return len(payload)

    def proc_pidfdinfo(self, pid, fd, flavor, record, size):
        self.release.wait()
        return 0


class _HealthyLibproc:
    """One pid, one vnode fd naming ``target`` with the given identity."""

    def __init__(self, target, identity):
        self._target = target
        self._identity = identity

    def proc_pidinfo(self, pid, flavor, arg, buf, size):
        payload = struct.pack("<iI", 7, 1)
        ctypes.memmove(buf, payload, len(payload))
        return len(payload)

    def proc_pidfdinfo(self, pid, fd, flavor, record, size):
        blob = bytearray(_DARWIN_FD_RECORD_SIZE)
        struct.pack_into("<I", blob, _DARWIN_FD_DEV_OFFSET, self._identity[0])
        struct.pack_into("<Q", blob, _DARWIN_FD_INO_OFFSET, self._identity[1])
        encoded = self._target.encode("utf-8")
        blob[_DARWIN_FD_PATH_OFFSET:_DARWIN_FD_PATH_OFFSET + len(encoded)] = encoded
        ctypes.memmove(record, bytes(blob), len(blob))
        return 1


def _run_guarded(func, *args):
    """Run ``func`` so a hang fails the test instead of hanging the suite."""
    outcome = []
    worker = threading.Thread(target=lambda: outcome.append(func(*args)), daemon=True)
    started = time.monotonic()
    worker.start()
    worker.join(HANG_GUARD)
    return worker, time.monotonic() - started, outcome


def test_hung_vnode_scan_fails_open_within_budget(monkeypatch, tmp_path, caplog):
    release = threading.Event()
    monkeypatch.setattr(dbfile, "_darwin_libproc", lambda: _HungLibproc(release))
    monkeypatch.setattr(dbfile, "_darwin_all_pids", lambda _lib: [4242])
    try:
        with caplog.at_level(logging.WARNING, logger="hermes_state"):
            worker, elapsed, outcome = _run_guarded(
                _iter_darwin_sidecar_holders, tmp_path / "state.db"
            )
        assert not worker.is_alive(), "scan blocked past the hang guard: no deadline"
        assert elapsed < HANG_GUARD
        assert outcome == [[]], "a timed-out scan must fail open (no holders, no refusal)"
        assert any(
            "timed out" in record.getMessage() for record in caplog.records
        ), "deadline expiry must log a visible warning"
    finally:
        release.set()
        scan_worker = getattr(dbfile, "_DARWIN_FD_SCAN_WORKER", None)
        if scan_worker is not None:
            scan_worker.join(2.0)


def test_healthy_scan_still_reports_holder(monkeypatch, tmp_path, caplog):
    """The timeout wrapper must not swallow a fast, successful scan."""
    watched = os.path.realpath(os.path.join(os.fspath(tmp_path), "state.db-wal"))
    # The watched path names nothing: stat fails, so the fd's identity is
    # judged a retired generation and the holder is reported.
    lib = _HealthyLibproc(watched, (0xDEAD, 0xBEEF))
    monkeypatch.setattr(dbfile, "_darwin_libproc", lambda: lib)
    monkeypatch.setattr(dbfile, "_darwin_all_pids", lambda _lib: [4242])
    # Prior test intentionally timed out; each test starts outside its brief cache.
    monkeypatch.setattr(dbfile, "_DARWIN_FD_SCAN_NEXT_RETRY", 0.0, raising=False)
    with caplog.at_level(logging.WARNING, logger="hermes_state"):
        worker, elapsed, outcome = _run_guarded(
            _iter_darwin_sidecar_holders, tmp_path / "state.db"
        )
    assert not worker.is_alive()
    assert elapsed < dbfile._DARWIN_FD_SCAN_TIMEOUT_SECONDS, "a fast scan must not wait out the budget"
    assert outcome == [[(4242, watched)]]
    assert not any(
        "timed out" in record.getMessage() for record in caplog.records
    ), "a successful scan must not log a timeout warning"


def test_stuck_scan_is_single_flight_across_callers(monkeypatch, tmp_path):
    """Regression for #153123: a timed-out scan cannot multiply libproc workers."""
    monkeypatch.setattr(dbfile, "_DARWIN_FD_SCAN_NEXT_RETRY", 0.0, raising=False)
    entered = threading.Event()
    release = threading.Event()
    scans = []

    def blocked_scan():
        scans.append(True)
        entered.set()
        release.wait()
        return iter(())

    monkeypatch.setattr(dbfile, "_iter_darwin_fd_targets", blocked_scan)
    results = []
    path = tmp_path / "state.db"
    first = threading.Thread(
        target=lambda: results.append(_iter_darwin_sidecar_holders(path)), daemon=True
    )
    second = threading.Thread(
        target=lambda: results.append(_iter_darwin_sidecar_holders(path)), daemon=True
    )
    third = threading.Thread(
        target=lambda: results.append(_iter_darwin_sidecar_holders(path)), daemon=True
    )
    first.start()
    try:
        assert entered.wait(2.0), "first libproc scan never started"
        second.start()
        second.join(2.0)
        assert not second.is_alive(), "a concurrent caller waited for a second scan timeout"
        assert len(scans) == 1, "a concurrent caller started another libproc worker"
        assert results == [[]], "a caller sharing a stuck scan must fail open"

        # Let the original caller hit its unchanged production deadline while
        # its libproc worker stays blocked inside the kernel-equivalent call.
        first.join(dbfile._DARWIN_FD_SCAN_TIMEOUT_SECONDS + 2.0)
        assert not first.is_alive(), "the initial caller exceeded its production deadline"
        third.start()
        third.join(2.0)
        assert not third.is_alive(), "a later caller started another timed-out scan"
        assert len(scans) == 1, "only one libproc worker may remain in flight"
        assert results == [[], [], []]
    finally:
        release.set()
        first.join(2.0)
        if second.ident is not None:
            second.join(2.0)
        if third.ident is not None:
            third.join(2.0)
        scan_worker = getattr(dbfile, "_DARWIN_FD_SCAN_WORKER", None)
        if scan_worker is not None:
            scan_worker.join(2.0)
