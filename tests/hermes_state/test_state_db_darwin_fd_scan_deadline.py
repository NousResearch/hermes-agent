"""Deadline on the macOS libproc fd scan (#115583).

Every ``sqlite3.connect`` on ``state.db`` runs the deleted-WAL guard, which on
macOS enumerates every fd of every process via libproc. A single process
holding thousands of vnode descriptors (e.g. Docker Desktop file sharing)
stalled the scan 20s+ with no bound. ``_iter_darwin_fd_targets`` must give up
after ~1s and return partial results fail-open (the guard's contract: no
holders found -> no refusal) instead of blocking the connect path.
"""

from __future__ import annotations

import ctypes
import logging
import struct
import time

import hermes_state_dbfile as dbfile
from hermes_state_dbfile import _iter_darwin_fd_targets, _iter_darwin_sidecar_holders

# 300 fds x 20ms per probe = ~6s unbounded; the ~1s budget must cut it short.
N_FDS = 300
PROBE_DELAY = 0.02
# Fixed code returns at ~1s; unbounded code takes ~6s. Wide margin either way.
ELAPSED_LIMIT = 4.0


class _SlowLibproc:
    """Fake libproc: one pid with N_FDS descriptors, each probe sleeps."""

    def __init__(self, n_fds=N_FDS, delay=PROBE_DELAY):
        self._payload = b"".join(
            struct.pack("<iI", fd, 0) for fd in range(3, 3 + n_fds)
        )
        self._delay = delay
        self.probes = 0

    def proc_pidinfo(self, pid, flavor, arg, buf, size):
        assert len(self._payload) < size
        ctypes.memmove(buf, self._payload, len(self._payload))
        return len(self._payload)

    def proc_pidfdinfo(self, pid, fd, flavor, record, size):
        time.sleep(self._delay)
        self.probes += 1
        return 0  # not a vnode of interest; skipped


def _patch_libproc(monkeypatch):
    lib = _SlowLibproc()
    monkeypatch.setattr(dbfile, "_darwin_libproc", lambda: lib)
    monkeypatch.setattr(dbfile, "_darwin_all_pids", lambda _lib: [4242])
    return lib


def test_fd_scan_aborts_on_budget_and_returns_partial(monkeypatch, caplog):
    lib = _patch_libproc(monkeypatch)
    with caplog.at_level(logging.WARNING, logger="hermes_state"):
        started = time.monotonic()
        found = list(_iter_darwin_fd_targets())
        elapsed = time.monotonic() - started
    assert elapsed < ELAPSED_LIMIT, f"unbounded scan took {elapsed:.1f}s"
    assert lib.probes < N_FDS, "scan ran past the deadline instead of aborting"
    assert found == []  # every probe skipped; abort surfaces as no holders
    assert any(
        "fd scan" in record.getMessage().lower() for record in caplog.records
    ), "deadline abort must log a warning"


def test_sidecar_holders_fail_open_on_slow_scan(monkeypatch, tmp_path, caplog):
    _patch_libproc(monkeypatch)
    with caplog.at_level(logging.WARNING, logger="hermes_state"):
        started = time.monotonic()
        holders = _iter_darwin_sidecar_holders(tmp_path / "state.db")
        elapsed = time.monotonic() - started
    assert elapsed < ELAPSED_LIMIT, f"guard stalled {elapsed:.1f}s on slow scan"
    assert holders == []  # fail-open: no refusal minted from a partial scan
