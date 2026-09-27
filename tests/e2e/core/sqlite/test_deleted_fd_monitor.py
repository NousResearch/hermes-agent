"""Deterministic semantics tests for the torture chamber's deleted-fd monitor (#125591).

SQLite's own final close unlinks ``-wal``/``-shm`` *before* closing their descriptors, so a monitor scan
that lands inside that interval observes a ``(deleted)`` sidecar the process is about to drop — that must
not count as a store-robbery hit. A live holder whose generation was unlinked beneath it (#121433) keeps
the descriptor for as long as its connection lives.

The tests drive the monitor scan by scan (``Chamber(monitor=False)`` + ``_scan_once()``) against the
``fdprobe`` role, which holds an unlinked fd for exactly as long as the test says, so both sides of the
distinction — and the immediate main-file hit — are deterministic on Linux, with no real SQLite race and
no reliance on wall-clock timing.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from tests.e2e.core.sqlite._helpers import DELETED_SIDECAR_CONFIRM_SCANS, Chamber

pytestmark = pytest.mark.skipif(not os.path.isdir("/proc/self/fd"),
                                reason="the /proc/<pid>/fd monitor semantics are Linux-only")


def _probe(chamber: Chamber, name: str, suffix: str = "-shm") -> None:
    """Spawn the ``fdprobe`` role on this chamber's own ``state.db<suffix>`` path — a placeholder file is
    enough, because the monitor matches path names, and the path stays inside the chamber's private home."""
    target = Path(f"{chamber.db}{suffix}")
    assert target.parent == chamber.hermes_home, target
    target.write_bytes(b"x" * 16)
    chamber.spawn("fdprobe", name, target=str(target))


def test_final_close_window_does_not_hit(tmp_path):
    """A deleted sidecar fd that is gone by the next scans is SQLite's final-close interval, not a hit."""
    ch = Chamber(tmp_path / "ch", monitor=False)
    try:
        _probe(ch, "closer")
        ch.wait_event("closer", "held")
        ch._scan_once()  # observes the deleted sidecar fd: pending, not yet a hit
        assert ch.deleted_hits_snapshot() == []
        ch.stop("closer")  # the fd closes with the process: the final-close outcome
        for _ in range(DELETED_SIDECAR_CONFIRM_SCANS):
            ch._scan_once()
        assert ch.deleted_hits_snapshot() == []
    finally:
        ch.shutdown()


def test_held_sidecar_confirms_exactly_one_hit(tmp_path):
    """A deleted sidecar fd held across the confirmation window is a live holder: exactly one hit."""
    ch = Chamber(tmp_path / "ch", monitor=False)
    try:
        _probe(ch, "holder")
        ch.wait_event("holder", "held")
        for _ in range(DELETED_SIDECAR_CONFIRM_SCANS - 1):
            ch._scan_once()
        assert ch.deleted_hits_snapshot() == []  # persistence not yet proven
        ch._scan_once()
        assert ch.deleted_hits_snapshot() == [
            ("holder", ch.procs["holder"].pid, f"{ch.db}-shm (deleted)")]
        ch.request_stop("holder")
        ch.reap("holder")
    finally:
        ch.shutdown()


def test_deleted_main_file_is_an_immediate_hit(tmp_path):
    """SQLite never unlinks the main file itself, so a deleted main-file descriptor stays an immediate hit."""
    ch = Chamber(tmp_path / "ch", monitor=False)
    try:
        _probe(ch, "swapped", suffix="")
        ch.wait_event("swapped", "held")
        ch._scan_once()
        assert ch.deleted_hits_snapshot() == [
            ("swapped", ch.procs["swapped"].pid, f"{ch.db} (deleted)")]
        ch.request_stop("swapped")
        ch.reap("swapped")
    finally:
        ch.shutdown()
