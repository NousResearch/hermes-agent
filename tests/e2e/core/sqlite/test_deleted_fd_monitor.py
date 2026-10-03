"""Deterministic semantics tests for the torture chamber's deleted-fd monitor (#125591).

SQLite's own final close unlinks ``-wal``/``-shm`` *before* closing their descriptors, so a monitor scan
that lands inside that interval observes a ``(deleted)`` sidecar the process is about to drop — that must
not count as a store-robbery hit. A live holder whose generation was unlinked beneath it (#121433) keeps
the descriptor for as long as its connection lives.

The tests drive the monitor scan by scan (``Chamber(monitor=False)`` + ``_scan_once()``) against the
``fdprobe`` role, which holds an unlinked fd for exactly as long as the test says (and can recycle one fd
number onto a second deleted file), so every side of the distinction — the immediate main-file hit included
— is deterministic on Linux, with no real SQLite race and no reliance on wall-clock timing.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from tests.e2e.core.sqlite._helpers import DELETED_SIDECAR_CONFIRM_SCANS, Chamber

pytestmark = pytest.mark.skipif(not os.path.isdir("/proc/self/fd"),
                                reason="the /proc/<pid>/fd monitor semantics are Linux-only")


def _probe(chamber: Chamber, name: str, suffix: str = "-shm", *, recycle_to_suffix: str | None = None) -> None:
    """Spawn the ``fdprobe`` role on this chamber's own ``state.db<suffix>`` path — a placeholder file is
    enough, because the monitor matches path names, and the path stays inside the chamber's private home."""
    target = Path(f"{chamber.db}{suffix}")
    assert target.parent == chamber.hermes_home, target
    target.write_bytes(b"x" * 16)
    extra: dict[str, str] = {}
    if recycle_to_suffix is not None:
        recycle_to = Path(f"{chamber.db}{recycle_to_suffix}")
        assert recycle_to.parent == chamber.hermes_home, recycle_to
        recycle_to.write_bytes(b"x" * 16)
        extra["recycle_to"] = str(recycle_to)
    chamber.spawn("fdprobe", name, target=str(target), **extra)


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


def test_recycled_fd_splices_no_streak(tmp_path):
    """An fd number is reused the moment its holder closes it: a deleted ``-shm`` for half the window and a
    deleted ``-wal`` on the *same* fd number for the rest are two generations, not one streak — no hit fires
    until a single generation survives the whole window on its own, and then the hit names the file held at
    the confirming scan, not the one frozen at first observation."""
    ch = Chamber(tmp_path / "ch", monitor=False)
    try:
        _probe(ch, "closer", recycle_to_suffix="-wal")
        ch.wait_event("closer", "held")
        half = DELETED_SIDECAR_CONFIRM_SCANS // 2
        for _ in range(half):
            ch._scan_once()  # the -shm generation alone: below the threshold
        ch.request_stop("closer")  # close the fd; the kernel hands the number to the -wal open
        recycled = ch.wait_event("closer", "recycled")
        assert recycled["fd_after"] == recycled["fd_before"], recycled  # same number, different generation
        for _ in range(half):
            ch._scan_once()  # the -wal generation alone: below the threshold too
        assert ch.deleted_hits_snapshot() == []  # neither generation earned a hit on its own
        for _ in range(half, DELETED_SIDECAR_CONFIRM_SCANS):
            ch._scan_once()  # the -wal generation reaches the threshold by itself
        assert ch.deleted_hits_snapshot() == [
            ("closer", ch.procs["closer"].pid, f"{ch.db}-wal (deleted)")]
        ch.request_stop("closer")
        ch.reap("closer")
    finally:
        ch.shutdown()
