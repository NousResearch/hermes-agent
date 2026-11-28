"""Board #741: EVERY wait on the connection-lifecycle lock is bounded, not just the raw read.

`offline_file_access` got a bounded acquire first, and that alone did not fix the class: the
guard builds its refusal message through `_unreadable_reason` -> `has_live_connection`, which
re-entered `_live_lock` UNBOUNDED. A 5.0s bound therefore produced a 300s scan — one acquire,
measured — because the second wait inherited the holder's wedge.

Each test below wedges the lock from another thread and asserts the caller returns on its own
budget. The lock is a distinct object per test, so a leaked holder cannot cross-contaminate.
"""
from __future__ import annotations

import threading
import time

import pytest

from hermes_cli import sqlite_safe_read


@pytest.fixture()
def wedged_lock(monkeypatch):
    """Hold the connection-lifecycle lock from another thread for the whole test."""
    lock = threading.RLock()
    monkeypatch.setattr(sqlite_safe_read, "_live_lock", lock)
    monkeypatch.setattr(sqlite_safe_read, "_live_connections", {})

    held, release = threading.Event(), threading.Event()

    def holder() -> None:
        lock.acquire()
        held.set()
        release.wait(30)
        try:
            lock.release()
        except RuntimeError:
            pass

    thread = threading.Thread(target=holder, daemon=True)
    thread.start()
    assert held.wait(5), "the wedging thread never took the lock"
    try:
        yield lock
    finally:
        release.set()
        thread.join(timeout=10)


def test_has_live_connection_does_not_wait_for_a_wedged_holder(wedged_lock):
    """The registry read is a DIAGNOSTIC: it must answer on its budget, not behind the wedge."""
    started = time.monotonic()
    answer = sqlite_safe_read.has_live_connection("/tmp/whatever.db", timeout=0.5)
    elapsed = time.monotonic() - started

    assert answer is None, "a wedged registry must answer 'cannot say', never True/False"
    assert elapsed < 5, f"registry read waited {elapsed:.1f}s on a 0.5s budget"


def test_the_default_budget_is_bounded(wedged_lock, monkeypatch):
    """No caller-supplied timeout must still be bounded by the module default."""
    monkeypatch.setattr(sqlite_safe_read, "_LIVE_LOCK_TIMEOUT_S", 0.3)
    started = time.monotonic()
    assert sqlite_safe_read.has_live_connection("/tmp/whatever.db") is None
    assert time.monotonic() - started < 5


def test_successful_read_still_reports_presence_and_absence():
    """The bound must not change the verdict when the lock is free."""
    assert sqlite_safe_read.has_live_connection("/tmp/definitely-not-tracked.db") is False
    sqlite_safe_read.track_connection("/tmp/tracked.db")
    try:
        assert sqlite_safe_read.has_live_connection("/tmp/tracked.db") is True
    finally:
        sqlite_safe_read.untrack_connection("/tmp/tracked.db")


def test_unreadable_reason_names_the_wedge_not_the_size(wedged_lock, monkeypatch):
    """A wedged registry is the real cause; blaming the scan cap sent readers chasing size."""
    from cron import lifecycle_guard

    monkeypatch.setattr(lifecycle_guard, "_MAX_REFERENCED_SCRIPT_BYTES", 1024)
    reason = lifecycle_guard._unreadable_reason(__import__("pathlib").Path("/tmp/script.sh"))

    assert "lock" in reason.lower(), reason
    assert "larger than the scan cap" not in reason, reason
