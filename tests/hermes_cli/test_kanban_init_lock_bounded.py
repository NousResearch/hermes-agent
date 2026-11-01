"""Tests for the bounded kanban init lock (issue #36644).

`connect()` wrapped its entire body in an unbounded blocking `flock(LOCK_EX)`
on every call. A single process stalled inside the critical section blocked the
long-lived gateway dispatcher's next-tick `connect()` forever — no timeout, no
recovery, board silently stops being worked.

Two fixes, both covered here:
1. Fast path: once a path is initialized in this process, `connect()` skips the
   cross-process init lock entirely (nothing left to serialize), so a held lock
   cannot block a steady-state connect.
2. Bounded acquire: even on first-init, `_cross_process_init_lock` retries a
   non-blocking acquire up to a deadline, then proceeds (with a WARNING) rather
   than hanging.

Both properties used to be asserted with a wall-clock *upper* bound
(`elapsed < 5.0` / `elapsed < 8.0`). Those are load flakes, not contracts: a
3-CPU cgroup on a saturated self-hosted runner stretched the 0.6s bounded
acquire to 4.38s on a self-hosted runner with nothing wrong; widening the
ceiling only moves the load level at which it recurs. Both
are now asserted as the ordering facts they stood in for — "the init lock was
never taken" and "the timeout branch ran" — with only a generous hard ceiling
(orders of magnitude above the behaviour) left to convert a real hang into a
named failure instead of a wedged runner.
"""

from __future__ import annotations

import contextlib
import logging
import threading
import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc

# Hard ceiling for "connect() returned at all". Two orders of magnitude above
# the bounded acquire under test (0.6s), so it is not a timing assertion — it
# exists solely so an unbounded acquire fails by name instead of hanging the
# whole test session. The holder threads below outlive it deliberately.
_HANG_CEILING_SECONDS = 60.0


def _connect_in_thread():
    """Run ``kbc.connect()`` off-thread; return (returned_event, elapsed_box, thread).

    ``returned`` is set on EVERY exit path, so "did the call come back?" is an
    ordering fact rather than a stopwatch reading.
    """
    returned = threading.Event()
    box: dict[str, object] = {}

    def _run():
        start = time.monotonic()
        try:
            conn = kbc.connect()
            conn.close()
        except BaseException as exc:  # pragma: no cover - surfaced via box
            box["error"] = exc
        finally:
            box["elapsed"] = time.monotonic() - start
            returned.set()

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return returned, box, t


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    db_path = kb.kanban_db_path(board="default")
    kbc._INITIALIZED_PATHS.discard(str(db_path.resolve()))
    return home


def _hold_init_lock(db_path: Path):
    """Return (release_event, thread) holding the init lock.

    The holder waits on ``release`` for far longer than ``_HANG_CEILING_SECONDS``
    so that "connect() came back" can never be explained by the holder having
    let go on its own.
    """
    holding = threading.Event()
    release = threading.Event()

    def _holder():
        with kbc._cross_process_init_lock(db_path):
            holding.set()
            release.wait(timeout=_HANG_CEILING_SECONDS * 3)

    t = threading.Thread(target=_holder, daemon=True)
    t.start()
    assert holding.wait(timeout=30), "holder thread never acquired the lock"
    return release, t


def test_initialized_path_connect_skips_init_lock(kanban_home, monkeypatch):
    """A connect to an already-initialized path must not take the init lock.

    The contract is structural — the fast path does not *enter*
    ``_cross_process_init_lock`` at all — so it is asserted by watching the
    lock, not by timing the call.
    """
    db_path = kb.kanban_db_path(board="default")
    # Initialize once.
    kbc.connect().close()
    assert str(db_path.resolve()) in kbc._INITIALIZED_PATHS

    # Hold the init lock; a fast-path connect must not even ask for it.
    release, holder = _hold_init_lock(db_path)

    lock_entries: list[str] = []
    real_lock = kbc._cross_process_init_lock
    t: threading.Thread | None = None

    @contextlib.contextmanager
    def _spy(path: Path):
        lock_entries.append(str(path))
        with real_lock(path):
            yield

    monkeypatch.setattr(kbc, "_cross_process_init_lock", _spy)

    try:
        returned, box, t = _connect_in_thread()
        returned.wait(timeout=_HANG_CEILING_SECONDS)
        # Ordering witness: the fast path never reaches the cross-process lock.
        assert lock_entries == [], (
            f"fast-path connect entered the cross-process init lock: {lock_entries}"
        )
        # Hard ceiling only — catches a real hang, not a slow runner.
        assert returned.is_set(), (
            "fast-path connect never returned while the init lock was held"
        )
        assert "error" not in box, f"fast-path connect raised: {box.get('error')!r}"
    finally:
        release.set()
        holder.join(timeout=30)
        if t is not None:
            t.join(timeout=30)


def test_first_init_connect_is_bounded_when_lock_held(kanban_home, monkeypatch, caplog):
    """First-init connect must time out the cross-process lock and proceed.

    The contract is "the bounded branch ran": the acquire gave up after the
    deadline and the init work went ahead anyway. That is witnessed by the
    timeout WARNING plus the fact that the connect returned while the holder
    still owns the lock — no upper wall-clock bound involved.
    """
    monkeypatch.setattr(kbc, "_INIT_LOCK_TIMEOUT_SECONDS", 0.6)
    db_path = kb.kanban_db_path(board="default")

    release, holder = _hold_init_lock(db_path)
    t: threading.Thread | None = None
    try:
        with caplog.at_level(logging.WARNING, logger="hermes_cli.kanban_db"):
            returned, box, t = _connect_in_thread()
            returned.wait(timeout=_HANG_CEILING_SECONDS)

        # Hard ceiling only — an unbounded acquire fails here by name instead
        # of wedging the session (the holder does not release until the
        # `finally` below).
        assert returned.is_set(), (
            "first-init connect never returned — the init lock acquire is unbounded"
        )
        assert not release.is_set(), "holder released before connect returned"
        assert "error" not in box, f"first-init connect raised: {box.get('error')!r}"

        # It genuinely waited for the deadline rather than skipping the lock.
        # A LOWER bound is load-safe: contention only makes it larger.
        elapsed = box["elapsed"]
        assert isinstance(elapsed, float) and elapsed >= 0.4, (
            f"connect did not wait out the {kbc._INIT_LOCK_TIMEOUT_SECONDS}s "
            f"acquire deadline (elapsed {elapsed}s)"
        )

        # The timeout branch of _cross_process_init_lock is the one that logs
        # this; acquiring the lock normally logs nothing.
        timeout_warnings = [
            r.getMessage()
            for r in caplog.records
            if r.levelno >= logging.WARNING and "init lock for" in r.getMessage()
        ]
        assert timeout_warnings, (
            "no 'init lock not acquired' warning — the bounded timeout branch "
            f"did not run. Warnings seen: {[r.getMessage() for r in caplog.records]}"
        )

        # ...and the init work proceeded anyway.
        assert str(db_path.resolve()) in kbc._INITIALIZED_PATHS
    finally:
        release.set()
        holder.join(timeout=30)
        if t is not None:
            t.join(timeout=30)
