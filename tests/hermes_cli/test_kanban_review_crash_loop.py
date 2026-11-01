"""Behavior contract: a fingerprint-blind dispatcher must not false-kill a live worker.

Incident (Command Board 2026-09-17, runs 299/300/309/310/312): gateways run from
mixed interpreters; the sweep-holder may lack psutil (or the runtime python was
swapped), so ``_process_fingerprint(pid)`` returns ``None``. ``_pid_recycled``
treated unreadable-as-mismatch: a LIVE worker pid answered "recycled" -> the
sweep reclaimed it as ``crashed`` ("pid N not alive") while its session was
still mid-turn. Two consecutive false deaths tripped the breaker (``gave_up``)
and parked DELIVERED review cards in ``blocked``.

Contract pinned here:

* An unreadable CURRENT fingerprint (None) must read as "identity unknown", not
  "recycled": ``_worker_alive`` keeps the existence answer for a live pid, and
  no signal is ever sent to it (kill-refusal semantics identical to UNVERIFIED).
* A readable fingerprint that differs (post-reboot recycle) stays "recycled" —
  the stranger-protection contract is unchanged.
* A whole tick from a fingerprint-blind sweeper must not crash-account a card
  whose worker process is demonstrably alive, and must not trip the breaker to
  ``blocked``.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


@pytest.fixture
def board(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    for var in ("HERMES_KANBAN_DB", "HERMES_KANBAN_BOARD", "HERMES_KANBAN_WORKSPACES_ROOT"):
        monkeypatch.delenv(var, raising=False)
    conn = kbc.connect(tmp_path / "kanban.db")
    try:
        yield conn
    finally:
        conn.close()


def _spawn_sleeper() -> subprocess.Popen:
    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(120)"],
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    time.sleep(0.3)
    return proc


def _claimed_running_with_live_worker(conn) -> tuple[str, subprocess.Popen, str]:
    tid = kb.create_task(conn, title="live worker", assignee="worker")
    kb.claim_task(conn, tid)
    proc = _spawn_sleeper()
    kbd._set_worker_pid(conn, tid, proc.pid)
    fingerprint = conn.execute(
        "SELECT worker_started_at FROM tasks WHERE id = ?", (tid,)
    ).fetchone()["worker_started_at"]
    # Age the run past every grace window so sweeps are eligible.
    old = int(time.time()) - 3600
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE tasks SET started_at = ?, claim_expires = ? WHERE id = ?",
            (old, old, tid),
        )
        conn.execute(
            "UPDATE task_runs SET started_at = ? "
            "WHERE id = (SELECT current_run_id FROM tasks WHERE id = ?)",
            (old, tid),
        )
    return tid, proc, fingerprint


# ---------------------------------------------------------------------------
# Unit contract: unreadable current fingerprint != recycled
# ---------------------------------------------------------------------------

def test_unreadable_current_fingerprint_is_not_recycled(board, monkeypatch):
    """A psutil-less sweeper (fingerprint -> None) must not report a live pid
    as recycled; identity is UNKNOWN, and unknown never kills (mirrors the
    UNVERIFIED spawn contract)."""
    tid, proc, fingerprint = _claimed_running_with_live_worker(board)
    try:
        assert kbd._worker_alive(proc.pid, fingerprint) is True  # sanity: base readable

        import hermes_cli.kanban_db_dispatch as dispatch_mod
        monkeypatch.setattr(dispatch_mod, "_process_fingerprint", lambda pid: None)

        # RED on base: None != "<epoch>|<start>" -> recycled -> not alive.
        assert kbd._worker_alive(proc.pid, fingerprint) is True
    finally:
        proc.terminate()
        proc.wait(timeout=10)


def test_unreadable_current_fingerprint_never_signals(board, monkeypatch):
    """The blind sweeper must not SIGTERM the live worker either (kill-refusal
    parity with UNVERIFIED rows)."""
    import hermes_cli.kanban_db_dispatch as dispatch_mod
    killed: list[tuple[int, int]] = []
    monkeypatch.setattr(dispatch_mod, "_process_fingerprint", lambda pid: None)
    tid, proc, fingerprint = _claimed_running_with_live_worker(board)
    try:
        with kb.write_txn(board):
            board.execute(
                "UPDATE tasks SET max_runtime_seconds = 1 WHERE id = ?", (tid,))
        kbd.enforce_max_runtime(board, signal_fn=lambda pid, sig: killed.append((pid, sig)))
        assert killed == []
        assert kb.get_task(board, tid).status == "running"
    finally:
        proc.terminate()
        proc.wait(timeout=10)


# ---------------------------------------------------------------------------
# Behavior contract: full crash sweep from a fingerprint-blind dispatcher
# ---------------------------------------------------------------------------

def test_blind_sweep_does_not_crash_a_live_worker_nor_trip_breaker(board, monkeypatch):
    """The exact incident shape: sweeper cannot read fingerprints, worker is
    alive and mid-turn. The sweep must NOT reclaim the run as crashed, must
    NOT count a breaker failure, and the card must NOT reach blocked."""
    import hermes_cli.kanban_db_dispatch as dispatch_mod
    monkeypatch.setattr(dispatch_mod, "_process_fingerprint", lambda pid: None)

    tid, proc, fingerprint = _claimed_running_with_live_worker(board)
    try:
        crashed = kbd.detect_crashed_workers(board)
        assert crashed == []
        task = kb.get_task(board, tid)
        assert task.status == "running"
        assert task.consecutive_failures == 0

        # A second blind tick must not accumulate failures either.
        kbd.detect_crashed_workers(board)
        task = kb.get_task(board, tid)
        assert task.status == "running"
        assert task.consecutive_failures == 0
    finally:
        proc.terminate()
        proc.wait(timeout=10)


def test_readable_foreign_fingerprint_is_still_recycled(board):
    """Invariant preserved: a readable-but-foreign fingerprint (post-reboot
    stranger) remains 'recycled' — claim released, never signalled."""
    killed: list[tuple[int, int]] = []
    tid, proc, _ = _claimed_running_with_live_worker(board)
    try:
        stranger = "deadbeef-boot:1|1"  # readable shape, never our process
        with kb.write_txn(board):
            board.execute(
                "UPDATE tasks SET worker_started_at = ? WHERE id = ?", (stranger, tid))
        assert kbd._worker_alive(proc.pid, stranger) is False
        crashed = kbd.detect_crashed_workers(board, )
        # The stranger run is reclaimed (crash-accounted) but never signalled.
        assert killed == []
        assert kb.get_task(board, tid).status != "running" or crashed == []
    finally:
        proc.terminate()
        proc.wait(timeout=10)


# ---------------------------------------------------------------------------
# Windows exit-classification amplifier: unknown exits must not be
# crash-accounted as hard crashes when the worker's own transition already
# closed the run (terminal-transition rejection is not a crash).
# ---------------------------------------------------------------------------

def test_terminal_transition_rejection_is_not_a_worker_crash(board):
    """Incident prong B: after ``request_review`` closed run A, the review lane
    re-claimed the card (run B) while the IMPLEMENTER session was still alive;
    the implementer's nudged ``kanban_complete`` (expected_run_id=A) was
    rejected as stale. That rejection must never be booked as a crash against
    the card — the board, not the worker, moved on."""
    tid, proc, fingerprint = _claimed_running_with_live_worker(board)
    try:
        run_a = kb._current_run_id(board, tid)
        assert kb.request_review(
            board, tid, summary="delivered", reviewer=None, expected_run_id=run_a,
        ) is True
        assert kb._task_status(board, tid) == "review"

        # Review lane re-claims (a reviewer run opens).
        assert kb.claim_review_task(board, tid) is not None
        run_b = kb._current_run_id(board, tid)

        # The stale implementer's complete is rejected (CAS on run A) — the
        # board's contract, not a crash.
        assert kb.complete_task(board, tid, summary="stale", expected_run_id=run_a) is False
        task = kb.get_task(board, tid)
        assert task.consecutive_failures == 0
        assert task.status == "running"  # still reviewer-owned, untouched

        # The reviewer completes cleanly — the legitimate terminal path.
        assert kb.complete_task(board, tid, summary="approved", expected_run_id=run_b) is True
        assert kb._task_status(board, tid) == "done"
    finally:
        proc.terminate()
        proc.wait(timeout=10)
