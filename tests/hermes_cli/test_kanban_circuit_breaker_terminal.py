"""Hermetic tests for the dispatcher retry / protocol circuit breaker.

Policy under test (school-system M9-P04 follow-up):

* The circuit breaker is TERMINAL: once a task trips (``gave_up``), the
  dispatcher's ``recompute_ready`` must never auto-promote it back to the
  source phase, regardless of the promotion limit the tick uses.
* Only an explicit ``kanban_unblock`` clears the hold.
* A clean worker exit without a terminal Kanban call is a protocol violation
  and counts as a failure.
* A worker whose PID disappears counts as a failure.
* The failure counter is persisted per task (survives recompute / restart).

Each test runs in an isolated ``HERMES_HOME`` with a fresh DB and a patched
liveness probe, so no production or shared Kanban state is touched.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Isolated HERMES_HOME with an empty kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    # Immediate crash reclaim: the 30s production grace window would defer the
    # reap for tests that claim a task and instantly reap it.
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _drive_worker_exit(conn, tid, fake_pid, raw_status):
    """Claim ``tid``, record ``raw_status`` for its dead worker pid, reap."""
    import hermes_cli.kanban_db as _kb
    from hermes_cli import kanban_db_dispatch as _kbd

    host_prefix = _kb._claimer_id().split(":", 1)[0]
    claimed = _kb.claim_task(conn, tid, claimer=f"{host_prefix}:mock")
    assert claimed is not None, "task was not claimable for the next attempt"
    _kbd._set_worker_pid(conn, tid, fake_pid)
    _kbd._record_worker_exit(fake_pid, raw_status)
    original_alive = _kb._pid_alive
    _kb._pid_alive = lambda p: False
    try:
        return _kbd.detect_crashed_workers(conn)
    finally:
        _kb._pid_alive = original_alive


def _drive_protocol_violation(conn, tid, fake_pid):
    """Clean exit (rc=0) without a terminal Kanban call."""
    return _drive_worker_exit(conn, tid, fake_pid, 0)


def _drive_nonzero_crash(conn, tid, fake_pid):
    """Non-zero exit crash."""
    return _drive_worker_exit(conn, tid, fake_pid, 256)


def test_protocol_violation_trip_is_terminal(kanban_home: Path) -> None:
    """Third protocol violation trips the breaker; no promotion at any limit."""
    from hermes_cli import kanban_db_dispatch as _kbd

    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="loop reproducer", assignee="worker")
        limit = _kbd._PROTOCOL_VIOLATION_FAILURE_LIMIT
        assert limit >= 2

        # Below budget: each violation leaves the task retryable.
        for i in range(limit - 1):
            _drive_protocol_violation(conn, tid, 992000 + i)
            assert kb.get_task(conn, tid).status == "ready"

        # Final violation trips the breaker -> terminal blocked.
        _drive_protocol_violation(conn, tid, 992999)
        assert kb.get_task(conn, tid).status == "blocked"

        gave_up = [e for e in kb.list_events(conn, tid) if e.kind == "gave_up"]
        assert len(gave_up) == 1

        # Dispatcher ticks at ANY promotion limit must not re-promote.
        for promotion_limit in (1, 2, 3, 5):
            assert kb.recompute_ready(conn, failure_limit=promotion_limit) == 0
            assert kb.get_task(conn, tid).status == "blocked"

        # No fourth attempt: the task is not claimable while held.
        assert kb.claim_task(conn, tid) is None


def test_explicit_unblock_clears_terminal_hold(kanban_home: Path) -> None:
    """Only an explicit kanban_unblock releases a breaker hold."""
    from hermes_cli import kanban_db_dispatch as _kbd

    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="hold then release", assignee="worker")
        for i in range(_kbd._PROTOCOL_VIOLATION_FAILURE_LIMIT):
            _drive_protocol_violation(conn, tid, 993000 + i)
        assert kb.get_task(conn, tid).status == "blocked"
        assert kb.recompute_ready(conn) == 0

        assert kb.unblock_task(conn, tid) is True
        assert kb.get_task(conn, tid).status == "ready"
        assert kb.get_task(conn, tid).consecutive_failures == 0


def test_worker_pid_gone_counts_as_failure(kanban_home: Path) -> None:
    """A disappeared worker PID is a failure, not a silent no-op."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="pid gone", assignee="worker")
        _drive_nonzero_crash(conn, tid, 994000)
        assert kb.get_task(conn, tid).consecutive_failures >= 1


def test_counter_persists_across_recompute(kanban_home: Path) -> None:
    """The per-task failure counter is durable across dispatcher ticks."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="persist", assignee="worker")
        _drive_nonzero_crash(conn, tid, 995000)
        before = kb.get_task(conn, tid).consecutive_failures
        assert before >= 1
        for _ in range(3):
            kb.recompute_ready(conn)
        assert kb.get_task(conn, tid).consecutive_failures == before


def test_regression_complete_is_done(kanban_home: Path) -> None:
    """A normal completion still lands in done and clears the counter."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="happy path", assignee="worker")
        assert kb.claim_task(conn, tid) is not None
        assert kb.complete_task(conn, tid, result="ok") is True
        assert kb.get_task(conn, tid).status == "done"
        assert kb.get_task(conn, tid).consecutive_failures == 0


def test_regression_worker_block_is_sticky(kanban_home: Path) -> None:
    """Worker-initiated blocks remain sticky and un-promoted."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="worker block", assignee="worker")
        kb.claim_task(conn, tid)
        assert kb.block_task(
            conn, tid, reason="review-required: human eyes",
            expected_run_id=kb.get_task(conn, tid).current_run_id,
        )
        assert kb.get_task(conn, tid).status == "blocked"
        assert kb.recompute_ready(conn) == 0
        assert kb.get_task(conn, tid).status == "blocked"
