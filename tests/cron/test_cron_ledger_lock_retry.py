"""Cron ledger contention: a transient SQLite lock must not fail a fire.

Regression cover for the 2026-11-16 13:00-13:55 window (overseer report):
``MASTER group skim`` was booked as "Restart-safe cron worker failed after handoff: database is
locked" while its worker had not adopted the row, and ``MASTER inbox triage`` reported the legacy
"cron fire ownership was lost" wording for a claim that was never taken over by another process.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path
from unittest.mock import Mock

import pytest


@pytest.fixture
def execution_ledger(tmp_path, monkeypatch):
    from cron import executions

    monkeypatch.setattr(executions, "EXECUTIONS_FILE", tmp_path / "executions.db")
    return executions


@pytest.fixture
def fast_retries(monkeypatch):
    from cron import executions

    monkeypatch.setattr(executions, "LEDGER_LOCK_RETRY_DELAYS", (0.01, 0.01, 0.01))
    return executions


def _locked() -> sqlite3.OperationalError:
    return sqlite3.OperationalError("database is locked")


def _raise_locked():
    raise _locked()


# --- ledger retry helper -----------------------------------------------------------------------

def test_ledger_operation_retries_transient_lock(fast_retries):
    calls = []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise _locked()
        return "committed"

    assert fast_retries.retry_on_lock(flaky) == "committed"
    assert len(calls) == 3


def test_ledger_operation_does_not_retry_a_real_error(fast_retries):
    calls = []

    def broken():
        calls.append(1)
        raise sqlite3.OperationalError("no such table: executions")

    with pytest.raises(sqlite3.OperationalError, match="no such table"):
        fast_retries.retry_on_lock(broken)
    assert len(calls) == 1


def test_ledger_operation_gives_up_after_the_retry_budget(fast_retries):
    calls = []

    def always_locked():
        calls.append(1)
        raise _locked()

    with pytest.raises(sqlite3.OperationalError, match="database is locked"):
        fast_retries.retry_on_lock(always_locked)
    assert len(calls) == 1 + len(fast_retries.LEDGER_LOCK_RETRY_DELAYS)


def test_ledger_connect_widens_the_busy_timeout(execution_ledger):
    """The 5s default expired under load; the ledger keeps a 15s window before SQLITE_BUSY."""
    conn = execution_ledger._connect()
    try:
        assert conn.execute("PRAGMA busy_timeout").fetchone()[0] == execution_ledger.LEDGER_BUSY_TIMEOUT_MS
    finally:
        conn.close()
    assert execution_ledger.LEDGER_BUSY_TIMEOUT_MS >= 15000


def test_handoff_ledger_write_survives_a_transient_lock(execution_ledger, monkeypatch):
    """A lock during the handoff CAS must be retried, not turned into a dispatch failure."""
    record = execution_ledger.create_execution("job-lock", source="builtin")
    real_transaction = execution_ledger._transaction
    state = {"failed": False}

    def flaky_transaction():
        if not state["failed"]:
            state["failed"] = True
            raise _locked()
        return real_transaction()

    monkeypatch.setattr(execution_ledger, "_transaction", flaky_transaction)
    monkeypatch.setattr(execution_ledger, "LEDGER_LOCK_RETRY_DELAYS", (0.01,))
    marked = execution_ledger.mark_execution_handoff_pending(record["id"])
    assert marked is not None and marked["handoff_pending"] == 1


# --- post-handoff waiter -----------------------------------------------------------------------

def test_waiter_survives_a_transient_ledger_lock(fast_retries, monkeypatch):
    from cron import scheduler

    statuses = iter(
        [
            _locked(),  # first ledger read loses the race with the worker's own write
            {"id": "exec-1", "status": "completed"},
            {"id": "exec-1", "status": "completed"},
        ]
    )

    def get(_execution_id):
        value = next(statuses)
        if isinstance(value, Exception):
            raise value
        return value

    monkeypatch.setattr(scheduler, "get_execution", Mock(side_effect=get))
    process = Mock()
    process.wait.side_effect = [__import__("subprocess").TimeoutExpired(cmd="worker", timeout=1.0), 0]
    process.poll.return_value = 0

    assert scheduler._wait_for_external_cron_worker(process, execution_id="exec-1") is True


def test_waiter_keeps_the_payload_while_the_worker_is_still_booting(tmp_path, fast_retries, monkeypatch):
    """Deleting the payload from the waiter's finally block killed a booting worker (#128509 class)."""
    from cron import scheduler

    payload = tmp_path / "exec-1.json"
    payload.write_text('{"job": {}}', encoding="utf-8")
    monkeypatch.setattr(scheduler, "get_execution", Mock(side_effect=lambda _id: _raise_locked()))
    process = Mock()
    process.wait.side_effect = __import__("subprocess").TimeoutExpired(cmd="worker", timeout=1.0)
    process.poll.return_value = None  # worker alive, still booting

    with pytest.raises(scheduler._ExternalWorkerPostHandoffError, match="database is locked"):
        scheduler._wait_for_external_cron_worker(
            process, execution_id="exec-1", handoff_files=(payload,))
    assert payload.exists(), "a live worker must still find its payload on disk"


def test_waiter_cleans_up_after_a_dead_worker(tmp_path, fast_retries, monkeypatch):
    from cron import scheduler

    payload = tmp_path / "exec-2.json"
    payload.write_text('{"job": {}}', encoding="utf-8")
    monkeypatch.setattr(
        scheduler, "get_execution",
        Mock(return_value={"id": "exec-2", "status": "completed"}))
    process = Mock()
    process.wait.return_value = 0
    process.poll.return_value = 0

    assert scheduler._wait_for_external_cron_worker(
        process, execution_id="exec-2", handoff_files=(payload,)) is True
    assert not payload.exists()


# --- truthful ownership-loss reason -------------------------------------------------------------

def test_lost_ownership_carries_its_cause(monkeypatch):
    """The run must report WHY the claim was lost, not the legacy fallback wording."""
    from cron import cancellation, scheduler

    monkeypatch.setattr(scheduler, "_RUN_CLAIM_HEARTBEAT_SECONDS", 0.01)
    monkeypatch.setattr(scheduler, "_FIRE_CLAIM_MISS_CONFIRM_SECONDS", 0.01)
    # First call is the pre-execution validation (owns the claim); every later heartbeat misses.
    calls = {"n": 0}

    def heartbeat(*_a, **_k):
        calls["n"] += 1
        return calls["n"] == 1

    monkeypatch.setattr(scheduler, "heartbeat_fire_claim", heartbeat)
    monkeypatch.setattr(scheduler, "self_removal_delivery_allowed", lambda _job_id: False)

    seen = {}

    def run(lost_ownership):
        deadline = time.monotonic() + 5.0
        while not lost_ownership.is_set() and time.monotonic() < deadline:
            time.sleep(0.01)
        seen["set"] = lost_ownership.is_set()
        seen["reason"] = cancellation.cancel_reason(lost_ownership)
        return True

    job = {"id": "job-1", "fire_claim": {"by": "host:1"}}
    assert scheduler._run_with_fire_claim_heartbeat(job, run) is True
    assert seen["set"] is True
    assert seen["reason"] == "another process now owns this job's fire claim"
    assert seen["reason"] != "cron fire ownership was lost"


def test_renewal_failure_names_the_underlying_error(monkeypatch):
    from cron import cancellation, scheduler

    monkeypatch.setattr(scheduler, "_RUN_CLAIM_HEARTBEAT_SECONDS", 0.01)
    monkeypatch.setattr(scheduler, "_FIRE_CLAIM_HEARTBEAT_GRACE_SECONDS", 0.0)

    calls = {"n": 0}

    def boom(*_a, **_k):
        calls["n"] += 1
        if calls["n"] == 1:
            return True  # pre-execution validation succeeds
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(scheduler, "heartbeat_fire_claim", boom)

    seen = {}

    def run(lost_ownership):
        deadline = time.monotonic() + 5.0
        while not lost_ownership.is_set() and time.monotonic() < deadline:
            time.sleep(0.01)
        seen["reason"] = cancellation.cancel_reason(lost_ownership)
        return True

    assert scheduler._run_with_fire_claim_heartbeat(
        {"id": "job-2", "fire_claim": {"by": "host:2"}}, run) is True
    assert "could not be renewed" in seen["reason"]
    assert "database is locked" in seen["reason"]
