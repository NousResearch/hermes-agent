"""Occurrence-scoped terminal outcomes for cron attempts.

The cron ``fire_claim`` is transport-level: a heartbeat can lose the local fence, or a later fire
can take the claim, AFTER the run has already delivered. Recording those as interrupted failures
produced the fleet-wide phantom rows of 2026-09-13 — a delivered run's ledger row written ``failed``
one second after its successful delivery, which flipped ``last_status`` to ``error``.

Contract pinned here: the attempt's own ledger row decides what the attempt is recorded as, and the
job's ``last_status`` belongs to the newest attempt of its occurrence. A delivered result is never
demoted to a plain failure; an attempt that a successor (or a completed sibling for the same
scheduled instant) has overtaken is recorded ``superseded`` instead of ``failed``.
"""

from __future__ import annotations

import pytest

INSTANT = "2026-09-13T20:00:00+00:00"


@pytest.fixture()
def executions(monkeypatch, tmp_path):
    import cron.executions as executions_mod

    monkeypatch.setattr(
        executions_mod, "EXECUTIONS_FILE", tmp_path / "cron" / "executions.db"
    )
    return executions_mod


def _running_attempt(executions, job_id: str, *, instant: str = INSTANT):
    row = executions.create_execution(job_id, source="builtin", scheduled_instant=instant)
    executions.mark_execution_running(row["id"])
    return row["id"]


def test_classify_is_a_pure_contract():
    from cron.attempt_outcome import (
        COMPLETED, INTERRUPTED, SUPERSEDED, classify_post_delivery_outcome,
    )

    # A delivered result under a latch that fired post hoc still ran: complete it.
    assert classify_post_delivery_outcome(
        delivered=True, owns_job_record=True, occurrence_completed=False) == COMPLETED
    # Delivered, but a later fire already owns the job record: superseded, never a plain failure.
    assert classify_post_delivery_outcome(
        delivered=True, owns_job_record=False, occurrence_completed=False) == SUPERSEDED
    # Nothing delivered and the occurrence already has a completed survivor: that row owns the
    # status, so this attempt must not be recorded as another failure.
    assert classify_post_delivery_outcome(
        delivered=False, owns_job_record=True, occurrence_completed=True) == SUPERSEDED
    assert classify_post_delivery_outcome(
        delivered=False, owns_job_record=False, occurrence_completed=False) == SUPERSEDED
    # Genuine loss: nothing delivered, this is still the newest attempt, nothing completed.
    assert classify_post_delivery_outcome(
        delivered=False, owns_job_record=True, occurrence_completed=False) == INTERRUPTED


def test_delivered_attempt_is_completed_not_the_interrupted_failure(executions):
    from cron.attempt_outcome import COMPLETED, record_post_delivery_outcome

    execution_id = _running_attempt(executions, "job-delivered")

    outcome = record_post_delivery_outcome(
        "job-delivered", execution_id, delivered=True,
        error="Interrupted by shutdown before terminal completion.",
    )

    assert outcome == COMPLETED
    row = executions.get_execution(execution_id)
    assert row["status"] == "completed"
    assert row["error"] is None


def test_delivered_attempt_overtaken_by_a_newer_fire_is_superseded(executions):
    from cron.attempt_outcome import SUPERSEDED, record_post_delivery_outcome

    stale = _running_attempt(executions, "job-overtaken")
    newer = executions.create_execution(
        "job-overtaken", source="builtin", scheduled_instant="2026-09-13T21:00:00+00:00")
    executions.mark_execution_running(newer["id"])

    outcome = record_post_delivery_outcome("job-overtaken", stale, delivered=True)

    assert outcome == SUPERSEDED
    row = executions.get_execution(stale)
    assert row["status"] == "superseded"
    assert "superseded" in (row["error"] or "").lower()
    # The successor's own attempt is untouched by the stale attempt's bookkeeping.
    assert executions.get_execution(newer["id"])["status"] == "running"


def _bind_occurrence_directly(executions, execution_id: str, instant: str) -> None:
    """Model the ledger shape an older build could leave behind.

    The provider path creates a row before the occurrence is known and binds the instant afterwards,
    so a ledger can hold a live attempt whose occurrence another attempt already completed. The
    classification must still refuse to report that attempt as a failure.
    """
    import sqlite3

    conn = sqlite3.connect(executions.EXECUTIONS_FILE)
    conn.execute(
        "UPDATE executions SET scheduled_instant=? WHERE id=?", (instant, execution_id))
    conn.commit()
    conn.close()


def test_failed_attempt_next_to_a_completed_survivor_is_superseded(executions):
    from cron.attempt_outcome import SUPERSEDED, record_post_delivery_outcome

    winner = executions.create_execution("job-double", source="builtin", scheduled_instant=INSTANT)
    executions.mark_execution_running(winner["id"])
    executions.finish_execution(winner["id"], success=True)
    loser = executions.create_execution("job-double", source="provider")
    executions.mark_execution_running(loser["id"])
    _bind_occurrence_directly(executions, loser["id"], INSTANT)

    outcome = record_post_delivery_outcome("job-double", loser["id"], delivered=False, error="boom")

    assert outcome == SUPERSEDED
    assert executions.get_execution(loser["id"])["status"] == "superseded"
    # The occurrence's surviving row stays the one that decides last_status.
    assert executions.occurrence_completed("job-double", INSTANT) is True
    assert executions.occurrence_winner("job-double", INSTANT) == "completed"


def test_undelivered_newest_attempt_keeps_the_failure_outcome(executions):
    from cron.attempt_outcome import INTERRUPTED, record_post_delivery_outcome

    execution_id = _running_attempt(executions, "job-genuine-loss")

    outcome = record_post_delivery_outcome(
        "job-genuine-loss", execution_id, delivered=False, error="transport died")

    assert outcome == INTERRUPTED
    row = executions.get_execution(execution_id)
    assert row["status"] == "failed"
    assert row["error"] == "transport died"


def test_job_status_write_is_refused_for_an_overtaken_attempt(executions):
    from cron.attempt_outcome import job_status_write_blocked

    stale = _running_attempt(executions, "job-gate")
    assert job_status_write_blocked("job-gate", stale) is None

    executions.create_execution(
        "job-gate", source="builtin", scheduled_instant="2026-09-13T21:00:00+00:00")

    reason = job_status_write_blocked("job-gate", stale)
    assert reason and "newer attempt" in reason


def test_job_status_write_is_refused_when_the_occurrence_already_completed(executions):
    from cron.attempt_outcome import job_status_write_blocked

    winner = executions.create_execution("job-done", source="builtin", scheduled_instant=INSTANT)
    executions.mark_execution_running(winner["id"])
    executions.finish_execution(winner["id"], success=True)
    loser = executions.create_execution("job-done", source="provider")
    executions.mark_execution_running(loser["id"])
    _bind_occurrence_directly(executions, loser["id"], INSTANT)

    reason = job_status_write_blocked("job-done", loser["id"])
    assert reason and "already completed" in reason
