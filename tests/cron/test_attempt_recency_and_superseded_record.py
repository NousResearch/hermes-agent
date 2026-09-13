"""The newest-attempt tiebreak and the SUPERSEDED job-record contract.

WH-CREATED-5A4D2A184BBA, legs AC4 and AC3.

AC4 — `cron.executions.newest_attempt_id` used `ORDER BY claimed_at DESC, id DESC`. `claimed_at` is
wall clock and `id` is a random uuid4, so two fires inside the SAME clock tick were ordered by
whichever uuid happened to sort higher: the attempt trusted to own the job record could be the older
one. The tiebreak is now `rowid DESC` (monotonic insert order), and every recency decision
(`attempt_is_newest`, `attempt_owns_job_record`, `job_status_write_blocked`) inherits it.

AC3 — a SUPERSEDED attempt is recorded on its OWN ledger row (`superseded=True`) and must NOT write
the job record: `last_status` keeps the value written by the attempt that owns it and `failure_streak`
is neither advanced nor reset, because the job record describes exactly one attempt's outcome.
`job_status_write_blocked` is what states that reason to the scheduler.
"""

from __future__ import annotations

import sqlite3

import pytest

INSTANT = "2026-09-13T20:00:00+00:00"
SAME_TICK = "2026-09-13T20:00:00.000000+00:00"


@pytest.fixture()
def executions(monkeypatch, tmp_path):
    import cron.executions as executions_mod

    monkeypatch.setattr(
        executions_mod, "EXECUTIONS_FILE", tmp_path / "cron" / "executions.db"
    )
    return executions_mod


def _rewrite_row(executions, old_id: str, new_id: str) -> None:
    """Force an id + a shared claimed_at so the ORDER BY tiebreak is the only discriminator."""
    con = sqlite3.connect(executions.EXECUTIONS_FILE)
    try:
        con.execute(
            "UPDATE executions SET id=?, claimed_at=? WHERE id=?", (new_id, SAME_TICK, old_id)
        )
        con.commit()
    finally:
        con.close()


def test_same_tick_fires_order_by_insert_not_by_uuid(executions):
    """RED before the fix: `id DESC` picked the lexicographically higher — but OLDER — attempt."""
    from cron.attempt_outcome import attempt_is_newest

    older = executions.create_execution("job-same-tick", source="builtin", scheduled_instant=INSTANT)
    newer = executions.create_execution("job-same-tick", source="builtin", scheduled_instant=INSTANT)
    # The older fire carries the HIGHER uuid: uuid4 ordering would call it the newest attempt.
    _rewrite_row(executions, older["id"], "ffffffff-0000-4000-8000-000000000001")
    _rewrite_row(executions, newer["id"], "00000000-0000-4000-8000-000000000002")
    older_id, newer_id = "ffffffff-0000-4000-8000-000000000001", "00000000-0000-4000-8000-000000000002"

    assert executions.newest_attempt_id("job-same-tick") == newer_id
    assert attempt_is_newest("job-same-tick", newer_id) is True
    assert attempt_is_newest("job-same-tick", older_id) is False


def test_superseded_attempt_is_ledger_only_and_its_reason_is_explicit(executions):
    from cron.attempt_outcome import (
        SUPERSEDED,
        finish_kwargs,
        job_status_write_blocked,
        record_post_delivery_outcome,
    )

    older = executions.create_execution("job-superseded", source="builtin", scheduled_instant=INSTANT)
    executions.mark_execution_running(older["id"])
    executions.create_execution("job-superseded", source="builtin", scheduled_instant=INSTANT)

    # AC3: a superseded attempt states WHY it must not rewrite the job record …
    assert job_status_write_blocked("job-superseded", older["id"]) == (
        "a newer attempt for this job owns its status"
    )
    # … and its terminal write is a superseded ledger row, never a plain failure.
    kwargs = finish_kwargs(SUPERSEDED, error="fire claim ownership lost")
    assert kwargs["superseded"] is True
    assert kwargs["success"] is False

    written: dict = {}

    def record(execution_id, **kw):
        written.update(kw)
        return None

    outcome = record_post_delivery_outcome(
        "job-superseded", older["id"], delivered=True,
        error="fire claim ownership lost", finish=record,
    )
    assert outcome == SUPERSEDED
    assert written["superseded"] is True
    # The job record is untouched by this path: nothing here writes last_status/failure_streak, so a
    # newer attempt's status survives (the scheduler passes owns_job_record=False on every
    # SUPERSEDED branch).
    assert "last_status" not in written and "failure_streak" not in written
