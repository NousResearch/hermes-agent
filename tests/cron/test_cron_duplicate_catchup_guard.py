"""Regression tests for #104312: the recurring catch-up path must not re-fire
an occurrence that already ran to completion.

After a gateway restart / process-handoff write race, a job's persisted
``next_run_at`` can point at a **past instant that already ran**. The
fast-forward path decided purely on wall-clock staleness and re-fired it —
at-least-once delivery became duplicate delivery. The
``_already_completed_occurrence`` guard consults the dispatch stamp
(string-exact ``scheduled_at`` vs the stored ``next_run_at``) and the
executions ledger (latest row completed, claimed within a tight window of
the scheduled instant), and fails open on any ledger/parse error.

Real SQLite ledger under a temp HERMES_HOME; no mocks.
"""

from datetime import datetime, timedelta, timezone

import pytest

from cron.executions import (
    create_execution,
    finish_execution,
    open_ledger,
)
from cron.executions import _hermes_now
from cron.jobs import _DueScan, _evaluate_due_job

_TZ = timezone(timedelta(hours=8))
_NOW = datetime(2026, 9, 6, 12, 0, 0, tzinfo=_TZ)
# On the "0 3 * * *" lattice (today 03:00 local) — 9h stale, past the 2h grace.
_STALE = _NOW - timedelta(hours=9)
# A DISTANT lattice occurrence (yesterday 03:00) for the rollback scenario.
_DISTANT = _NOW - timedelta(hours=33)


def _iso(dt: datetime) -> str:
    return dt.isoformat()


def _job(next_run_dt: datetime, *, stamp: bool = True, job_id: str = "job-dup") -> dict:
    job = {
        "id": job_id,
        "name": "nightly",
        "enabled": True,
        "schedule": {"kind": "cron", "expr": "0 3 * * *"},
        "next_run_at": _iso(next_run_dt),
    }
    if stamp:
        job["last_dispatch"] = {
            "scheduled_at": _iso(next_run_dt),
            "dispatched_at": _iso(next_run_dt + timedelta(seconds=30)),
            "lateness_seconds": 30.0,
            "kind": "late",
        }
    return job


def _scan(job: dict) -> _DueScan:
    return _DueScan(raw_jobs=[job], now=_NOW)


def _completed_execution(job_id: str, claimed_at: datetime) -> str:
    rec = create_execution(job_id, source="test")
    finish_execution(rec["id"], success=True)
    # Pin claimed_at to the exact instant the occurrence was served.
    with open_ledger(_ledger_path()) as conn:
        conn.execute(
            "UPDATE executions SET claimed_at=? WHERE id=?",
            (_iso(claimed_at), rec["id"]),
        )
    return rec["id"]


def _ledger_path():
    from hermes_constants import get_hermes_home
    import cron.executions as exec_mod

    return exec_mod.EXECUTIONS_FILE or (
        get_hermes_home().resolve() / "cron" / "executions.db"
    )


def test_already_completed_occurrence_is_not_re_fired(monkeypatch, tmp_path):
    """The incident: stale next_run_at + a completed ledger row for the same
    served instant → the job re-anchors to the next occurrence without
    re-firing."""
    job = _job(_STALE)
    exec_id = _completed_execution(job["id"], _STALE + timedelta(seconds=30))

    fired = _evaluate_due_job(job, _scan(job), run_claim_ttl=900.0)

    assert fired is False, "an already-completed occurrence must not re-fire"
    assert job["next_run_at"] != _iso(_STALE), "next_run_at must be re-anchored forward"
    # No new execution row was created for this tick.
    with open_ledger(_ledger_path()) as conn:
        n = conn.execute(
            "SELECT COUNT(*) FROM executions WHERE job_id=?", (job["id"],)
        ).fetchone()[0]
    assert n == 1  # only the historical completed row


def test_fresh_occurrence_without_history_still_fires(monkeypatch, tmp_path):
    """No dispatch stamp and no completed row: a genuinely missed occurrence
    still fires (the fast-forward is not over-suppressed)."""
    job = _job(_STALE, stamp=False)
    job.pop("last_dispatch", None)

    fired = _evaluate_due_job(job, _scan(job), run_claim_ttl=900.0)

    assert fired is True


def test_failed_last_dispatch_still_retries(monkeypatch, tmp_path):
    """A stale occurrence whose latest ledger row FAILED must retry, not be
    suppressed by the guard."""
    job = _job(_STALE)
    rec = create_execution(job["id"], source="test")
    finish_execution(rec["id"], success=False)
    with open_ledger(_ledger_path()) as conn:
        conn.execute(
            "UPDATE executions SET claimed_at=? WHERE id=?",
            (_iso(_STALE + timedelta(seconds=30)), rec["id"]),
        )

    fired = _evaluate_due_job(job, _scan(job), run_claim_ttl=900.0)

    assert fired is True, "a failed attempt must retry rather than be skipped"


def test_distant_rollback_is_not_suppressed(monkeypatch, tmp_path):
    """next_run_at rolled back to a DISTANT occurrence while the latest
    completed row is a LATER round: the tight claimed_at window must not
    swallow the genuinely-missed round."""
    job = _job(_DISTANT)  # distant stale lattice instant (yesterday 03:00)
    _completed_execution(job["id"], _NOW - timedelta(hours=9))  # a later lattice round

    fired = _evaluate_due_job(job, _scan(job), run_claim_ttl=900.0)

    assert fired is True, "a distant rollback must not be suppressed by a later completion"


def test_late_catchup_claim_within_grace_is_suppressed(monkeypatch, tmp_path):
    """The gateway was down past the scheduled instant: the occurrence is
    claimed and completes 30 minutes late — inside the daily job's 2h catch-up
    grace but far beyond the 300s ticker slack. The stale re-armed
    next_run_at must still be suppressed (PR #104323 review, late-claim
    variant)."""
    job = _job(_STALE)
    _completed_execution(job["id"], _STALE + timedelta(minutes=30))

    fired = _evaluate_due_job(job, _scan(job), run_claim_ttl=900.0)

    assert fired is False, "a late-but-legitimate catch-up claim is still this occurrence"
    assert job["next_run_at"] != _iso(_STALE), "next_run_at must be re-anchored forward"


def test_claim_beyond_grace_still_fails_open(monkeypatch, tmp_path):
    """A completion claimed beyond the job's catch-up grace is indeterminate
    (no legitimate claim lands there): the guard fails open rather than
    over-suppressing."""
    job = _job(_STALE)
    _completed_execution(job["id"], _STALE + timedelta(hours=3))  # grace is 2h for daily

    fired = _evaluate_due_job(job, _scan(job), run_claim_ttl=900.0)

    assert fired is True, "beyond-grace claims stay fail-open"
