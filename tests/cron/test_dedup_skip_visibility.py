"""A due slot consumed by the completed-occurrence dedup must leave a visible trace (#111414).

An off-tick run that stamps a FUTURE occurrence's identity onto its execution row (the fixed
#105704 / dashboard-trigger bug class) leaves behind a completed row that the dedup gates later
use to consume the REAL slot: next_run_at advances, nothing dispatches, and — before the fix —
no log line, no execution row, and no error anywhere. Report #111414: a weekly job silently
missed its fire with `last_status: ok` because a week-old completed row already carried the due
instant's identity.

Contract pinned here: the skip is always logged with the matching row, and when the matching
row was claimed well before the skipped instant — impossible for a legitimate run of that slot,
which is claimed at/after it — the anomaly is also stamped as ``last_fire_error`` on the job
record so `hermes cron list` surfaces it. A timely row (legitimate re-delivery dedup) skips
quietly with a log line but must NOT raise the alarm.
"""

import json
import logging
import sqlite3
from datetime import timedelta
from pathlib import Path

import pytest

from hermes_time import now


@pytest.fixture()
def cron_world(tmp_path, monkeypatch):
    """Isolated cron store + executions ledger; yields (jobs module, cron dir)."""
    from cron import executions, jobs

    monkeypatch.setattr(executions, "EXECUTIONS_FILE", tmp_path / "executions.db")
    cron_dir = tmp_path / "cron"
    with jobs.use_cron_store(tmp_path):
        yield jobs, cron_dir


def _store_job(cron_dir: Path, job_id: str, slot: str) -> None:
    job = {
        "id": job_id,
        "name": job_id,
        "prompt": "",
        "schedule": {"kind": "interval", "minutes": 60},
        "next_run_at": slot,
        "enabled": True,
        "state": "scheduled",
        "deliver": "local",
        "repeat": {"times": None, "completed": 0},
    }
    cron_dir.mkdir(parents=True, exist_ok=True)
    (cron_dir / "jobs.json").write_text(json.dumps({"jobs": [job]}))


def _completed_row(executions, job_id: str, slot: str, claimed_at):
    """A completed execution row claiming *slot*, with its claim backdated to *claimed_at*."""
    row = executions.create_execution(job_id, source="builtin", scheduled_instant=slot)
    assert executions.finish_execution(row["id"], success=True) is not None
    if claimed_at is not None:
        with sqlite3.connect(executions.EXECUTIONS_FILE) as conn:
            conn.execute(
                "UPDATE executions SET claimed_at=? WHERE id=?",
                (claimed_at.isoformat(), row["id"]),
            )
    return row


def _load_job(jobs, job_id: str) -> dict:
    return next(j for j in jobs.load_jobs() if j["id"] == job_id)


def test_backdated_completed_row_skips_slot_aloud(cron_world, caplog):
    """The #111414 shape: a week-old completed row carries the due slot's identity — the skip
    must advance past the slot AND stamp last_fire_error (fail-visible, not silent)."""
    from cron import executions

    jobs, cron_dir = cron_world
    slot = (now() - timedelta(minutes=1)).isoformat()
    _store_job(cron_dir, "weekly", slot)
    row = _completed_row(executions, "weekly", slot, now() - timedelta(days=7))

    with caplog.at_level(logging.WARNING, logger="cron.occurrences"):
        due = jobs.get_due_jobs()

    assert [j["id"] for j in due] == [], "polluted identity must consume the slot"
    stored = _load_job(jobs, "weekly")
    assert stored["next_run_at"] != slot, "skip advances past the consumed slot"
    alarm = stored.get("last_fire_error")
    assert alarm, "backdated identity skip must be surfaced as last_fire_error"
    assert row["id"][:8] in alarm["detail"]
    assert slot[:19] in alarm["detail"]
    assert alarm["at"]
    assert any(row["id"][:8] in r.getMessage() for r in caplog.records), (
        "the skip itself must be logged"
    )


def test_timely_completed_row_skips_slot_without_alarm(cron_world, caplog):
    """Legitimate re-delivery dedup: the matching row was claimed AFTER the slot it covers —
    skip and advance, but no last_fire_error (the run already happened)."""
    from cron import executions

    jobs, cron_dir = cron_world
    slot = (now() - timedelta(minutes=5)).isoformat()
    _store_job(cron_dir, "weekly", slot)
    _completed_row(executions, "weekly", slot, None)  # claimed now, after the slot

    with caplog.at_level(logging.WARNING, logger="cron.occurrences"):
        due = jobs.get_due_jobs()

    assert [j["id"] for j in due] == []
    stored = _load_job(jobs, "weekly")
    assert stored["next_run_at"] != slot
    assert not stored.get("last_fire_error"), "timely dedup is not a missed fire"
    assert any("weekly" in r.getMessage() for r in caplog.records)


def test_claim_gate_backdated_skip_also_alarms(cron_world):
    """The fire-claim dedup gate (external re-delivery path) carries the same contract."""
    from cron import executions

    jobs, cron_dir = cron_world
    slot = (now() - timedelta(minutes=1)).isoformat()
    _store_job(cron_dir, "weekly", slot)
    _completed_row(executions, "weekly", slot, now() - timedelta(days=7))

    claimed = jobs.claim_job_for_fire("weekly", return_job=True)

    assert claimed is False, "polluted identity must consume the slot"
    stored = _load_job(jobs, "weekly")
    assert stored["next_run_at"] != slot
    assert stored.get("last_fire_error"), "backdated identity skip must be surfaced"


def test_future_slot_identity_is_not_reported_before_it_is_due(cron_world, caplog):
    """A slot that has not come due yet is not a missed fire.

    A retained completed row can already carry the identity of a FUTURE next_run_at (the same
    stale off-tick stamping shape as the backdated case above). The skip reporting must wait for
    the occurrence to be due, or an ordinary early scan alarms on — and a one-shot re-alarms on
    every tick, since it never advances next_run_at.
    """
    from cron import executions

    jobs, cron_dir = cron_world
    future = (now() + timedelta(minutes=30)).isoformat()
    _store_job(cron_dir, "weekly", future)
    _completed_row(executions, "weekly", future, now() - timedelta(days=7))

    with caplog.at_level(logging.WARNING, logger="cron.occurrences"):
        due = jobs.get_due_jobs()

    assert [j["id"] for j in due] == []
    stored = _load_job(jobs, "weekly")
    assert not stored.get("last_fire_error"), "a slot that is not due yet is not a missed fire"
    assert not [r for r in caplog.records if r.name == "cron.occurrences"], (
        "no skip report until the occurrence is due"
    )
