"""Due-scan dispatch-limit guard: removal of a re-armed consumed one-shot is
operator-visible (#93524 follow-up to #93615).

Pre-#93615 stores (or hand edits) can hold a record that already completed a
run (last_run_at set) but was re-armed without a budget reset. The due-scan
guard removes it without firing — correct, since #93615's policy routes
re-runs through `cron resume` — but the removal must log at WARNING with a
remediation hint, never a silent INFO delete.
"""

import logging
from datetime import timedelta

import pytest

from cron.jobs import (
    claim_dispatch,
    create_job,
    get_due_jobs,
    load_jobs,
    mark_job_run,
    save_jobs,
    _hermes_now,
)

@pytest.fixture()
def temp_home(tmp_path, monkeypatch):
    """Redirect cron storage to a temp dir (same pattern as the sibling
    grace-gate tests)."""
    monkeypatch.setattr("cron.jobs.CRON_DIR", tmp_path / "cron")
    monkeypatch.setattr("cron.jobs.JOBS_FILE", tmp_path / "cron" / "jobs.json")
    monkeypatch.setattr("cron.jobs.OUTPUT_DIR", tmp_path / "cron" / "output")
    return tmp_path

def test_guard_warns_on_rearmed_consumed_record(temp_home, caplog):
    job = create_job(
        prompt="x",
        schedule=(_hermes_now() + timedelta(hours=1)).isoformat(),
        name="warn-guard",
        deliver="local",
    )
    jid = job["id"]
    assert claim_dispatch(jid)
    mark_job_run(jid, True)

    # Simulate a pre-#93615 re-arm: enabled+scheduled+due, budget still spent.
    jobs = load_jobs()
    for j in jobs:
        if j["id"] == jid:
            j["enabled"] = True
            j["state"] = "scheduled"
            j["next_run_at"] = (_hermes_now() - timedelta(seconds=5)).isoformat()
    save_jobs(jobs)

    with caplog.at_level(logging.INFO, logger="cron"):
        due = get_due_jobs()

    assert jid not in [d["id"] for d in due]
    # Patch 018: a consumed one-shot is RETAINED and disabled, never silently
    # deleted — a real financial deadline (job f98f9fcf2561, 2026-08-20) was
    # lost that way. What matters is that it cannot fire again, not that the
    # record disappears.
    retained = [j for j in load_jobs() if j["id"] == jid]
    assert len(retained) == 1
    assert retained[0]["enabled"] is False
    assert retained[0]["next_run_at"] is None
    warnings = [
        r for r in caplog.records
        # Patch 018 reworded this: a consumed one-shot is disarmed in place and
        # retained, never deleted. This record HAS a completed run, so it takes
        # the "already completed a run — disarming in place" arm.
        if r.levelno == logging.WARNING and "not deleting" in r.getMessage()
    ]
    assert warnings, "expected WARNING on removal of a re-armed consumed one-shot"
    assert "cron resume" in warnings[0].getMessage()
