"""Due-scan dispatch-limit guard: a re-armed consumed one-shot never fires twice.

Pre-#93615 stores (or hand edits) can hold a record that already completed a
run (last_run_at set) but was re-armed without a budget reset. The due-scan
guard removes it without firing — re-runs go through `cron resume`.
"""

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


def test_rearmed_consumed_oneshot_is_removed_without_firing(temp_home):
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

    due = get_due_jobs()

    assert jid not in [d["id"] for d in due]
    assert jid not in [j["id"] for j in load_jobs()]


