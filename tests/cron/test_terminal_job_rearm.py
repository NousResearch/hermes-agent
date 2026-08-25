"""Behavioral coverage for terminal cron jobs and explicit one-shot re-arm."""

from datetime import datetime, timedelta, timezone
import copy

import pytest

from cron.jobs import (
    advance_next_run,
    create_job,
    get_due_jobs,
    get_job,
    load_jobs,
    mark_job_run,
    save_jobs,
    trigger_job,
    update_job,
)


@pytest.fixture()
def tmp_cron_dir(tmp_path, monkeypatch):
    monkeypatch.setattr("cron.jobs.CRON_DIR", tmp_path / "cron")
    monkeypatch.setattr("cron.jobs.JOBS_FILE", tmp_path / "cron" / "jobs.json")
    monkeypatch.setattr("cron.jobs.OUTPUT_DIR", tmp_path / "cron" / "output")
    return tmp_path


def test_completed_oneshot_trigger_is_refused_and_disk_record_is_unchanged(tmp_cron_dir):
    job = create_job("done", "30m", name="done", repeat=1)
    mark_job_run(job["id"], success=True)
    before = copy.deepcopy(load_jobs())

    with pytest.raises(ValueError, match="terminal"):
        trigger_job(job["id"])

    assert load_jobs() == before
    assert get_job(job["id"]) == before[0]


def test_exhausted_recurring_job_trigger_is_refused(tmp_cron_dir):
    job = create_job("done", "every 1h", repeat=1)
    mark_job_run(job["id"], success=True)

    with pytest.raises(ValueError, match="terminal"):
        trigger_job(job["id"])


def test_wedged_claimed_oneshot_remains_triggerable(tmp_cron_dir):
    now = datetime.now(timezone.utc)
    job = create_job("wedged", "30m", repeat=2)
    record = get_job(job["id"])
    record.update({
        "run_claim": {"at": now.isoformat(), "by": "dead-worker"},
        "state": "scheduled",
        "enabled": True,
        "next_run_at": (now - timedelta(minutes=1)).isoformat(),
    })
    save_jobs([record])

    triggered = trigger_job(job["id"])
    assert triggered["state"] == "scheduled"
    assert triggered["enabled"] is True


def test_paused_job_run_override_remains_allowed(tmp_cron_dir):
    job = create_job("paused", "every 1h")
    from cron.jobs import pause_job

    pause_job(job["id"])
    triggered = trigger_job(job["id"])
    assert triggered["state"] == "scheduled"
    assert triggered["enabled"] is True


def test_terminal_jobs_are_not_due_or_advanced(tmp_cron_dir):
    job = create_job("done", "every 1h", repeat=1)
    mark_job_run(job["id"], success=True)
    before = copy.deepcopy(load_jobs())

    assert get_due_jobs() == []
    assert advance_next_run(job["id"]) is False
    assert load_jobs() == before


def test_terminal_refusal_survives_reload(tmp_cron_dir):
    job = create_job("done", "30m", repeat=1)
    mark_job_run(job["id"], success=True)
    before = copy.deepcopy(load_jobs())
    assert get_job(job["id"])["state"] == "completed"

    with pytest.raises(ValueError):
        trigger_job(job["id"])
    assert load_jobs() == before


def test_update_cannot_reactivate_terminal_record(tmp_cron_dir):
    job = create_job("done", "30m", repeat=1)
    mark_job_run(job["id"], success=True)
    with pytest.raises(ValueError, match="terminal"):
        update_job(job["id"], {"enabled": True})
    with pytest.raises(ValueError, match="terminal"):
        update_job(job["id"], {"schedule": "every 1h"})


def test_rearm_completed_oneshot_restores_schedule_and_preserves_history(tmp_cron_dir):
    from cron.jobs import rearm_oneshot

    job = create_job("done", "30m", repeat=3)
    mark_job_run(job["id"], success=True)
    finished = get_job(job["id"])
    run_at = (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat()

    rearmed = rearm_oneshot(job["id"], run_at)
    assert rearmed["schedule"]["kind"] == "once"
    assert rearmed["repeat"]["times"] == 3
    assert rearmed["repeat"]["completed"] == 0
    assert rearmed["state"] == "scheduled"
    assert rearmed["enabled"] is True
    assert rearmed["next_run_at"] == rearmed["schedule"]["run_at"]
    assert rearmed["last_run_at"] == finished["last_run_at"]
    assert rearmed["last_status"] == finished["last_status"]


def test_rearm_refuses_recurring_and_live_claim(tmp_cron_dir):
    from cron.jobs import rearm_oneshot

    recurring = create_job("recurring", "every 1h")
    future = (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat()
    with pytest.raises(ValueError, match="one-shot"):
        rearm_oneshot(recurring["id"], future)

    oneshot = create_job("claimed", "30m")
    record = get_job(oneshot["id"])
    record["run_claim"] = {"at": datetime.now(timezone.utc).isoformat(), "by": "live"}
    save_jobs([record])
    with pytest.raises(ValueError, match="claim"):
        rearm_oneshot(oneshot["id"], future)
