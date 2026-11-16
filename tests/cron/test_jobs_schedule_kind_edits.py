"""A schedule edit that changes the job's kind (recurring <-> one-shot) or lands while it is paused."""

from datetime import UTC, datetime, timedelta

import pytest

from cron.jobs import (
    _hermes_now,
    claim_job_for_fire,
    create_job,
    get_due_jobs,
    get_job,
    load_jobs,
    mark_job_run,
    pause_job,
    resume_job,
    update_job,
)


@pytest.fixture()
def tmp_cron_dir(tmp_path, monkeypatch):
    monkeypatch.setattr("cron.jobs.CRON_DIR", tmp_path / "cron")
    monkeypatch.setattr("cron.jobs.JOBS_FILE", tmp_path / "cron" / "jobs.json")
    monkeypatch.setattr("cron.jobs.OUTPUT_DIR", tmp_path / "cron" / "output")
    return tmp_path


@pytest.mark.parametrize("via", ["update_job", "update_job+repeat", "cronjob_tool+repeat", "bounded"])
def test_recurring_job_that_ran_turned_oneshot_fires_instead_of_being_deleted(
    tmp_cron_dir, monkeypatch, via,
):
    """Runs of the old recurring schedule must not spend the one-shot's budget: at
    completed >= times the due scan deletes the job without firing it. The tool and
    ``cron edit --repeat`` pass an explicit repeat carrying the stored counter. A bounded
    recurring count is not the one-shot's budget either: the one-shot fires once."""
    import json
    from tools.cronjob_tools import cronjob

    job = create_job(prompt="water the plants", schedule="every 1h",
                     repeat=5 if via == "bounded" else None)
    for _ in range(3):
        mark_job_run(job["id"], success=True)
    if via in ("update_job", "bounded"):
        update_job(job["id"], {"schedule": "in 5m"})
    elif via == "update_job+repeat":
        update_job(job["id"], {"schedule": "in 5m", "repeat": 1})
    else:
        result = json.loads(cronjob(action="update", job_id=job["id"], schedule="in 5m", repeat=1))
        assert result["success"], result

    assert get_job(job["id"])["repeat"] == {"times": 1, "completed": 0}
    later = _hermes_now() + timedelta(minutes=5, seconds=10)
    monkeypatch.setattr("cron.jobs._hermes_now", lambda: later)
    assert job["id"] in {j["id"] for j in get_due_jobs()}
    assert get_job(job["id"]) is not None

@pytest.mark.parametrize("run", ["fire_claim", "scheduler_run"])
def test_turning_a_job_oneshot_mid_run_is_refused(tmp_cron_dir, run):
    """The in-flight run would land after the reset and retire the new one-shot unfired. A
    ticker-dispatched recurring run holds no claim; this process's running set sees it."""
    from cron.scheduler import release_running_job, try_register_running_job

    job = create_job(prompt="digest", schedule="every 1h")
    mark_job_run(job["id"], success=True)
    if run == "fire_claim":
        assert claim_job_for_fire(job["id"])
    else:
        assert try_register_running_job(job["id"])
    try:
        with pytest.raises(ValueError, match="while a run is in progress"):
            update_job(job["id"], {"schedule": "in 5m"})
        unchanged = get_job(job["id"])
        assert unchanged["schedule"]["kind"] == "interval"
        assert unchanged["repeat"] == {"times": None, "completed": 1}
    finally:
        if run == "scheduler_run":
            release_running_job(job["id"])
    if run == "fire_claim":
        # The claimed run still lands on the recurring job, and the flip then succeeds.
        owner = next(j for j in load_jobs() if j["id"] == job["id"])["fire_claim"]["by"]
        assert mark_job_run(job["id"], success=True, expected_fire_owner=owner)
    assert update_job(job["id"], {"schedule": "in 5m"})["repeat"] == {"times": 1, "completed": 0}


@pytest.mark.parametrize("new_schedule", ["every 1d", "0 9 * * *"])
def test_schedule_edit_while_paused_resumes_on_the_new_schedule(
    tmp_cron_dir, monkeypatch, new_schedule,
):
    """A slot that elapsed while paused stays due on resume only while it belongs to the job's
    schedule. After the schedule is edited, that slot belongs to the replaced cadence: resume
    runs on the new schedule, exactly as the same edit on an unpaused job does."""
    clock = {"now": datetime(2026, 10, 5, 10, 0, 0, tzinfo=UTC)}
    monkeypatch.setattr("cron.jobs._hermes_now", lambda: clock["now"])
    edited = create_job(prompt="noisy", schedule="every 30m", deliver="local")
    resaved = create_job(prompt="steady", schedule="every 30m", deliver="local")
    for job in (edited, resaved):
        pause_job(job["id"])
    clock["now"] += timedelta(hours=3)  # the 10:30 slot elapses while paused
    old_slot = get_job(resaved["id"])["next_run_at"]

    update_job(edited["id"], {"schedule": new_schedule})
    update_job(resaved["id"], {"schedule": "every 30m"})  # a re-save is not an edit

    resumed = resume_job(edited["id"])
    assert datetime.fromisoformat(resumed["next_run_at"]) > clock["now"]
    assert resume_job(resaved["id"])["next_run_at"] == old_slot
    assert {j["id"] for j in get_due_jobs()} == {resaved["id"]}
