"""A resumed job starts its failure count over; the scheduler's own pause lift does not.

``failure_streak`` is failed runs in a row: ``hermes cron list`` prints "N failures in a row" from it
and the failure delivery appends "This job has failed N runs in a row ... pause it"
(``scheduler._failure_streak_nudge``). A job an operator paused after two failures and resumed has
run nothing since, so its first failure after the repair was reported as the third in a row.
"""
import pytest

from cron import jobs
from cron.scheduler import _failure_streak_nudge
from cron.scheduler_authority import _lift_pause, _pause_identity


@pytest.fixture(autouse=True)
def store(tmp_path):
    with jobs.use_cron_store(tmp_path):
        yield tmp_path


def _failed_twice_and_paused(reason="paused after two failures"):
    job = jobs.create_job("check", "every 1h", name="check")
    for _ in range(2):
        jobs.mark_job_run(job["id"], success=False, error="Script exited with code 1")
    assert jobs.get_job(job["id"])["failure_streak"] == 2
    return job["id"], jobs.pause_job(job["id"], reason=reason)


def test_resume_starts_the_failure_count_over():
    job_id, _ = _failed_twice_and_paused()

    resumed = jobs.resume_job(job_id)

    assert resumed["enabled"] is True and resumed["state"] == "scheduled"
    assert resumed["failure_streak"] == 0


def test_the_first_failure_after_a_resume_is_not_nudged_as_the_third_in_a_row(monkeypatch):
    from cron import scheduler
    monkeypatch.setattr(scheduler, "load_config", lambda: {"cron": {"failure_nudge_threshold": 3}})
    job_id, _ = _failed_twice_and_paused()

    jobs.resume_job(job_id)

    assert _failure_streak_nudge(jobs.get_job(job_id)) == ""


def test_resume_keeps_the_record_of_the_last_failure():
    job_id, _ = _failed_twice_and_paused()
    before = jobs.get_job(job_id)

    after = jobs.resume_job(job_id)

    assert after["last_status"] == before["last_status"] == "error"
    assert after["last_error"] == before["last_error"]
    assert after.get("last_failure") == before.get("last_failure")


def test_two_failed_runs_after_a_resume_count_again():
    job_id, _ = _failed_twice_and_paused()
    jobs.resume_job(job_id)

    jobs.mark_job_run(job_id, success=False, error="Script exited with code 1")
    assert jobs.get_job(job_id)["failure_streak"] == 1
    jobs.mark_job_run(job_id, success=False, error="Script exited with code 1")
    assert jobs.get_job(job_id)["failure_streak"] == 2


def test_the_schedulers_own_pause_lift_keeps_the_count():
    """A pause around an execution nobody could verify is not a repair: a job that keeps failing
    must not have its count wiped each time the scheduler pauses and lifts it."""
    job_id, paused = _failed_twice_and_paused(reason="execution unverified")

    _lift_pause(job_id, _pause_identity(paused))

    lifted = jobs.get_job(job_id)
    assert lifted["state"] == "scheduled" and lifted["enabled"] is True
    assert lifted["failure_streak"] == 2
