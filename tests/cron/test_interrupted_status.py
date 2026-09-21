"""A shutdown-interrupted cron run is its own outcome, never a failure.

Gateway shutdown marks every in-flight job interrupted after kill_all() (#60432).
The run's own outcome is unknown - its tool subprocess was killed mid-flight -
so neither ``ok`` nor ``error`` is honest, and recording ``error`` (the status
the operator and every consumer of the cron record reads) reports an
interrupted run as a failed one. Seen live 2026-09-15: a long report job was
marked ``error`` at shutdown while the pipeline it had detached kept running
and finished 14 minutes later.

Contract asserted here, through the real producer -> store -> operator path:
  * the run is recorded as ``interrupted``, not ``error``;
  * the cron doctor says "interrupted", not "last run failed";
  * an interruption is streak-neutral, while a genuine agent error still
    increments ``failure_streak``.
"""

import pytest


@pytest.fixture(autouse=True)
def _reset_scheduler_state():
    """The in-flight registries are module-level globals shared per process."""
    import cron.scheduler as sched

    sched._running_job_ids.clear()
    sched._running_fire_owners.clear()
    sched._interrupted_job_ids.clear()
    yield
    sched._running_job_ids.clear()
    sched._running_fire_owners.clear()
    sched._interrupted_job_ids.clear()


SHUTDOWN_REASON = (
    "Gateway shutdown (post-interrupt) killed the job's tool subprocess "
    "before the run finished."
)


def _claimed_job(jobs, home, *, prompt="x", streak=0):
    """A job with a live fire claim, plus its owner, in *home*'s store."""
    created = jobs.create_job(prompt=prompt, schedule="every 1h", name=prompt)
    job_id = created["id"]
    if streak:
        jobs.update_job(job_id, {"failure_streak": streak})
    claimed = jobs.claim_job_for_fire(job_id, force=True, return_job=True)
    return job_id, claimed["fire_claim"]["by"]


def test_interrupted_run_reads_as_interrupted_not_error(tmp_path):
    import cron.jobs as jobs
    import cron.scheduler as sched
    from hermes_cli.cron import _cron_doctor_issues_for_job

    home = tmp_path / "profile"
    home.mkdir()
    with jobs.use_cron_store(home):
        job_id, owner = _claimed_job(jobs, home, streak=2)
        inflight_key = sched._inflight_key(job_id, home)
        sched._running_job_ids.add(inflight_key)
        sched._running_fire_owners[inflight_key] = {object(): (owner, home)}

        marked = sched.mark_running_jobs_interrupted(SHUTDOWN_REASON)
        record = jobs.get_job(job_id)
        assert record is not None

    assert marked == [job_id]
    assert record["last_status"] == "interrupted"
    assert record["last_error"] == SHUTDOWN_REASON
    # The operator-facing health check must not call an interruption a failure.
    issues = " ".join(_cron_doctor_issues_for_job(record))
    assert "last run failed" not in issues
    assert "interrupted" in issues


def test_interruption_is_streak_neutral_but_a_real_error_is_not(tmp_path):
    """Interruption says nothing about whether the agent failed, so it must not
    inflate the failure streak that drives the repeated-failure nudge; a genuine
    agent error still does."""
    import cron.jobs as jobs

    home = tmp_path / "profile"
    home.mkdir()
    with jobs.use_cron_store(home):
        interrupted_id, interrupted_owner = _claimed_job(
            jobs, home, prompt="interrupted", streak=2)
        failed_id, failed_owner = _claimed_job(jobs, home, prompt="failed", streak=2)

        assert jobs.mark_job_run(
            interrupted_id, False, SHUTDOWN_REASON, status="interrupted",
            expected_fire_owner=interrupted_owner)
        assert jobs.mark_job_run(
            failed_id, False, "Provider returned error",
            expected_fire_owner=failed_owner)

        interrupted = jobs.get_job(interrupted_id)
        assert interrupted is not None
        failed = jobs.get_job(failed_id)
        assert failed is not None

    assert interrupted["last_status"] == "interrupted"
    assert interrupted["failure_streak"] == 2
    assert failed["last_status"] == "error"
    assert failed["failure_streak"] == 3
