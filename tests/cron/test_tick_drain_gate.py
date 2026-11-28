"""The in-process ticker must honour a drain that starts while a tick is already running."""

from unittest.mock import patch

def test_tick_rechecks_drain_gate_after_collecting_due_jobs():
    """A drain that starts mid-tick must still stop dispatch.

    ``can_dispatch`` used to be read once at the top of the tick. Collecting due jobs and the
    pre-dispatch housekeeping can take seconds on a loaded host, and a SIGTERM shutdown that set
    ``_draining`` in that window still got the job dispatched: the drain snapshot saw
    ``cron_at_start=0``, final-cleanup then killed the job and marked it interrupted, and its
    script kept running as an orphan after the gateway exited. Re-checking before the schedule
    is advanced also leaves the job due, so it can run after the restart.
    """
    from cron.scheduler import tick

    job = {
        "id": "drain-race-job",
        "name": "drain race job",
        "schedule": {"kind": "interval", "seconds": 60},
        "next_run_at": "2020-01-01T00:00:00+00:00",
        "enabled": True,
    }
    draining = {"on": False}

    def _due_jobs():
        draining["on"] = True  # shutdown begins while the tick is collecting due jobs
        return [job]

    with patch("cron.scheduler.get_due_jobs", side_effect=_due_jobs), patch(
        "cron.scheduler.claim_job_for_fire", return_value=True
    ) as claim, patch("cron.scheduler.run_one_job") as run_one, patch(
        "cron.scheduler.advance_next_runs"
    ) as advance:
        assert tick(verbose=False, sync=True, can_dispatch=lambda: not draining["on"]) == 0

    advance.assert_not_called()
    claim.assert_not_called()
    run_one.assert_not_called()
