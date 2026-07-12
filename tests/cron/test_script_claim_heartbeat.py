"""Regression coverage for one-shot claims during blocking cron scripts."""

from datetime import datetime, timedelta, timezone
import threading
from unittest.mock import MagicMock, patch

import pytest


@pytest.mark.parametrize(
    ("no_agent", "script_output"),
    [
        (True, "watchdog complete"),
        (False, '{"wakeAgent": false}'),
    ],
    ids=("script-only-job", "pre-agent-script"),
)
def test_long_running_script_refreshes_owned_claim_in_profile_store(
    tmp_path, monkeypatch, no_agent, script_output
):
    """Both blocking script paths keep their one-shot claim alive.

    The real store update runs on the heartbeat thread.  A second store holds
    the same job ID, proving the thread inherited the active profile's
    ContextVar instead of falling back to another profile's default paths.
    """
    import cron.jobs as jobs
    import cron.scheduler as scheduler

    profile_home = tmp_path / "profile"
    default_cron = tmp_path / "default" / "cron"
    default_cron.mkdir(parents=True)
    profile_home.mkdir()

    monkeypatch.setattr(jobs, "CRON_DIR", default_cron)
    monkeypatch.setattr(jobs, "JOBS_FILE", default_cron / "jobs.json")
    monkeypatch.setattr(jobs, "OUTPUT_DIR", default_cron / "output")
    monkeypatch.setattr(scheduler, "_RUN_CLAIM_HEARTBEAT_SECONDS", 0.01)

    original_timestamp = "2026-07-12T12:00:00+00:00"
    original_time = datetime.fromisoformat(original_timestamp)
    claim_ttl = jobs._oneshot_run_claim_ttl_seconds()
    current_time = [original_time + timedelta(seconds=claim_ttl - 60)]
    monkeypatch.setattr(jobs, "_hermes_now", lambda: current_time[0])

    def _job() -> dict:
        return {
            "id": "long-script",
            "name": "long script",
            "prompt": "inspect the script output",
            "script": "watchdog.py",
            "no_agent": no_agent,
            "schedule": {
                "kind": "once",
                "run_at": original_timestamp,
            },
            "next_run_at": original_timestamp,
            "enabled": True,
            "run_claim": {
                "at": original_timestamp,
                "by": "dispatch-owner",
            },
        }

    # Safe fallback store: if ContextVars are not propagated to the heartbeat
    # thread, this record would be modified instead of the profile record.
    jobs.save_jobs([_job()])
    with jobs.use_cron_store(profile_home):
        jobs.save_jobs([_job()])
        claimed_job = jobs.get_job("long-script")

    heartbeat_seen = threading.Event()
    real_heartbeat = jobs.heartbeat_run_claim
    second_scheduler_scan = {}

    def _observed_heartbeat(job_id: str, *, expected_owner: str) -> bool:
        updated = real_heartbeat(job_id, expected_owner=expected_owner)
        # A different scheduler scans after the ORIGINAL claim's TTL while the
        # script is still blocked. The refreshed claim must keep the job out of
        # the due set and preserve its durable record.
        current_time[0] = original_time + timedelta(seconds=claim_ttl + 10)
        second_scheduler_scan["due"] = jobs.get_due_jobs()
        second_scheduler_scan["record_present"] = jobs.get_job(job_id) is not None
        heartbeat_seen.set()
        return updated

    def _blocking_script(_script_path: str) -> tuple[bool, str]:
        assert heartbeat_seen.wait(timeout=2), (
            "claim was not refreshed while script blocked"
        )
        return True, script_output

    monkeypatch.setattr(scheduler, "heartbeat_run_claim", _observed_heartbeat)
    monkeypatch.setattr(scheduler, "_run_job_script", _blocking_script)

    with (
        jobs.use_cron_store(profile_home),
        patch("hermes_state.SessionDB", return_value=MagicMock()),
    ):
        success, _doc, _response, error = scheduler.run_job(claimed_job)
        profile_claim = jobs.get_job("long-script")["run_claim"]

    assert success is True
    assert error is None
    assert profile_claim["at"] != original_timestamp
    assert profile_claim["by"] == "dispatch-owner"
    assert second_scheduler_scan == {"due": [], "record_present": True}
    assert jobs.get_job("long-script")["run_claim"] == {
        "at": original_timestamp,
        "by": "dispatch-owner",
    }


def test_script_heartbeat_uses_captured_claim_owner(tmp_path, monkeypatch):
    """A stale script runner cannot refresh a replacement owner's claim."""
    import cron.jobs as jobs
    import cron.scheduler as scheduler

    profile_home = tmp_path / "profile"
    profile_home.mkdir()
    original_timestamp = "2026-07-12T12:00:00+00:00"
    replacement_timestamp = "2026-07-12T12:00:30+00:00"
    job = {
        "id": "reclaimed-script",
        "script": "watchdog.py",
        "schedule": {"kind": "once", "run_at": original_timestamp},
        "run_claim": {"at": original_timestamp, "by": "original-owner"},
    }

    with jobs.use_cron_store(profile_home):
        jobs.save_jobs([
            {
                **job,
                "run_claim": {
                    "at": replacement_timestamp,
                    "by": "replacement-owner",
                },
            }
        ])

    heartbeat_seen = threading.Event()
    real_heartbeat = jobs.heartbeat_run_claim

    def _observed_heartbeat(job_id: str, *, expected_owner: str) -> bool:
        updated = real_heartbeat(job_id, expected_owner=expected_owner)
        heartbeat_seen.set()
        return updated

    def _blocking_script(_script_path: str) -> tuple[bool, str]:
        assert heartbeat_seen.wait(timeout=2)
        return True, "done"

    monkeypatch.setattr(scheduler, "_RUN_CLAIM_HEARTBEAT_SECONDS", 0.01)
    monkeypatch.setattr(scheduler, "heartbeat_run_claim", _observed_heartbeat)
    monkeypatch.setattr(scheduler, "_run_job_script", _blocking_script)

    with jobs.use_cron_store(profile_home):
        assert scheduler._run_job_script_with_claim_heartbeat(job, "watchdog.py") == (
            True,
            "done",
        )
        assert jobs.get_job("reclaimed-script")["run_claim"] == {
            "at": replacement_timestamp,
            "by": "replacement-owner",
        }
