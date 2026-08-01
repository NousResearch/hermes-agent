"""Regression coverage for one-shot claims during blocking cron scripts."""

from datetime import datetime, timedelta, timezone
import os
import threading
import time
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
                "id": "dispatch-token",
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

    def _observed_heartbeat(job_id: str, *, expected_claim_id: str) -> bool:
        updated = real_heartbeat(job_id, expected_claim_id=expected_claim_id)
        # A different scheduler scans after the ORIGINAL claim's TTL while the
        # script is still blocked. The refreshed claim must keep the job out of
        # the due set and preserve its durable record.
        current_time[0] = original_time + timedelta(seconds=claim_ttl + 10)
        second_scheduler_scan["due"] = jobs.get_due_jobs()
        second_scheduler_scan["record_present"] = jobs.get_job(job_id) is not None
        heartbeat_seen.set()
        return updated

    def _blocking_script(_script_path: str, **kwargs) -> tuple[bool, str]:
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
    assert profile_claim["id"] == "dispatch-token"
    assert profile_claim["at"] != original_timestamp
    assert profile_claim["by"] == "dispatch-owner"
    assert second_scheduler_scan == {"due": [], "record_present": True}
    assert jobs.get_job("long-script")["run_claim"] == {
        "id": "dispatch-token",
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

    def _blocking_script(_script_path: str, **kwargs) -> tuple[bool, str]:
        assert heartbeat_seen.wait(timeout=2)
        return True, "done"

    monkeypatch.setattr(scheduler, "_RUN_CLAIM_HEARTBEAT_SECONDS", 0.01)
    monkeypatch.setattr(scheduler, "heartbeat_run_claim", _observed_heartbeat)
    monkeypatch.setattr(scheduler, "_run_job_script", _blocking_script)

    with jobs.use_cron_store(profile_home):
        success, error = scheduler._run_job_script_with_claim_heartbeat(
            job, "watchdog.py"
        )
        assert success is False
        assert "ownership" in error.lower()
        assert jobs.get_job("reclaimed-script")["run_claim"] == {
            "at": replacement_timestamp,
            "by": "replacement-owner",
        }


def test_recurring_script_heartbeats_owned_fire_token(monkeypatch):
    """Recurring manual/provider scripts keep the universal fire claim alive."""
    import cron.scheduler as scheduler

    heartbeat_seen = threading.Event()
    calls = []

    def _heartbeat(job_id: str, *, expected_claim_id: str) -> bool:
        calls.append((job_id, expected_claim_id))
        heartbeat_seen.set()
        return True

    def _blocking_script(_script_path: str, **_kwargs) -> tuple[bool, str]:
        assert heartbeat_seen.wait(timeout=2)
        return True, "done"

    monkeypatch.setattr(scheduler, "_RUN_CLAIM_HEARTBEAT_SECONDS", 0.01)
    monkeypatch.setattr(scheduler, "heartbeat_fire_claim", _heartbeat)
    monkeypatch.setattr(scheduler, "_run_job_script", _blocking_script)

    result = scheduler._run_job_script_with_claim_heartbeat(
        {
            "id": "recurring-script",
            "schedule": {"kind": "interval", "minutes": 5},
            "_fire_claim_id": "fire-token-1",
        },
        "watchdog.py",
    )

    assert result == (True, "done")
    assert calls
    assert calls[0] == ("recurring-script", "fire-token-1")


def test_fire_claim_loss_terminates_running_script(tmp_path, monkeypatch):
    """A replaced fire owner must not let its child continue side effects."""
    import cron.scheduler as scheduler

    home = tmp_path / "profile"
    scripts = home / "scripts"
    scripts.mkdir(parents=True)
    late_effect = tmp_path / "late-effect"
    script = scripts / "slow.py"
    script.write_text(
        "import pathlib, time\n"
        "time.sleep(1)\n"
        f"pathlib.Path({str(late_effect)!r}).write_text('stale side effect')\n"
    )
    monkeypatch.setattr(scheduler, "_get_hermes_home", lambda: home)
    monkeypatch.setattr(scheduler, "_RUN_CLAIM_HEARTBEAT_SECONDS", 0.01)
    monkeypatch.setattr(scheduler, "heartbeat_fire_claim", lambda *a, **k: False)

    started = time.monotonic()
    success, output = scheduler._run_job_script_with_claim_heartbeat(
        {
            "id": "lost-script-owner",
            "schedule": {"kind": "interval", "minutes": 5},
            "_fire_claim_id": "stale-token",
        },
        "slow.py",
    )
    elapsed = time.monotonic() - started

    assert success is False
    assert "ownership" in output.lower()
    assert elapsed < 0.5
    time.sleep(1.0)
    assert not late_effect.exists()


def test_running_claimed_script_refreshes_parent_worker_pulse(tmp_path, monkeypatch):
    """A live pre-run/no-agent script is not mistaken for an idle agent."""
    import cron.scheduler as scheduler

    home = tmp_path / "profile"
    scripts = home / "scripts"
    scripts.mkdir(parents=True)
    (scripts / "slow.py").write_text("import time\ntime.sleep(0.15)\nprint('done')\n")
    pulse = tmp_path / "script.pulse"
    monkeypatch.setattr(scheduler, "_get_hermes_home", lambda: home)
    monkeypatch.setenv(scheduler._CRON_WORKER_PULSE_ENV, str(pulse))

    result = scheduler._run_job_script(
        "slow.py",
        abort_event=threading.Event(),
    )

    assert result == (True, "done")
    assert pulse.exists()


def test_claimed_script_timeout_kills_descendants(tmp_path, monkeypatch):
    """The claimed-script Popen path preserves timeout tree cleanup."""
    import cron.scheduler as scheduler

    home = tmp_path / "profile"
    scripts = home / "scripts"
    scripts.mkdir(parents=True)
    marker = tmp_path / "script-descendant-survived"
    descendant = (
        "import pathlib,signal,time;"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN);"
        "time.sleep(1.4);"
        f"pathlib.Path({str(marker)!r}).write_text('bad')"
    )
    (scripts / "timeout.py").write_text(
        "import subprocess,sys,time\n"
        f"subprocess.Popen([sys.executable, '-c', {descendant!r}])\n"
        "time.sleep(5)\n"
    )
    monkeypatch.setattr(scheduler, "_get_hermes_home", lambda: home)
    monkeypatch.setattr(scheduler, "_SCRIPT_TIMEOUT", 1)
    monkeypatch.setattr(scheduler, "_CRON_WORKER_TERMINATE_GRACE_SECONDS", 0.03)

    success, output = scheduler._run_job_script(
        "timeout.py",
        abort_event=threading.Event(),
    )

    assert success is False
    assert "timed out" in output.lower()
    time.sleep(0.5)
    assert not marker.exists()


@pytest.mark.live_system_guard_bypass
@pytest.mark.skipif(os.name == "nt", reason="POSIX detached-session regression")
def test_claimed_script_timeout_kills_detached_descendants(tmp_path, monkeypatch):
    """A script child cannot escape timeout cleanup by creating a session."""
    import cron.scheduler as scheduler

    home = tmp_path / "profile"
    scripts = home / "scripts"
    scripts.mkdir(parents=True)
    marker = tmp_path / "detached-script-descendant-survived"
    descendant = (
        "import pathlib,signal,time;"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN);"
        "time.sleep(1.4);"
        f"pathlib.Path({str(marker)!r}).write_text('bad')"
    )
    (scripts / "detached-timeout.py").write_text(
        "import subprocess,sys,time\n"
        f"subprocess.Popen([sys.executable, '-c', {descendant!r}], "
        "start_new_session=True)\n"
        "time.sleep(5)\n"
    )
    monkeypatch.setattr(scheduler, "_get_hermes_home", lambda: home)
    monkeypatch.setattr(scheduler, "_SCRIPT_TIMEOUT", 1)
    monkeypatch.setattr(
        scheduler,
        "_CRON_WORKER_TERMINATE_GRACE_SECONDS",
        0.03,
    )

    success, output = scheduler._run_job_script(
        "detached-timeout.py",
        abort_event=threading.Event(),
    )

    assert success is False
    assert "timed out" in output.lower()
    time.sleep(0.5)
    assert not marker.exists()
