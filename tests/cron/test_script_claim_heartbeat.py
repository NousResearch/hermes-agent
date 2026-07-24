"""Regression coverage for one-shot claims during blocking cron scripts."""

from datetime import datetime, timedelta, timezone
import contextlib
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


def test_run_one_job_refreshes_fire_claim_in_profile_store(tmp_path, monkeypatch):
    """The shared execute/save/deliver body keeps its durable fire claim alive."""
    import cron.jobs as jobs
    import cron.scheduler as scheduler

    profile_home = tmp_path / "profile"
    profile_home.mkdir()
    with jobs.use_cron_store(profile_home):
        job = jobs.create_job(prompt="x", schedule="every 5m", name="agent-run")
        assert jobs.claim_job_for_fire(job["id"]) is True
        claimed_job = jobs.get_job(job["id"])
        original_claim = dict(claimed_job["fire_claim"])

    heartbeat_seen = threading.Event()
    real_heartbeat = jobs.heartbeat_fire_claim

    def _observed_heartbeat(job_id: str, *, expected_owner: str) -> bool:
        updated = real_heartbeat(job_id, expected_owner=expected_owner)
        heartbeat_seen.set()
        return updated

    def _blocking_body(job, **kwargs):
        assert heartbeat_seen.wait(timeout=2)
        return True

    monkeypatch.setattr(scheduler, "_RUN_CLAIM_HEARTBEAT_SECONDS", 0.01)
    monkeypatch.setattr(scheduler, "heartbeat_fire_claim", _observed_heartbeat)
    monkeypatch.setattr(scheduler, "_run_one_job_body", _blocking_body)

    with jobs.use_cron_store(profile_home):
        assert scheduler.run_one_job(claimed_job) is True
        refreshed = jobs.get_job(job["id"])["fire_claim"]

    assert refreshed["at"] != original_claim["at"]
    assert refreshed["by"] == original_claim["by"]


def test_lost_fire_claim_stops_stale_delivery(monkeypatch):
    """A runner that loses its durable owner must not deliver its stale result."""
    import cron.scheduler as scheduler

    lost_seen = threading.Event()
    heartbeat_calls = 0

    def _heartbeat(job_id: str, *, expected_owner: str) -> bool:
        nonlocal heartbeat_calls
        heartbeat_calls += 1
        if heartbeat_calls == 1:
            return True
        lost_seen.set()
        return False

    def _run_job(job, *, defer_agent_teardown=None, cancel_event=None):
        assert lost_seen.wait(timeout=2)
        return True, "stale output", "stale response", None

    job = {
        "id": "reclaimed-agent",
        "name": "reclaimed agent",
        "prompt": "work",
        "execution_id": "stale-execution",
        "fire_claim": {"at": "2026-07-12T12:00:00+00:00", "by": "stale-owner"},
    }
    monkeypatch.setattr(scheduler, "_RUN_CLAIM_HEARTBEAT_SECONDS", 0.01)
    monkeypatch.setattr(scheduler, "heartbeat_fire_claim", _heartbeat)
    monkeypatch.setattr(scheduler, "run_job", _run_job)
    monkeypatch.setattr(scheduler, "claim_dispatch", lambda job_id: True)
    monkeypatch.setattr(scheduler, "mark_execution_running", lambda execution_id: None)
    monkeypatch.setattr(scheduler, "finish_execution", lambda *args, **kwargs: None)
    save_output = MagicMock()
    deliver_result = MagicMock()
    mark_run = MagicMock()
    monkeypatch.setattr(scheduler, "save_job_output", save_output)
    monkeypatch.setattr(scheduler, "_deliver_result", deliver_result)
    monkeypatch.setattr(scheduler, "mark_job_run", mark_run)

    with patch("agent.secret_scope.set_secret_scope", return_value=None), \
         patch("agent.secret_scope.build_profile_secret_scope", return_value=None), \
         patch("agent.secret_scope.reset_secret_scope"):
        assert scheduler.run_one_job(job) is True

    save_output.assert_not_called()
    deliver_result.assert_not_called()
    mark_run.assert_not_called()


def test_initially_lost_fire_claim_finishes_execution_without_running(monkeypatch):
    """A stale claimed snapshot rejected before body entry must close its ledger row."""
    import cron.scheduler as scheduler

    run_body = MagicMock(return_value=True)
    finish = MagicMock()
    job = {
        "id": "already-reclaimed",
        "execution_id": "stale-execution",
        "fire_claim": {"at": "2026-07-12T12:00:00+00:00", "by": "stale-owner"},
    }
    monkeypatch.setattr(scheduler, "heartbeat_fire_claim", lambda *args, **kwargs: False)
    monkeypatch.setattr(scheduler, "_run_one_job_body", run_body)
    monkeypatch.setattr(scheduler, "finish_execution", finish)

    assert scheduler.run_one_job(job) is True

    run_body.assert_not_called()
    finish.assert_called_once_with(
        "stale-execution",
        success=False,
        error="Fire claim ownership lost before execution started.",
    )


def test_initially_lost_claim_does_not_run_when_ledger_write_fails(monkeypatch):
    """A ledger I/O error cannot turn a confirmed ownership loss into execution."""
    import cron.scheduler as scheduler

    run_body = MagicMock(return_value=True)
    job = {
        "id": "already-reclaimed",
        "execution_id": "stale-execution",
        "fire_claim": {"at": "2026-07-12T12:00:00+00:00", "by": "stale-owner"},
    }
    monkeypatch.setattr(scheduler, "heartbeat_fire_claim", lambda *args, **kwargs: False)
    monkeypatch.setattr(scheduler, "_run_one_job_body", run_body)
    monkeypatch.setattr(
        scheduler,
        "finish_execution",
        MagicMock(side_effect=OSError("ledger unavailable")),
    )

    assert scheduler.run_one_job(job) is True
    run_body.assert_not_called()


def test_initial_heartbeat_exception_does_not_start_execution(monkeypatch):
    """Unconfirmed initial ownership must fail closed before any side effect."""
    import cron.scheduler as scheduler

    run_body = MagicMock(return_value=True)
    finish = MagicMock()
    job = {
        "id": "validation-error",
        "execution_id": "validation-execution",
        "fire_claim": {"at": "2026-07-12T12:00:00+00:00", "by": "owner"},
    }
    monkeypatch.setattr(
        scheduler,
        "heartbeat_fire_claim",
        MagicMock(side_effect=OSError("store unavailable")),
    )
    monkeypatch.setattr(scheduler, "_run_one_job_body", run_body)
    monkeypatch.setattr(scheduler, "finish_execution", finish)

    assert scheduler.run_one_job(job) is True

    run_body.assert_not_called()
    finish.assert_called_once_with(
        "validation-execution",
        success=False,
        error="Fire claim ownership could not be validated before execution started.",
    )


def test_heartbeat_thread_start_failure_does_not_start_execution(monkeypatch):
    """A claimed job cannot run when no renewal monitor protects its lease."""
    import cron.scheduler as scheduler

    run_body = MagicMock(return_value=True)
    finish = MagicMock()
    job = {
        "id": "thread-start-error",
        "execution_id": "thread-execution",
        "fire_claim": {"at": "2026-07-12T12:00:00+00:00", "by": "owner"},
    }
    monkeypatch.setattr(scheduler, "heartbeat_fire_claim", lambda *args, **kwargs: True)
    monkeypatch.setattr(scheduler, "_run_one_job_body", run_body)
    monkeypatch.setattr(scheduler, "finish_execution", finish)
    monkeypatch.setattr(
        scheduler.threading.Thread,
        "start",
        MagicMock(side_effect=RuntimeError("cannot start thread")),
    )

    assert scheduler.run_one_job(job) is True

    run_body.assert_not_called()
    finish.assert_called_once_with(
        "thread-execution",
        success=False,
        error="Fire claim heartbeat could not be started; execution was not run.",
    )


def test_repeated_heartbeat_errors_cancel_after_bounded_grace(monkeypatch):
    """Store uncertainty cannot let a run outlive its last confirmed lease forever."""
    import cron.scheduler as scheduler

    calls = 0

    def heartbeat(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return True
        raise OSError("store unavailable")

    def run_body(_job, **kwargs):
        assert kwargs["fire_claim_lost"].wait(timeout=0.5)
        return True

    job = {
        "id": "heartbeat-errors",
        "fire_claim": {"at": "2026-07-12T12:00:00+00:00", "by": "owner"},
    }
    monkeypatch.setattr(scheduler, "heartbeat_fire_claim", heartbeat)
    monkeypatch.setattr(scheduler, "_run_one_job_body", run_body)
    monkeypatch.setattr(scheduler, "_RUN_CLAIM_HEARTBEAT_SECONDS", 0.01)
    monkeypatch.setattr(scheduler, "_FIRE_CLAIM_HEARTBEAT_GRACE_SECONDS", 0.03)

    assert scheduler.run_one_job(job) is True
    assert calls >= 3


def test_terminal_owner_cas_failure_marks_ledger_ownership_lost(monkeypatch):
    """A replacement owner cannot leave the stale ledger recorded as success."""
    import cron.scheduler as scheduler

    @contextlib.contextmanager
    def owned_fence(*_args, **_kwargs):
        yield True

    job = {
        "id": "terminal-cas",
        "execution_id": "execution-cas",
        "name": "terminal-cas",
        "fire_claim": {"at": "2026-07-12T12:00:00+00:00", "by": "owner"},
    }
    finish = MagicMock()
    monkeypatch.setattr(scheduler, "heartbeat_fire_claim", lambda *args, **kwargs: True)
    monkeypatch.setattr(scheduler, "claim_dispatch", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(scheduler, "mark_execution_running", lambda *_args: None)
    monkeypatch.setattr(
        scheduler,
        "run_job",
        lambda *_args, **_kwargs: (True, "output", "response", None),
    )
    monkeypatch.setattr(scheduler, "fire_claim_fence", owned_fence, raising=False)
    monkeypatch.setattr(scheduler, "save_job_output", lambda *_args: "output.md")
    monkeypatch.setattr(scheduler, "_deliver_result", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(scheduler, "mark_job_run", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(scheduler, "finish_execution", finish)

    with patch("agent.secret_scope.set_secret_scope", return_value=None), \
         patch("agent.secret_scope.build_profile_secret_scope", return_value=None), \
         patch("agent.secret_scope.reset_secret_scope"):
        assert scheduler.run_one_job(job) is True

    finish.assert_called_once_with(
        "execution-cas",
        success=False,
        error="Fire claim ownership lost before terminal completion.",
    )
