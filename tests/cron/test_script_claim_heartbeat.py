"""Regression coverage for one-shot claims during blocking cron scripts."""

from datetime import datetime, timedelta, timezone
import contextlib
import sys
import threading
import time
from unittest.mock import MagicMock, patch

import pytest


def test_cancel_event_terminates_script_process_tree(tmp_path, monkeypatch):
    """Losing a fire claim must stop both the script and its descendants."""
    import cron.scheduler as scheduler

    monkeypatch.setattr(scheduler, "_get_hermes_home", lambda: tmp_path)
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    started = tmp_path / "started"
    child_done = tmp_path / "child-done"
    script = scripts_dir / "blocking.py"
    child_code = (
        "import time; from pathlib import Path; "
        f"time.sleep(1); Path({str(child_done)!r}).write_text('done')"
    )
    script.write_text(
        "import subprocess, sys, time\n"
        f"subprocess.Popen([sys.executable, '-c', {child_code!r}])\n"
        f"open({str(started)!r}, 'w').close()\n"
        "time.sleep(30)\n",
        encoding="utf-8",
    )

    cancel = threading.Event()
    result = []
    errors = []

    def _run() -> None:
        try:
            result.append(
                scheduler._run_job_script(
                    str(script),
                    workdir=str(tmp_path),
                    cancel_event=cancel,
                )
            )
        except Exception as exc:
            errors.append(exc)

    thread = threading.Thread(target=_run)
    thread.start()
    deadline = time.monotonic() + 5
    while not started.exists() and not errors and time.monotonic() < deadline:
        time.sleep(0.01)
    assert errors == []
    assert started.exists(), "script did not start"

    cancel.set()
    thread.join(timeout=3)

    assert errors == []
    assert not thread.is_alive(), "script ignored cancellation"
    assert result and result[0][0] is False
    assert "cancel" in result[0][1].lower()
    time.sleep(1.2)
    assert not child_done.exists(), "script descendant survived cancellation"


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process-group semantics")
def test_cancel_event_kills_sigterm_ignoring_descendant(tmp_path, monkeypatch):
    """A SIGTERM-ignoring grandchild must not wedge the cancellation path:
    the tree kill escalates to SIGKILL for surviving group members, and the
    pipe drain is bounded even if a descendant still holds the write ends."""
    import cron.scheduler as scheduler

    monkeypatch.setattr(scheduler, "_get_hermes_home", lambda: tmp_path)
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    started = tmp_path / "started"
    script = scripts_dir / "stubborn.py"
    child_code = (
        "import signal, time; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        f"open({str(started)!r}, 'w').close(); "
        "time.sleep(60)"
    )
    script.write_text(
        "import subprocess, sys, time\n"
        f"subprocess.Popen([sys.executable, '-c', {child_code!r}])\n"
        "time.sleep(60)\n",
        encoding="utf-8",
    )

    cancel = threading.Event()
    result = []
    errors = []

    def _run() -> None:
        try:
            result.append(
                scheduler._run_job_script(
                    str(script),
                    workdir=str(tmp_path),
                    cancel_event=cancel,
                )
            )
        except Exception as exc:
            errors.append(exc)

    thread = threading.Thread(target=_run)
    thread.start()
    deadline = time.monotonic() + 5
    while not started.exists() and not errors and time.monotonic() < deadline:
        time.sleep(0.01)
    assert errors == []
    assert started.exists(), "script did not spawn its descendant"

    cancel.set()
    # TERM grace (1s) + KILL + bounded drain (5s) + margin: must return well
    # before the unbounded-communicate hang this regresses against.
    thread.join(timeout=10)

    assert errors == []
    assert not thread.is_alive(), "cancellation wedged on a SIGTERM-ignoring descendant"
    assert result and result[0][0] is False
    assert "cancel" in result[0][1].lower()


def test_no_agent_forwards_cancel_event_to_script_runner(monkeypatch):
    import cron.scheduler as scheduler

    cancel = threading.Event()
    observed = []

    def _script_runner(job, script_path, workdir=None, cancel_event=None):
        observed.append(cancel_event)
        return True, ""

    monkeypatch.setattr(
        scheduler,
        "_run_job_script_with_claim_heartbeat",
        _script_runner,
    )

    success, _output, _response, error = scheduler.run_job(
        {
            "id": "cancel-aware-script",
            "name": "cancel aware",
            "script": "watchdog.py",
            "no_agent": True,
        },
        cancel_event=cancel,
    )

    assert success is True
    assert error is None
    assert observed == [cancel]


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

    def _blocking_script(_script_path: str, **kwargs) -> tuple[bool, str]:
        assert heartbeat_seen.wait(timeout=2), (
            "claim was not refreshed while script blocked"
        )
        return True, script_output

    monkeypatch.setattr(scheduler, "heartbeat_run_claim", _observed_heartbeat)
    monkeypatch.setattr(scheduler, "_run_job_script", _blocking_script)

    with (
        jobs.use_cron_store(profile_home),
        patch("hermes_state.get_shared_session_db", return_value=MagicMock()),
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

    def _blocking_script(_script_path: str, **kwargs) -> tuple[bool, str]:
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
        assert isinstance(claimed_job, dict)
        assert scheduler.run_one_job(claimed_job) is True
        refreshed = jobs.get_job(job["id"])["fire_claim"]

    assert refreshed["at"] != original_claim["at"]
    assert refreshed["by"] == original_claim["by"]


def test_slow_owned_delivery_does_not_false_interrupt_completed_run(
    tmp_path, monkeypatch
):
    """An owner-fenced delivery may outlast one heartbeat lock attempt.

    The worker still owns the unchanged fire claim throughout. A transient
    failure to acquire its own delivery fence must not become ownership loss.
    """
    import cron.executions as executions
    import cron.jobs as jobs
    import cron.scheduler as scheduler

    profile_home = tmp_path / "profile"
    profile_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(profile_home))
    monkeypatch.setattr(
        executions, "EXECUTIONS_FILE", profile_home / "cron" / "executions.db"
    )

    with jobs.use_cron_store(profile_home):
        job = jobs.create_job(prompt="x", schedule="every 5m", name="slow delivery")
        assert jobs.claim_job_for_fire(job["id"]) is True
        claimed_job = jobs.get_job(job["id"])
        assert claimed_job is not None
        execution = executions.create_execution(job["id"], source="builtin")
        claimed_job["execution_id"] = execution["id"]

        monkeypatch.setenv("_HERMES_CRON_EXTERNAL_WORKER", execution["id"])
        monkeypatch.setattr(scheduler, "_RUN_CLAIM_HEARTBEAT_SECONDS", 0.01)
        monkeypatch.setattr(jobs, "_JOBS_LOCK_TIMEOUT_SECONDS", 0.01)
        monkeypatch.setattr(scheduler, "claim_dispatch", lambda _job_id: True)
        monkeypatch.setattr(
            scheduler,
            "run_job",
            lambda *_args, **_kwargs: (True, "output", "completed", None),
        )
        monkeypatch.setattr(
            scheduler, "save_job_output", lambda *_args, **_kwargs: tmp_path / "output.md"
        )

        false_heartbeat_seen = threading.Event()
        real_heartbeat = jobs.heartbeat_fire_claim

        expected_owner = claimed_job["fire_claim"]["by"]

        def _observed_heartbeat(job_id: str, *, expected_owner: str) -> bool:
            renewed = real_heartbeat(job_id, expected_owner=expected_owner)
            if not renewed:
                false_heartbeat_seen.set()
            return renewed

        def _slow_delivery(*_args, **_kwargs):
            assert false_heartbeat_seen.wait(timeout=2)
            still_owned = jobs.get_job(job["id"])
            assert still_owned is not None
            assert still_owned["fire_claim"]["by"] == expected_owner
            return None

        monkeypatch.setattr(scheduler, "heartbeat_fire_claim", _observed_heartbeat)
        monkeypatch.setattr(scheduler, "_deliver_result", _slow_delivery)

        assert scheduler.run_one_job(claimed_job) is True

        finished = executions.get_execution(execution["id"])
        persisted_job = jobs.get_job(job["id"])
        assert finished is not None
        assert persisted_job is not None

    assert (
        finished["status"],
        finished["error"],
        persisted_job["last_status"],
    ) == ("completed", None, "ok")


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

    def _run_job(
        job,
        *,
        defer_agent_teardown=None,
        extra_prompt=None,
        cancel_event=None,
        execution_id=None,
    ):
        assert execution_id == job["execution_id"]
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
    monkeypatch.setattr(scheduler, "mark_execution_running", lambda execution_id: {})
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
    monkeypatch.setattr(scheduler, "mark_execution_running", lambda *_args: {})
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


def test_heartbeat_lock_contention_is_unconfirmed_not_ownership_loss(monkeypatch):
    """Fence contention (None) is unconfirmed ownership, not a loss (c-027).

    During slow fenced delivery the worker's own heartbeat thread cannot
    acquire the process-local fire fence. That must not become an
    authoritative ownership-loss cancellation: None rides the grace window
    while only an explicit False interrupts immediately.
    """
    import cron.scheduler as scheduler

    calls = []

    def heartbeat(*_args, **_kwargs):
        # Initial validation renews (True); afterwards the fence stays
        # contended for the whole run: never False, never raises.
        calls.append("none" if calls else "true")
        return True if len(calls) == 1 else None

    ran = threading.Event()

    def run_body(_job, **kwargs):
        # Far beyond several heartbeat ticks, well under the grace window.
        assert not kwargs["fire_claim_lost"].wait(timeout=0.08)
        ran.set()
        return True

    job = {
        "id": "fence-contended",
        "fire_claim": {"at": "2026-07-12T12:00:00+00:00", "by": "owner"},
    }
    monkeypatch.setattr(scheduler, "heartbeat_fire_claim", heartbeat)
    monkeypatch.setattr(scheduler, "_run_one_job_body", run_body)
    monkeypatch.setattr(scheduler, "_RUN_CLAIM_HEARTBEAT_SECONDS", 0.01)

    assert scheduler.run_one_job(job) is True

    assert ran.is_set()
    assert calls, "heartbeat loop never probed the fire claim"


def test_pre_delivery_ownership_probe_treats_none_as_not_lost(tmp_path, monkeypatch):
    """A contended (None) post-run probe must not discard a completed run (c-027).

    Between run_job returning and delivery, ``_fire_claim_ownership_lost()``
    probes the claim. A None probe (fence contended, likely by our own
    fenced save) is not a loss: delivery must proceed and the run completes.
    """
    import cron.executions as executions
    import cron.jobs as jobs
    import cron.scheduler as scheduler

    profile_home = tmp_path / "profile"
    profile_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(profile_home))
    monkeypatch.setattr(
        executions, "EXECUTIONS_FILE", profile_home / "cron" / "executions.db"
    )

    with jobs.use_cron_store(profile_home):
        job = jobs.create_job(prompt="x", schedule="every 5m", name="probe none")
        assert jobs.claim_job_for_fire(job["id"]) is True
        claimed_job = jobs.get_job(job["id"])
        execution = executions.create_execution(job["id"], source="builtin")
        claimed_job["execution_id"] = execution["id"]

        monkeypatch.setenv("_HERMES_CRON_EXTERNAL_WORKER", execution["id"])
        monkeypatch.setattr(scheduler, "claim_dispatch", lambda _job_id: True)
        monkeypatch.setattr(
            scheduler,
            "run_job",
            lambda *_args, **_kwargs: (True, "output", "done", None),
        )
        monkeypatch.setattr(
            scheduler, "save_job_output", lambda *_args, **_kwargs: "output.md"
        )

        real_probe = jobs.heartbeat_fire_claim
        probes = []

        def _contended_probe(job_id: str, *, expected_owner: str):
            outcome = real_probe(job_id, expected_owner=expected_owner)
            probes.append(outcome)
            # First probe (initial validation) is clean; post-run probes hit
            # a contended fence.
            return True if len(probes) == 1 else None

        monkeypatch.setattr(scheduler, "heartbeat_fire_claim", _contended_probe)
        delivered = []

        def _deliver(*_args, **_kwargs):
            delivered.append(True)
            return None

        monkeypatch.setattr(scheduler, "_deliver_result", _deliver)

        assert scheduler.run_one_job(claimed_job) is True

        assert delivered, "completed run was not delivered"

    finished = executions.get_execution(execution["id"])
    persisted_job = jobs.get_job(job["id"])
    assert finished is not None and persisted_job is not None
    assert (
        finished["status"],
        finished["error"],
        persisted_job["last_status"],
    ) == ("completed", None, "ok")


def test_post_delivery_ownership_probe_treats_none_as_not_lost(tmp_path, monkeypatch):
    """A contended (None) probe after delivery must not overwrite the result (c-027)."""
    import cron.executions as executions
    import cron.jobs as jobs
    import cron.scheduler as scheduler

    profile_home = tmp_path / "profile"
    profile_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(profile_home))
    monkeypatch.setattr(
        executions, "EXECUTIONS_FILE", profile_home / "cron" / "executions.db"
    )

    with jobs.use_cron_store(profile_home):
        job = jobs.create_job(prompt="x", schedule="every 5m", name="post none")
        assert jobs.claim_job_for_fire(job["id"]) is True
        claimed_job = jobs.get_job(job["id"])
        execution = executions.create_execution(job["id"], source="builtin")
        claimed_job["execution_id"] = execution["id"]

        monkeypatch.setenv("_HERMES_CRON_EXTERNAL_WORKER", execution["id"])
        monkeypatch.setattr(scheduler, "claim_dispatch", lambda _job_id: True)
        monkeypatch.setattr(
            scheduler,
            "run_job",
            lambda *_args, **_kwargs: (True, "output", "done", None),
        )
        monkeypatch.setattr(
            scheduler, "save_job_output", lambda *_args, **_kwargs: "output.md"
        )

        real_probe = jobs.heartbeat_fire_claim
        probes = []

        def _contended_probe(job_id: str, *, expected_owner: str):
            outcome = real_probe(job_id, expected_owner=expected_owner)
            probes.append(outcome)
            return True if len(probes) == 1 else None

        monkeypatch.setattr(scheduler, "heartbeat_fire_claim", _contended_probe)

        def _deliver(*_args, **_kwargs):
            return None

        monkeypatch.setattr(scheduler, "_deliver_result", _deliver)

        assert scheduler.run_one_job(claimed_job) is True

    finished = executions.get_execution(execution["id"])
    persisted_job = jobs.get_job(job["id"])
    assert finished is not None and persisted_job is not None
    assert (
        finished["status"],
        finished["error"],
        persisted_job["last_status"],
    ) == ("completed", None, "ok")


def test_interrupted_path_none_probe_does_not_claim_confirmed_loss(monkeypatch):
    """Grace-exhausted stop + contended probe must not claim a confirmed loss (c-027)."""
    import cron.scheduler as scheduler

    calls = []

    def heartbeat(*_args, **_kwargs):
        calls.append(1)
        return True if len(calls) == 1 else None

    def _run_job(job, *, defer_agent_teardown=None, extra_prompt=None,
                 cancel_event=None, execution_id=None):
        # Sustained None heartbeats exhaust the grace window -> lost_ownership
        # set as "uncertain, stop". Never an explicit False.
        assert cancel_event is not None
        assert cancel_event.wait(timeout=1.0)
        return True, "output", "response", None

    finish = MagicMock()
    mark = MagicMock(return_value=True)
    job = {
        "id": "grace-none",
        "execution_id": "grace-none-exec",
        "fire_claim": {"at": "2026-07-12T12:00:00+00:00", "by": "owner"},
    }
    monkeypatch.setattr(scheduler, "heartbeat_fire_claim", heartbeat)
    monkeypatch.setattr(scheduler, "run_job", _run_job)
    monkeypatch.setattr(scheduler, "_RUN_CLAIM_HEARTBEAT_SECONDS", 0.01)
    monkeypatch.setattr(scheduler, "_FIRE_CLAIM_HEARTBEAT_GRACE_SECONDS", 0.03)
    monkeypatch.setattr(scheduler, "claim_dispatch", lambda *_a: True)
    monkeypatch.setattr(scheduler, "mark_execution_running", lambda *_a: {})
    monkeypatch.setattr(scheduler, "mark_job_run", mark)
    monkeypatch.setattr(scheduler, "finish_execution", finish)

    with patch("agent.secret_scope.set_secret_scope", return_value=None), \
         patch("agent.secret_scope.build_profile_secret_scope", return_value=None), \
         patch("agent.secret_scope.reset_secret_scope"):
        assert scheduler.run_one_job(job) is True

    errors = [call.kwargs.get("error") for call in finish.call_args_list]
    assert errors, "execution ledger never closed"
    for err in errors:
        assert "ownership lost" not in err.lower(), (
            f"None probe adjudicated as confirmed ownership loss: {err!r}"
        )


def test_terminal_write_none_does_not_mark_ownership_lost(monkeypatch):
    """An unconfirmed (None) terminal write must not be read as CAS loss (c-027)."""
    import cron.scheduler as scheduler

    @contextlib.contextmanager
    def owned_fence(*_args, **_kwargs):
        yield True

    finish = MagicMock()
    job = {
        "id": "terminal-none",
        "execution_id": "terminal-none-exec",
        "name": "terminal-none",
        "fire_claim": {"at": "2026-07-12T12:00:00+00:00", "by": "owner"},
    }
    monkeypatch.setattr(scheduler, "heartbeat_fire_claim", lambda *a, **k: True)
    monkeypatch.setattr(scheduler, "claim_dispatch", lambda *_a: True)
    monkeypatch.setattr(scheduler, "mark_execution_running", lambda *_a: {})
    monkeypatch.setattr(
        scheduler,
        "run_job",
        lambda *_a, **_k: (True, "output", "response", None),
    )
    monkeypatch.setattr(scheduler, "fire_claim_fence", owned_fence, raising=False)
    monkeypatch.setattr(scheduler, "save_job_output", lambda *_a: "output.md")
    monkeypatch.setattr(scheduler, "_deliver_result", lambda *_a, **_k: None)
    # Terminal write hits a contended fence -> None (unconfirmed, not a CAS loss).
    monkeypatch.setattr(scheduler, "mark_job_run", lambda *_a, **_k: None)
    monkeypatch.setattr(scheduler, "finish_execution", finish)

    with patch("agent.secret_scope.set_secret_scope", return_value=None), \
         patch("agent.secret_scope.build_profile_secret_scope", return_value=None), \
         patch("agent.secret_scope.reset_secret_scope"):
        assert scheduler.run_one_job(job) is True

    for call in finish.call_args_list:
        assert call.kwargs.get("error") != (
            "Fire claim ownership lost before terminal completion."
        ), "unconfirmed terminal write adjudicated as ownership loss"


def test_recovery_execution_fails_closed_while_original_delivery_settles(
    tmp_path, monkeypatch
):
    """RSI-loop case (c-027): a recovery dispatch of the same job while the
    original holds the fence in slow delivery must (a) fail closed without
    starting a duplicate run and (b) not disturb the original's claim.
    """
    import cron.executions as executions
    import cron.jobs as jobs
    import cron.scheduler as scheduler

    profile_home = tmp_path / "profile"
    profile_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(profile_home))
    monkeypatch.setattr(
        executions, "EXECUTIONS_FILE", profile_home / "cron" / "executions.db"
    )

    with jobs.use_cron_store(profile_home):
        job = jobs.create_job(prompt="x", schedule="every 5m", name="rsi loop")
        assert jobs.claim_job_for_fire(job["id"]) is True
        claimed_job = jobs.get_job(job["id"])
        original_owner = claimed_job["fire_claim"]["by"]
        execution = executions.create_execution(job["id"], source="builtin")
        claimed_job["execution_id"] = execution["id"]

        monkeypatch.setenv("_HERMES_CRON_EXTERNAL_WORKER", execution["id"])
        monkeypatch.setattr(scheduler, "claim_dispatch", lambda _job_id: True)
        monkeypatch.setattr(
            scheduler,
            "run_job",
            lambda *_args, **_kwargs: (True, "output", "done", None),
        )
        monkeypatch.setattr(
            scheduler, "save_job_output", lambda *_args, **_kwargs: "output.md"
        )

        in_delivery = threading.Event()
        release_delivery = threading.Event()

        def _slow_delivery(*_args, **_kwargs):
            in_delivery.set()
            assert release_delivery.wait(timeout=5)
            return None

        monkeypatch.setattr(scheduler, "_deliver_result", _slow_delivery)
        # Short fence timeout so the recovery dispatch's contended validation
        # resolves as None quickly instead of blocking on the real 30s wait.
        monkeypatch.setattr(jobs, "_JOBS_LOCK_TIMEOUT_SECONDS", 0.02)

        runner_done = threading.Event()
        runner_result = []

        def _original_runner():
            runner_result.append(scheduler.run_one_job(dict(claimed_job)))
            runner_done.set()

        runner = threading.Thread(target=_original_runner, daemon=True)
        runner.start()
        assert in_delivery.wait(timeout=5)

        # A recovery execution of the SAME job (new execution id, no
        # external-worker marker) attempts to start while the fence is held.
        monkeypatch.delenv("_HERMES_CRON_EXTERNAL_WORKER", raising=False)
        recovery_execution = executions.create_execution(job["id"], source="builtin")
        recovery_job = dict(claimed_job)
        recovery_job["execution_id"] = recovery_execution["id"]
        recovery_job["fire_claim"] = dict(claimed_job["fire_claim"])

        recovery_ran = []

        def _must_not_run(*_args, **_kwargs):
            recovery_ran.append(True)
            return True, "duplicate", "duplicate", None

        monkeypatch.setattr(scheduler, "run_job", _must_not_run)

        assert scheduler.run_one_job(recovery_job) is True
        assert not recovery_ran, "recovery dispatch ran a duplicate side effect"
        recovery_row = executions.get_execution(recovery_execution["id"])
        assert recovery_row is not None
        assert recovery_row["status"] == "failed"

        # The original must be untouched and still complete its delivery.
        current = jobs.get_job(job["id"])
        assert current is not None
        assert current["fire_claim"]["by"] == original_owner

        release_delivery.set()
        assert runner_done.wait(timeout=5)
        assert runner_result == [True]

    finished = executions.get_execution(execution["id"])
    persisted_job = jobs.get_job(job["id"])
    assert finished is not None and persisted_job is not None
    assert (
        finished["status"],
        finished["error"],
        persisted_job["last_status"],
    ) == ("completed", None, "ok")


def test_terminal_write_after_slow_owned_delivery_succeeds(tmp_path, monkeypatch):
    """TrustMRR case (c-027): the terminal write runs after fenced delivery
    releases and must record the successful run, not a false loss.
    """
    import cron.executions as executions
    import cron.jobs as jobs
    import cron.scheduler as scheduler

    profile_home = tmp_path / "profile"
    profile_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(profile_home))
    monkeypatch.setattr(
        executions, "EXECUTIONS_FILE", profile_home / "cron" / "executions.db"
    )

    with jobs.use_cron_store(profile_home):
        job = jobs.create_job(prompt="x", schedule="every 5m", name="trustmrr")
        assert jobs.claim_job_for_fire(job["id"]) is True
        claimed_job = jobs.get_job(job["id"])
        execution = executions.create_execution(job["id"], source="builtin")
        claimed_job["execution_id"] = execution["id"]

        monkeypatch.setenv("_HERMES_CRON_EXTERNAL_WORKER", execution["id"])
        monkeypatch.setattr(scheduler, "claim_dispatch", lambda _job_id: True)
        monkeypatch.setattr(
            scheduler,
            "run_job",
            lambda *_args, **_kwargs: (True, "output", "done", None),
        )
        monkeypatch.setattr(
            scheduler, "save_job_output", lambda *_args, **_kwargs: "output.md"
        )

        def _slow_delivery(*_args, **_kwargs):
            # Hold the fence past several heartbeat intervals so the real
            # heartbeat thread contends on the lock the delivery itself
            # holds (the production shape).
            time.sleep(0.08)
            return None

        monkeypatch.setattr(scheduler, "_deliver_result", _slow_delivery)

        mark_calls = []

        real_mark = jobs.mark_job_run

        def _observed_mark(job_id, success, error=None, **kwargs):
            result = real_mark(job_id, success, error, **kwargs)
            mark_calls.append((success, error, result))
            return result

        monkeypatch.setattr(scheduler, "mark_job_run", _observed_mark)

        assert scheduler.run_one_job(claimed_job) is True

    finished = executions.get_execution(execution["id"])
    persisted_job = jobs.get_job(job["id"])
    assert finished is not None and persisted_job is not None
    assert finished["status"] == "completed"
    assert persisted_job["last_status"] == "ok"
    terminal = [call for call in mark_calls if call[2] is not None]
    assert terminal and terminal[-1][0] is True, (
        f"terminal write never recorded success: {mark_calls!r}"
    )


def test_none_probe_cannot_authorize_unfenced_exception_delivery(
    tmp_path, monkeypatch
):
    """A contended ownership probe is unconfirmed; only the side-effect
    fence may authorize the exception-path failure notification."""
    import cron.scheduler as scheduler

    probes = iter([True, None])
    delivered = MagicMock(return_value=None)
    fence = MagicMock(
        side_effect=lambda *a, **k: contextlib.nullcontext(False)
    )
    monkeypatch.setattr(
        scheduler, "heartbeat_fire_claim", lambda *a, **k: next(probes)
    )
    monkeypatch.setattr(scheduler, "_RUN_CLAIM_HEARTBEAT_SECONDS", 60.0)
    monkeypatch.setattr(scheduler, "claim_dispatch", lambda *a, **k: True)
    monkeypatch.setattr(scheduler, "mark_execution_running", lambda *a, **k: {})
    monkeypatch.setattr(
        scheduler, "run_job", MagicMock(side_effect=RuntimeError("run failure"))
    )
    monkeypatch.setattr(scheduler, "_deliver_result", delivered)
    monkeypatch.setattr(scheduler, "fire_claim_fence", fence)
    monkeypatch.setattr(scheduler, "mark_job_run", lambda *a, **k: True)
    monkeypatch.setattr(scheduler, "finish_execution", MagicMock())
    monkeypatch.setattr(
        scheduler, "_upsert_incident_for_failure", lambda *a, **k: (False, None)
    )
    monkeypatch.setattr(scheduler, "_mark_incident_alerted", lambda *a, **k: None)
    monkeypatch.setattr(scheduler, "_get_hermes_home", lambda: tmp_path)

    job = {
        "id": "stale-worker",
        "name": "stale worker",
        "execution_id": "exec-stale",
        "prompt": "x",
        "schedule_display": "manual",
        "deliver": "telegram:123",
        "fire_claim": {"by": "old-owner"},
    }

    with patch("agent.secret_scope.set_secret_scope", return_value=None), \
         patch("agent.secret_scope.build_profile_secret_scope", return_value=None), \
         patch("agent.secret_scope.reset_secret_scope"), \
         patch("tools.terminal_scope.install_profile_terminal_scope", return_value=None):
        assert scheduler.run_one_job(job) is False

    fence.assert_called_once_with("stale-worker", expected_owner="old-owner")
    delivered.assert_not_called()
