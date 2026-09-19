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
    from cron import scheduler_script as sched_script

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
                sched_script._run_job_script(
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
    from cron import scheduler_script as sched_script

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
                sched_script._run_job_script(
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
    from cron import scheduler_script as sched_script

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
    from cron import scheduler_script as sched_script

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
    monkeypatch.setattr(sched_script, "_run_job_script", _blocking_script)

    with (
        jobs.use_cron_store(profile_home),
        patch("hermes_state_registry.acquire", return_value=MagicMock()),
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
    from cron import scheduler_script as sched_script

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
    monkeypatch.setattr(sched_script, "_run_job_script", _blocking_script)

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
    from cron import scheduler_script as sched_script

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


def test_lost_fire_claim_stops_stale_delivery(monkeypatch):
    """A runner that loses its durable owner must not deliver its stale result."""
    import cron.scheduler as scheduler
    from cron import scheduler_script as sched_script

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


def _run_claimed_job_with_mid_run_action(
    tmp_path, monkeypatch, mid_run, *, execution_id, stub_output=True, crash=None,
    expect_result=True,
):
    """Fire a claimed job through run_one_job with a stubbed agent run that performs ``mid_run``
    on its own record, keeps working past one fire-claim heartbeat tick, then completes (or
    raises ``crash``)."""
    import cron.jobs as jobs
    import cron.scheduler as scheduler

    def _run_job(job, **_kwargs):
        mid_run(jobs, job)
        time.sleep(0.3)
        if crash is not None:
            raise crash
        return True, "saved output", "D1 is promoting", None

    delivered = MagicMock(return_value=None)
    finished = MagicMock()
    monkeypatch.setattr(scheduler, "_RUN_CLAIM_HEARTBEAT_SECONDS", 0.05)
    monkeypatch.setattr(scheduler, "run_job", _run_job)
    monkeypatch.setattr(scheduler, "claim_dispatch", lambda *_args: True)
    monkeypatch.setattr(scheduler, "mark_execution_running", lambda *_args: {})
    monkeypatch.setattr(scheduler, "finish_execution", finished)
    if stub_output:
        monkeypatch.setattr(scheduler, "save_job_output", lambda *_args: "output.md")
    monkeypatch.setattr(scheduler, "_deliver_result", delivered)

    with jobs.use_cron_store(tmp_path):
        job = jobs.create_job(
            prompt="work", schedule="every 5m", name="remove self", deliver="telegram")
        assert jobs.claim_job_for_fire(job["id"])
        claimed = jobs.get_job(job["id"])
        claimed["execution_id"] = execution_id

        with patch("agent.secret_scope.set_secret_scope", return_value=None), \
             patch("agent.secret_scope.build_profile_secret_scope", return_value=None), \
             patch("agent.secret_scope.reset_secret_scope"):
            assert scheduler.run_one_job(claimed) is expect_result
    return delivered, finished


def test_self_removed_job_still_delivers_after_post_removal_heartbeat(tmp_path, monkeypatch):
    """A run that removes its own job (cronjob remove on its own id) and keeps working past a
    heartbeat tick must still deliver its final response and complete its ledger row (#111039)."""
    import cron.jobs as jobs

    delivered, finished = _run_claimed_job_with_mid_run_action(
        tmp_path, monkeypatch,
        lambda jobs_mod, job: jobs_mod.remove_job(job["id"]),
        execution_id="self-removal-heartbeat-execution")

    delivered.assert_called_once()
    finished.assert_called_once_with(
        "self-removal-heartbeat-execution",
        success=True, error=None, delivery_outcome="delivered", delivery_error=None)
    with jobs.use_cron_store(tmp_path):
        assert jobs.load_jobs() == []


def test_self_removed_job_leaves_no_output_directory(tmp_path, monkeypatch):
    """remove_job() deletes <cron>/output/<job_id>/; the finishing run must not re-create it
    (an orphan directory per self-removing job), so 'only the job record is gone' stays true."""
    import cron.jobs as jobs

    delivered, _finished = _run_claimed_job_with_mid_run_action(
        tmp_path, monkeypatch,
        lambda jobs_mod, job: jobs_mod.remove_job(job["id"]),
        execution_id="self-removal-output-execution", stub_output=False)

    delivered.assert_called_once()
    with jobs.use_cron_store(tmp_path):
        assert jobs.load_jobs() == []
    assert list((tmp_path / "cron" / "output").glob("*")) == []


def test_self_removed_job_crash_skips_mark_job_run(tmp_path, monkeypatch):
    """A run that crashes after removing its own record has no record to mark: the crash path
    must skip mark_job_run like the completion path does, not probe a missing record."""
    import cron.scheduler as scheduler

    marked = MagicMock(return_value=True)
    monkeypatch.setattr(scheduler, "mark_job_run", marked)
    _delivered, finished = _run_claimed_job_with_mid_run_action(
        tmp_path, monkeypatch,
        lambda jobs_mod, job: jobs_mod.remove_job(job["id"]),
        execution_id="self-removal-crash-execution",
        crash=RuntimeError("boom after self-removal"), expect_result=False)

    marked.assert_not_called()
    finished.assert_called_once_with(
        "self-removal-crash-execution", success=False,
        error="boom after self-removal", delivery_outcome="delivered", delivery_error=None)


def test_self_removal_followed_by_replacement_record_stays_fail_closed(tmp_path, monkeypatch):
    """Self-removal only excuses a MISSING record: once another owner's record reclaims the id,
    the run is stale again and its result must be discarded, never delivered."""

    def _remove_then_replace(jobs_mod, job):
        assert jobs_mod.remove_job(job["id"])
        replacement = {k: v for k, v in job.items() if k != "execution_id"}
        replacement["fire_claim"] = {"at": job["fire_claim"]["at"], "by": "other-machine:owner"}
        jobs_mod.save_jobs(jobs_mod.load_jobs() + [replacement])

    delivered, finished = _run_claimed_job_with_mid_run_action(
        tmp_path, monkeypatch, _remove_then_replace, execution_id="replacement-execution")

    delivered.assert_not_called()
    finished.assert_called_once_with(
        "replacement-execution", success=False,
        error="Fire claim ownership lost; stale result was discarded.")


def test_initially_lost_fire_claim_finishes_execution_without_running(monkeypatch):
    """A stale claimed snapshot rejected before body entry must close its ledger row."""
    import cron.scheduler as scheduler
    from cron import scheduler_script as sched_script

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
    from cron import scheduler_script as sched_script

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
    from cron import scheduler_script as sched_script

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
    from cron import scheduler_script as sched_script

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
    """Store uncertainty cannot let a run outlive its last confirmed lease forever.

    The contract is elapsed-time based (grace since the last confirmed renewal), not a renewal
    count: on a slow host the first wake can land after the grace, so cancellation after a single
    failed renewal is correct (#111471). Assert the contract, never a minimum attempt count."""
    import cron.scheduler as scheduler
    from cron import scheduler_script as sched_script

    last_confirmed_at = []
    cancellation_after = []
    calls = 0

    def heartbeat(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            last_confirmed_at.append(time.monotonic())
            return True
        raise OSError("store unavailable")

    def run_body(_job, **kwargs):
        assert kwargs["claim_lost"].wait(timeout=0.5)
        cancellation_after.append(time.monotonic() - last_confirmed_at[0])
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
    assert calls >= 2, "cancellation must follow at least one failed renewal"
    assert cancellation_after[0] >= scheduler._FIRE_CLAIM_HEARTBEAT_GRACE_SECONDS


def test_terminal_owner_cas_failure_marks_ledger_ownership_lost(monkeypatch):
    """A replacement owner cannot leave the stale ledger recorded as success."""
    import cron.scheduler as scheduler
    from cron import scheduler_script as sched_script

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
        delivery_error=None,
    )
