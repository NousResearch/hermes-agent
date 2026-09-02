"""Regression for #86721 — a one-shot `hermes cron run` invocation's
dispatched runner thread dies with the exiting process, leaving a stale
'claimed'/'running' row in cron/executions.db that blocks every subsequent
manual run of the same job. recover_interrupted_executions() already
existed and is correctly implemented, but was only ever called at the
long-lived scheduler ticker's own startup -- a one-shot CLI invocation has
no equivalent moment of its own, so the self-heal never ran for it.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def test_stale_claim_from_a_dead_one_shot_process_blocks_new_execution_creation(
    tmp_path,
):
    """Sanity/negative-control: reproduces the exact reported symptom in
    isolation -- a genuinely dead-owner 'claimed' row for a job, with no
    recovery step run, simply sits there. (create_execution() doesn't
    itself gate on prior rows for the same job_id; the blocking happens
    at a higher layer that checks for an existing claimed/running row --
    this test establishes the stale row exists and stays 'claimed'
    without recovery, matching the bug report's own SQL evidence.)"""
    home = tmp_path / "home"
    repo = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env["HERMES_HOME"] = str(home)
    env["PYTHONPATH"] = str(repo)

    # Simulate the dispatched runner's owner process dying mid-flight,
    # exactly as issue #86721 describes: a one-shot process creates the
    # execution row (status='claimed') and exits before finishing it.
    create = subprocess.run(
        [
            sys.executable, "-c",
            "from cron.executions import create_execution; "
            "r=create_execution('cron-run-job', source='direct'); "
            "print(r['id'])",
        ],
        cwd=repo, env=env, text=True, capture_output=True, check=True,
    )
    execution_id = create.stdout.strip()

    # Without recovery, the row is exactly where the bug report found it.
    check = subprocess.run(
        [
            sys.executable, "-c",
            "import json; from cron.executions import list_executions; "
            "print(json.dumps(list_executions(job_id='cron-run-job')))",
        ],
        cwd=repo, env=env, text=True, capture_output=True, check=True,
    )
    records = json.loads(check.stdout.strip())
    assert len(records) == 1
    assert records[0]["id"] == execution_id
    assert records[0]["status"] == "claimed"  # stranded, matching the report


def test_recover_interrupted_executions_reaps_the_stale_claim_from_a_dead_process(
    tmp_path,
):
    """The self-heal this issue's fix now triggers per one-shot
    invocation: recover_interrupted_executions() correctly identifies and
    clears a stale claim left by a process that has genuinely exited
    (real dead PID, not a mock), unblocking the job for a new run."""
    home = tmp_path / "home"
    repo = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env["HERMES_HOME"] = str(home)
    env["PYTHONPATH"] = str(repo)

    create = subprocess.run(
        [
            sys.executable, "-c",
            "from cron.executions import create_execution; "
            "r=create_execution('cron-run-job-2', source='direct'); "
            "print(r['id'])",
        ],
        cwd=repo, env=env, text=True, capture_output=True, check=True,
    )
    execution_id = create.stdout.strip()

    # This is what #86721's fix now calls, from a FRESH process, mirroring
    # exactly what a subsequent `hermes cron run` invocation would trigger
    # before attempting its own claim.
    recover = subprocess.run(
        [
            sys.executable, "-c",
            "import json; "
            "from cron.executions import recover_interrupted_executions, list_executions; "
            "print(recover_interrupted_executions()); "
            "print(json.dumps(list_executions(job_id='cron-run-job-2')))",
        ],
        cwd=repo, env=env, text=True, capture_output=True, check=True,
    )
    lines = recover.stdout.strip().splitlines()
    assert lines[0] == "1", "exactly one stale execution must be reaped"
    records = json.loads(lines[1])
    assert records[0]["id"] == execution_id
    assert records[0]["status"] == "unknown", (
        "the stale claim from the dead owner must be reclassified, "
        "no longer blocking a fresh claim attempt for this job"
    )


def test_try_dispatch_background_run_calls_recovery_before_claiming(monkeypatch):
    """Direct unit check on the fix's exact insertion point: the one-shot
    dispatch path must call recover_interrupted_executions() before
    proceeding to claim/create an execution for THIS run, so a stale row
    left by a prior dead one-shot invocation is cleared first."""
    import tools.cronjob_tools as cronjob_tools

    calls = []
    monkeypatch.setattr(
        "cron.executions.recover_interrupted_executions",
        lambda: calls.append("recovered") or 0,
    )
    # Force the function past its own early-return guards so execution
    # reaches the point where recovery is called, without needing a real
    # gateway/session runtime.
    monkeypatch.setattr(
        "gateway.session_context.async_delivery_supported", lambda: True
    )
    monkeypatch.setattr(
        "tools.approval.get_current_session_key", lambda default="": ""
    )

    job = {"id": "unit-test-job", "name": "unit test job", "deliver": "local"}
    # session_id is intentionally empty/None too, matching the direct-
    # caller ("hermes cron run", tests) early return documented just
    # below the recovery call -- this test only needs to confirm recovery
    # ran before that point, not exercise the full dispatch.
    cronjob_tools._try_dispatch_background_run(job, session_id=None)

    assert calls == ["recovered"], (
        "recover_interrupted_executions() must be called before any claim "
        "attempt in the one-shot dispatch path"
    )


def test_hermes_cron_run_never_reaches_the_86721_recovery_call(monkeypatch):
    """Follow-up regression: a real `hermes cron run` invocation declares its
    session stateless (hermes_cli/cron.py::_job_action), so
    async_delivery_supported() is False and _try_dispatch_background_run()
    returns None before ever reaching the #86721 recovery call above --
    exactly the opposite of what test_try_dispatch_background_run_calls_
    recovery_before_claiming forces with its async_delivery_supported=True
    monkeypatch. This pins that the #86721 fix's own reachable-path claim
    (proven by the previous test) does NOT translate to real `hermes cron
    run` invocations, which is the gap the follow-up fix in _execute_job_now
    closes."""
    import tools.cronjob_tools as cronjob_tools

    calls = []
    monkeypatch.setattr(
        "cron.executions.recover_interrupted_executions",
        lambda: calls.append("recovered") or 0,
    )
    monkeypatch.setattr(
        "gateway.session_context.async_delivery_supported", lambda: False
    )

    job = {"id": "unit-test-job", "name": "unit test job", "deliver": "local"}
    result = cronjob_tools._try_dispatch_background_run(job, session_id=None)

    assert result is None, "background dispatch must decline for a stateless session"
    assert calls == [], (
        "the #86721 recovery call is unreachable on the real hermes cron run "
        "path -- confirms the caller must fall back to _execute_job_now(), "
        "which needs its own recovery call"
    )


def test_execute_job_now_calls_recovery_before_claiming(monkeypatch):
    """Follow-up fix: _execute_job_now() -- what a one-shot `hermes cron run`
    actually falls through to once _try_dispatch_background_run() declines
    (see test_hermes_cron_run_never_reaches_the_86721_recovery_call above)
    -- must reap stale execution rows itself before attempting its own
    claim, the same self-heal #86721 gave the background-dispatch path."""
    import tools.cronjob_tools as cronjob_tools

    calls = []
    monkeypatch.setattr(
        "cron.executions.recover_interrupted_executions",
        lambda: calls.append("recovered") or 0,
    )
    # Short-circuit right after the point under test so this stays a unit
    # check on ordering, not a full job execution.
    monkeypatch.setattr(
        cronjob_tools, "claim_job_for_fire", lambda job_id, return_job=True: False
    )
    monkeypatch.setattr(cronjob_tools, "get_job", lambda job_id: None)

    job = {"id": "unit-test-job-2", "name": "unit test job 2"}
    result = cronjob_tools._execute_job_now(job)

    assert calls == ["recovered"], (
        "recover_interrupted_executions() must be called before any claim "
        "attempt in _execute_job_now(), the path a stateless `hermes cron "
        "run` session actually reaches"
    )
    assert result["claimed"] is False


def test_execute_job_now_reaps_a_real_stale_claim_from_a_dead_process(tmp_path):
    """End-to-end version of the two unit tests above: a stale 'claimed' row
    left by a genuinely dead one-shot process is cleared by the time
    _execute_job_now() attempts its own claim -- the actual fix for #86721's
    reported symptom on the real `hermes cron run` code path (no gateway,
    async_delivery_supported() False)."""
    import os
    import subprocess
    import sys

    home = tmp_path / "home"
    repo = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env["HERMES_HOME"] = str(home)
    env["PYTHONPATH"] = str(repo)

    create = subprocess.run(
        [
            sys.executable, "-c",
            "from cron.executions import create_execution; "
            "r=create_execution('cron-run-job-3', source='direct'); "
            "print(r['id'])",
        ],
        cwd=repo, env=env, text=True, capture_output=True, check=True,
    )
    execution_id = create.stdout.strip()

    check = subprocess.run(
        [
            sys.executable, "-c",
            "import json; "
            "from tools.cronjob_tools import _execute_job_now; "
            "from cron.executions import list_executions; "
            "_execute_job_now({'id': 'no-such-job', 'name': 'no-such-job'}); "
            "print(json.dumps(list_executions(job_id='cron-run-job-3')))",
        ],
        cwd=repo, env=env, text=True, capture_output=True, check=True,
    )
    records = json.loads(check.stdout.strip())
    assert records[0]["id"] == execution_id
    assert records[0]["status"] == "unknown", (
        "a stale execution row from a dead one-shot process must be reaped "
        "by _execute_job_now() itself, not just the background-dispatch path"
    )
