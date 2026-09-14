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
        "tools.approval_context.get_current_session_key", lambda default="": ""
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
