"""Durable evidence when a restart-safe cron worker dies AFTER adopting its attempt.

Regression for #128509. A ``cronjob(action='run')`` fired inside a gateway agent turn
returns ``executed=true / execution_mode=background``, then the attempt dies. The waiter
learns the adopted worker exited and calls ``recover_interrupted_executions()``, which only
knows "the owner is gone" and stamps the generic ``_OWNER_GONE_REASON`` -- "Scheduler
restarted after this execution's owner exited" -- a restart that never happened. The waiter
then returns success, so ``run_one_job`` short-circuits and never calls ``mark_job_run``:

* the job's ``fire_claim`` is left behind, so the next manual fire is refused with
  "Job is already being fired by the scheduler" until the 300s lease expires, and
* ``last_error`` stays empty, so the operator has no error and no cause anywhere.

The worker is not in the calling turn's process group (``start_new_session=True`` plus a
``systemd-run --user --scope`` cgroup), so turn teardown never signalled it: the child
crashed on its own. What the gateway still owes the operator is a truthful cause and a
released claim.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

# A real worker process: it adopts the durable row (the ownership rewrite that makes the
# reported status line reachable), publishes the acknowledgement, then dies on its own.
# The sleep keeps the parent's ack poll (every 50ms) deterministically ahead of the exit.
_WORKER_THAT_DIES_AFTER_ADOPTION = """
import json, os, sys, time
from pathlib import Path

argv = sys.argv[1:]
payload_path = Path(argv[argv.index("--external-worker-file") + 1])
ack_path = Path(argv[argv.index("--ack-file") + 1])

import cron.executions as executions

job = json.loads(payload_path.read_text(encoding="utf-8-sig"))["job"]
assert executions.adopt_claimed_execution(job["execution_id"]) is not None
ack_path.write_text(
    json.dumps({"pid": os.getpid(), "execution_id": job["execution_id"]}),
    encoding="utf-8",
)
time.sleep(1.5)
os._exit(9)
"""


@pytest.fixture
def profile_store(tmp_path, monkeypatch):
    """A real profile-scoped cron store, ledger and scheduler home, all on one temp path."""
    from cron.jobs import create_job, use_cron_store

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    import cron.executions as executions
    import cron.scheduler as scheduler

    monkeypatch.setattr(scheduler, "_hermes_home", home)
    monkeypatch.setattr(executions, "EXECUTIONS_FILE", home / "cron" / "executions.db")
    monkeypatch.setattr(scheduler, "load_config_readonly", lambda: {})
    return home


def test_worker_dying_after_adoption_reports_its_cause_and_releases_the_claim(
    tmp_path, monkeypatch, profile_store
):
    """The run is lost either way; the ledger must not blame a restart that never
    happened, the job must be claimable again, and the operator must get a cause."""
    import cron.executions as executions
    import cron.scheduler as scheduler
    from cron.jobs import claim_job_for_fire, create_job, get_job, use_cron_store
    from tools.process_registry import GatewayChildDispatch

    home = profile_store
    with use_cron_store(home):
        job = create_job(prompt="summarise the day", schedule="every 1h", name="daily brief")
        claimed = claim_job_for_fire(job["id"], manual=True, return_job=True)
    assert claimed, "the manual claim must succeed or this test proves nothing"

    worker = tmp_path / "worker_that_dies.py"
    worker.write_text(_WORKER_THAT_DIES_AFTER_ADOPTION, encoding="utf-8")

    def dispatch(command, *, unit_suffix, require_restart_safe_scope=False):
        # Keep the launcher's real payload/ack argv; swap only the child's body so the
        # spawn, the acknowledgement poll and the wait all run for real.
        flags = command[command.index("--external-worker-file"):]
        return GatewayChildDispatch(
            "degraded", [sys.executable, str(worker), *flags])

    monkeypatch.setattr(
        "tools.process_registry.restart_safe_gateway_child_argv", dispatch)

    assert scheduler.run_one_job(claimed, adapters=None) is True
    execution_id = claimed["execution_id"]

    with use_cron_store(home):
        record = get_job(job["id"])
        # The claim is what makes the next manual fire say "Job is already being fired";
        # only mark_job_run retires it, and the lost waiter used to never call it.
        assert record["fire_claim"] is None, "the fire claim outlived its lost execution"
        assert claim_job_for_fire(job["id"], manual=True) is True

        # A truthful cause: this waiter never lost its scheduler, it watched ITS worker
        # exit. The generic dead-owner sweep can only assert a restart.
        assert record["last_error"], "a lost run recorded no error at all"
        assert "Scheduler restarted" not in record["last_error"]
        assert "exit 9" in record["last_error"]

    # Still `unknown`, not a fabricated durable state: side effects really are unknown.
    row = executions.get_execution(execution_id)
    assert row["status"] == "unknown"
    assert "Scheduler restarted" not in (row["error"] or "")


def test_terminalizer_refuses_an_attempt_whose_worker_is_still_alive(
    profile_store,
):
    """The targeted terminalizer must not steal a LIVE worker's attempt: refusing a live
    owner is what keeps the fix from becoming a second way to lose side effects."""
    import cron.executions as executions

    script = (
        "import sys, time\n"
        "from pathlib import Path\n"
        "import cron.executions as executions\n"
        f"executions.EXECUTIONS_FILE = Path({str(executions.EXECUTIONS_FILE)!r})\n"
        "assert executions.adopt_claimed_execution(sys.argv[1]) is not None\n"
        "sys.stdout.write('adopted\\n')\n"
        "sys.stdout.flush()\n"
        "time.sleep(60)\n"
    )
    record = executions.create_execution("job-live", source="direct")
    assert executions.mark_execution_handoff_pending(record["id"]) is not None
    live_worker = subprocess.Popen(
        [sys.executable, "-c", script, record["id"]],
        stdout=subprocess.PIPE, text=True)
    try:
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline:
            if executions.get_execution(record["id"])["status"] == "running":
                break
            time.sleep(0.05)
        assert executions.get_execution(record["id"])["status"] == "running"

        assert executions.terminalize_dead_owner(
            record["id"], reason="exit 9") is False
        assert executions.get_execution(record["id"])["status"] == "running"
    finally:
        live_worker.kill()
        live_worker.wait(timeout=30)
