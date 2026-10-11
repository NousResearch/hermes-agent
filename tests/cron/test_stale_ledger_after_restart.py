"""A run whose owner died must not leave `jobs.json` permanently stale.

Regression for the silent stale-ledger class found 2026-08-08.

A ``no_agent`` script is spawned with ``start_new_session=True``, so it survives
as an orphan when the gateway is SIGKILLed mid-drain and it RUNS TO COMPLETION —
but the process that would have called ``mark_job_run`` is gone. Nothing ever
writes the outcome, so ``jobs.json`` keeps the timestamp from the PREVIOUS run.

``recover_interrupted_executions`` already repaired the ``executions`` ledger,
but ``jobs.json`` is what ``hermes cron`` and any status
dashboard actually READ — so two jobs that had run perfectly (leaving their artifacts
on disk) displayed a 36h-stale ``last_run_at`` indefinitely.
"""

from __future__ import annotations

import time


def _isolate(monkeypatch, tmp_path):
    """Point BOTH stores at the tmp dir: the ledger DB and the jobs file."""
    import cron.executions as executions

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(
        executions, "EXECUTIONS_FILE", tmp_path / "cron" / "executions.db"
    )
    (tmp_path / "cron").mkdir(parents=True, exist_ok=True)
    (tmp_path / "scripts").mkdir(parents=True, exist_ok=True)
    return executions


def _make_script_job(name="orphan-bridge"):
    from cron import jobs as J

    return J.create_job(
        prompt="", schedule="45 2 * * *", script="slow.sh",
        no_agent=True, name=name,
    )


def _abandon(executions, execution_id):
    """Simulate the owner process being SIGKILLed: no terminal state is written."""
    with executions._transaction() as conn:
        conn.execute(
            "UPDATE executions SET process_id='DEAD-GATEWAY', pid=999999 WHERE id=?",
            (execution_id,),
        )


def test_orphaned_run_advances_jobs_json(monkeypatch, tmp_path):
    """The core bug: a killed owner must not freeze last_run_at forever."""
    executions = _isolate(monkeypatch, tmp_path)
    from cron import jobs as J
    from cron.scheduler_provider import InProcessCronScheduler

    job_id = _make_script_job()["id"]
    J.mark_job_run(job_id, True)  # a healthy run recorded earlier
    stale = next(j for j in J.load_jobs() if j["id"] == job_id)["last_run_at"]

    time.sleep(1.05)  # ensure a strictly later timestamp is observable
    execution = executions.create_execution(job_id, source="builtin")
    executions.mark_execution_running(execution["id"])
    _abandon(executions, execution["id"])  # gateway dies here; script orphan-completes

    # Precondition: without recovery the row is indistinguishable from "never ran again".
    assert next(j for j in J.load_jobs() if j["id"] == job_id)["last_run_at"] == stale

    recovered = InProcessCronScheduler().recover_interrupted()

    assert recovered == 1
    job = next(j for j in J.load_jobs() if j["id"] == job_id)
    assert job["last_run_at"] > stale, "last_run_at must advance past the abandoned run"
    # Honest, not optimistic: the orphan's exit status is genuinely unknowable.
    assert "Interrupted" in (job["last_error"] or "")
    assert "outlives the gateway" in (job["last_error"] or "")


def test_job_that_recorded_its_own_outcome_is_not_clobbered(monkeypatch, tmp_path):
    """Negative control: the normal path must never be overwritten by recovery."""
    executions = _isolate(monkeypatch, tmp_path)
    from cron import jobs as J
    from cron.scheduler_provider import InProcessCronScheduler

    job_id = _make_script_job()["id"]
    execution = executions.create_execution(job_id, source="builtin")
    executions.mark_execution_running(execution["id"])
    _abandon(executions, execution["id"])
    # ...but the job DID manage to record its own success (e.g. a later run).
    J.mark_job_run(job_id, True)
    good = next(j for j in J.load_jobs() if j["id"] == job_id)

    InProcessCronScheduler().recover_interrupted()

    after = next(j for j in J.load_jobs() if j["id"] == job_id)
    assert after["last_run_at"] == good["last_run_at"]
    assert after["last_status"] == "ok", "a healthy job must never be marked error"


def test_live_execution_is_never_reconciled(monkeypatch, tmp_path):
    """Negative control: an IN-FLIGHT run owned by this process must be untouched."""
    executions = _isolate(monkeypatch, tmp_path)
    from cron import jobs as J
    from cron.scheduler_provider import InProcessCronScheduler

    job_id = _make_script_job()["id"]
    J.mark_job_run(job_id, True)
    before = next(j for j in J.load_jobs() if j["id"] == job_id)

    execution = executions.create_execution(job_id, source="builtin")
    executions.mark_execution_running(execution["id"])  # still owned by THIS process

    assert InProcessCronScheduler().recover_interrupted() == 0

    after = next(j for j in J.load_jobs() if j["id"] == job_id)
    assert after["last_run_at"] == before["last_run_at"]
    assert after["last_status"] == "ok"
