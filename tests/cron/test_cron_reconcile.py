"""Reconciling a shutdown-interrupted cron run against evidence.

A shutdown kills a run mid-flight, so the ledger can honestly record only that the attempt is
``unknown``: its owner is gone and nothing durable says how it ended. Honest is not final, though,
because the outcome is often knowable from outside (the detached pipeline's own report, an
idempotency check against the system it wrote to). Reconcile closes that row ONCE, against named
evidence, and repairs the job record that was left reading ``interrupted``.

Contract asserted here, through the real ledger and job store:
  * only an ``unknown`` attempt can be closed, and the evidence is recorded with it, not asserted;
  * a refusal writes nothing at all;
  * a job record is repaired only when the reconciled attempt is its most recent execution;
  * a reconcile is not a run: it moves neither the schedule nor the fire claim;
  * a failed reconcile keeps the interruption's reason and the streak it left alone;
  * a closed row settles the occurrence it was claimed for, which is what the missed-occurrence
    scan reads.
"""

import hashlib
import itertools
from datetime import datetime, timedelta, timezone

import pytest

SHUTDOWN_REASON = (
    "Gateway shutdown (post-interrupt) killed the job's tool subprocess "
    "before the run finished."
)

EVIDENCE = "pipeline report: finished 14:22, exit 0\n"


def _profile(tmp_path):
    home = tmp_path / "profile"
    home.mkdir()
    return home


def _ledger(monkeypatch, tmp_path, home):
    import cron.executions as executions

    monkeypatch.setattr(executions, "EXECUTIONS_FILE", home / "cron" / "executions.db")
    return executions


def _evidence_file(tmp_path, text=EVIDENCE):
    path = tmp_path / "pipeline-report.txt"
    path.write_text(text, encoding="utf-8")
    return path


def _abandoned(executions, monkeypatch, job_id, *, generation):
    """One attempt whose owning process is provably gone: claim it, then recover it ``unknown``."""
    record = executions.create_execution(str(job_id), source="builtin")
    monkeypatch.setattr(executions, "_PROCESS_ID", f"replacement-gateway-{generation}")
    monkeypatch.setattr(executions, "_owner_is_live", lambda _pid, _started: False)
    assert executions.recover_interrupted_executions() == 1
    return executions.get_execution(record["id"])


def _frozen_clock(executions, monkeypatch):
    """Strictly increasing time, so "that job's most recent execution" cannot be a coin flip."""
    base = datetime(2026, 9, 15, 4, 0, 0, tzinfo=timezone.utc)
    ticks = itertools.count()
    monkeypatch.setattr(executions, "_hermes_now", lambda: base + timedelta(seconds=next(ticks)))


def _interrupted_job(jobs, *, streak=0):
    """A job the shutdown left mid-flight: ``last_status = interrupted``, claims already cleared."""
    created = jobs.create_job(prompt="report", schedule="every 1h", name="report")
    job_id = created["id"]
    if streak:
        jobs.update_job(job_id, {"failure_streak": streak})
    claimed = jobs.claim_job_for_fire(job_id, force=True, return_job=True)
    assert jobs.mark_job_run(
        job_id, False, SHUTDOWN_REASON, status="interrupted",
        expected_fire_owner=claimed["fire_claim"]["by"])
    return job_id


def test_reconcile_closes_an_unknown_attempt_with_its_evidence(monkeypatch, tmp_path):
    import cron.jobs as jobs

    home = _profile(tmp_path)
    with jobs.use_cron_store(home):
        executions = _ledger(monkeypatch, tmp_path, home)
        job_id = jobs.create_job(prompt="report", schedule="every 1h", name="report")["id"]
        abandoned = _abandoned(executions, monkeypatch, job_id, generation=1)
        assert abandoned["status"] == "unknown"

        evidence = _evidence_file(tmp_path)
        reconciled = executions.reconcile_execution(
            abandoned["id"], status="completed", evidence=str(evidence),
            note="pipeline exited 0")

        # The row is no longer `unknown`, so its evidence is not silently replaceable.
        assert executions.reconcile_execution(
            abandoned["id"], status="failed", evidence=str(evidence)) is None

    assert reconciled["status"] == "completed"
    assert reconciled["reconciled_evidence"] == str(evidence)
    assert reconciled["reconciled_evidence_sha256"] == hashlib.sha256(
        EVIDENCE.encode("utf-8")).hexdigest()
    assert reconciled["reconciled_note"] == "pipeline exited 0"
    assert reconciled["reconciled_at"]
    assert reconciled["reconciled_by"]  # provenance names the caller, even if it is "custom"
    assert reconciled["error"] is None
    assert executions.get_execution(abandoned["id"])["status"] == "completed"


def test_reconcile_refuses_every_state_but_unknown(monkeypatch, tmp_path):
    """`claimed`/`running` have a live owner; `completed`/`failed` are the ledger's own record."""
    import cron.jobs as jobs

    home = _profile(tmp_path)
    with jobs.use_cron_store(home):
        executions = _ledger(monkeypatch, tmp_path, home)
        evidence = str(_evidence_file(tmp_path))

        for state in ("claimed", "running", "completed", "failed"):
            record = executions.create_execution(f"job-{state}", source="builtin")
            if state in ("running", "completed", "failed"):
                assert executions.mark_execution_running(record["id"]) is not None
            if state in ("completed", "failed"):
                assert executions.finish_execution(
                    record["id"], success=state == "completed") is not None
            before = executions.get_execution(record["id"])

            assert executions.reconcile_execution(
                record["id"], status="completed", evidence=evidence) is None
            # Refused means untouched: not one column moved, provenance included.
            assert executions.get_execution(record["id"]) == before


def test_reconcile_needs_readable_evidence_for_a_real_outcome(monkeypatch, tmp_path):
    """Without evidence the verb would be substituting an assertion; it refuses instead."""
    import cron.jobs as jobs

    home = _profile(tmp_path)
    with jobs.use_cron_store(home):
        executions = _ledger(monkeypatch, tmp_path, home)
        abandoned = _abandoned(executions, monkeypatch, "job-1", generation=1)
        evidence = str(_evidence_file(tmp_path))

        for unusable in (str(tmp_path / "gone.txt"), str(tmp_path)):
            with pytest.raises(ValueError):
                executions.reconcile_execution(
                    abandoned["id"], status="completed", evidence=unusable)
        # The ledger records outcomes, not causes: "interrupted" is not one of them.
        with pytest.raises(ValueError):
            executions.reconcile_execution(
                abandoned["id"], status="interrupted", evidence=evidence)

        assert executions.get_execution(abandoned["id"]) == abandoned


def test_job_record_is_repaired_only_for_its_most_recent_interrupted_run(monkeypatch, tmp_path):
    import cron.jobs as jobs

    home = _profile(tmp_path)
    with jobs.use_cron_store(home):
        executions = _ledger(monkeypatch, tmp_path, home)
        _frozen_clock(executions, monkeypatch)
        job_id = _interrupted_job(jobs)
        older = _abandoned(executions, monkeypatch, job_id, generation=1)
        newer = _abandoned(executions, monkeypatch, job_id, generation=2)

        after_interruption = jobs.get_job(job_id)
        before = (after_interruption["next_run_at"], after_interruption["fire_claim"],
                  after_interruption["repeat"])

        # An older attempt says nothing about the state the job is in now.
        assert not jobs.reconcile_job_record(job_id, execution_id=older["id"], success=True)
        assert jobs.get_job(job_id)["last_status"] == "interrupted"

        assert jobs.reconcile_job_record(
            job_id, execution_id=newer["id"], success=True, note="pipeline exited 0")
        repaired = jobs.get_job(job_id)

    assert repaired["last_status"] == "ok"
    assert repaired["last_error"] is None
    assert repaired["failure_streak"] == 0
    # A reconcile is not a run: the schedule and the at-most-once claims are exactly where they were.
    assert (repaired["next_run_at"], repaired["fire_claim"], repaired["repeat"]) == before


def test_failed_reconcile_keeps_the_interruption_reason_and_the_streak(monkeypatch, tmp_path):
    import cron.jobs as jobs

    home = _profile(tmp_path)
    with jobs.use_cron_store(home):
        executions = _ledger(monkeypatch, tmp_path, home)
        job_id = _interrupted_job(jobs, streak=2)
        abandoned = _abandoned(executions, monkeypatch, job_id, generation=1)
        evidence = _evidence_file(tmp_path, "pipeline report: exit 3, destination rejected it\n")

        reconciled = executions.reconcile_execution(
            abandoned["id"], status="failed", evidence=str(evidence),
            note="destination rejected it")
        assert jobs.reconcile_job_record(
            job_id, execution_id=abandoned["id"], success=False, note="destination rejected it")
        repaired = jobs.get_job(job_id)

    assert reconciled["error"] == "destination rejected it"
    assert repaired["last_status"] == "error"
    assert repaired["last_error"] == SHUTDOWN_REASON
    # The interruption was deliberately streak-neutral; neither is its repair a failure to count.
    assert repaired["failure_streak"] == 2


def test_cron_reconcile_cli_closes_the_row_and_shows_the_provenance(
    monkeypatch, tmp_path, capsys
):
    """The operator surface, end to end, including the digest a human compares by eye."""
    import argparse

    import cron.jobs as jobs
    from hermes_cli.cron import _CRON_SUBCOMMANDS, cron_runs

    home = _profile(tmp_path)
    with jobs.use_cron_store(home):
        executions = _ledger(monkeypatch, tmp_path, home)
        abandoned = _abandoned(executions, monkeypatch, "job-1", generation=1)
        evidence = _evidence_file(tmp_path)

        exit_code = _CRON_SUBCOMMANDS["reconcile"](argparse.Namespace(
            execution=abandoned["id"], status="completed", evidence=str(evidence),
            note="pipeline exited 0"))
        output = capsys.readouterr().out
        closed = executions.get_execution(abandoned["id"])
        cron_runs()
        listing = capsys.readouterr().out

    assert exit_code == 0
    assert closed["status"] == "completed"
    assert abandoned["id"] in output
    assert hashlib.sha256(EVIDENCE.encode("utf-8")).hexdigest() in output
    # The row stays readable afterwards: history shows it was closed from evidence, not by luck.
    assert "reconciled" in listing


def test_a_closed_row_also_settles_the_occurrence_it_was_claimed_for(monkeypatch, tmp_path):
    """The missed-occurrence scan reads `completed` rows: closing one settles that slot too."""
    import cron.jobs as jobs
    from cron.occurrences import completed_occurrence, scheduled_instant

    home = _profile(tmp_path)
    instant = scheduled_instant("2026-09-15T04:00:00+00:00")
    with jobs.use_cron_store(home):
        executions = _ledger(monkeypatch, tmp_path, home)
        job_id = jobs.create_job(prompt="report", schedule="every 1h", name="report")["id"]
        record = executions.create_execution(
            job_id, source="builtin", scheduled_instant=instant)
        monkeypatch.setattr(executions, "_PROCESS_ID", "replacement-gateway-1")
        monkeypatch.setattr(executions, "_owner_is_live", lambda _pid, _started: False)
        assert executions.recover_interrupted_executions() == 1
        abandoned = executions.get_execution(record["id"])
        assert abandoned is not None
        assert abandoned["scheduled_instant"] == instant

        # `unknown` proves nothing, so the slot stays eligible for a restore...
        assert not completed_occurrence(jobs.get_job(job_id), instant)

        # ...and the outcome established from evidence is what settles it.
        assert executions.reconcile_execution(
            record["id"], status="completed",
            evidence=str(_evidence_file(tmp_path))) is not None
        assert completed_occurrence(jobs.get_job(job_id), instant)
