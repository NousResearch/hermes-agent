"""Item 57: transient failures of the cron external-worker adopt handoff.

Field shape (executions.db, 2026-09-18/19): rows fail with
``Restart-safe cron worker dispatch failed: cron external worker exited
before ownership acknowledgement (exit 1|0)`` clustered at the same second
across multiple jobs (dispatch bursts + CI/campaign load contending the
executions ledger's 5s busy timeout).

Two defects, both in the same handoff window:

1. *Unlogged worker death.* In the worker process
   (``_run_external_worker_payload``), ``adopt_claimed_execution`` is an
   sqlite UPDATE under a 5s busy timeout. A transient
   ``sqlite3.OperationalError`` (or any exception from the adopt/ack seam)
   escaped the worker entirely: stdout/stderr are DEVNULL, so the traceback
   vanished and the process exited 1 with zero durable evidence. The
   gateway-side catch books "exited before ownership acknowledgement", which
   names the symptom, never the cause.

2. *Lost tick.* The gateway-side catch terminalizes the execution as failed
   and returns; the tick is simply gone. The job's next occurrence fires at
   the NEXT schedule slot — for a 10-minute monitor that is a 10-minute
   blind window, and the failures cluster exactly when the box is busiest.

Fixes under test here:

* worker side — bounded same-tick retry (2 extra attempts, 0.5s/1.0s
  backoff) around the adopt for transient ``sqlite3.OperationalError``;
  unexpected exceptions from the adopt seam are logged and the worker exits
  1 WITH a durable log line instead of dying silently.
* gateway side — when a worker exits before acking, the ledger decides:
  a terminal row is trusted (the exit-0 ack-race family: never book a
  failure over the worker's recorded outcome), a provably never-adopted
  claimed row (no started_at) gets ONE same-tick re-dispatch on the SAME
  execution row (attempt 1 never wrote anything, so no side-effect risk),
  and only after the retry budget is the handoff declared failed.
"""

from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
import unittest.mock as mock
from pathlib import Path

import pytest


@pytest.fixture
def execution_ledger(tmp_path, monkeypatch):
    import cron.executions as executions

    monkeypatch.setattr(executions, "EXECUTIONS_FILE", tmp_path / "executions.db")
    return executions


def _fresh_record(execution_ledger):
    record = execution_ledger.create_execution("job-1", source="builtin")
    assert (
        execution_ledger.mark_execution_handoff_pending(record["id"]) is not None
    )
    return record


def _write_payload(payload: Path, execution_id: str) -> None:
    payload.write_text(
        json.dumps(
            {
                "job": {"id": "job-1", "execution_id": execution_id},
                "profile_home": str(payload.parent / "profile"),
            }
        ),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Worker side: the adopt seam
# ---------------------------------------------------------------------------


def test_adopt_retry_absorbs_transient_busy(execution_ledger, tmp_path, monkeypatch):
    """First adopt raises OperationalError (busy ledger), retry adopts.

    OLD behavior: the exception escaped _run_external_worker_payload, the
    worker died with no log line, and the tick was lost.
    """
    import cron.scheduler as scheduler

    payload = tmp_path / "payload.json"
    ack = tmp_path / "ready.json"
    record = _fresh_record(execution_ledger)
    _write_payload(payload, record["id"])
    monkeypatch.setattr(execution_ledger, "_process_start_time", lambda _pid: 9876)

    real_adopt = execution_ledger.adopt_claimed_execution
    calls = []

    def contended_adopt(execution_id):
        calls.append(execution_id)
        if len(calls) == 1:
            raise sqlite3.OperationalError("database is locked")
        return real_adopt(execution_id)

    monkeypatch.setattr("cron.executions.adopt_claimed_execution", contended_adopt)
    run = mock.Mock(return_value=True)
    monkeypatch.setattr(scheduler, "run_one_job", run)

    assert scheduler._run_external_worker_payload(payload, ack) is True

    assert len(calls) == 2
    run.assert_called_once()
    assert ack.exists()
    row = execution_ledger.get_execution(record["id"])
    assert row["status"] == "running"
    assert row["started_at"] is not None


def test_adopt_retry_gives_up_after_budget_and_logs(
    execution_ledger, tmp_path, monkeypatch, caplog
):
    """Persistent OperationalError: worker logs + exits 1, never a silent crash."""
    import logging

    import cron.scheduler as scheduler

    payload = tmp_path / "payload.json"
    ack = tmp_path / "ready.json"
    record = _fresh_record(execution_ledger)
    _write_payload(payload, record["id"])

    def always_busy(_execution_id):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr("cron.executions.adopt_claimed_execution", always_busy)
    run = mock.Mock()
    monkeypatch.setattr(scheduler, "run_one_job", run)

    with caplog.at_level(logging.ERROR, logger="cron.scheduler"):
        result = scheduler._run_external_worker_payload(payload, ack)

    assert result is False
    run.assert_not_called()
    assert not ack.exists()
    messages = [rec.getMessage() for rec in caplog.records]
    assert any("could not adopt execution" in m for m in messages), messages
    row = execution_ledger.get_execution(record["id"])
    assert row["status"] == "claimed"
    assert row["started_at"] is None


def test_adopt_refusal_is_immediate_not_retried(
    execution_ledger, tmp_path, monkeypatch
):
    """A None refusal (lost adoption race) must not consume retry budget."""
    import cron.scheduler as scheduler

    payload = tmp_path / "payload.json"
    ack = tmp_path / "ready.json"
    record = _fresh_record(execution_ledger)
    _write_payload(payload, record["id"])

    calls = []

    def refuses(execution_id):
        calls.append(execution_id)
        return None

    monkeypatch.setattr("cron.executions.adopt_claimed_execution", refuses)
    run = mock.Mock()
    monkeypatch.setattr(scheduler, "run_one_job", run)

    assert scheduler._run_external_worker_payload(payload, ack) is False

    assert len(calls) == 1
    run.assert_not_called()
    assert not ack.exists()


def test_real_worker_process_logs_and_exits_cleanly_on_busy_adopt(
    tmp_path, monkeypatch
):
    """End-to-end through the REAL worker entrypoint with a locked ledger.

    Boots ``python -m cron.scheduler --external-worker-file ...`` against a
    real executions.db whose write lock is held by another connection past
    every retry. The worker must exit 1 WITH the adopt failure logged under
    the temp home (durable evidence), never die on an unlogged traceback.
    This is the item-57 field shape. (profile_home == the ledger home, as in
    production: the payload's profile_home IS the home that owns the row.)
    """
    import os

    import cron.executions as executions

    # Same path the worker resolves: <home>/cron/executions.db (no
    # EXECUTIONS_FILE override in production; the worker lands here via
    # use_cron_store(profile_home) + get_hermes_home()).
    ledger = tmp_path / "cron" / "executions.db"
    monkeypatch.setattr(executions, "EXECUTIONS_FILE", ledger)
    record = executions.create_execution("job-1", source="builtin")
    assert (
        executions.mark_execution_handoff_pending(record["id"]) is not None
    )

    payload = tmp_path / "payload.json"
    payload.write_text(
        json.dumps(
            {
                "job": {"id": "job-1", "execution_id": record["id"]},
                "profile_home": str(tmp_path),
            }
        ),
        encoding="utf-8",
    )
    ack = tmp_path / "ready.json"

    # Hold an exclusive write transaction on the ledger, past every retry
    # (0.5s + 1.0s backoff + the 5s busy timeout per fresh connection).
    blocker = sqlite3.connect(ledger, timeout=0.1)
    blocker.execute("BEGIN IMMEDIATE")
    blocker.execute(
        "INSERT INTO executions (id, job_id, source, process_id, pid, status,"
        " claimed_at) VALUES ('blocker', 'j', 'builtin', 'x', 1, 'claimed', 't')"
    )

    try:
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "cron.scheduler",
                "--external-worker-file",
                str(payload),
                "--ack-file",
                str(ack),
            ],
            cwd=str(Path(__file__).resolve().parents[2]),
            env={
                **os.environ,
                "HERMES_HOME": str(tmp_path),
            },
            capture_output=True,
            text=True,
            timeout=300,
        )
    finally:
        blocker.rollback()
        blocker.close()

    assert proc.returncode == 1
    assert not ack.exists()
    row = executions.get_execution(record["id"])
    assert row["status"] == "claimed"
    assert row["started_at"] is None
    # Durable evidence: the worker's own log line landed under the temp home.
    log_text = ""
    for candidate in (
        tmp_path / "logs" / "agent.log",
        tmp_path / "logs" / "errors.log",
    ):
        if candidate.exists():
            log_text += candidate.read_text(encoding="utf-8", errors="replace")
    assert "could not adopt execution" in log_text, log_text[-2000:]


# ---------------------------------------------------------------------------
# Gateway side: the lost tick
# ---------------------------------------------------------------------------


class _FakeWorkerProcess:
    def __init__(self, returncode=None):
        self.returncode = returncode

    def poll(self):
        return self.returncode

    def wait(self, timeout=1.0):
        if self.returncode is None:
            raise subprocess.TimeoutExpired(cmd="worker", timeout=timeout)
        return self.returncode


def _install_worker_scenarios(scheduler, monkeypatch, scenarios):
    """Spawn one FakeWorkerProcess per scenario.

    Each scenario: ``{"exit": N}`` (dies before ack) or ``{"ack": True}``
    (acks and stays alive until the ledger says terminal).
    Returns (spawned, statuses) where statuses drives get_execution.
    """
    spawned = []
    pending = list(scenarios)

    def popen(command, **_kwargs):
        scenario = pending.pop(0) if pending else {"exit": 1}
        if scenario.get("ack"):
            ack_index = command.index("--ack-file") + 1
            Path(command[ack_index]).write_text(
                json.dumps({"pid": 4321, "execution_id": "exec-1"}),
                encoding="utf-8",
            )
            proc = _FakeWorkerProcess(returncode=None)
        else:
            proc = _FakeWorkerProcess(returncode=scenario["exit"])
        spawned.append(proc)
        return proc

    monkeypatch.setattr(scheduler.subprocess, "Popen", popen)

    statuses = []

    def get_execution(_execution_id):
        # Consumed from the FRONT: each append stages the next ledger read
        # in call order (disposition check, then waiter polls).
        return statuses.pop(0) if statuses else None

    monkeypatch.setattr(scheduler, "get_execution", get_execution)
    return spawned, statuses


def _gateway_job(tmp_path, monkeypatch, scheduler):
    monkeypatch.setattr(scheduler, "_get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr(scheduler, "HANDOFF_ADOPTION_GRACE_SECONDS", 0.05)

    def fake_create(job_id, source="builtin", scheduled_instant=None):
        return {
            "id": "exec-1",
            "job_id": job_id,
            "status": "claimed",
            "handoff_pending": 0,
            "started_at": None,
        }

    monkeypatch.setattr(scheduler, "create_execution", fake_create)
    monkeypatch.setattr(
        scheduler,
        "mark_execution_handoff_pending",
        lambda execution_id: {"id": execution_id, "handoff_pending": 1},
    )
    monkeypatch.setattr(
        "tools.process_registry.restart_safe_gateway_child_argv",
        lambda command, **_kw: type(
            "D", (), {"mode": "scoped", "argv": list(command)}
        )(),
    )
    return {"id": "job-1", "execution_id": "exec-1", "prompt": "work"}


def test_gateway_retries_once_when_worker_never_adopted(tmp_path, monkeypatch):
    """Exit-1-before-ack with the row provably never adopted → same-tick retry.

    OLD behavior: the RuntimeError escaped _launch_external_cron_worker,
    run_one_job caught it, terminalized the attempt failed, and the tick was
    lost (job blind until its next schedule slot).
    """
    import cron.scheduler as scheduler

    job = _gateway_job(tmp_path, monkeypatch, scheduler)
    spawned, statuses = _install_worker_scenarios(
        scheduler, monkeypatch, [{"exit": 1}, {"ack": True}]
    )
    # Ledger reads in call order: disposition check on attempt 1 (claimed →
    # never adopted → retry), then the attempt-2 waiter (running → completed).
    statuses.append({"id": "exec-1", "status": "claimed", "started_at": None})
    statuses.append({"id": "exec-1", "status": "running", "started_at": "t"})
    statuses.append({"id": "exec-1", "status": "completed", "started_at": "t"})

    assert scheduler._launch_external_cron_worker(job) is True
    assert len(spawned) == 2, "expected one same-tick re-dispatch"
    assert spawned[0].returncode == 1


def test_gateway_trusts_terminal_outcome_over_waiter_race(tmp_path, monkeypatch):
    """Worker completed its row but the ack race lost → trust the outcome.

    Field shape (exit 0 rows): the payload ran to completion and the worker
    returned True, but the gateway's grace loop was thread-starved and saw
    exit before reading the ack. OLD code books a second failure over the
    worker's honest success; NEW code reads the ledger first.
    """
    import cron.scheduler as scheduler

    job = _gateway_job(tmp_path, monkeypatch, scheduler)
    spawned, statuses = _install_worker_scenarios(
        scheduler, monkeypatch, [{"exit": 0}]
    )
    # Disposition check reads the worker's terminal row: trust it, no retry.
    statuses.append(
        {"id": "exec-1", "status": "completed", "started_at": "t"}
    )

    assert scheduler._launch_external_cron_worker(job) is True
    # Trust the terminal row: exactly ONE dispatch, no retry.
    assert len(spawned) == 1


def test_gateway_retry_is_bounded(tmp_path, monkeypatch):
    """Both attempts die pre-adoption → the handoff error stands, no loop."""
    import cron.scheduler as scheduler

    job = _gateway_job(tmp_path, monkeypatch, scheduler)
    spawned, statuses = _install_worker_scenarios(
        scheduler, monkeypatch, [{"exit": 1}, {"exit": 1}]
    )
    # Both disposition checks see the row still claimed (never adopted).
    statuses.append({"id": "exec-1", "status": "claimed", "started_at": None})
    statuses.append({"id": "exec-1", "status": "claimed", "started_at": None})

    with pytest.raises(RuntimeError, match="exited before ownership"):
        scheduler._launch_external_cron_worker(job)
    assert len(spawned) == 2


def test_gateway_does_not_retry_when_worker_adopted_then_died(tmp_path, monkeypatch):
    """Row shows adoption (running, started_at) → side effects possible → no retry."""
    import cron.scheduler as scheduler

    job = _gateway_job(tmp_path, monkeypatch, scheduler)
    spawned, statuses = _install_worker_scenarios(
        scheduler, monkeypatch, [{"exit": 1}]
    )
    # Disposition check: running with started_at → adoption advanced → uncertain.
    statuses.append({"id": "exec-1", "status": "running", "started_at": "t"})

    with pytest.raises(RuntimeError, match="exited before ownership"):
        scheduler._launch_external_cron_worker(job)
    assert len(spawned) == 1
