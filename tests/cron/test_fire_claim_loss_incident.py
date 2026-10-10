"""A gateway restart must not release the fire claim of a live restart-safe worker (#136188).

``claim_job_for_fire`` stamps ``fire_claim.by`` with the dispatching gateway's pid, so once that
gateway exits, ``_claim_owner_is_dead`` declares the claim stale and the replacement gateway
re-fires the job while the restart-safe worker it spawned is still running — the worker's result
is then discarded and no incident is ever opened. The worker must move the claim to its own pid
at adoption, and every claim-loss ending must leave a ``cron_incidents`` row.

These exercise the real jobs/executions/incidents stores under a temp HERMES_HOME (no store
mocks) per the E2E-over-mocks discipline for file-touching code.
"""

from __future__ import annotations

import json
import os
import socket

import pytest


@pytest.fixture
def temp_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME so jobs.json/executions.db never touch the real stores."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    yield tmp_path


def _dead_gateway_pid() -> int:
    """A host pid that provably does not exist: the stand-in for the exited dispatching gateway.

    Walking up from this process's pid is deterministic on the sparse pid space a test host has,
    and the found pid feeds the same ``_pid_exists`` probe ``_claim_owner_is_dead`` uses."""
    from gateway.status import _pid_exists

    candidate = os.getpid() + 1
    while _pid_exists(candidate):
        candidate += 1
    return candidate


def _claim_held_by_dead_gateway(monkeypatch, job_id: str) -> dict:
    """Claim the job exactly as a dispatching gateway would, under a pid that has exited."""
    from cron.jobs import claim_job_for_fire

    with monkeypatch.context() as m:
        m.setattr(
            "cron.jobs._machine_id",
            lambda: f"{socket.gethostname()}:{_dead_gateway_pid()}",
        )
        claimed = claim_job_for_fire(job_id, return_job=True)
    assert isinstance(claimed, dict)
    return claimed


def test_gateway_restart_cannot_reclaim_live_worker_claim(
    temp_home, tmp_path, monkeypatch
):
    """The adopting worker re-stamps the claim under its own pid, so the replacement gateway's
    tick loses the claim race instead of re-firing the job under the live worker."""
    from cron import scheduler
    from cron.executions import create_execution, mark_execution_handoff_pending
    from cron.jobs import claim_job_for_fire, create_job, get_job

    job = create_job(prompt="x", schedule="every 5m", name="worker-adopt")
    claimed = _claim_held_by_dead_gateway(monkeypatch, job["id"])
    record = create_execution(job["id"], source="builtin")
    assert mark_execution_handoff_pending(record["id"]) is not None

    payload_job = dict(claimed)
    payload_job["execution_id"] = record["id"]
    payload = tmp_path / "payload.json"
    payload.write_text(
        json.dumps({
            "job": payload_job,
            "profile_home": str(temp_home),
            "multiplex_active": False,
        }),
        encoding="utf-8",
    )

    seen: dict = {}

    def probe_run_one_job(running_job, **_kwargs):
        # The replacement gateway's next tick tries to re-claim the just-dispatched job.
        seen["replacement_claimed"] = claim_job_for_fire(running_job["id"])
        seen["worker_owner"] = running_job["fire_claim"]["by"]
        store_claim = get_job(running_job["id"])["fire_claim"]
        seen["store_owner"] = store_claim["by"]
        seen["adopted_by"] = store_claim.get("adopted_by")
        return True

    monkeypatch.setattr(scheduler, "run_one_job", probe_run_one_job)

    ack = tmp_path / f"{record['id']}.ready"
    assert scheduler._run_external_worker_payload(payload, ack) is True

    assert seen["replacement_claimed"] is False
    assert seen["worker_owner"] == seen["store_owner"]
    # Liveness moved to the adopting worker while the CAS token (``by``) stayed stable.
    assert seen["adopted_by"]


def test_worker_refuses_execution_when_claim_was_already_retaken(
    temp_home, tmp_path, monkeypatch
):
    """A claim re-taken by the replacement gateway before adoption cannot be stolen back: the
    worker refuses to run instead of double-firing the job."""
    from cron import scheduler
    from cron.executions import (
        create_execution,
        latest_execution,
        mark_execution_handoff_pending,
    )
    from cron.jobs import create_job, get_job, load_jobs, save_jobs
    from hermes_time import now as hermes_now

    job = create_job(prompt="x", schedule="every 5m", name="retaken")
    # The store's claim was already re-stamped by the replacement gateway; the payload still
    # carries the dead gateway's owner string.
    jobs = load_jobs()
    for row in jobs:
        if row["id"] == job["id"]:
            row["fire_claim"] = {
                "at": hermes_now().isoformat(),
                "by": "other-gateway:1:tok",
            }
    save_jobs(jobs)

    record = create_execution(job["id"], source="builtin")
    assert mark_execution_handoff_pending(record["id"]) is not None
    payload_job = dict(get_job(job["id"]))
    payload_job["execution_id"] = record["id"]
    payload_job["fire_claim"] = {
        "at": hermes_now().isoformat(),
        "by": "dead-gateway:1:tok",
    }

    payload = tmp_path / "payload.json"
    payload.write_text(
        json.dumps({
            "job": payload_job,
            "profile_home": str(temp_home),
            "multiplex_active": False,
        }),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        scheduler, "run_one_job", lambda *_a, **_k: pytest.fail("must not run")
    )

    assert (
        scheduler._run_external_worker_payload(payload, tmp_path / "ack.ready") is False
    )

    finished = latest_execution(job["id"])
    assert finished["status"] == "failed"
    assert "ownership lost before the worker started" in finished["error"]


def test_claim_already_held_opens_incident(temp_home):
    """A due fire that loses the claim to a live holder (e.g. a manual run overlapping the tick)
    leaves an incident row, not just an executions ledger line."""
    from cron import scheduler
    from cron.executions import create_execution, latest_execution
    from cron.incidents import list_incidents
    from cron.jobs import claim_job_for_fire, create_job, get_job

    job = create_job(prompt="x", schedule="every 5m", name="held")
    assert claim_job_for_fire(job["id"]) is True  # the other run holds the claim
    record = create_execution(job["id"], source="builtin")
    due_job = dict(get_job(job["id"]))
    due_job["execution_id"] = record["id"]

    assert (
        scheduler._process_due_job(due_job, adapters=None, loop=None, verbose=False)
        is True
    )

    finished = latest_execution(job["id"])
    assert finished["status"] == "failed"
    assert "not started" in finished["error"]
    incidents = list_incidents()
    assert len(incidents) == 1
    assert incidents[0]["job_id"] == job["id"]


def test_stale_owner_before_execution_opens_incident(temp_home):
    """A run whose fire claim was already re-taken before the heartbeat started never runs and
    leaves an incident row."""
    from cron import scheduler
    from cron.executions import create_execution, latest_execution
    from cron.incidents import list_incidents
    from cron.jobs import create_job, get_job, load_jobs, save_jobs
    from hermes_time import now as hermes_now

    job = create_job(prompt="x", schedule="every 5m", name="stale-owner")
    jobs = load_jobs()
    for row in jobs:
        if row["id"] == job["id"]:
            row["fire_claim"] = {
                "at": hermes_now().isoformat(),
                "by": "other-gateway:1:tok",
            }
    save_jobs(jobs)

    record = create_execution(job["id"], source="builtin")
    job_dict = dict(get_job(job["id"]))
    job_dict["execution_id"] = record["id"]
    job_dict["fire_claim"] = {
        "at": hermes_now().isoformat(),
        "by": "dead-gateway:1:tok",
    }

    ran: list = []
    assert (
        scheduler._run_with_fire_claim_heartbeat(
            job_dict, lambda lost: ran.append(lost) or True
        )
        is True
    )
    assert ran == []

    finished = latest_execution(job["id"])
    assert finished["status"] == "failed"
    assert "ownership lost before execution started" in finished["error"]
    incidents = list_incidents()
    assert len(incidents) == 1
    assert incidents[0]["job_id"] == job["id"]


def test_discarded_stale_result_opens_incident(temp_home):
    """A finished run whose result was discarded because ownership was lost mid-run leaves an
    incident row for the lost work."""
    from cron import scheduler
    from cron.executions import create_execution, latest_execution
    from cron.incidents import list_incidents
    from cron.jobs import create_job, get_job

    job = create_job(prompt="x", schedule="every 5m", name="discarded")
    record = create_execution(job["id"], source="builtin")
    job_dict = dict(get_job(job["id"]))
    job_dict["execution_id"] = record["id"]

    scheduler._record_fire_ownership_lost(job_dict, None, record["id"])

    finished = latest_execution(job["id"])
    assert finished["status"] == "failed"
    assert "stale result was discarded" in finished["error"]
    incidents = list_incidents()
    assert len(incidents) == 1
    assert incidents[0]["job_id"] == job["id"]
