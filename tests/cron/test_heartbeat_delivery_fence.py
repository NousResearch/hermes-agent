"""Renewal must not contend with delivery, without weakening owner fencing."""

import threading

import pytest


@pytest.mark.parametrize("claim_kind", ["fire", "run"])
def test_claim_renewal_during_side_effect_fence(tmp_path, monkeypatch, claim_kind):
    from cron import jobs

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(jobs, "_JOBS_LOCK_TIMEOUT_SECONDS", 0.05)
    job = jobs.create_job(prompt="x", schedule="in 30m")
    claimed = jobs.claim_job_for_fire(job["id"], return_job=True)
    owner = claimed["fire_claim"]["by"]
    field = f"{claim_kind}_claim"
    if claim_kind == "run":
        jobs.update_job(job["id"], {field: dict(claimed["fire_claim"])})
    before = jobs.get_job(job["id"])[field]
    heartbeat = getattr(jobs, f"heartbeat_{claim_kind}_claim")
    results = []
    errors = []
    done = threading.Event()

    def renew():
        try:
            results.append(heartbeat(job["id"], expected_owner=owner))
            # A competing dispatch and terminal write must still fail closed.
            results.append(jobs.claim_job_for_fire(job["id"], claim_ttl_seconds=0))
            results.append(jobs.mark_job_run(job["id"], True, expected_fire_owner=owner))
        except BaseException as exc:
            errors.append(exc)
        finally:
            done.set()

    with jobs.fire_claim_fence(job["id"], expected_owner=owner) as owned:
        assert owned
        thread = threading.Thread(target=renew)
        thread.start()
        try:
            assert done.wait(5), "renewal blocked behind delivery's fire fence"
        finally:
            thread.join(timeout=5)
    assert not thread.is_alive()
    assert not errors
    assert results == [True, False, False]
    refreshed = jobs.get_job(job["id"])[field]
    assert refreshed["by"] == before["by"]
    assert refreshed["at"] != before["at"]

    # Replaced or cleared claims must never be renewed by the stale owner.
    replacement = {"by": "replacement", "at": refreshed["at"]}
    jobs.update_job(job["id"], {field: replacement})
    assert heartbeat(job["id"], expected_owner=owner) is False
    assert jobs.get_job(job["id"])[field] == replacement
    jobs.update_job(job["id"], {field: None})
    assert heartbeat(job["id"], expected_owner=owner) is False
    assert jobs.get_job(job["id"])[field] is None


@pytest.mark.parametrize("replace_owner", [False, True])
def test_run_one_job_delivery_and_real_ownership_loss(tmp_path, monkeypatch, replace_owner):
    from cron import executions, jobs, scheduler

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(jobs, "_JOBS_LOCK_TIMEOUT_SECONDS", 0.05)
    monkeypatch.setattr(scheduler, "_RUN_CLAIM_HEARTBEAT_SECONDS", 0.01)
    job = jobs.create_job(prompt="x", schedule="every 5m", deliver="discord:123")
    claimed = jobs.claim_job_for_fire(job["id"], return_job=True)
    execution = executions.create_execution(job["id"], source="direct")
    claimed["execution_id"] = execution["id"]
    delivering = threading.Event()
    renewed = threading.Event()
    renewals = []
    delivered = []
    real_heartbeat = jobs.heartbeat_fire_claim

    def observe_heartbeat(job_id, *, expected_owner):
        result = real_heartbeat(job_id, expected_owner=expected_owner)
        if delivering.is_set():
            renewals.append(result)
            renewed.set()
        return result

    def run_job(_job, *, cancel_event, **kwargs):
        if replace_owner:
            assert jobs.claim_job_for_fire(job["id"], claim_ttl_seconds=0)
            assert cancel_event.wait(5), "real replacement did not cancel the stale runner"
        return True, "saved output", "final response", None

    def deliver(_job, content, **kwargs):
        delivered.append(content)
        delivering.set()
        try:
            # Delivery holds the real fire fence until the monitor has attempted renewal.
            assert renewed.wait(5), "heartbeat never completed during delivery"
        finally:
            delivering.clear()
        return None

    monkeypatch.setattr(scheduler, "heartbeat_fire_claim", observe_heartbeat)
    monkeypatch.setattr(scheduler, "run_job", run_job)
    monkeypatch.setattr(scheduler, "_deliver_result", deliver)
    assert scheduler.run_one_job(claimed) is True

    persisted = jobs.get_job(job["id"])
    ledger = executions.get_execution(execution["id"])
    if replace_owner:
        assert delivered == []
        assert ledger["status"] == "failed"
        assert "ownership lost" in ledger["error"].lower()
        assert persisted["fire_claim"]["by"] != claimed["fire_claim"]["by"]
        assert persisted.get("last_run_at") is None
    else:
        assert delivered == ["final response"]
        assert renewals and all(renewals)
        assert persisted["last_status"] == "ok"
        assert persisted["fire_claim"] is None
        assert ledger["status"] == "completed"
        assert ledger["error"] is None
        assert list((tmp_path / "cron" / "output" / job["id"]).glob("*.md"))
