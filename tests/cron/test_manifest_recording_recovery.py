"""A post-send ledger fault must never become transport failure or trigger resend."""
import pytest
from cron import executions, scheduler, delivery_queue, bot_chat_delivery, jobs
from tests.cron.test_mixed_delivery_retention import setup


@pytest.mark.parametrize("external", [False, True])
@pytest.mark.parametrize("persistent_fault", [False, True])
def test_manifest_fault_retains_child_and_real_send_outcome(setup, monkeypatch, external, persistent_fault):
    home, cfg, run, sent = setup
    original = executions.record_delivery_manifest
    failures = []
    def fail(eid, manifest):
        if "bot" in manifest:
            failures.append(eid)
            raise OSError("fixture ledger recorder unavailable")
        return original(eid, manifest)
    monkeypatch.setattr(executions, "record_delivery_manifest", fail)
    job, before = run(external)
    if external:
        scheduler.drain_delivery_queue({}, None)
        assert delivery_queue.get_status(before["id"])["status"] == "delivered"
    assert len(sent) == 1
    assert failures
    mid = executions.get_execution(before["id"])
    assert mid["delivery_outcome"] == "queued"
    assert jobs.get_job(job["id"])["last_delivery_queued"]
    if not persistent_fault:
        monkeypatch.setattr(executions, "record_delivery_manifest", original)
    cfg(True)
    bot_chat_delivery.drain()
    executions.reconcile_delivery_projections()
    after = executions.get_execution(before["id"])
    assert after["delivery_outcome"] == "delivered"
    assert after["error"] == before["error"]
    assert after["finished_at"] == before["finished_at"]
    assert not jobs.get_job(job["id"])["last_delivery_queued"]
    assert len(sent) == 1
    monkeypatch.setattr(executions, "record_delivery_manifest", original)
    executions.reconcile_delivery_projections()
    executions.reconcile_delivery_projections()
    assert len(sent) == 1
