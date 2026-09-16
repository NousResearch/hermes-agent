"""A receipt belongs to one incident occurrence, not every reuse of its ID."""
import pytest
from cron import executions, incidents, delivery_queue
from tests.cron.test_deferred_warning_projection import _queued_execution


@pytest.mark.parametrize("finish_before_recovery", [False, True])
def test_old_receipt_cannot_alert_reopened_incident(tmp_path, monkeypatch, finish_before_recovery):
    job, execution, incident_id = _queued_execution(tmp_path, monkeypatch)
    if finish_before_recovery:
        delivery_queue.drain(lambda *a: None)
        assert incidents.get_incident(incident_id)["state"] == "alerted"
    assert incidents.close_incidents_for_recovered_job(job["id"]) == 1
    reopened, is_new = incidents.upsert_incident(job["id"], "raw failure")
    assert reopened == incident_id and is_new
    if not finish_before_recovery:
        delivery_queue.drain(lambda *a: None)
    executions.reconcile_delivery_projections()
    executions.reconcile_delivery_projections()
    assert executions.get_execution(execution["id"])["delivery_outcome"] == "delivered"
    assert incidents.get_incident(incident_id)["state"] == "detected"


def test_receipt_for_current_reopened_occurrence_can_alert(tmp_path, monkeypatch):
    job, old, incident_id = _queued_execution(tmp_path, monkeypatch)
    incidents.close_incidents_for_recovered_job(job["id"])
    incidents.upsert_incident(job["id"], "raw failure")
    current = executions.create_execution(job["id"], source="new failure")
    executions.bind_delivery_incident(current["id"], incident_id)
    executions.record_delivery_manifest(current["id"], {"external": True})
    delivery_queue.enqueue(current["id"], dict(job, execution_id=current["id"]), "diagnostic", for_failure=True)
    executions.finish_execution(current["id"], success=False, error="raw failure", delivery_outcome="queued")
    delivery_queue.drain(lambda *a: None)
    assert incidents.get_incident(incident_id)["state"] == "alerted"


def test_upgrade_does_not_invent_old_receipts_incident_generation(tmp_path, monkeypatch):
    import sqlite3
    job, execution, incident_id = _queued_execution(tmp_path, monkeypatch)
    before = executions.get_execution(execution["id"])
    # Exact pre-generation schema shape, with a pending receipt already persisted.
    with sqlite3.connect(tmp_path / "cron" / "executions.db") as conn:
        conn.execute("ALTER TABLE executions DROP COLUMN incident_generation")
        conn.execute("DROP TRIGGER IF EXISTS cron_incident_recurrence_generation")
        conn.execute("ALTER TABLE cron_incidents DROP COLUMN generation")
    delivery_queue.drain(lambda *a: None)
    after = executions.get_execution(execution["id"])
    assert after["delivery_outcome"] == "delivered"
    assert after["incident_generation"] is None
    assert after["error"] == before["error"]
    assert incidents.get_incident(incident_id)["state"] == "detected"
    assert incidents.get_incident(incident_id)["generation"] == 1


def test_concurrent_reconcilers_cannot_alert_reopened_occurrence(tmp_path, monkeypatch):
    import os
    import subprocess
    import sys
    job, execution, incident_id = _queued_execution(tmp_path, monkeypatch)
    delivery_queue.drain(lambda *a: None)
    incidents.close_incidents_for_recovered_job(job["id"])
    incidents.upsert_incident(job["id"], "raw failure")
    code = "from cron.executions import reconcile_delivery_projections; reconcile_delivery_projections()"
    children = [subprocess.Popen([sys.executable, "-c", code], env=os.environ.copy()) for _ in range(4)]
    for child in children:
        assert child.wait(timeout=30) == 0
    assert incidents.get_incident(incident_id)["state"] == "detected"
    assert executions.get_execution(execution["id"])["delivery_outcome"] == "delivered"
