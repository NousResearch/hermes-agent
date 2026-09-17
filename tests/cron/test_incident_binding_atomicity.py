"""An execution owns the occurrence created in its incident transaction."""
import os
import subprocess
import sys

import pytest

from cron import executions, incidents, scheduler
from tests.cron.test_mixed_delivery_retention import setup


def test_reopen_after_upsert_return_cannot_rebind_old_execution(setup, monkeypatch):
    home, cfg, run, sent = setup
    original = incidents.upsert_incident
    def interleave(*args, **kwargs):
        result = original(*args, **kwargs)
        iid = result[0]
        script = home / "reopen.py"
        script.write_text('from cron import incidents\nimport sys\ni=incidents.get_incident(sys.argv[1])\nassert incidents.close_incidents_for_recovered_job(i["job_id"])==1\nincidents.upsert_incident(i["job_id"],i["error"])\n')
        subprocess.run([sys.executable, str(script), iid], env={**os.environ, "PYTHONPATH": os.getcwd()}, check=True, timeout=20)
        return result
    monkeypatch.setattr(incidents, "upsert_incident", interleave)
    job, before = run(True, deliver="telegram:test")
    assert before["incident_generation"] == 1
    assert scheduler.drain_delivery_queue({}, None) == 1
    incident = incidents.get_incident(before["incident_id"])
    assert incident["generation"] == 2
    assert incident["state"] == "detected"
    assert len(sent) == 1


def test_pre_generation_writer_cannot_reuse_occurrence(setup):
    home, cfg, run, sent = setup
    job, before = run(True, deliver="telegram:test")
    assert scheduler.drain_delivery_queue({}, None) == 1
    iid = before["incident_id"]
    assert incidents.get_incident(iid)["state"] == "alerted"
    incidents.close_incidents_for_recovered_job(job["id"])
    # Exact old writer, not a replacement approximation of its SQL.
    source = subprocess.check_output(['git', 'show', 'd738176e09d65a58252d5c4f2110e95f43570602:cron/incidents.py'], text=True)
    (home / "old_incidents.py").write_text(source)
    script = home / "old_writer.py"
    script.write_text('import old_incidents as old\nimport sys\ni=old.get_incident(sys.argv[1])\nassert old.upsert_incident(i["job_id"],i["error"])[1]\n')
    subprocess.run([sys.executable, str(script), iid], env={**os.environ, "PYTHONPATH": os.getcwd()}, check=True, timeout=20)
    executions.reconcile_delivery_projections()
    assert incidents.get_incident(iid)["generation"] == 2
    assert incidents.get_incident(iid)["state"] == "detected"


def test_incident_and_execution_bind_roll_back_together(setup, monkeypatch):
    home, cfg, run, sent = setup
    execution = executions.create_execution("bind-rollback", source="fixture")
    def fail(*args):
        raise OSError("fixture bind failure")
    with monkeypatch.context() as scoped:
        scoped.setattr(executions, "_bind_delivery_incident_unlocked", fail)
        with pytest.raises(OSError, match="fixture bind failure"):
            incidents.upsert_incident("bind-rollback", "fixture", execution_id=execution["id"])
    assert not [i for i in incidents.list_incidents() if i["job_id"] == "bind-rollback"]
    assert executions.get_execution(execution["id"])["incident_id"] is None
    iid, new = incidents.upsert_incident("bind-rollback", "fixture", execution_id=execution["id"])
    assert new
    assert executions.get_execution(execution["id"])["incident_id"] == iid
    assert executions.get_execution(execution["id"])["incident_generation"] == 1
