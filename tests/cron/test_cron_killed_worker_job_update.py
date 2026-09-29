"""#120328: a cron worker killed after adoption must not leave the job untouched.

A restart-safe worker that dies after adopting its execution (e.g. systemd
OOM-kills its scope) is recovered to ``unknown`` by the gateway waiter — but
nothing updates the job: ``last_run_at``/``last_status``/``last_error`` keep
the previous run's values, no ``cron_incidents`` row opens, nothing is
delivered, and the gateway logs no error.
"""
from __future__ import annotations

import logging
from unittest.mock import Mock


def test_killed_worker_after_adopt_updates_job_incident_and_log(
    tmp_path, monkeypatch, caplog
):
    import cron.scheduler as scheduler
    import cron.executions as executions
    import cron.incidents as incidents
    from cron.jobs import (
        claim_job_for_fire, create_job, get_job, use_cron_store,
    )

    monkeypatch.setattr(
        executions, "EXECUTIONS_FILE", tmp_path / "executions.db")
    with use_cron_store(tmp_path):
        job = create_job(prompt="x", schedule="every 5m", name="killed")
        jid = job["id"]
        assert claim_job_for_fire(jid) is True
        assert get_job(jid)["fire_claim"]["by"]

        recovery_error = executions._OWNER_GONE_REASON
        states = iter([
            {"id": "exec-1", "job_id": jid, "status": "running"},
            {"id": "exec-1", "job_id": jid, "status": "unknown",
             "error": recovery_error},
            {"id": "exec-1", "job_id": jid, "status": "unknown",
             "error": recovery_error},
        ])
        monkeypatch.setattr(
            scheduler, "get_execution", lambda _eid: next(states))
        recover = Mock(return_value=1)
        monkeypatch.setattr(
            scheduler, "recover_interrupted_executions", recover)
        delivered = Mock(return_value=None)
        monkeypatch.setattr(scheduler, "_deliver_result", delivered)

        process = Mock()
        process.wait.return_value = 137
        process.poll.return_value = 137

        with caplog.at_level(logging.ERROR):
            assert scheduler._wait_for_external_cron_worker(
                process, execution_id="exec-1", job_id=jid) is True

        after = get_job(jid)
        assert after["last_run_at"] is not None
        assert after["last_status"] == "error"
        assert "137" in (after["last_error"] or "")
        rows = [r for r in incidents.list_incidents()
                if r["job_id"] == jid]
        assert rows, "killed run must open a cron_incidents row"
        assert delivered.called, "killed run must attempt failure delivery"
        assert any(jid in r.getMessage() or "exec-1" in r.getMessage()
                   for r in caplog.records), \
            "killed run must log a gateway error"
