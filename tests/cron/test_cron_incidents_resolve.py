"""A successful run resolves the job's open incidents; the same error afterwards re-opens them.

The ledger only ever grew: a one-off failure (drift skip, provider outage) stayed ``detected``/``alerted``
after the job had been green for weeks, so ``hermes cron incidents`` listed 32 "open" incidents on an
install whose every job was healthy. ``resolved`` is auto and re-openable; ``closed`` is the operator's
ack and stays silent.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import cron.incidents as incidents
import cron.scheduler as sched


def _point_db(monkeypatch, tmp_path):
    monkeypatch.setattr(incidents, "EXECUTIONS_FILE", tmp_path / "cron" / "executions.db")
    return incidents


def test_successful_run_resolves_open_incidents_and_a_repeat_reopens_them(monkeypatch, tmp_path):
    inc = _point_db(monkeypatch, tmp_path)
    inc_id, _ = inc.upsert_incident("job-1", "provider 503")
    sched._mark_incident_alerted(inc_id)
    other_id, _ = inc.upsert_incident("job-2", "provider 503")

    content, *_ = sched._compose_run_delivery(
        {"id": "job-1", "name": "j1"}, success=True, error=None, final_response="all good", output_file=None)

    assert content == "all good"
    assert inc.get_incident(inc_id)["state"] == "resolved"
    assert inc.get_incident(inc_id)["closed_at"]
    assert inc.get_incident(other_id)["state"] == "detected"  # another job's incident is untouched
    assert inc.count_incidents("resolved") == 1

    same_id, reopened = inc.upsert_incident("job-1", "provider 503")
    assert same_id == inc_id and reopened is True
    assert inc.get_incident(inc_id)["state"] == "detected"
    assert inc.get_incident(inc_id)["closed_at"] is None


def test_operator_ack_survives_a_successful_run(monkeypatch, tmp_path):
    inc = _point_db(monkeypatch, tmp_path)
    inc_id, _ = inc.upsert_incident("job-1", "known flaky")
    inc.ack_incident(inc_id)

    assert inc.close_incidents_for_recovered_job("job-1") == 0
    assert inc.get_incident(inc_id)["state"] == "closed"
    _, reopened = inc.upsert_incident("job-1", "known flaky")
    assert reopened is False and inc.get_incident(inc_id)["state"] == "closed"
