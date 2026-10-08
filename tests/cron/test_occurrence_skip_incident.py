"""The occurrence-dedup gate consumes a due slot without a run; that skip must leave a durable
incident naming the execution that consumed it (#134858), and a broken incident store must never
change the gate's answer."""

from __future__ import annotations

import sys
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import cron.executions as cron_executions
import cron.incidents as incidents
import cron.occurrences as occurrences
from hermes_time import now as _hermes_now


def _job(**overrides):
    job = {
        "id": "skip-incident-test",
        "name": "skip incident test",
        "prompt": "hello",
        "enabled": True,
        "state": "scheduled",
        "schedule": {"kind": "interval", "minutes": 5, "display": "every 5m"},
        "deliver": "local",
    }
    job.update(overrides)
    return job


def _point_ledgers(monkeypatch, tmp_path):
    db = tmp_path / "cron" / "executions.db"
    monkeypatch.setattr(cron_executions, "EXECUTIONS_FILE", db)
    monkeypatch.setattr(incidents, "EXECUTIONS_FILE", db)
    return db


def _seed_completed_execution(job_id, instant, *, finished_at=None):
    record = cron_executions.create_execution(
        job_id, source="test", scheduled_instant=instant)
    # Same terminal row shape finish_execution writes, without its live-owner guards.
    finish = finished_at or _hermes_now().isoformat()
    with cron_executions._transaction() as conn:
        conn.execute(
            "UPDATE executions SET status='completed', finished_at=? WHERE id=?",
            (finish, record["id"]))
    return record["id"]


def test_consumed_slot_records_incident_naming_the_execution(monkeypatch, tmp_path):
    _point_ledgers(monkeypatch, tmp_path)
    slot = (_hermes_now() - timedelta(minutes=30)).isoformat()
    execution_id = _seed_completed_execution("skip-incident-test", slot)

    assert occurrences.completed_occurrence(_job(), slot) is True

    open_incidents = [i for i in incidents.list_incidents() if i["state"] != "resolved"]
    assert len(open_incidents) == 1
    incident = open_incidents[0]
    assert incident["job_id"] == "skip-incident-test"
    assert incident["failure_type"] == "skipped_occurrence"
    # The incident carries the canonical UTC identity of the consumed occurrence.
    assert occurrences.scheduled_instant(slot) in incident["error"]
    assert execution_id in incident["error"]


def test_healthy_slot_records_no_incident(monkeypatch, tmp_path):
    _point_ledgers(monkeypatch, tmp_path)
    slot = (_hermes_now() - timedelta(minutes=30)).isoformat()
    # A failed attempt for the same slot proves nothing; the occurrence stays eligible.
    record = cron_executions.create_execution(
        "skip-incident-test", source="test", scheduled_instant=slot)
    cron_executions.finish_execution(record["id"], success=False, error="boom")

    assert occurrences.completed_occurrence(_job(), slot) is False
    assert incidents.list_incidents() == []


def test_repeat_skip_refreshes_one_incident_not_many(monkeypatch, tmp_path):
    _point_ledgers(monkeypatch, tmp_path)
    slot = (_hermes_now() - timedelta(minutes=30)).isoformat()
    _seed_completed_execution("skip-incident-test", slot)

    assert occurrences.completed_occurrence(_job(), slot) is True
    assert occurrences.completed_occurrence(_job(), slot) is True

    stored = incidents.list_incidents()
    assert len(stored) == 1


def test_poison_row_does_not_consume_and_records_nothing(monkeypatch, tmp_path):
    _point_ledgers(monkeypatch, tmp_path)
    slot = (_hermes_now() - timedelta(minutes=30)).isoformat()
    # Completed far BEFORE its claimed occurrence: the known poison shape the gate ignores.
    poison_finished = (_hermes_now() - timedelta(days=2)).isoformat()
    _seed_completed_execution("skip-incident-test", slot, finished_at=poison_finished)

    assert occurrences.completed_occurrence(_job(), slot) is False
    assert incidents.list_incidents() == []


def test_broken_incident_store_never_changes_the_gate(monkeypatch, tmp_path):
    _point_ledgers(monkeypatch, tmp_path)
    slot = (_hermes_now() - timedelta(minutes=30)).isoformat()
    _seed_completed_execution("skip-incident-test", slot)

    with patch("cron.incidents.upsert_incident", side_effect=RuntimeError("store gone")):
        # The dedup verdict must stay True even when the incident cannot be recorded.
        assert occurrences.completed_occurrence(_job(), slot) is True
    assert incidents.list_incidents() == []
