from __future__ import annotations

import json
import time

import pytest

from hermes_cli import governance_findings as findings
from hermes_cli import kanban_db as kb
from hermes_cli import profile_activity_ledger as ledger


def _payload():
    return {
        "finding_id": "finding-kanban-001",
        "detector": "test.detector",
        "subject_type": "profile",
        "subject_id": "octacon",
        "severity": "high",
        "source_observed_at": 100,
        "evidence_refs": ["test:tests/hermes_cli/test_governance_findings_kanban.py"],
        "owner": "octacon",
        "task_id": None,
        "resolution_ref": None,
        "dedupe_key": "test.detector:profile:octacon",
    }


@pytest.fixture
def kanban_connection(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(kb, "get_current_board", lambda: "default")
    path = home / "kanban.db"
    conn = kb.connect(path)
    yield conn
    conn.close()


def test_finding_task_link_is_idempotent_and_completion_does_not_resolve_finding(
    kanban_connection, monkeypatch
):
    monkeypatch.setattr(ledger, "is_enabled", lambda cfg=None: True)
    payload = _payload()

    first = findings.link_finding_to_task(kanban_connection, payload)
    second = findings.link_finding_to_task(kanban_connection, payload)

    assert first == second
    task_events = kanban_connection.execute(
        "SELECT kind, payload FROM task_events WHERE task_id = ? AND kind = 'finding_linked'",
        (first,),
    ).fetchall()
    assert len(task_events) == 1
    assert json.loads(task_events[0]["payload"]) == {
        "dedupe_key": payload["dedupe_key"],
        "finding_id": payload["finding_id"],
        "severity": "high",
    }

    assert kb.complete_task(kanban_connection, first, result="implementation complete") is True
    current = findings.reduce_findings(ledger.query_events())
    assert current[payload["dedupe_key"]]["state"] == "open"
    assert current[payload["dedupe_key"]]["task_id"] == first


def test_resolution_requires_explicit_typed_resolution_evidence_and_links_back(
    kanban_connection, monkeypatch
):
    monkeypatch.setattr(ledger, "is_enabled", lambda cfg=None: True)
    payload = _payload()
    task_id = findings.link_finding_to_task(kanban_connection, payload)
    assert kb.complete_task(kanban_connection, task_id, result="green") is True

    with pytest.raises(findings.FindingValidationError, match="resolution_ref"):
        findings.resolve_finding(payload, "not-a-reference", task_id=task_id)

    resolution_payload = dict(payload, task_id=task_id)
    event_id = findings.resolve_finding(
        resolution_payload,
        "test:tests/hermes_cli/test_governance_findings_kanban.py",
        task_id=task_id,
        conn=kanban_connection,
    )
    assert event_id
    current = findings.reduce_findings(ledger.query_events())
    item = current[payload["dedupe_key"]]
    assert item["state"] == "resolved"
    assert item["resolution_ref"].startswith("test:")


def test_recurrence_reopens_the_same_finding_and_reuses_the_same_task(
    kanban_connection, monkeypatch
):
    monkeypatch.setattr(ledger, "is_enabled", lambda cfg=None: True)
    payload = _payload()
    task_id = findings.link_finding_to_task(kanban_connection, payload)
    findings.resolve_finding(
        dict(payload, task_id=task_id),
        "report:resolution-1",
        task_id=task_id,
        conn=kanban_connection,
    )

    reopened = dict(payload, source_observed_at=int(time.time()) + 1, task_id=None)
    same_task = findings.link_finding_to_task(kanban_connection, reopened)

    assert same_task == task_id
    current = findings.reduce_findings(ledger.query_events())
    assert current[payload["dedupe_key"]]["state"] == "open"
    assert kanban_connection.execute(
        "SELECT COUNT(*) FROM tasks WHERE idempotency_key = ?",
        ("governance:finding:" + payload["dedupe_key"],),
    ).fetchone()[0] == 1


def test_linkage_queries_only_finding_events(kanban_connection, monkeypatch):
    monkeypatch.setattr(ledger, "is_enabled", lambda cfg=None: True)
    calls = []
    real_query_events = ledger.query_events

    def query_events(**kwargs):
        calls.append(kwargs)
        return real_query_events(**kwargs)

    monkeypatch.setattr(ledger, "query_events", query_events)
    findings.link_finding_to_task(kanban_connection, _payload())

    assert calls == [{"event_types": sorted(findings.FINDING_EVENT_TYPES)}]
