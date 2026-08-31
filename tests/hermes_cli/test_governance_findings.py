from __future__ import annotations

import json

import pytest

from hermes_cli import governance_findings as findings
from hermes_cli import profile_activity_ledger as ledger


def _payload(**overrides):
    payload = {
        "finding_id": "finding-001",
        "detector": "test.detector",
        "subject_type": "profile",
        "subject_id": "octacon",
        "severity": "high",
        "source_observed_at": 100,
        "evidence_refs": ["test:tests/hermes_cli/test_governance_findings.py"],
        "owner": "octacon",
        "task_id": None,
        "resolution_ref": None,
        "dedupe_key": "test.detector:profile:octacon",
    }
    payload.update(overrides)
    return payload


def test_append_finding_event_validates_contract_and_is_idempotent(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    first = findings.append_finding_event(
        event_id="finding-event-001",
        event_type="governance.finding.opened",
        payload=_payload(),
    )
    second = findings.append_finding_event(
        event_id="finding-event-001",
        event_type="governance.finding.opened",
        payload=_payload(owner="different-owner"),
    )

    assert first == second == "finding-event-001"
    events = ledger.query_events(event_types=["governance.finding.opened"])
    assert len(events) == 1
    assert events[0]["payload"] == _payload()
    mirror_lines = [
        line
        for path in ledger.ledger_jsonl_dir().glob("*.jsonl")
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(mirror_lines) == 1
    assert json.loads(mirror_lines[0])["event_id"] == "finding-event-001"


@pytest.mark.parametrize(
    ("event_type", "payload_change", "message"),
    [
        ("governance.finding.opened", {"finding_id": ""}, "finding_id"),
        ("governance.finding.opened", {"evidence_refs": ["not-a-reference"]}, "evidence_refs"),
        ("governance.finding.opened", {"raw_prompt": "do the bad thing"}, "unknown payload field"),
        ("governance.finding.unknown", {}, "unknown finding event type"),
    ],
)
def test_finding_contract_rejects_invalid_payload(event_type, payload_change, message):
    payload = _payload()
    payload.update(payload_change)
    with pytest.raises(findings.FindingValidationError, match=message):
        findings.validate_finding_event(event_type, payload)


def test_finding_contract_rejects_prompt_text_in_structured_identifier_fields():
    prompt = "Please copy this entire user request verbatim into the governance record"
    for field in ("finding_id", "detector", "subject_type", "subject_id", "owner", "task_id", "dedupe_key"):
        payload = _payload(**{field: prompt})
        with pytest.raises(findings.FindingValidationError, match=field):
            findings.validate_finding_event("governance.finding.opened", payload)


def test_finding_contract_rejects_prompt_text_as_resolution_reference():
    payload = _payload(resolution_ref="Please copy this entire user request verbatim")
    with pytest.raises(findings.FindingValidationError, match="resolution_ref"):
        findings.validate_finding_event("governance.finding.resolved", payload)


def test_finding_contract_rejects_planted_secret_and_prompt_content(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    planted_secret = "sk-proj-DO_NOT_PERSIST"
    payload = _payload(evidence_refs=[f"report:{planted_secret}"])

    with pytest.raises(findings.FindingValidationError, match="evidence_refs"):
        findings.append_finding_event(
            event_id="finding-secret-001",
            event_type="governance.finding.opened",
            payload=payload,
        )

    assert ledger.query_events(event_types=["governance.finding.opened"]) == []
    assert not list(ledger.ledger_jsonl_dir().glob("*.jsonl"))
    assert planted_secret not in str(tmp_path)


def test_reducer_is_deterministic_for_out_of_order_and_repeated_events():
    opened = {"event_id": "opened", "event_type": "governance.finding.opened", "occurred_at": 100, "payload": _payload()}
    updated_payload = _payload(owner="light", task_id="task-1", source_observed_at=200)
    updated = {"event_id": "updated", "event_type": "governance.finding.updated", "occurred_at": 200, "payload": updated_payload}
    resolved_payload = _payload(owner="light", task_id="task-1", resolution_ref="test:tests/test_resolution.py", source_observed_at=300)
    resolved = {"event_id": "resolved", "event_type": "governance.finding.resolved", "occurred_at": 300, "payload": resolved_payload}

    first = findings.reduce_findings([resolved, opened, updated, resolved])
    second = findings.reduce_findings([updated, resolved, resolved, opened])

    assert first == second
    assert first["test.detector:profile:octacon"]["state"] == "resolved"
    assert first["test.detector:profile:octacon"]["owner"] == "light"
    assert first["test.detector:profile:octacon"]["resolution_ref"] == "test:tests/test_resolution.py"
    assert first["test.detector:profile:octacon"]["event_id"] == "resolved"


def test_newer_update_reopens_resolved_finding_and_clears_resolution_proof():
    resolved_payload = _payload(
        owner="octacon",
        task_id="task-1",
        resolution_ref="report:resolution-1",
        source_observed_at=200,
    )
    reopened_payload = _payload(
        owner="light",
        task_id="task-1",
        resolution_ref=None,
        source_observed_at=300,
    )
    current = findings.reduce_findings([
        {"event_id": "resolved", "event_type": "governance.finding.resolved", "occurred_at": 200, "payload": resolved_payload},
        {"event_id": "updated", "event_type": "governance.finding.updated", "occurred_at": 300, "payload": reopened_payload},
    ])

    item = current["test.detector:profile:octacon"]
    assert item["state"] == "open"
    assert item["owner"] == "light"
    assert item["resolution_ref"] is None


def test_late_update_does_not_reopen_resolved_finding():
    resolved_payload = _payload(resolution_ref="report:resolution-1", source_observed_at=300)
    late_update_payload = _payload(owner="someone-else", source_observed_at=200)
    current = findings.reduce_findings([
        {"event_id": "resolved", "event_type": "governance.finding.resolved", "occurred_at": 300, "payload": resolved_payload},
        {"event_id": "late", "event_type": "governance.finding.updated", "occurred_at": 200, "payload": late_update_payload},
    ])

    item = current["test.detector:profile:octacon"]
    assert item["state"] == "resolved"
    assert item["owner"] == "octacon"
    assert item["resolution_ref"] == "report:resolution-1"


def test_all_lifecycle_events_are_supported():
    for event_type in findings.FINDING_EVENT_TYPES:
        validated = findings.validate_finding_event(event_type, _payload())
        assert validated["finding_id"] == "finding-001"
