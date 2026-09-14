from __future__ import annotations

import json
from types import SimpleNamespace

from agent.clinical_trace import begin_trace, delivery, emit, snapshot


def test_clinical_trace_is_correlated_and_contains_no_content(caplog, monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    agent = SimpleNamespace()
    secret_prompt = "patient-specific secret prompt"
    secret_answer = "patient-specific secret answer"

    with caplog.at_level("INFO", logger="hermes.clinical_delivery_trace"):
        trace = begin_trace(agent, protected=True, platform="telegram")
        emit(agent, "repair", outcome="block", repair_count=1)
        emit(agent, "decision", outcome="allow")
        final = snapshot(agent)
        delivery(final, outcome="delivered")

    assert trace is not None
    assert final is not None
    assert final["trace_id"] == trace["trace_id"]
    assert final["repair_count"] == 1
    assert final["decision"] == "allow"
    payloads = [json.loads(record.message.removeprefix("clinical_delivery_trace ")) for record in caplog.records]
    assert {payload["trace_id"] for payload in payloads} == {trace["trace_id"]}
    assert [payload["event"] for payload in payloads] == ["scope", "repair", "decision", "delivery"]
    rendered = "\n".join(record.message for record in caplog.records)
    assert secret_prompt not in rendered
    assert secret_answer not in rendered
    assert set(final) == {"trace_id", "schema", "protected", "platform", "repair_count", "decision"}
    trace_path = tmp_path / "logs" / "clinical_delivery_trace.jsonl"
    persisted = [json.loads(line) for line in trace_path.read_text().splitlines()]
    assert [record["trace_id"] for record in persisted] == [trace["trace_id"]] * 4
    assert {record["event"] for record in persisted} == {"scope", "repair", "decision", "delivery"}
    assert trace_path.stat().st_mode & 0o777 == 0o600


def test_non_protected_turn_has_no_clinical_trace(caplog):
    agent = SimpleNamespace()
    with caplog.at_level("INFO", logger="hermes.clinical_delivery_trace"):
        assert begin_trace(agent, protected=False, platform="acp") is None
        emit(agent, "decision", outcome="allow")
        delivery(None, outcome="delivered")
    assert snapshot(agent) is None
    assert not caplog.records
