import json
from types import SimpleNamespace

from agent.beta.orchestrator import orchestrate_request
from agent.beta.risk import ApprovalGate
from agent.beta_identity import BETA_MODE, HERMES_MODE, ResolvedIdentity
from tools import delegate_tool


REQUEST = "Chefe, verifique por que o PostgreSQL está lento."


def _specialist_payload(task, **overrides):
    specialist_id = task["specialist_id"]
    data = {
        "task_id": task["task_id"],
        "specialist_id": specialist_id,
        "correlation_id": task["correlation_id"],
        "status": "completed",
        "summary": f"{specialist_id} completed its investigation.",
        "evidence": [f"evidence:{specialist_id}"],
        "facts": [],
        "hypotheses": [],
        "confidence": 0.85,
        "recommended_actions": [],
    }
    data.update(overrides)
    return json.dumps(data)


def _delegate_from(findings, calls):
    def fake_delegate(**kwargs):
        task_data = [json.loads(entry["context"])["task"] for entry in kwargs["tasks"]]
        calls.append(tuple(task["specialist_id"] for task in task_data))
        results = []
        for index, task in enumerate(task_data):
            finding = findings[task["specialist_id"]]
            if finding.get("outer_status"):
                results.append(
                    {
                        "task_index": index,
                        "status": finding["outer_status"],
                        "error": finding.get("error", "specialist failed"),
                    }
                )
            else:
                results.append(
                    {
                        "task_index": index,
                        "status": "completed",
                        "summary": _specialist_payload(task, **finding),
                    }
                )
        return json.dumps({"results": results})

    return fake_delegate


def _base_findings():
    cause = "A long-running query holding a lock generated sustained disk I/O"
    return {
        "dba": {
            "facts": ["PostgreSQL session 42 has a long-running query holding a lock"],
            "hypotheses": [cause],
            "evidence": ["pg_stat_activity:session=42 wait_event=Lock"],
            "confidence": 0.94,
            "recommended_actions": [
                "Terminate PostgreSQL session 42",
                "Wait for PostgreSQL session 42 to complete",
            ],
        },
        "infra": {
            "facts": ["CPU usage is normal", "Disk writes are high"],
            "evidence": ["host:cpu=37% disk_writes=high"],
            "confidence": 0.9,
        },
        "monitoring": {
            "facts": ["Latency and disk writes spiked at the query start time"],
            "hypotheses": [cause],
            "evidence": ["metrics:latency_and_writes_spike=10:42Z"],
            "confidence": 0.92,
        },
        "qa-auditor": {
            "summary": "QA validated the cross-source evidence and recommendation boundary.",
            "facts": ["Evidence timestamps correlate across database and monitoring sources"],
            "evidence": ["qa:cross_source_correlation=validated"],
            "confidence": 0.91,
        },
    }


def test_postgresql_diagnosis_runs_parallel_batch_qa_and_exact_approval():
    calls = []
    requested = []

    def deny(operation):
        requested.append(operation)
        return {"approved": False}

    gate = ApprovalGate(requester=deny)
    run = orchestrate_request(
        REQUEST,
        object(),
        delegate=_delegate_from(_base_findings(), calls),
        approval_gate=gate,
    )

    assert run.decision.intent == "diagnosis"
    assert run.decision.specialists == ("dba", "infra", "monitoring")
    assert calls == [("dba", "infra", "monitoring"), ("qa-auditor",)]
    assert run.response.qa_performed is True
    assert run.response.probable_cause == "A long-running query holding a lock generated sustained disk I/O"
    assert run.response.confidence >= 0.8
    assert run.response.authorization_required is True
    assert run.response.recommendation == (
        "Terminate PostgreSQL session 42",
        "Wait for PostgreSQL session 42 to complete",
    )
    assert requested[0].action == "Terminate PostgreSQL session 42"
    assert [event.event for event in gate.events][-2:] == ["requested", "denied"]
    assert run.executed_actions == ()


def test_contradiction_activates_qa_validation():
    findings = _base_findings()
    findings["dba"] = {"facts": ["CPU usage is high"], "confidence": 0.9}
    findings["infra"] = {"facts": ["CPU usage is normal"], "confidence": 0.9}
    findings["monitoring"] = {"facts": ["Disk writes are high"], "confidence": 0.8}
    calls = []

    run = orchestrate_request(REQUEST, object(), delegate=_delegate_from(findings, calls))

    assert run.response.contradictions
    assert run.response.qa_performed is True
    assert calls[-1] == ("qa-auditor",)


def test_specialist_failure_keeps_valid_sibling_evidence():
    findings = _base_findings()
    findings["dba"]["recommended_actions"] = []
    findings["infra"] = {"outer_status": "timeout", "error": "deadline"}
    calls = []

    run = orchestrate_request(REQUEST, object(), delegate=_delegate_from(findings, calls))

    assert run.response.partial_failures == ("infra: deadline",)
    assert "pg_stat_activity:session=42 wait_event=Lock" in run.response.evidence
    assert calls == [("dba", "infra", "monitoring")]


def test_beta_delegate_handler_uses_orchestrator(monkeypatch):
    calls = []
    findings = _base_findings()
    findings["dba"]["recommended_actions"] = []
    monkeypatch.setattr(delegate_tool, "delegate_task", _delegate_from(findings, calls))
    parent = SimpleNamespace(_resolved_identity=ResolvedIdentity(BETA_MODE, "Beta"))

    payload = json.loads(delegate_tool._handle_delegate_call({"goal": REQUEST}, parent_agent=parent))

    assert payload["decision"]["specialists"] == ["dba", "infra", "monitoring"]
    assert calls == [("dba", "infra", "monitoring")]


def test_hermes_delegate_handler_preserves_legacy_path(monkeypatch):
    captured = {}

    def legacy_delegate(**kwargs):
        captured.update(kwargs)
        return "legacy"

    monkeypatch.setattr(delegate_tool, "delegate_task", legacy_delegate)
    parent = SimpleNamespace(_resolved_identity=ResolvedIdentity(HERMES_MODE, "Hermes"))

    result = delegate_tool._handle_delegate_call({"goal": "inspect"}, parent_agent=parent)

    assert result == "legacy"
    assert captured["goal"] == "inspect"
    assert captured["parent_agent"] is parent
