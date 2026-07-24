from agent.beta.delegation import SpecialistResult
from agent.beta.orchestrator import orchestrate_request
from agent.beta.risk import ApprovalGate


def test_failed_mandatory_qa_blocks_approval_and_execution(monkeypatch):
    specialist_result = SpecialistResult(
        task_id="specialist-1",
        specialist_id="dba-postgresql",
        correlation_id="qa-gate",
        status="completed",
        summary="PostgreSQL restart is recommended",
        evidence=("postgres is unresponsive",),
        facts=("postgres service is unavailable",),
        recommended_actions=("restart PostgreSQL in production",),
        confidence=0.9,
    )
    qa_failure = SpecialistResult(
        task_id="qa-1",
        specialist_id="qa-auditor",
        correlation_id="qa-gate",
        status="failed",
        errors=("QA unavailable",),
    )

    calls = {"delegate": 0, "approval": 0, "executor": 0}

    def fake_execute(tasks, _parent_agent, **_kwargs):
        calls["delegate"] += 1
        if tasks[0].specialist_id == "qa-auditor":
            return (qa_failure,)
        return (specialist_result,)

    def request_approval(_operation):
        calls["approval"] += 1
        return {"approved": True}

    def executor(_operation):
        calls["executor"] += 1
        return "should not run"

    monkeypatch.setattr("agent.beta.orchestrator.execute_delegations", fake_execute)
    gate = ApprovalGate(requester=request_approval)

    run = orchestrate_request(
        "Diagnostique e corrija o PostgreSQL de produção",
        object(),
        approval_gate=gate,
        executor=executor,
    )

    assert run.response.qa_required is True
    assert run.response.qa_performed is False
    assert run.response.next_step == "Run QA validation before acting."
    assert run.approval_requests == ()
    assert run.approval_receipts == ()
    assert run.executed_actions == ()
    assert calls == {"delegate": 2, "approval": 0, "executor": 0}
