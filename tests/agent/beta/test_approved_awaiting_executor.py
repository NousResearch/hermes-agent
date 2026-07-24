import json

from agent.beta.orchestrator import orchestrate_request
from agent.beta.risk import ApprovalGate


REQUEST = "Chefe, verifique por que o PostgreSQL está lento."


def _delegate(**kwargs):
    results = []
    for index, entry in enumerate(kwargs["tasks"]):
        task = json.loads(entry["context"])["task"]
        specialist_id = task["specialist_id"]
        recommendations = ["Terminate PostgreSQL session 42"] if specialist_id == "dba" else []
        payload = {
            "task_id": task["task_id"],
            "specialist_id": specialist_id,
            "correlation_id": task["correlation_id"],
            "status": "completed",
            "summary": f"{specialist_id} completed",
            "evidence": [f"evidence:{specialist_id}"],
            "facts": ["PostgreSQL session 42 is blocking work"],
            "hypotheses": [],
            "confidence": 0.9,
            "recommended_actions": recommendations,
        }
        results.append({"task_index": index, "status": "completed", "summary": json.dumps(payload)})
    return json.dumps({"results": results})


def test_approved_operation_without_executor_is_not_reported_as_pending_authorization():
    gate = ApprovalGate(requester=lambda _operation: {"approved": True})

    run = orchestrate_request(
        REQUEST,
        object(),
        delegate=_delegate,
        approval_gate=gate,
        executor=None,
    )

    assert run.approval_receipts
    assert run.executed_actions == ()
    assert run.response.result == "Approved operations awaiting executor"
    assert run.response.authorization_required is False
    assert "executor" in run.response.next_step.lower()
    assert "executed and validated" not in run.response.result.lower()
