from __future__ import annotations

from agent.task_orchestration import (
    HermesTaskOrchestrator,
    TaskEnvelope,
    WorkflowStep,
)


def test_legacy_orchestrator_executes_step_and_records_trace():
    orchestrator = HermesTaskOrchestrator(backend="legacy")
    envelope = TaskEnvelope(
        session_id="sess-1",
        task_id="task-1",
        workflow="gateway.message",
        backend="legacy",
    )

    result = orchestrator.run(
        envelope,
        steps=[WorkflowStep(name="conversation")],
        executor=lambda: {"final_response": "hello"},
    )

    assert result["status"] == "completed"
    assert result["result"]["final_response"] == "hello"
    assert result["trace"]["steps"][0]["name"] == "conversation"
    assert result["trace"]["steps"][0]["status"] == "completed"
    assert result["completed_steps"] == ["conversation"]


def test_langgraph_backend_requires_optional_dependency():
    orchestrator = HermesTaskOrchestrator(backend="langgraph")
    envelope = TaskEnvelope(
        session_id="sess-1",
        task_id="task-1",
        workflow="gateway.message",
        backend="langgraph",
    )

    try:
        orchestrator.run(
            envelope,
            steps=[WorkflowStep(name="conversation")],
            executor=lambda: {"final_response": "hello"},
        )
    except RuntimeError as exc:
        message = str(exc)
    else:
        raise AssertionError("expected RuntimeError when langgraph is unavailable")

    assert "langgraph" in message.lower()
    assert "optional" in message.lower()
