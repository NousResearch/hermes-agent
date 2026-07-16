import json

from agent.beta.delegation import (
    SpecialistResult,
    create_delegation_tasks,
    execute_delegations,
)
from agent.beta.router import route_request


def _completed(task, **overrides):
    data = SpecialistResult(
        task_id=task.task_id,
        specialist_id=task.specialist_id,
        correlation_id=task.correlation_id,
        status="completed",
        summary=f"{task.specialist_id} result",
        evidence=(f"evidence:{task.specialist_id}",),
        facts=(f"fact:{task.specialist_id}",),
        confidence=0.8,
    ).model_dump()
    data.update(overrides)
    return json.dumps(data)


def test_simple_task_uses_existing_delegate_contract():
    decision = route_request("Diagnose a query PostgreSQL lenta")
    task = next(task for task in create_delegation_tasks("Diagnose a query PostgreSQL lenta", decision) if task.specialist_id == "dba")
    captured = {}

    def fake_delegate(**kwargs):
        captured.update(kwargs)
        return json.dumps({"results": [{"task_index": 0, "status": "completed", "summary": _completed(task)}]})

    result = execute_delegations([task], object(), delegate=fake_delegate)
    assert result[0].status == "completed"
    assert captured["tasks"][0]["role"] == "leaf"
    assert "memory" not in task.allowed_tools
    assert captured["background"] is False


def test_batch_preserves_valid_results_when_one_times_out():
    decision = route_request("Verifique por que o PostgreSQL está lento")
    tasks = create_delegation_tasks(
        "Verifique por que o PostgreSQL está lento",
        decision,
        correlation_id="corr-1",
    )

    def fake_delegate(**kwargs):
        assert len(kwargs["tasks"]) == len(tasks)
        return json.dumps(
            {
                "results": [
                    {"task_index": 0, "status": "completed", "summary": _completed(tasks[0])},
                    {"task_index": 1, "status": "timeout", "error": "deadline"},
                    {"task_index": 2, "status": "completed", "summary": _completed(tasks[2])},
                ]
            }
        )

    results = execute_delegations(tasks, object(), delegate=fake_delegate)
    assert [result.status for result in results] == ["completed", "timeout", "completed"]
    assert {result.correlation_id for result in results} == {"corr-1"}


def test_invalid_specialist_response_is_contract_error():
    decision = route_request("Diagnose PostgreSQL lento")
    task = create_delegation_tasks("Diagnose PostgreSQL lento", decision)[0]

    def fake_delegate(**_kwargs):
        return json.dumps({"results": [{"task_index": 0, "status": "completed", "summary": "not json"}]})

    result = execute_delegations([task], object(), delegate=fake_delegate)[0]
    assert result.status == "contract_error"
    assert "invalid specialist response" in result.errors[0]


def test_mismatched_correlation_is_contract_error():
    decision = route_request("Diagnose PostgreSQL lento")
    task = create_delegation_tasks("Diagnose PostgreSQL lento", decision)[0]

    def fake_delegate(**_kwargs):
        return json.dumps(
            {"results": [{"task_index": 0, "status": "completed", "summary": _completed(task, correlation_id="wrong")}]}
        )

    assert execute_delegations([task], object(), delegate=fake_delegate)[0].status == "contract_error"

