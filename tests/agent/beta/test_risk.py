import json

from agent.beta.delegation import SpecialistResult, create_delegation_tasks, execute_delegations, task_operation
from agent.beta.risk import ApprovalGate, Operation, RiskLevel, classify_risk
from agent.beta.router import route_request


def test_read_only_logs_are_low_risk():
    assert classify_risk("Read service logs", "terminal") == RiskLevel.LOW


def test_restart_is_blocked_before_delegation_without_approval():
    decision = route_request("Reinicie o PostgreSQL no servidor db-1")
    task = create_delegation_tasks("Reinicie o PostgreSQL no servidor db-1", decision)[0]
    called = False

    def fake_delegate(**_kwargs):
        nonlocal called
        called = True
        return "{}"

    result = execute_delegations([task], object(), delegate=fake_delegate)[0]
    assert called is False
    assert result.authorization_required is True


def test_exact_approved_operation_can_execute():
    decision = route_request("Reinicie o PostgreSQL no servidor db-1")
    task = create_delegation_tasks("Reinicie o PostgreSQL no servidor db-1", decision)[0]
    gate = ApprovalGate(requester=lambda _operation: {"approved": True})
    receipt = gate.request(task_operation(task))

    response = SpecialistResult(
        task_id=task.task_id,
        specialist_id=task.specialist_id,
        correlation_id=task.correlation_id,
        status="completed",
        summary="Restart completed",
    )

    def fake_delegate(**_kwargs):
        return json.dumps({"results": [{"task_index": 0, "status": "completed", "summary": response.model_dump_json()}]})

    result = execute_delegations(
        [task],
        object(),
        delegate=fake_delegate,
        approval_gate=gate,
        approval_receipts={task.task_id: receipt},
    )[0]
    assert result.status == "completed"
    assert [event.event for event in gate.events] == ["requested", "approved", "authorized"]


def test_denied_and_expired_approvals_fail_closed():
    operation = Operation(
        target="db-1",
        action="restart postgres",
        impact="brief outage",
        rollback="start previous service",
        risk=RiskLevel.HIGH,
    )
    denied = ApprovalGate(requester=lambda _operation: {"approved": False})
    assert denied.request(operation) is None

    now = [100.0]
    expiring = ApprovalGate(requester=lambda _operation: {"approved": True}, clock=lambda: now[0])
    receipt = expiring.request(operation, ttl_seconds=5)
    now[0] = 106.0
    assert expiring.authorized(operation, receipt) is False
    assert expiring.events[-1].event == "expired"


def test_approval_does_not_cross_target_or_changed_action():
    gate = ApprovalGate(requester=lambda _operation: {"approved": True})
    original = Operation(
        target="db-1",
        action="restart postgres",
        impact="brief outage",
        rollback="start previous service",
        risk=RiskLevel.HIGH,
    )
    receipt = gate.request(original)
    assert gate.authorized(original.model_copy(update={"target": "db-2"}), receipt) is False
    assert gate.authorized(original.model_copy(update={"action": "drop database"}), receipt) is False

