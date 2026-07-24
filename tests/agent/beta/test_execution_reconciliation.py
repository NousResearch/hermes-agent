from agent.beta.consolidation import ConsolidatedResponse
from agent.beta.orchestrator import ExecutedAction, _reconcile_execution_response
from agent.beta.risk import Operation, RiskLevel


def _response() -> ConsolidatedResponse:
    return ConsolidatedResponse(
        understanding="Restart PostgreSQL after diagnosis",
        agents_activated=("dba",),
        result="Execution requires explicit approval",
        evidence=("diagnosis:lock confirmed",),
        facts=("Session 42 is blocking writes",),
        hypotheses=(),
        probable_cause="Blocking session",
        confidence=0.95,
        risk=RiskLevel.HIGH,
        recommendation=("Terminate PostgreSQL session 42",),
        authorization_required=True,
        next_step="Request approval from the Chief",
    )


def _operation() -> Operation:
    return Operation(
        target="postgresql/db-1",
        action="Terminate PostgreSQL session 42",
        impact="Release blocked writes",
        rollback="Reconnect the application session",
        risk=RiskLevel.HIGH,
    )


def test_completed_execution_clears_authorization_and_adds_evidence():
    operation = _operation()
    action = ExecutedAction(
        operation_fingerprint=operation.fingerprint,
        action=operation.action,
        status="completed",
        evidence="executor:session 42 terminated",
    )

    response = _reconcile_execution_response(_response(), (operation,), (action,))

    assert response.authorization_required is False
    assert response.result == "Approved operations executed and validated"
    assert response.next_step == "Execution completed; monitor the resulting system state"
    assert "execution:Terminate PostgreSQL session 42: executor:session 42 terminated" in response.evidence


def test_failed_execution_reports_failure_without_claiming_success():
    operation = _operation()
    action = ExecutedAction(
        operation_fingerprint=operation.fingerprint,
        action=operation.action,
        status="failed",
        evidence="executor error: connection refused",
    )

    response = _reconcile_execution_response(_response(), (operation,), (action,))

    assert response.authorization_required is False
    assert response.result == "Approved execution failed validation"
    assert response.next_step == "Review executor failure evidence and replan before any retry"
    assert response.partial_failures == (
        "execution:Terminate PostgreSQL session 42: executor error: connection refused",
    )


def test_partial_failure_preserves_authorization_for_pending_operation():
    failed_operation = _operation()
    pending_operation = Operation(
        target="postgresql/db-1",
        action="Restart PostgreSQL service",
        impact="Brief database outage",
        rollback="Start the previous service state",
        risk=RiskLevel.HIGH,
    )
    failed_action = ExecutedAction(
        operation_fingerprint=failed_operation.fingerprint,
        action=failed_operation.action,
        status="failed",
        evidence="executor error: connection refused",
    )

    response = _reconcile_execution_response(
        _response(),
        (failed_operation, pending_operation),
        (failed_action,),
    )

    assert response.authorization_required is True
    assert response.result == "Approved execution partially failed validation"
    assert "authorize remaining operations" in response.next_step
    assert response.partial_failures == (
        "execution:Terminate PostgreSQL session 42: executor error: connection refused",
    )
