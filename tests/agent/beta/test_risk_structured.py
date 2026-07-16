from agent.beta.risk import ApprovalGate, Operation, RiskContext, RiskLevel, classify_risk


def test_structured_production_context_overrides_read_only_words():
    risk = classify_risk(
        "inspect then apply",
        context=RiskContext(read_only=False, changes_state=True, production=True),
    )
    assert risk == RiskLevel.HIGH


def test_explicit_read_only_context_is_low():
    assert classify_risk(
        "query production status",
        context=RiskContext(read_only=True, production=False),
    ) == RiskLevel.LOW


def test_receipt_only_authorizes_exact_operation():
    now = [100.0]
    gate = ApprovalGate(
        requester=lambda operation: {"approved": True, "approved_by": "chief"},
        clock=lambda: now[0],
    )
    operation = Operation(
        target="server-a", action="restart nginx", impact="brief outage",
        rollback="start previous unit", risk=RiskLevel.HIGH,
    )
    receipt = gate.request(operation, ttl_seconds=10)
    assert gate.authorized(operation, receipt)
    other = operation.model_copy(update={"target": "server-b"})
    assert not gate.authorized(other, receipt)
    now[0] = 111.0
    assert not gate.authorized(operation, receipt)
