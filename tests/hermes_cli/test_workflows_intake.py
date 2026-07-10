from hermes_cli.workflows_spec import WorkflowSpec


def _trigger(input_schema=None, intake=None):
    spec = WorkflowSpec.model_validate({
        "id": "intake_demo",
        "name": "Intake Demo",
        "version": 1,
        "triggers": [{
            "type": "manual",
            "id": "kickoff",
            "input_schema": input_schema or {},
            "intake": intake or {},
        }],
        "nodes": {"start": {"type": "pass"}},
    })
    return spec.triggers[0]


def test_evaluate_intake_requires_fields_and_lengths():
    from hermes_cli.workflows_intake import evaluate_intake

    result = evaluate_intake(
        _trigger({"brief": {"kind": "long_text", "required": True, "min_length": 5}}),
        {"brief": "hey"},
    )

    assert result.ready is False
    assert result.status == "needs_input"
    assert result.messages == ["brief must be at least 5 characters"]


def test_evaluate_intake_treats_blank_required_text_as_missing():
    from hermes_cli.workflows_intake import evaluate_intake

    result = evaluate_intake(
        _trigger({"brief": {"kind": "long_text", "required": True}}),
        {"brief": "   "},
    )

    assert result.ready is False
    assert result.status == "needs_input"
    assert result.messages == ["brief is required"]


def test_evaluate_intake_enforces_numeric_min_and_max():
    from hermes_cli.workflows_intake import evaluate_intake

    low = evaluate_intake(
        _trigger({"score": {"kind": "number", "required": True, "min": 3, "max": 5}}),
        {"score": 2},
    )
    high = evaluate_intake(
        _trigger({"score": {"kind": "number", "required": True, "min": 3, "max": 5}}),
        {"score": 6},
    )
    ok = evaluate_intake(
        _trigger({"score": {"kind": "number", "required": True, "min": 3, "max": 5}}),
        {"score": 4},
    )

    assert low.status == "needs_input"
    assert low.messages == ["score must be at least 3"]
    assert high.status == "needs_input"
    assert high.messages == ["score must be at most 5"]
    assert ok.status == "queued"


def test_evaluate_intake_rejects_boolean_for_numeric_fields():
    from hermes_cli.workflows_intake import evaluate_intake

    result = evaluate_intake(
        _trigger({"score": {"kind": "number", "required": True, "min": 0}}),
        {"score": True},
    )

    assert result.ready is False
    assert result.status == "needs_input"
    assert result.messages == ["score must be a number"]


def test_evaluate_intake_handles_ready_when_runtime_errors(monkeypatch):
    import hermes_cli.workflows_intake as intake

    def raise_runtime_error(_condition, _data):
        raise RuntimeError("boom")

    monkeypatch.setattr(intake, "eval_condition", raise_runtime_error)

    result = intake.evaluate_intake(
        _trigger(
            {"brief": {"kind": "text", "required": True}},
            {"ready_when": {"op": "exists", "path": "$.input.brief"}},
        ),
        {"brief": "ship it"},
    )

    assert result.ready is False
    assert result.status == "needs_input"
    assert result.messages == ["ready_when invalid: boom"]


def test_evaluate_intake_ready_when_uses_workflow_conditions():
    from hermes_cli.workflows_intake import evaluate_intake

    result = evaluate_intake(
        _trigger(
            {"brief": {"kind": "text", "required": True}},
            {"ready_when": {"op": "eq", "left": {"path": "$.input.accepted"}, "right": True}},
        ),
        {"brief": "ship it", "accepted": True},
    )

    assert result.ready is True
    assert result.status == "queued"
    assert result.messages == []
    assert result.criteria["ready_when"] is True
