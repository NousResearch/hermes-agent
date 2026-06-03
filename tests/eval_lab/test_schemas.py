from agent.eval_lab.schemas import (
    EvalScenario,
    EvalScore,
    TrajectoryAttempt,
    TrajectoryGroup,
    TrajectoryStep,
)


def test_eval_schema_round_trip_preserves_grouped_attempts():
    scenario = EvalScenario(
        id="tool_required_math",
        title="Tool required math",
        prompt="Use a tool to calculate 19 * 23.",
        tags=["smoke", "tool-use"],
        expected_artifacts=[],
        blocked_actions=["send_message"],
        success_criteria=["final response contains 437"],
    )
    step = TrajectoryStep(
        role="tool",
        content="437",
        tool_name="execute_code",
        tool_args_redacted={"code": "print(19 * 23)"},
        duration_ms=12,
        error=None,
    )
    attempt = TrajectoryAttempt(
        attempt_id="attempt-1",
        scenario_id=scenario.id,
        started_at="2026-05-25T10:00:00Z",
        finished_at="2026-05-25T10:00:01Z",
        status="completed",
        final_response="437",
        steps=[step],
        metadata={"latency_ms": 1000},
    )
    group = TrajectoryGroup(group_id="group-1", scenario_id=scenario.id, attempts=[attempt])
    score = EvalScore(
        attempt_id=attempt.attempt_id,
        total=1.0,
        criteria={"completion": 1.0, "tool_use_required": 1.0},
        notes=["ok"],
    )

    scenario_payload = scenario.to_dict()
    group_payload = group.to_dict()
    score_payload = score.to_dict()

    assert EvalScenario.from_dict(scenario_payload) == scenario
    assert TrajectoryGroup.from_dict(group_payload) == group
    assert EvalScore.from_dict(score_payload) == score
    assert group_payload["attempts"][0]["steps"][0]["tool_args_redacted"] == {"code": "print(19 * 23)"}


def test_eval_scenario_requires_core_fields():
    try:
        EvalScenario.from_dict({"id": "missing_prompt", "title": "Missing prompt"})
    except ValueError as exc:
        message = str(exc)
    else:
        raise AssertionError("Expected invalid scenario to raise ValueError")

    assert "prompt" in message
    assert "tags" in message
