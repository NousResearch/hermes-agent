"""Tests for durable-state promotion candidate detection."""

from agent.durable_candidate_detection import detect_durable_candidates


def test_detects_skill_candidate_after_many_tool_calls_and_error_recovery():
    result = detect_durable_candidates(
        tool_call_count=7,
        fixed_error=True,
        user_corrected_procedure=False,
        discovered_project_fact=False,
        produced_artifacts=["N:/Hermes/reports/fix.json"],
    )

    assert result["suggest_skill"] is True
    assert "many-tool-calls" in result["reasons"]
    assert "error-recovery" in result["reasons"]
    assert result["artifacts"] == ["N:/Hermes/reports/fix.json"]


def test_detects_project_note_for_stable_project_fact():
    result = detect_durable_candidates(
        tool_call_count=2,
        discovered_project_fact=True,
        project_fact="Project uses uv-run pytest with -n 0 on Windows",
    )

    assert result["suggest_project_note"] is True
    assert result["project_facts"] == ["Project uses uv-run pytest with -n 0 on Windows"]
    assert result["suggest_memory"] is False


def test_detects_memory_for_stable_user_preference_only():
    result = detect_durable_candidates(
        stable_user_preference="User prefers compact chat rotation when durable state is preserved",
    )

    assert result["suggest_memory"] is True
    assert result["user_preferences"] == [
        "User prefers compact chat rotation when durable state is preserved"
    ]


def test_rejects_stale_artifacts_from_memory_candidates():
    result = detect_durable_candidates(
        stable_user_preference="PR #123 was opened",
        produced_artifacts=["commit abc123", "N:/Hermes/reports/result.json"],
    )

    assert result["suggest_memory"] is False
    assert "PR #123 was opened" in result["do_not_carry"]
    assert "commit abc123" in result["do_not_carry"]
    assert "N:/Hermes/reports/result.json" in result["artifacts"]
