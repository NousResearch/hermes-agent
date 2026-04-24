from agent.gap_check import GapCheckResult, analyze_gap, should_skip_next_iteration


def test_analyze_gap_returns_complete_for_fully_evidenced_plan():
    result = analyze_gap(
        plan={
            "steps": [
                "Add gap-check module",
                "Write focused tests",
            ]
        },
        result={
            "summary": "Implemented add gap-check module and write focused tests",
            "completed": ["Add gap-check module", "Write focused tests"],
        },
        evidence=[
            "agent/gap_check.py created",
            "tests/agent/test_gap_check.py created",
        ],
    )

    assert isinstance(result, GapCheckResult)
    assert result.status == "complete"
    assert result.missing_items == []
    assert result.blocked_items == []
    assert result.remediation_tasks == []
    assert result.next_prompt is None
    assert result.read_only_safe is True
    assert should_skip_next_iteration(result) is True


def test_analyze_gap_returns_missing_with_remediation_and_next_prompt():
    result = analyze_gap(
        plan=["Create module", "Add tests", "Run focused verification"],
        result="Created module and added tests.",
        evidence=["agent/gap_check.py", "tests/agent/test_gap_check.py"],
    )

    assert result.status == "missing"
    assert result.missing_items == ["Run focused verification"]
    assert result.blocked_items == []
    assert result.remediation_tasks == [
        "Complete missing plan item: Run focused verification"
    ]
    assert "Run focused verification" in result.next_prompt
    assert should_skip_next_iteration(result) is False


def test_analyze_gap_returns_blocked_when_blocker_is_detected():
    result = analyze_gap(
        plan=["Run focused verification"],
        result={
            "summary": "Run focused verification is blocked by missing uv binary.",
            "blocked": ["Run focused verification"],
        },
        evidence=["Blocked: uv binary not found"],
    )

    assert result.status == "blocked"
    assert result.missing_items == []
    assert result.blocked_items == ["Run focused verification"]
    assert result.remediation_tasks == [
        "Resolve blocker for plan item: Run focused verification"
    ]
    assert "blocked item" in result.next_prompt.lower()
    assert should_skip_next_iteration(result) is False


def test_analyze_gap_is_read_only_safe_for_reviewer_and_verifier_calls():
    reviewer_result = analyze_gap(
        plan="Document outcome",
        result="Document outcome completed",
        evidence={"artifacts": ["summary note"]},
        caller_role="reviewer",
    )
    verifier_result = analyze_gap(
        plan="Document outcome",
        result="Document outcome completed",
        evidence={"artifacts": ["summary note"]},
        caller_role="verifier",
    )

    assert reviewer_result.status == "complete"
    assert reviewer_result.read_only_safe is True
    assert verifier_result.status == "complete"
    assert verifier_result.read_only_safe is True
