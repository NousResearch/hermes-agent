from gateway.run import (
    apply_pr_created_claim_guard,
    validate_pr_created_claim_evidence,
)


def test_pr_created_claim_guard_blocks_claim_without_url():
    decision = validate_pr_created_claim_evidence("PR作成済みです。", {})

    assert decision["passed"] is False
    assert decision["guard"] == "pr-created-claim-evidence"
    assert decision["reason"] == "pr_created_claim_without_concrete_pr_url"


def test_pr_created_claim_guard_blocks_placeholder_url():
    decision = validate_pr_created_claim_evidence(
        "PR作成済み。https://github.com/foo/bar/pull/123 これは仮の値です",
        {
            "tool_trace": [
                {
                    "command": "gh pr create --title test",
                    "exit_code": 0,
                    "stdout": "https://github.com/foo/bar/pull/123",
                }
            ]
        },
    )

    assert decision["passed"] is False
    assert decision["reason"] == "pr_created_claim_with_placeholder_url"


def test_pr_created_claim_guard_blocks_claim_without_create_trace():
    decision = validate_pr_created_claim_evidence(
        "PR作成済みです。https://github.com/foo/bar/pull/123",
        {"tool_trace": []},
    )

    assert decision["passed"] is False
    assert decision["reason"] == "pr_created_claim_without_create_trace"


def test_pr_created_claim_guard_blocks_view_only_trace():
    decision = validate_pr_created_claim_evidence(
        "PR作成済みです。https://github.com/foo/bar/pull/123",
        {
            "tool_trace": [
                {
                    "command": "gh pr view 123",
                    "exit_code": 0,
                    "stdout": "https://github.com/foo/bar/pull/123",
                }
            ]
        },
    )

    assert decision["passed"] is False
    assert decision["reason"] == "pr_created_claim_without_create_trace"


def test_pr_created_claim_guard_allows_create_trace_with_matching_url():
    decision = validate_pr_created_claim_evidence(
        "PR作成済みです。https://github.com/foo/bar/pull/123",
        {
            "tool_trace": [
                {
                    "command": "gh pr create --title test",
                    "exit_code": 0,
                    "stdout": "https://github.com/foo/bar/pull/123",
                }
            ]
        },
    )

    assert decision["passed"] is True
    assert decision["reason"] == "pr_created_claim_has_create_trace_and_url"


def test_pr_created_claim_guard_allows_honest_not_created_response():
    decision = validate_pr_created_claim_evidence(
        "PR作成は未実行です。PR本文案と作成手順までは用意できます。",
        {},
    )

    assert decision["passed"] is True
    assert decision["reason"] == "no_pr_created_claim"


def test_pr_created_claim_guard_replaces_blocked_response():
    agent_result = {"tool_trace": []}
    response = apply_pr_created_claim_guard("PR作成済みです。", agent_result)

    assert response.startswith("PR作成は未実行です。")
    assert agent_result["runtime_guards"][0]["passed"] is False
