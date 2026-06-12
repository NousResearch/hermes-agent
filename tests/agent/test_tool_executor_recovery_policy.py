from agent.failure_policy import RecoveryAction
from agent.tool_executor import _tool_failure_recovery_decision


def test_tool_failure_recovery_decision_for_empty_search_result() -> None:
    decision = _tool_failure_recovery_decision("web_search", '{"results": []}')

    assert decision is not None
    assert decision.primary_action == RecoveryAction.retry_smaller_scope


def test_tool_failure_recovery_decision_for_unavailable_tool() -> None:
    decision = _tool_failure_recovery_decision(
        "ha_get_state",
        '{"success": false, "error": "Tool not available: Home Assistant token missing"}',
    )

    assert decision is not None
    assert decision.primary_action == RecoveryAction.fallback_tool
    assert decision.retryable is False


def test_tool_failure_recovery_decision_returns_none_for_success() -> None:
    assert _tool_failure_recovery_decision("read_file", '{"content": "hello"}') is None
