import json

from agent.error_classifier import ClassifiedError, FailoverReason
from agent.failure_policy import (
    FailureClass,
    RecoveryAction,
    decide_api_recovery,
    decide_tool_recovery,
)


def test_api_context_overflow_prefers_compression() -> None:
    err = ClassifiedError(
        reason=FailoverReason.context_overflow,
        retryable=True,
        should_compress=True,
    )

    decision = decide_api_recovery(err)

    assert decision.failure_class == FailureClass.context_overflow
    assert decision.primary_action == RecoveryAction.compress_context
    assert decision.secondary_actions == (RecoveryAction.retry_smaller_scope,)
    assert decision.retryable is True


def test_api_auth_prefers_rotation_then_fallback() -> None:
    err = ClassifiedError(
        reason=FailoverReason.auth,
        retryable=False,
        should_fallback=True,
        should_rotate_credential=True,
    )

    decision = decide_api_recovery(err)

    assert decision.failure_class == FailureClass.auth
    assert decision.primary_action == RecoveryAction.rotate_credential
    assert decision.secondary_actions == (RecoveryAction.fallback_provider,)


def test_api_content_policy_asks_user_not_retry() -> None:
    err = ClassifiedError(reason=FailoverReason.content_policy_blocked, retryable=False)

    decision = decide_api_recovery(err)

    assert decision.failure_class == FailureClass.content_policy
    assert decision.primary_action == RecoveryAction.ask_user
    assert decision.retryable is False


def test_tool_structured_nonzero_exit_prefers_degrade_then_fallback() -> None:
    result = json.dumps({"exit_code": 2, "output": "Traceback: nope"})

    decision = decide_tool_recovery("terminal", result)

    assert decision.failure_class == FailureClass.tool_failed
    assert decision.primary_action == RecoveryAction.degrade_mode
    assert decision.secondary_actions == (RecoveryAction.fallback_tool,)
    assert decision.retryable is True


def test_tool_empty_results_prefers_narrower_scope() -> None:
    result = json.dumps({"results": []})

    decision = decide_tool_recovery("web_search", result)

    assert decision.failure_class == FailureClass.empty_result
    assert decision.primary_action == RecoveryAction.retry_smaller_scope
    assert decision.secondary_actions == (RecoveryAction.fallback_tool,)


def test_tool_missing_service_is_unavailable() -> None:
    result = json.dumps({"success": False, "error": "Tool not available: Home Assistant token missing"})

    decision = decide_tool_recovery("ha_get_state", result)

    assert decision.failure_class == FailureClass.tool_unavailable
    assert decision.primary_action == RecoveryAction.fallback_tool
    assert decision.retryable is False


def test_tool_invalid_json_args_becomes_parse_error() -> None:
    result = "JSON parse error: expected object"

    decision = decide_tool_recovery("execute_code", result)

    assert decision.failure_class == FailureClass.parse_error
    assert decision.primary_action == RecoveryAction.degrade_mode
    assert decision.secondary_actions == (RecoveryAction.retry_same,)


def test_tool_plain_unknown_text_retries_then_falls_back() -> None:
    decision = decide_tool_recovery("browser_navigate", "unexpected upstream weirdness")

    assert decision.failure_class == FailureClass.unknown
    assert decision.primary_action == RecoveryAction.retry_same
    assert decision.secondary_actions == (RecoveryAction.fallback_tool,)
