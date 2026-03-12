from agent.self_correction import (
    build_shadow_critique_event,
    classify_failure,
    get_retry_policy,
)


def test_classify_provider_failure_from_retryable_timeout():
    result = classify_failure(
        tool_name="web_extract",
        error_type="TimeoutError",
        error_summary="request timeout from upstream",
        retryable_hint=True,
    )
    assert result.failure_class == "provider_failure"
    assert result.retryable is True


def test_classify_unsafe_action_block_for_permission_denied():
    result = classify_failure(
        tool_name="terminal",
        error_type="PermissionError",
        error_summary="permission denied writing /etc/hosts",
        retryable_hint=False,
    )
    assert result.failure_class == "unsafe_action_block"
    assert result.retryable is False


def test_build_shadow_critique_event_contains_recommendation():
    payload = build_shadow_critique_event(
        tool_name="browser_click",
        error_type="ToolExecutionError",
        error_summary="element not found",
        retryable_hint=False,
    )
    assert payload["shadow_mode"] is True
    assert payload["failure_class"] in {"missing_context", "tool_misuse"}
    assert isinstance(payload["recommendation"], str) and payload["recommendation"]


def test_retry_policy_provider_tool_gets_two_retries():
    failure = classify_failure(
        tool_name="web_extract",
        error_type="TimeoutError",
        error_summary="connection timeout",
        retryable_hint=True,
    )
    policy = get_retry_policy(tool_name="web_extract", failure=failure)
    assert policy.max_retries == 2
    assert policy.reason == "provider_backoff"


def test_retry_policy_unsafe_block_has_zero_retries():
    failure = classify_failure(
        tool_name="write_file",
        error_type="PermissionError",
        error_summary="permission denied",
        retryable_hint=False,
    )
    policy = get_retry_policy(tool_name="write_file", failure=failure)
    assert policy.max_retries == 0
