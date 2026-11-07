"""Extracted SDK error shapes preserve provider-owned auxiliary retry decisions."""

import pytest


@pytest.mark.parametrize("status,code,transient,rate_limited", [
    (503, "ServiceUnavailableException", True, False),
    (400, "ThrottlingException", False, True),
])
def test_structured_sdk_status_reaches_auxiliary_retry_classifiers(status, code, transient, rate_limited):
    from agent.auxiliary_client import _exc_http_status, _is_rate_limit_error, _is_transient_transport_error

    error = RuntimeError("Structured SDK refusal")
    error.response = {"ResponseMetadata": {"HTTPStatusCode": status}, "Error": {"Code": code}}
    assert _exc_http_status(error) == status
    assert _is_transient_transport_error(error) is transient
    assert _is_rate_limit_error(error) is rate_limited


@pytest.mark.parametrize("status", [429, 503])
def test_plan_policy_denial_is_not_retried_as_generic_transient_or_pool_recovery(status):
    from agent.auxiliary_client import _is_transient_transport_error
    from agent.auxiliary_error_policy import require_provider_recovery
    from run_agent import _StreamErrorEvent

    error = _StreamErrorEvent("App allowance exhausted", status_code=status,
                              code="subscription_sharing_usage_limit_exceeded")
    assert not _is_transient_transport_error(error, "openai-chatgpt", model="gpt-test",
                                              base_url="https://api.openai.com/v1")
    with pytest.raises(type(error)) as caught:
        require_provider_recovery(error, "openai-chatgpt", model="gpt-test",
                                  base_url="https://api.openai.com/v1")
    assert caught.value is error
