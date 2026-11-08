"""Extracted SDK error shapes preserve provider-owned auxiliary retry decisions."""

import pytest


@pytest.mark.parametrize("status,code,transient,rate_limited", [
    (503, "ServiceUnavailableException", True, False),
    (400, "ThrottlingException", False, True),
])
def test_structured_sdk_status_reaches_auxiliary_retry_classifiers(status, code, transient, rate_limited):
    from agent import auxiliary_error_policy as policy

    # ClientError carries both ResponseMetadata and the symbolic code in its SDK message.
    error = RuntimeError(f"An error occurred ({code}) when calling the Converse operation: Structured SDK refusal")
    error.response = {"ResponseMetadata": {"HTTPStatusCode": status}, "Error": {"Code": code}}
    verdict = policy.classify_attempt(error, "bedrock")
    assert verdict.status_code == status
    assert policy.same_provider_retry(verdict) is transient
    assert policy.same_key_retry(verdict) is rate_limited


@pytest.mark.parametrize("status", [429, 503])
def test_plan_policy_denial_is_not_retried_as_generic_transient_or_pool_recovery(status):
    from agent import auxiliary_error_policy as policy
    from run_agent import _StreamErrorEvent

    error = _StreamErrorEvent("App allowance exhausted", status_code=status,
                              code="subscription_sharing_usage_limit_exceeded")
    verdict = policy.classify_attempt(error, "openai-chatgpt", model="gpt-test",
                                       base_url="https://api.openai.com/v1")
    failure = policy.Failure(error, verdict)
    assert not policy.same_provider_retry(verdict)
    assert not policy.ladder_accepts(failure)
    assert policy.resend_cannot_help(verdict)
