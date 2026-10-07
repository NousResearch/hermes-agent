"""OpenRouter guardrail routing failures must skip retries and fall back (#133850)."""

from types import SimpleNamespace

import pytest

from agent.error_classifier import FailoverReason, classify_api_error


class _APIError(Exception):
    def __init__(self, message, body):
        super().__init__(message)
        self.status_code = 404
        self.body = body
        self.response = SimpleNamespace(headers={})


@pytest.mark.parametrize(
    "metadata",
    [
        {"failed_routing_step": "Filter by Guardrails"},
        {"ineligibility_reasons": [{"reason": "model-ignored-by-guardrail"}]},
        {"ineligibility_reasons": [{"reason": "zdr-violation-by-account"}]},
    ],
)
def test_openrouter_guardrail_metadata_falls_back_without_retry(metadata):
    error = _APIError(
        "routing failed",
        {"error": {"message": "routing failed", "code": 404, "metadata": metadata}},
    )

    result = classify_api_error(
        error,
        provider="openrouter",
        model="nvidia/nemotron-3.5-lightning",
    )

    assert result.reason is FailoverReason.provider_policy_blocked
    assert result.retryable is False
    assert result.should_fallback is True
    assert result.should_rotate_credential is False


def test_openrouter_current_guardrail_wording_falls_back_without_retry():
    message = (
        "0 endpoints out of 3 requested are available matching your guardrail "
        "restrictions and data policy. Model blocked by guardrail: 3 endpoints excluded"
    )
    error = _APIError(message, {"error": {"message": message, "code": 404}})

    result = classify_api_error(
        error,
        provider="openrouter",
        model="nvidia/nemotron-3.5-lightning",
    )

    assert result.reason is FailoverReason.provider_policy_blocked
    assert result.retryable is False
    assert result.should_fallback is True
