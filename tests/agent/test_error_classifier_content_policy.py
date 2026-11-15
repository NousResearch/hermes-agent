"""Provider safety-filter refusals classify as content_policy_blocked, never as a retryable outage."""

import pytest

from agent.error_classifier import FailoverReason, classify_api_error


class _APIError(Exception):
    def __init__(self, message, status_code=None):
        super().__init__(message)
        self.status_code = status_code
        self.body = {}


@pytest.mark.parametrize("status, message", [
    # Output guardrail relayed by OpenRouter / the Nous Portal as an in-band 502 marked provider_unavailable.
    (502, "Upstream error from Alibaba: Output data may contain inappropriate content."),
    (None, "Upstream error from Alibaba: Output data may contain inappropriate content."),
    # Input guardrail: DashScope's documented 400 (code data_inspection_failed).
    (400, "Error code: 400 - {'error': {'message': 'Input data may contain inappropriate content.', "
          "'type': 'data_inspection_failed', 'code': 'data_inspection_failed'}}"),
])
def test_alibaba_guardrail_is_a_content_policy_block_not_an_outage(status, message):
    """Alibaba Model Studio's guardrail refuses this content; retrying the same request repeats the
    refusal, so it must not ride the 5xx / unknown retry ladder."""
    result = classify_api_error(_APIError(message, status), provider="nous", model="qwen/qwen3.8-flash")
    assert result.reason == FailoverReason.content_policy_blocked
    assert result.retryable is False
