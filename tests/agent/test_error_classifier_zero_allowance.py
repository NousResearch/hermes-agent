"""A 429 whose ``x-ratelimit-limit-*`` header is ``0`` means the plan grants no allowance for the
model (Mistral code 1300): no backoff refills it, so it must rotate/fall back, not wait (omp#19301)."""

import httpx
import openai
import pytest

from agent.error_classifier import FailoverReason, classify_api_error


@pytest.mark.parametrize(("limit", "reason", "retryable"), [
    ("0", FailoverReason.billing, False),  # zero allowance: waiting never helps
    ("60", FailoverReason.rate_limit, True),  # spent window with a real limit: still a backoff
])
def test_429_zero_ratelimit_limit_is_not_a_wait(limit, reason, retryable):
    body = {"object": "error", "message": "Rate limit exceeded", "type": "rate_limited", "code": "1300"}
    response = httpx.Response(
        429,
        request=httpx.Request("POST", "https://api.mistral.ai/v1/chat/completions"),
        headers={"x-ratelimit-limit-req-minute": limit, "x-ratelimit-remaining-req-minute": "0"},
        json=body,
    )
    error = openai.RateLimitError(f"Error code: 429 - {body}", response=response, body=body)

    result = classify_api_error(error, provider="mistral", model="mistral-medium-latest")

    assert (result.reason, result.retryable) == (reason, retryable)
    assert result.should_fallback
