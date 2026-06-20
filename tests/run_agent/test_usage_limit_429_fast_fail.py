"""Regression tests for account usage-limit 429 handling.

A plan/account usage cap that resets hours later is not a transient per-minute
rate limit. Retrying the same request burns requests/tokens and still cannot
succeed before the reset window.
"""

from types import SimpleNamespace

from agent.error_classifier import FailoverReason, classify_api_error


class _UsageLimit429(Exception):
    status_code = 429

    def __init__(self):
        super().__init__("HTTP 429: The usage limit has been reached")
        self.body = {
            "error": {
                "type": "usage_limit_reached",
                "message": "The usage limit has been reached",
                "resets_in_seconds": 14_000,
            }
        }
        self.response = SimpleNamespace(json=lambda: self.body)


def test_usage_limit_429_is_not_retried_as_transient_rate_limit():
    classified = classify_api_error(
        _UsageLimit429(),
        provider="openai-codex",
        model="gpt-5.5",
    )

    assert classified.reason == FailoverReason.billing
    assert classified.retryable is False
    assert classified.should_rotate_credential is True
    assert classified.should_fallback is True


class _Normal429(Exception):
    status_code = 429

    def __init__(self):
        super().__init__("HTTP 429: rate limit exceeded, retry later")
        self.body = {
            "error": {
                "type": "rate_limit_exceeded",
                "message": "Rate limit exceeded; retry later",
            }
        }
        self.response = SimpleNamespace(json=lambda: self.body)


def test_normal_429_still_retries_as_rate_limit():
    classified = classify_api_error(
        _Normal429(),
        provider="openai-codex",
        model="gpt-5.5",
    )

    assert classified.reason == FailoverReason.rate_limit
    assert classified.retryable is True
