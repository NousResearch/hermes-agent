"""A response marked ``x-should-retry: false`` is not replayed by hermes' own retry loop.

A gateway that already ran its retries (Hermes' API server for ``admission_queued`` included)
answers 503 + ``x-should-retry: false``. The vendor Python SDKs honour that header, but hermes
builds them with ``max_retries=0`` and retries in its own loop, which read only the status.
"""
import anthropic
import httpx
import openai
import pytest

from agent.error_classifier import RETRYABLE_CLIENT_REASONS, FailoverReason, classify_api_error


def _chat_error(status, headers):
    req = httpx.Request("POST", "http://127.0.0.1:9/v1/chat/completions")
    body = {"error": {"message": "upstream unavailable", "type": "server_error"}}
    resp = httpx.Response(status, headers=headers, json=body, request=req)
    return openai.APIStatusError(f"Error code: {status} - upstream unavailable", response=resp, body=body)


def _messages_error(status, headers):
    req = httpx.Request("POST", "http://127.0.0.1:9/v1/messages")
    body = {"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}}
    resp = httpx.Response(status, headers=headers, json=body, request=req)
    return anthropic.APIStatusError("Error code: 529 - Overloaded", response=resp, body=body)


@pytest.mark.parametrize("make, status", [(_chat_error, 503), (_messages_error, 529)])
def test_should_retry_false_ends_the_replay_but_keeps_fallback(make, status):
    result = classify_api_error(make(status, {"x-should-retry": "false"}), provider="custom")
    # The retry loop backs off on RETRYABLE_CLIENT_REASONS regardless of ``retryable``.
    assert result.retryable is False and result.reason not in RETRYABLE_CLIENT_REASONS
    assert result.should_fallback is True


@pytest.mark.parametrize("make, status, headers, reason", [
    (_chat_error, 503, {}, FailoverReason.overloaded),
    (_chat_error, 503, {"x-should-retry": "true"}, FailoverReason.overloaded),
    (_messages_error, 529, {}, FailoverReason.overloaded),
    # A rate limit's recovery (rotation / reset wait) changes the next request: the header leaves it alone.
    (_chat_error, 429, {"x-should-retry": "false"}, FailoverReason.rate_limit),
])
def test_other_verdicts_unchanged(make, status, headers, reason):
    result = classify_api_error(make(status, headers), provider="custom")
    assert result.reason == reason and result.retryable is True
