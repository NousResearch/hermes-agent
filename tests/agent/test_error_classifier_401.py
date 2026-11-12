"""401 overload wrappers must back off without changing genuine auth failures."""

import pytest

from agent.error_classifier import FailoverReason, classify_api_error
from tests.agent.test_error_classifier import MockAPIError


@pytest.mark.parametrize("message,body", [
    ('HTTP 401: authentication failed (status=503): {"error":"Too many concurrent requests"}', {}),
    ("Unauthorized", {"error": {"message": "too many concurrent requests"}}),
    ("Unauthorized", {"error": "Too many concurrent requests"}),
    ("Rate limit exceeded", {}),
    ("HTTP 401: authentication failed (status=503): Service unavailable", {}),
])
def test_401_overload_is_retryable_rate_limit(message, body):
    result = classify_api_error(MockAPIError(message, status_code=401, body=body), provider="openai-codex")
    assert result.reason == FailoverReason.rate_limit
    assert result.retryable is True
    assert result.is_auth is False


@pytest.mark.parametrize("message,body", [
    ("Invalid API key", {"error": {"code": "invalid_api_key"}}),
    ("Unauthorized", {}),
    ("Token expired", {}),
    ("HTTP 401: authentication failed (status=401): Unauthorized", {}),
    ("Unauthorized: unrelated status=503", {}),
    ("HTTP 401: authentication failed (status=5030): Unauthorized", {}),
])
def test_401_auth_is_unchanged(message, body):
    result = classify_api_error(MockAPIError(message, status_code=401, body=body), provider="openai-codex")
    assert result.reason == FailoverReason.auth
    assert result.retryable is False
    assert result.should_rotate_credential is True
    assert result.should_fallback is True
