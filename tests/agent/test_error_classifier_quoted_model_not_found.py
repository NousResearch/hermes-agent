"""llama.cpp's router words a missing model ``model '<id>' not found`` (HTTP 400).

The id sits between "model" and "not found", which no plain substring matched, so the turn was
reported as a malformed request ("rejected this request") instead of a model that isn't served.
"""

from types import SimpleNamespace

from agent.error_classifier import FailoverReason, classify_api_error


class _APIError(Exception):
    def __init__(self, message, status_code):
        super().__init__(message)
        self.status_code = status_code
        self.body = {}
        self.response = SimpleNamespace(headers={})


def test_400_quoted_model_not_found_is_model_not_found():
    result = classify_api_error(_APIError("model 'Big-33B-Q4_K_M' not found", 400))
    assert result.reason == FailoverReason.model_not_found
    assert result.should_fallback is True
    assert result.retryable is False


def test_400_quoted_non_model_not_found_stays_format_error():
    """The quoted-id rule is anchored on the word "model": a missing tool is still a bad request."""
    assert classify_api_error(_APIError("tool 'browser_click' not found", 400)).reason == FailoverReason.format_error
