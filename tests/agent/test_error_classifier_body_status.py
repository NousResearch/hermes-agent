"""Body-carried HTTP status extraction — an in-stream SSE error object's numeric
``code`` must classify like the equivalent HTTP response (#121270)."""

from types import SimpleNamespace

from agent.error_classifier import (
    FailoverReason,
    classify_api_error,
    _extract_status_code,
)


class MockAPIError(Exception):
    """Simulates a status-less OpenAI SDK APIError raised mid-stream."""

    def __init__(self, message, status_code=None, body=None, headers=None):
        super().__init__(message)
        self.status_code = status_code
        self.body = body or {}
        self.response = SimpleNamespace(headers=headers or {})


_BAN_BODY = {
    "error": {
        "code": 403,
        "message": "Your account has been banned by the upstream provider",
        "metadata": {"provider_name": "acme"},
    }
}


class TestBodyCarriedStatusExtraction:
    def test_top_level_code_and_status_keys_also_count(self):
        assert _extract_status_code(MockAPIError("x", body={"code": 503})) == 503
        assert (
            _extract_status_code(
                MockAPIError("x", body={"error": {"http_status": 502}})
            )
            == 502
        )


class TestInStreamErrorClassification:
    def test_403_ban_is_auth_not_transient_retry(self):
        result = classify_api_error(
            MockAPIError("Error code: 403", body=_BAN_BODY), provider="custom"
        )
        assert result.status_code == 403
        assert result.reason == FailoverReason.auth
        assert result.retryable is False
        assert result.should_fallback is True
