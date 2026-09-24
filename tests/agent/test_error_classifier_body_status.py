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
    def test_in_stream_error_code_is_extracted_as_status(self):
        assert (
            _extract_status_code(MockAPIError("Error code: 403", body=_BAN_BODY)) == 403
        )

    def test_exception_status_still_wins_over_body_code(self):
        err = MockAPIError("Too Many Requests", status_code=429, body=_BAN_BODY)
        assert _extract_status_code(err) == 429

    def test_symbolic_string_codes_are_not_statuses(self):
        body = {"error": {"code": "insufficient_quota", "message": "quota exceeded"}}
        assert _extract_status_code(MockAPIError("quota", body=body)) is None

    def test_out_of_range_and_non_int_codes_are_ignored(self):
        for code in (99, 600, "403", True, 403.0):
            body = {"error": {"code": code, "message": "x"}}
            assert _extract_status_code(MockAPIError("x", body=body)) is None, code

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

    def test_502_in_stream_error_stays_retryable(self):
        body = {"error": {"code": 502, "message": "upstream connect error"}}
        result = classify_api_error(
            MockAPIError("Error code: 502", body=body), provider="custom"
        )
        assert result.status_code == 502
        assert result.retryable is True

    def test_status_less_body_less_error_stays_unknown(self):
        result = classify_api_error(
            MockAPIError("weird failure", body={}), provider="custom"
        )
        assert result.status_code is None
        assert result.reason == FailoverReason.unknown
