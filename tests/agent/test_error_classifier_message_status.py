"""Last-resort HTTP status recovery from error message text (#66358)."""

from agent.error_classifier import (
    FailoverReason,
    classify_api_error,
    _extract_status_code_from_message,
)


class TestExtractStatusCodeFromMessage:
    """Last-resort status recovery for in-stream error frames.

    Routers/proxies (e.g. continuum-router) can deliver an upstream failure
    as an error frame inside an HTTP-200 SSE stream; the raised exception
    then has no status attribute and only the message carries the code.
    """

    def test_client_error_http_pattern(self):
        assert _extract_status_code_from_message("client error: http 400") == 400

    def test_http_status_pattern(self):
        assert _extract_status_code_from_message("upstream http status 502") == 502

    def test_status_code_pattern(self):
        assert _extract_status_code_from_message("failed with status code: 503") == 503

    def test_error_code_pattern(self):
        assert _extract_status_code_from_message("error code: 429 - rate limited") == 429

    def test_rejects_success_status(self):
        """2xx/3xx mentions are not failure statuses."""
        assert _extract_status_code_from_message("expected http 200 but got eof") is None

    def test_rejects_bare_numbers(self):
        """Numbers without a status/HTTP prefix must not match."""
        assert _extract_status_code_from_message("generated 400 tokens in 429 ms") is None

    def test_returns_none_for_plain_message(self):
        assert _extract_status_code_from_message("connection reset by peer") is None

    def test_classify_in_stream_400_fails_fast(self):
        """The end-to-end case from the report: a proxied 400 delivered as an
        in-stream error frame must classify as a non-retryable format_error
        instead of unknown/retryable (which burned all retries)."""
        e = Exception("Client error: HTTP 400")
        result = classify_api_error(e, provider="aigo-router", model="local-model")
        assert result.status_code == 400
        assert result.reason == FailoverReason.format_error
        assert result.retryable is False

    def test_classify_plain_message_still_unknown(self):
        """No status signal in the message keeps the retryable unknown path."""
        e = Exception("something odd happened")
        result = classify_api_error(e, provider="p", model="m")
        assert result.status_code is None
        assert result.reason == FailoverReason.unknown
        assert result.retryable is True
