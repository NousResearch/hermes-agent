"""Structured 400 bodies from API gateway (new-api style) channel gating.

A distributor in front of an OpenAI-compatible upstream surfaces transient
channel gating as HTTP 400 — ``{"error": {"message": "The channel has been
suspension (ID: …)", "type": "new_api_error", "code": "new_api_error"}}`` —
which is the SAME transience class as the structured ``upstream_unavailable``
code already honoured on a 403 (#75388): the identical request succeeds minutes
later, so the request shape is fine and the route must be retried with backoff
rather than aborted to the fallback chain. Pinned-route consumers that disable
fallback (workflow agent nodes) die outright on such a storm when the verdict
is ``format_error``.

``new_api_error`` is that gateway's GENERIC wrapper code — quota, model-not-
found and overflow 400s carry it too — so it only earns the transient class
alongside channel-gating prose. The specific ``channel_unavailable`` code
stands alone.
"""

from types import SimpleNamespace

from agent.error_classifier import FailoverReason, classify_api_error


class MockAPIError(Exception):
    """Simulates an OpenAI SDK APIStatusError."""

    def __init__(self, message, status_code=None, body=None, headers=None):
        super().__init__(message)
        self.status_code = status_code
        self.body = body or {}
        self.response = SimpleNamespace(headers=headers or {})


def _gateway_error(message, code):
    return MockAPIError(
        "Error code: 400",
        status_code=400,
        body={"error": {"message": message, "type": code, "code": code}},
    )


class TestGatewayChannelSuspensionIsTransient:
    def test_channel_suspension_body_retries_same_route(self):
        result = classify_api_error(
            _gateway_error("The channel has been suspension (ID: 1022)", "new_api_error"),
            provider="custom",
        )
        assert result.reason == FailoverReason.overloaded
        assert result.retryable is True
        assert result.should_rotate_credential is False
        assert result.should_fallback is False

    def test_specific_channel_unavailable_code_needs_no_prose(self):
        # The precise code identifies the class on its own — a proxy that strips
        # the prose must not fall back to format_error.
        result = classify_api_error(
            MockAPIError("Error code: 400", status_code=400,
                         body={"error": {"type": "channel_unavailable",
                                         "code": "channel_unavailable"}}),
            provider="custom",
        )
        assert result.reason == FailoverReason.overloaded
        assert result.retryable is True
        assert result.should_rotate_credential is False
        assert result.should_fallback is False

    def test_no_available_channel_wording_is_transient(self):
        result = classify_api_error(
            _gateway_error("No available channel for model gpt-x under group default",
                           "new_api_error"),
            provider="custom",
        )
        assert result.reason == FailoverReason.overloaded
        assert result.retryable is True
        assert result.should_fallback is False


class TestGenericWrapperCodeDoesNotSwallowRealFaults:
    """``new_api_error`` is the gateway's generic wrapper — the prose decides."""

    def test_generic_code_with_validation_prose_stays_format_error(self):
        result = classify_api_error(
            _gateway_error("Invalid schema for function 'x': 'temperature' is not supported",
                           "new_api_error"),
            provider="custom",
        )
        assert result.reason == FailoverReason.format_error
        assert result.retryable is False
        assert result.should_fallback is True

    def test_generic_code_wrapping_an_overflow_stays_context_overflow(self):
        # The transient gate must not preempt overflow detection: a new-api-wrapped
        # "maximum context length" 400 has to reach the compression path, never a
        # blind same-route retry.
        result = classify_api_error(
            _gateway_error("This model's maximum context length is 8192 tokens, however you "
                           "requested 10000 tokens", "new_api_error"),
            provider="custom",
        )
        assert result.reason == FailoverReason.context_overflow
        assert result.should_compress is True

    def test_generic_code_with_model_not_found_prose_stays_model_not_found(self):
        result = classify_api_error(
            _gateway_error("The model gpt-9 does not exist or you do not have access to it",
                           "new_api_error"),
            provider="custom",
        )
        assert result.reason == FailoverReason.model_not_found
        assert result.retryable is False
