"""Comprehensive tests for agent.provider_errors — structured error classification.

Covers:
- Every ProviderErrorReason classification path
- Edge cases: generic 400 + large session, Anthropic OAuth generic 400,
  server disconnect heuristic, long-context tier gate
- Retryable vs non-retryable classification
- suggested_action() for each reason
- Priority ordering (e.g. BILLING before RATE_LIMIT for 429 + extra usage)
- RateLimitState tracking and pre-emptive throttling
- parse_rate_limit_headers() with various response shapes
"""

import time
from types import SimpleNamespace

import pytest

from agent.provider_errors import (
    ProviderError,
    ProviderErrorReason,
    RateLimitState,
    classify_provider_error,
    is_retryable,
    parse_rate_limit_headers,
    suggested_action,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class FakeAPIError(Exception):
    """Simulates an API error with status_code and optional body."""
    def __init__(self, message, status_code=None, body=None):
        super().__init__(message)
        self.status_code = status_code
        self.body = body


class FakeReadError(Exception):
    """Simulates httpcore.ReadError — no status_code."""
    pass

# Trick: we need to test error_type name matching, so create classes with
# the exact names the classifier checks.

def _make_named_error(class_name, message):
    """Create an exception with a specific class name for type-name matching."""
    cls = type(class_name, (Exception,), {})
    return cls(message)


# =========================================================================
# 1. BILLING — Anthropic long-context tier gate
# =========================================================================

class TestBillingClassification:
    def test_429_extra_usage_long_context(self):
        err = FakeAPIError("Extra usage is required for long context requests", status_code=429)
        result = classify_provider_error(
            err, error_msg=str(err).lower(), status_code=429, model="anthropic/claude-3.5-sonnet"
        )
        assert result.reason == ProviderErrorReason.BILLING
        assert result.retryable is False

    def test_429_extra_usage_without_long_context_is_rate_limit(self):
        """429 with 'extra usage' but without 'long context' is a rate limit."""
        err = FakeAPIError("Extra usage tier required", status_code=429)
        result = classify_provider_error(err, error_msg=str(err).lower(), status_code=429)
        assert result.reason == ProviderErrorReason.RATE_LIMIT

    def test_billing_preserves_original_error(self):
        err = FakeAPIError("Extra usage for long context", status_code=429)
        result = classify_provider_error(err, error_msg=str(err).lower(), status_code=429)
        assert result.original_error is err


# =========================================================================
# 2. RATE_LIMIT
# =========================================================================

class TestRateLimitClassification:
    def test_status_429(self):
        err = FakeAPIError("Too many requests", status_code=429)
        result = classify_provider_error(err, error_msg=str(err).lower(), status_code=429)
        assert result.reason == ProviderErrorReason.RATE_LIMIT
        assert result.retryable is True

    def test_rate_limit_in_message(self):
        err = FakeAPIError("You hit a rate limit, please slow down")
        result = classify_provider_error(err, error_msg=str(err).lower())
        assert result.reason == ProviderErrorReason.RATE_LIMIT

    def test_too_many_requests_in_message(self):
        err = FakeAPIError("Too many requests to the API")
        result = classify_provider_error(err, error_msg=str(err).lower())
        assert result.reason == ProviderErrorReason.RATE_LIMIT

    def test_rate_limit_underscore(self):
        err = FakeAPIError("rate_limit exceeded")
        result = classify_provider_error(err, error_msg=str(err).lower())
        assert result.reason == ProviderErrorReason.RATE_LIMIT

    def test_usage_limit(self):
        err = FakeAPIError("usage limit exceeded for this key")
        result = classify_provider_error(err, error_msg=str(err).lower())
        assert result.reason == ProviderErrorReason.RATE_LIMIT

    def test_quota(self):
        err = FakeAPIError("quota exceeded for project")
        result = classify_provider_error(err, error_msg=str(err).lower())
        assert result.reason == ProviderErrorReason.RATE_LIMIT


# =========================================================================
# 3. PAYLOAD_TOO_LARGE
# =========================================================================

class TestPayloadTooLargeClassification:
    def test_status_413(self):
        err = FakeAPIError("Request entity too large", status_code=413)
        result = classify_provider_error(err, error_msg=str(err).lower(), status_code=413)
        assert result.reason == ProviderErrorReason.PAYLOAD_TOO_LARGE
        assert result.retryable is True

    def test_request_entity_too_large_message(self):
        err = FakeAPIError("request entity too large for this endpoint")
        result = classify_provider_error(err, error_msg=str(err).lower())
        # Without status code, context-length phrases match first since
        # "request entity too large" is in _CONTEXT_LENGTH_PHRASES too.
        assert result.reason in (ProviderErrorReason.PAYLOAD_TOO_LARGE, ProviderErrorReason.CONTEXT_OVERFLOW)

    def test_payload_too_large_message(self):
        err = FakeAPIError("payload too large")
        result = classify_provider_error(err, error_msg=str(err).lower())
        assert result.reason == ProviderErrorReason.PAYLOAD_TOO_LARGE

    def test_error_code_413_in_message(self):
        err = FakeAPIError("Server returned error code: 413 entity too large")
        result = classify_provider_error(err, error_msg=str(err).lower())
        assert result.reason == ProviderErrorReason.PAYLOAD_TOO_LARGE


# =========================================================================
# 4. CONTEXT_OVERFLOW — explicit keywords
# =========================================================================

class TestContextOverflowClassification:
    @pytest.mark.parametrize("phrase", [
        "context length exceeded",
        "context size has been exceeded",
        "maximum context window",
        "token limit reached",
        "too many tokens in request",
        "reduce the length of the messages",
        "exceeds the limit of 128000",
        "context window exceeded",
        "prompt is too long: 200000 tokens > 128000 maximum",
        "prompt exceeds max length",
    ])
    def test_context_length_phrases(self, phrase):
        err = FakeAPIError(phrase, status_code=400)
        result = classify_provider_error(
            err, error_msg=phrase.lower(), status_code=400
        )
        assert result.reason == ProviderErrorReason.CONTEXT_OVERFLOW
        assert result.retryable is True

    def test_context_overflow_no_status_code(self):
        err = FakeAPIError("context length exceeded")
        result = classify_provider_error(err, error_msg="context length exceeded")
        assert result.reason == ProviderErrorReason.CONTEXT_OVERFLOW


# =========================================================================
# 5. CONTEXT_OVERFLOW — generic 400 + large session heuristic
# =========================================================================

class TestGeneric400LargeSessionHeuristic:
    def test_generic_400_large_tokens(self):
        err = FakeAPIError("Error", status_code=400)
        result = classify_provider_error(
            err, error_msg="error", status_code=400,
            approx_tokens=100_000, num_messages=10, context_length=200_000,
        )
        assert result.reason == ProviderErrorReason.CONTEXT_OVERFLOW

    def test_generic_400_many_messages(self):
        err = FakeAPIError("Error", status_code=400)
        result = classify_provider_error(
            err, error_msg="error", status_code=400,
            approx_tokens=1000, num_messages=100, context_length=200_000,
        )
        assert result.reason == ProviderErrorReason.CONTEXT_OVERFLOW

    def test_generic_400_small_session_not_context_overflow(self):
        """Small session + generic 400 should NOT be context overflow."""
        err = FakeAPIError("Error", status_code=400)
        result = classify_provider_error(
            err, error_msg="error", status_code=400,
            approx_tokens=1000, num_messages=5, context_length=200_000,
        )
        # Should fall through to the Anthropic OAuth generic 400 check
        assert result.reason != ProviderErrorReason.CONTEXT_OVERFLOW

    def test_generic_400_long_error_message_not_heuristic(self):
        """If the error message is long/descriptive, don't use heuristic."""
        err = FakeAPIError("This is a detailed error message about something specific", status_code=400)
        result = classify_provider_error(
            err, error_msg=str(err).lower(), status_code=400,
            approx_tokens=100_000, num_messages=100, context_length=200_000,
        )
        # Long message → not generic → doesn't trigger heuristic
        assert result.reason != ProviderErrorReason.CONTEXT_OVERFLOW


# =========================================================================
# 6. CONTEXT_OVERFLOW — server disconnect + large session heuristic
# =========================================================================

class TestServerDisconnectHeuristic:
    def test_server_disconnected_large_session(self):
        err = FakeAPIError("Server disconnected without sending a response")
        result = classify_provider_error(
            err, error_msg=str(err).lower(), status_code=None,
            approx_tokens=150_000, num_messages=50, context_length=200_000,
        )
        assert result.reason == ProviderErrorReason.CONTEXT_OVERFLOW

    def test_peer_closed_large_session(self):
        err = FakeAPIError("peer closed connection without response")
        result = classify_provider_error(
            err, error_msg=str(err).lower(), status_code=None,
            approx_tokens=150_000, num_messages=50, context_length=200_000,
        )
        assert result.reason == ProviderErrorReason.CONTEXT_OVERFLOW

    def test_read_error_type_large_session(self):
        err = _make_named_error("ReadError", "connection reset")
        result = classify_provider_error(
            err, error_msg=str(err).lower(), status_code=None,
            approx_tokens=150_000, num_messages=50, context_length=200_000,
        )
        assert result.reason == ProviderErrorReason.CONTEXT_OVERFLOW

    def test_remote_protocol_error_type_large_session(self):
        err = _make_named_error("RemoteProtocolError", "peer closed connection")
        result = classify_provider_error(
            err, error_msg=str(err).lower(), status_code=None,
            approx_tokens=150_000, num_messages=50, context_length=200_000,
        )
        assert result.reason == ProviderErrorReason.CONTEXT_OVERFLOW

    def test_server_disconnected_error_type_large_session(self):
        err = _make_named_error("ServerDisconnectedError", "disconnected")
        result = classify_provider_error(
            err, error_msg=str(err).lower(), status_code=None,
            approx_tokens=150_000, num_messages=50, context_length=200_000,
        )
        assert result.reason == ProviderErrorReason.CONTEXT_OVERFLOW

    def test_server_disconnected_small_session_is_stream_drop(self):
        """Small session disconnect should be STREAM_DROP, not CONTEXT_OVERFLOW."""
        err = FakeAPIError("Server disconnected without sending a response")
        result = classify_provider_error(
            err, error_msg=str(err).lower(), status_code=None,
            approx_tokens=1000, num_messages=5, context_length=200_000,
        )
        assert result.reason == ProviderErrorReason.STREAM_DROP


# =========================================================================
# 7. OVERLOADED (529)
# =========================================================================

class TestOverloadedClassification:
    def test_status_529(self):
        err = FakeAPIError("Overloaded", status_code=529)
        result = classify_provider_error(err, error_msg=str(err).lower(), status_code=529)
        assert result.reason == ProviderErrorReason.OVERLOADED
        assert result.retryable is True


# =========================================================================
# 8. MODEL_NOT_FOUND
# =========================================================================

class TestModelNotFoundClassification:
    def test_not_valid_model(self):
        err = FakeAPIError("'gpt-5' is not a valid model for this endpoint")
        result = classify_provider_error(err, error_msg=str(err).lower())
        assert result.reason == ProviderErrorReason.MODEL_NOT_FOUND
        assert result.retryable is False

    def test_invalid_model(self):
        err = FakeAPIError("Invalid model: xyz-model")
        result = classify_provider_error(err, error_msg=str(err).lower())
        assert result.reason == ProviderErrorReason.MODEL_NOT_FOUND

    def test_model_not_found_message(self):
        err = FakeAPIError("Model not found: some-model")
        result = classify_provider_error(err, error_msg=str(err).lower())
        assert result.reason == ProviderErrorReason.MODEL_NOT_FOUND


# =========================================================================
# 9. AUTH_PERMANENT
# =========================================================================

class TestAuthPermanentClassification:
    def test_invalid_api_key(self):
        err = FakeAPIError("Invalid API key provided")
        result = classify_provider_error(err, error_msg=str(err).lower())
        assert result.reason == ProviderErrorReason.AUTH_PERMANENT
        assert result.retryable is False

    def test_invalid_api_key_underscore(self):
        err = FakeAPIError("invalid_api_key")
        result = classify_provider_error(err, error_msg=str(err).lower())
        assert result.reason == ProviderErrorReason.AUTH_PERMANENT

    def test_authentication_error(self):
        err = FakeAPIError("Authentication failed for this request")
        result = classify_provider_error(err, error_msg=str(err).lower())
        assert result.reason == ProviderErrorReason.AUTH_PERMANENT


# =========================================================================
# 10. AUTH (401/403)
# =========================================================================

class TestAuthClassification:
    def test_status_401(self):
        err = FakeAPIError("Unauthorized", status_code=401)
        result = classify_provider_error(err, error_msg="unauthorized", status_code=401)
        # "unauthorized" phrase triggers AUTH_PERMANENT before the 401 check
        # BUT in the original code, 401 is handled by credential refresh first,
        # then by the generic 4xx check. Let's verify classification:
        # "unauthorized" in error_msg -> hits "unauthorized" in the 4xx phrase list
        # but AUTH_PERMANENT check for "invalid api key"/"authentication" doesn't match "unauthorized"
        # So 401 status_code → AUTH
        assert result.reason == ProviderErrorReason.AUTH
        assert result.retryable is False

    def test_status_403(self):
        err = FakeAPIError("Forbidden", status_code=403)
        result = classify_provider_error(err, error_msg="forbidden", status_code=403)
        assert result.reason == ProviderErrorReason.AUTH

    def test_401_with_authentication_message(self):
        """401 + 'authentication' → AUTH_PERMANENT takes priority."""
        err = FakeAPIError("Authentication error: invalid token", status_code=401)
        result = classify_provider_error(
            err, error_msg=str(err).lower(), status_code=401,
        )
        assert result.reason == ProviderErrorReason.AUTH_PERMANENT


# =========================================================================
# 11. FORMAT_ERROR — ValueError/TypeError
# =========================================================================

class TestFormatErrorClassification:
    def test_value_error(self):
        err = ValueError("invalid literal for int()")
        result = classify_provider_error(err, error_msg=str(err).lower())
        assert result.reason == ProviderErrorReason.FORMAT_ERROR
        assert result.retryable is False

    def test_type_error(self):
        err = TypeError("expected str, got int")
        result = classify_provider_error(err, error_msg=str(err).lower())
        assert result.reason == ProviderErrorReason.FORMAT_ERROR

    def test_unicode_encode_error_is_not_format_error(self):
        """UnicodeEncodeError is a ValueError subclass but should NOT be FORMAT_ERROR."""
        err = UnicodeEncodeError("utf-8", "", 0, 1, "surrogates not allowed")
        result = classify_provider_error(err, error_msg=str(err).lower())
        assert result.reason != ProviderErrorReason.FORMAT_ERROR


# =========================================================================
# 12. SERVER_ERROR — Anthropic OAuth generic 400
# =========================================================================

class TestAnthropicOAuthGeneric400:
    def test_generic_400_with_error_body(self):
        err = FakeAPIError(
            "Error",
            status_code=400,
            body={"error": {"type": "invalid_request_error", "message": "Error"}},
        )
        result = classify_provider_error(
            err, error_msg="error", status_code=400,
            approx_tokens=1000, num_messages=5, context_length=200_000,
        )
        assert result.reason == ProviderErrorReason.SERVER_ERROR
        assert result.retryable is True

    def test_generic_400_empty_message_body(self):
        err = FakeAPIError(
            "",
            status_code=400,
            body={"error": {"type": "invalid_request_error", "message": ""}},
        )
        result = classify_provider_error(
            err, error_msg="", status_code=400,
            approx_tokens=1000, num_messages=5, context_length=200_000,
        )
        assert result.reason == ProviderErrorReason.SERVER_ERROR
        assert result.retryable is True

    def test_400_with_descriptive_message_is_not_transient(self):
        """A 400 with a real descriptive message is a real client error."""
        err = FakeAPIError(
            "max_tokens must be less than 4096",
            status_code=400,
            body={"error": {"type": "invalid_request_error", "message": "max_tokens must be less than 4096"}},
        )
        result = classify_provider_error(
            err, error_msg=str(err).lower(), status_code=400,
            approx_tokens=1000, num_messages=5, context_length=200_000,
        )
        # Not generic → not SERVER_ERROR → falls to other 4xx handler
        assert result.reason == ProviderErrorReason.AUTH_PERMANENT
        assert result.retryable is False


# =========================================================================
# 13. Other 4xx → AUTH_PERMANENT
# =========================================================================

class TestOther4xxClassification:
    def test_status_404(self):
        err = FakeAPIError("Not found", status_code=404)
        result = classify_provider_error(err, error_msg="not found", status_code=404)
        # "not found" phrase matches before generic 4xx
        assert result.reason == ProviderErrorReason.AUTH_PERMANENT
        assert result.retryable is False

    def test_status_422(self):
        err = FakeAPIError("Unprocessable entity", status_code=422)
        result = classify_provider_error(err, error_msg="unprocessable entity", status_code=422)
        assert result.reason == ProviderErrorReason.AUTH_PERMANENT

    def test_error_code_phrases(self):
        err = FakeAPIError("error code: 401 unauthorized")
        result = classify_provider_error(err, error_msg=str(err).lower())
        # "error code: 401" phrase + "unauthorized" phrase → AUTH_PERMANENT
        assert result.reason == ProviderErrorReason.AUTH_PERMANENT


# =========================================================================
# 14. SERVER_ERROR (5xx)
# =========================================================================

class TestServerErrorClassification:
    def test_status_500(self):
        err = FakeAPIError("Internal server error", status_code=500)
        result = classify_provider_error(err, error_msg=str(err).lower(), status_code=500)
        assert result.reason == ProviderErrorReason.SERVER_ERROR
        assert result.retryable is True

    def test_status_502(self):
        err = FakeAPIError("Bad gateway", status_code=502)
        result = classify_provider_error(err, error_msg=str(err).lower(), status_code=502)
        assert result.reason == ProviderErrorReason.SERVER_ERROR

    def test_status_503(self):
        err = FakeAPIError("Service unavailable", status_code=503)
        result = classify_provider_error(err, error_msg=str(err).lower(), status_code=503)
        assert result.reason == ProviderErrorReason.SERVER_ERROR


# =========================================================================
# 15. STREAM_DROP — connection errors
# =========================================================================

class TestStreamDropClassification:
    @pytest.mark.parametrize("phrase", [
        "connection lost during streaming",
        "connection reset by peer",
        "connection closed unexpectedly",
        "network connection failed",
        "network error occurred",
        "request terminated by server",
    ])
    def test_stream_drop_phrases(self, phrase):
        err = FakeAPIError(phrase)
        result = classify_provider_error(err, error_msg=phrase.lower(), status_code=None)
        assert result.reason == ProviderErrorReason.STREAM_DROP
        assert result.retryable is True

    def test_connection_error_with_status_code_is_not_stream_drop(self):
        """If there's a status code, it should be classified by status, not as stream drop."""
        err = FakeAPIError("connection lost", status_code=500)
        result = classify_provider_error(err, error_msg="connection lost", status_code=500)
        assert result.reason == ProviderErrorReason.SERVER_ERROR


# =========================================================================
# 16. UNKNOWN
# =========================================================================

class TestUnknownClassification:
    def test_generic_exception(self):
        err = Exception("Something unexpected happened")
        result = classify_provider_error(err, error_msg=str(err).lower())
        assert result.reason == ProviderErrorReason.UNKNOWN
        assert result.retryable is True

    def test_runtime_error(self):
        err = RuntimeError("Unexpected internal failure")
        result = classify_provider_error(err, error_msg=str(err).lower())
        assert result.reason == ProviderErrorReason.UNKNOWN


# =========================================================================
# is_retryable() tests
# =========================================================================

class TestIsRetryable:
    @pytest.mark.parametrize("reason,expected", [
        (ProviderErrorReason.AUTH, True),
        (ProviderErrorReason.AUTH_PERMANENT, False),
        (ProviderErrorReason.RATE_LIMIT, True),
        (ProviderErrorReason.OVERLOADED, True),
        (ProviderErrorReason.BILLING, False),
        (ProviderErrorReason.MODEL_NOT_FOUND, False),
        (ProviderErrorReason.CONTEXT_OVERFLOW, True),
        (ProviderErrorReason.PAYLOAD_TOO_LARGE, True),
        (ProviderErrorReason.FORMAT_ERROR, False),
        (ProviderErrorReason.TIMEOUT, True),
        (ProviderErrorReason.SERVER_ERROR, True),
        (ProviderErrorReason.STREAM_DROP, True),
        (ProviderErrorReason.UNKNOWN, True),
    ])
    def test_retryable(self, reason, expected):
        assert is_retryable(reason) is expected


# =========================================================================
# suggested_action() tests
# =========================================================================

class TestSuggestedAction:
    @pytest.mark.parametrize("reason,expected", [
        (ProviderErrorReason.AUTH, "refresh_credentials"),
        (ProviderErrorReason.AUTH_PERMANENT, "abort"),
        (ProviderErrorReason.RATE_LIMIT, "retry_with_backoff"),
        (ProviderErrorReason.OVERLOADED, "retry_with_backoff"),
        (ProviderErrorReason.BILLING, "abort"),
        (ProviderErrorReason.MODEL_NOT_FOUND, "fallback"),
        (ProviderErrorReason.CONTEXT_OVERFLOW, "compress"),
        (ProviderErrorReason.PAYLOAD_TOO_LARGE, "compress"),
        (ProviderErrorReason.FORMAT_ERROR, "abort"),
        (ProviderErrorReason.TIMEOUT, "retry"),
        (ProviderErrorReason.SERVER_ERROR, "retry"),
        (ProviderErrorReason.STREAM_DROP, "retry"),
        (ProviderErrorReason.UNKNOWN, "retry"),
    ])
    def test_suggested_action(self, reason, expected):
        assert suggested_action(reason) == expected


# =========================================================================
# ProviderError dataclass tests
# =========================================================================

class TestProviderErrorDataclass:
    def test_defaults(self):
        pe = ProviderError(reason=ProviderErrorReason.UNKNOWN)
        assert pe.status_code is None
        assert pe.message == ""
        assert pe.retryable is True
        assert pe.provider is None
        assert pe.original_error is None

    def test_full_construction(self):
        orig = Exception("test")
        pe = ProviderError(
            reason=ProviderErrorReason.RATE_LIMIT,
            status_code=429,
            message="Rate limited",
            retryable=True,
            provider="openai",
            original_error=orig,
        )
        assert pe.reason == ProviderErrorReason.RATE_LIMIT
        assert pe.status_code == 429
        assert pe.message == "Rate limited"
        assert pe.provider == "openai"
        assert pe.original_error is orig

    def test_provider_passed_through(self):
        err = FakeAPIError("error", status_code=500)
        result = classify_provider_error(
            err, error_msg="error", status_code=500, provider="anthropic"
        )
        assert result.provider == "anthropic"


# =========================================================================
# Priority / edge case tests
# =========================================================================

class TestClassificationPriority:
    def test_billing_before_rate_limit(self):
        """429 + extra usage + long context → BILLING, not RATE_LIMIT."""
        err = FakeAPIError(
            "Extra usage is required for long context requests",
            status_code=429,
        )
        result = classify_provider_error(
            err, error_msg=str(err).lower(), status_code=429,
        )
        assert result.reason == ProviderErrorReason.BILLING

    def test_context_overflow_before_generic_4xx(self):
        """400 + context length phrase → CONTEXT_OVERFLOW, not AUTH_PERMANENT."""
        err = FakeAPIError("context length exceeded", status_code=400)
        result = classify_provider_error(
            err, error_msg=str(err).lower(), status_code=400,
        )
        assert result.reason == ProviderErrorReason.CONTEXT_OVERFLOW

    def test_payload_too_large_before_context_overflow_with_413(self):
        """413 → PAYLOAD_TOO_LARGE (not CONTEXT_OVERFLOW even though phrases overlap)."""
        err = FakeAPIError("request entity too large", status_code=413)
        result = classify_provider_error(
            err, error_msg=str(err).lower(), status_code=413,
        )
        assert result.reason == ProviderErrorReason.PAYLOAD_TOO_LARGE

    def test_model_not_found_without_status_code(self):
        """Model not found from error message without HTTP status."""
        err = Exception("The model 'xyz' is not a valid model")
        result = classify_provider_error(err, error_msg=str(err).lower())
        assert result.reason == ProviderErrorReason.MODEL_NOT_FOUND

    def test_auth_permanent_takes_priority_over_auth_for_invalid_key(self):
        """'invalid api key' with 401 → AUTH_PERMANENT, not AUTH."""
        err = FakeAPIError("Invalid API key", status_code=401)
        result = classify_provider_error(
            err, error_msg=str(err).lower(), status_code=401,
        )
        assert result.reason == ProviderErrorReason.AUTH_PERMANENT

    def test_context_overflow_heuristic_400_before_anthropic_oauth_check(self):
        """Generic 400 + large session → CONTEXT_OVERFLOW even if body is 'Error'."""
        err = FakeAPIError(
            "Error", status_code=400,
            body={"error": {"type": "invalid_request_error", "message": "Error"}},
        )
        result = classify_provider_error(
            err, error_msg="error", status_code=400,
            approx_tokens=100_000, num_messages=90, context_length=200_000,
        )
        # Large session + generic error → CONTEXT_OVERFLOW (heuristic #5)
        # takes priority over Anthropic OAuth generic 400 (heuristic #12)
        assert result.reason == ProviderErrorReason.CONTEXT_OVERFLOW

    def test_error_msg_auto_computed_if_empty(self):
        """If error_msg is not provided, it should be computed from str(error)."""
        err = FakeAPIError("Rate limit exceeded")
        result = classify_provider_error(err)
        assert result.reason == ProviderErrorReason.RATE_LIMIT


# =========================================================================
# ProviderErrorReason enum completeness
# =========================================================================

class TestEnumCompleteness:
    def test_all_reasons_have_retryable_mapping(self):
        for reason in ProviderErrorReason:
            # Should not raise
            is_retryable(reason)

    def test_all_reasons_have_action_mapping(self):
        for reason in ProviderErrorReason:
            action = suggested_action(reason)
            assert action in (
                "retry", "retry_with_backoff", "compress",
                "fallback", "refresh_credentials", "abort",
            )

    def test_enum_values(self):
        expected = {
            "AUTH", "AUTH_PERMANENT", "RATE_LIMIT", "OVERLOADED",
            "BILLING", "MODEL_NOT_FOUND", "CONTEXT_OVERFLOW",
            "PAYLOAD_TOO_LARGE", "FORMAT_ERROR", "TIMEOUT",
            "SERVER_ERROR", "STREAM_DROP", "UNKNOWN",
        }
        assert {r.name for r in ProviderErrorReason} == expected


# =========================================================================
# RateLimitState — should_throttle()
# =========================================================================

class TestRateLimitStateShouldThrottle:
    def test_below_10_percent(self):
        state = RateLimitState(remaining=5, limit=100)
        assert state.should_throttle() is True

    def test_at_10_percent(self):
        state = RateLimitState(remaining=10, limit=100)
        # 10/100 == 0.10, not < 0.10
        assert state.should_throttle() is False

    def test_above_10_percent(self):
        state = RateLimitState(remaining=50, limit=100)
        assert state.should_throttle() is False

    def test_zero_remaining(self):
        state = RateLimitState(remaining=0, limit=100)
        assert state.should_throttle() is True

    def test_remaining_none(self):
        state = RateLimitState(remaining=None, limit=100)
        assert state.should_throttle() is False

    def test_limit_none(self):
        state = RateLimitState(remaining=5, limit=None)
        assert state.should_throttle() is False

    def test_limit_zero(self):
        state = RateLimitState(remaining=0, limit=0)
        assert state.should_throttle() is False

    def test_both_none(self):
        state = RateLimitState()
        assert state.should_throttle() is False


# =========================================================================
# RateLimitState — suggested_delay()
# =========================================================================

class TestRateLimitStateSuggestedDelay:
    def test_reset_in_future(self):
        future = time.time() + 10.0
        state = RateLimitState(remaining=5, limit=100, reset_at=future)
        delay = state.suggested_delay()
        assert 9.0 < delay <= 10.0  # should be close to 10s

    def test_reset_far_future_capped_at_30(self):
        future = time.time() + 120.0
        state = RateLimitState(remaining=5, limit=100, reset_at=future)
        delay = state.suggested_delay()
        assert delay == 30.0

    def test_reset_in_past(self):
        past = time.time() - 5.0
        state = RateLimitState(remaining=5, limit=100, reset_at=past)
        # reset_at in past => falls through to should_throttle() check
        delay = state.suggested_delay()
        assert delay == 2.0  # remaining/limit = 5/100 = 5% < 10%

    def test_reset_none_throttle_needed(self):
        state = RateLimitState(remaining=3, limit=100, reset_at=None)
        delay = state.suggested_delay()
        assert delay == 2.0

    def test_reset_none_no_throttle(self):
        state = RateLimitState(remaining=50, limit=100, reset_at=None)
        delay = state.suggested_delay()
        assert delay == 0.0

    def test_all_none(self):
        state = RateLimitState()
        assert state.suggested_delay() == 0.0


# =========================================================================
# parse_rate_limit_headers()
# =========================================================================

class TestParseRateLimitHeaders:
    def test_response_with_headers_attr(self):
        resp = SimpleNamespace(headers={
            "x-ratelimit-remaining": "42",
            "x-ratelimit-limit": "1000",
            "x-ratelimit-reset": "1700000000",
        })
        state = parse_rate_limit_headers(resp)
        assert state is not None
        assert state.remaining == 42
        assert state.limit == 1000
        assert state.reset_at == 1700000000.0

    def test_response_with_capitalized_headers(self):
        resp = SimpleNamespace(headers={
            "X-RateLimit-Remaining": "10",
            "X-RateLimit-Limit": "500",
        })
        state = parse_rate_limit_headers(resp)
        assert state is not None
        assert state.remaining == 10
        assert state.limit == 500
        assert state.reset_at is None

    def test_response_with_relative_reset(self):
        resp = SimpleNamespace(headers={
            "x-ratelimit-remaining": "5",
            "x-ratelimit-limit": "100",
            "x-ratelimit-reset": "60",  # 60 seconds from now
        })
        before = time.time()
        state = parse_rate_limit_headers(resp)
        after = time.time()
        assert state is not None
        # reset_at should be approximately now + 60
        assert before + 59.5 < state.reset_at < after + 60.5

    def test_response_with_http_response_attr(self):
        inner = SimpleNamespace(headers={
            "ratelimit-remaining": "7",
            "ratelimit-limit": "200",
        })
        resp = SimpleNamespace(http_response=inner)
        state = parse_rate_limit_headers(resp)
        assert state is not None
        assert state.remaining == 7
        assert state.limit == 200

    def test_response_with_response_attr(self):
        inner = SimpleNamespace(headers={
            "x-ratelimit-remaining": "1",
            "x-ratelimit-limit": "50",
        })
        resp = SimpleNamespace(response=inner)
        state = parse_rate_limit_headers(resp)
        assert state is not None
        assert state.remaining == 1
        assert state.limit == 50

    def test_no_headers_returns_none(self):
        resp = SimpleNamespace(model="gpt-4", usage=None)
        state = parse_rate_limit_headers(resp)
        assert state is None

    def test_empty_headers_returns_none(self):
        resp = SimpleNamespace(headers={})
        state = parse_rate_limit_headers(resp)
        assert state is None

    def test_none_response(self):
        assert parse_rate_limit_headers(None) is None

    def test_non_numeric_header_values(self):
        resp = SimpleNamespace(headers={
            "x-ratelimit-remaining": "not-a-number",
            "x-ratelimit-limit": "also-not",
            "x-ratelimit-reset": "bad",
        })
        state = parse_rate_limit_headers(resp)
        assert state is None

    def test_partial_headers(self):
        resp = SimpleNamespace(headers={
            "x-ratelimit-remaining": "15",
        })
        state = parse_rate_limit_headers(resp)
        assert state is not None
        assert state.remaining == 15
        assert state.limit is None
        assert state.reset_at is None

    def test_updated_at_is_set(self):
        resp = SimpleNamespace(headers={
            "x-ratelimit-remaining": "10",
            "x-ratelimit-limit": "100",
        })
        before = time.time()
        state = parse_rate_limit_headers(resp)
        after = time.time()
        assert state is not None
        assert before <= state.updated_at <= after
