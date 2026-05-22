"""Tests for agent.error_classifier — structured API error classification."""

import pytest
from agent.error_classifier import (
    ClassifiedError,
    FailoverReason,
    classify_api_error,
    _extract_status_code,
    _extract_error_body,
    _extract_error_code,
    _classify_402,
)


# ── Helper: mock API errors ────────────────────────────────────────────

class MockAPIError(Exception):
    """Simulates an OpenAI SDK APIStatusError."""
    def __init__(self, message, status_code=None, body=None):
        super().__init__(message)
        self.status_code = status_code
        self.body = body or {}


class MockTransportError(Exception):
    """Simulates a transport-level error with a specific type name."""
    pass


class ReadTimeout(MockTransportError):
    pass


class ConnectError(MockTransportError):
    pass


class RemoteProtocolError(MockTransportError):
    pass


class ServerDisconnectedError(MockTransportError):
    pass


# ── Test: FailoverReason enum ──────────────────────────────────────────

class TestFailoverReason:
    def test_all_reasons_have_string_values(self):
        for reason in FailoverReason:
            assert isinstance(reason.value, str)

    def test_enum_members_exist(self):
        expected = {
            "auth", "auth_permanent", "billing", "rate_limit",
            "overloaded", "server_error", "timeout",
            "context_overflow", "payload_too_large", "image_too_large",
            "model_not_found", "format_error",
            "provider_policy_blocked",
            "thinking_signature", "long_context_tier",
            "oauth_long_context_beta_forbidden",
            "llama_cpp_grammar_pattern",
            "unknown",
        }
        actual = {r.value for r in FailoverReason}
        assert expected == actual


# ── Test: ClassifiedError ──────────────────────────────────────────────

class TestClassifiedError:
    def test_is_auth_property(self):
        e1 = ClassifiedError(reason=FailoverReason.auth)
        assert e1.is_auth is True

        e2 = ClassifiedError(reason=FailoverReason.auth_permanent)
        assert e2.is_auth is True

        e3 = ClassifiedError(reason=FailoverReason.billing)
        assert e3.is_auth is False

    def test_defaults(self):
        e = ClassifiedError(reason=FailoverReason.unknown)
        assert e.retryable is True
        assert e.should_compress is False
        assert e.should_rotate_credential is False
        assert e.should_fallback is False
        assert e.status_code is None
        assert e.message == ""


# ── Test: Status code extraction ───────────────────────────────────────

class TestExtractStatusCode:
    def test_from_status_code_attr(self):
        e = MockAPIError("fail", status_code=429)
        assert _extract_status_code(e) == 429

    def test_from_status_attr(self):
        class ErrWithStatus(Exception):
            status = 503
        assert _extract_status_code(ErrWithStatus()) == 503

    def test_from_cause_chain(self):
        inner = MockAPIError("inner", status_code=401)
        outer = Exception("outer")
        outer.__cause__ = inner
        assert _extract_status_code(outer) == 401

    def test_none_when_missing(self):
        assert _extract_status_code(Exception("generic")) is None

    def test_rejects_non_http_status(self):
        """Integers outside 100-599 on .status should be ignored."""
        class ErrWeirdStatus(Exception):
            status = 42
        assert _extract_status_code(ErrWeirdStatus()) is None


# ── Test: Error body extraction ────────────────────────────────────────

class TestExtractErrorBody:
    def test_from_body_attr(self):
        e = MockAPIError("fail", body={"error": {"message": "bad"}})
        assert _extract_error_body(e) == {"error": {"message": "bad"}}

    def test_empty_when_no_body(self):
        assert _extract_error_body(Exception("generic")) == {}


# ── Test: Error code extraction ────────────────────────────────────────

class TestExtractErrorCode:
    def test_from_nested_error_code(self):
        body = {"error": {"code": "rate_limit_exceeded"}}
        assert _extract_error_code(body) == "rate_limit_exceeded"

    def test_from_nested_error_type(self):
        body = {"error": {"type": "invalid_request_error"}}
        assert _extract_error_code(body) == "invalid_request_error"

    def test_from_top_level_code(self):
        body = {"code": "model_not_found"}
        assert _extract_error_code(body) == "model_not_found"

    def test_empty_when_no_code(self):
        assert _extract_error_code({}) == ""
        assert _extract_error_code({"error": {"message": "oops"}}) == ""


# ── Test: 402 disambiguation ───────────────────────────────────────────

class TestClassify402:
    """The critical 402 billing vs rate_limit disambiguation."""

    def test_billing_exhaustion(self):
        """Plain 402 = billing."""
        result = _classify_402(
            "payment required",
            lambda reason, **kw: ClassifiedError(reason=reason, **kw),
        )
        assert result.reason == FailoverReason.billing
        assert result.should_rotate_credential is True

    def test_transient_usage_limit(self):
        """402 with 'usage limit' + 'try again' = rate limit, not billing."""
        result = _classify_402(
            "usage limit exceeded. try again in 5 minutes",
            lambda reason, **kw: ClassifiedError(reason=reason, **kw),
        )
        assert result.reason == FailoverReason.rate_limit
        assert result.should_rotate_credential is True

    def test_quota_with_retry(self):
        """402 with 'quota' + 'retry' = rate limit."""
        result = _classify_402(
            "quota exceeded, please retry after the window resets",
            lambda reason, **kw: ClassifiedError(reason=reason, **kw),
        )
        assert result.reason == FailoverReason.rate_limit

    def test_quota_without_retry(self):
        """402 with just 'quota' but no transient signal = billing."""
        result = _classify_402(
            "quota exceeded",
            lambda reason, **kw: ClassifiedError(reason=reason, **kw),
        )
        assert result.reason == FailoverReason.billing

    def test_insufficient_credits(self):
        result = _classify_402(
            "insufficient credits to complete request",
            lambda reason, **kw: ClassifiedError(reason=reason, **kw),
        )
        assert result.reason == FailoverReason.billing


# ── Test: Full classification pipeline ─────────────────────────────────

class TestClassifyApiError:
    """End-to-end classification tests."""

    # ── Auth errors ──

    def test_401_classified_as_auth(self):
        e = MockAPIError("Unauthorized", status_code=401)
        result = classify_api_error(e, provider="openrouter")
        assert result.reason == FailoverReason.auth
        assert result.should_rotate_credential is True
        # 401 is non-retryable on its own — credential rotation runs
        # before the retryability check in the agent loop.
        assert result.retryable is False
        assert result.should_fallback is True

    def test_403_classified_as_auth(self):
        e = MockAPIError("Forbidden", status_code=403)
        result = classify_api_error(e, provider="anthropic")
        assert result.reason == FailoverReason.auth
        assert result.should_fallback is True

    def test_403_key_limit_classified_as_billing(self):
        """OpenRouter 403 'key limit exceeded' is billing, not auth."""
        e = MockAPIError("Key limit exceeded for this key", status_code=403)
        result = classify_api_error(e, provider="openrouter")
        assert result.reason == FailoverReason.billing
        assert result.should_rotate_credential is True
        assert result.should_fallback is True

    def test_403_spending_limit_classified_as_billing(self):
        e = MockAPIError("spending limit reached", status_code=403)
        result = classify_api_error(e, provider="openrouter")
        assert result.reason == FailoverReason.billing

    # ── Billing ──

    def test_402_plain_billing(self):
        e = MockAPIError("Payment Required", status_code=402)
        result = classify_api_error(e)
        assert result.reason == FailoverReason.billing
        assert result.retryable is False

    def test_402_transient_usage_limit(self):
        e = MockAPIError("usage limit exceeded, try again later", status_code=402)
        result = classify_api_error(e)
        assert result.reason == FailoverReason.rate_limit
        assert result.retryable is True

    # ── Rate limit ──

    def test_429_rate_limit(self):
        e = MockAPIError("Too Many Requests", status_code=429)
        result = classify_api_error(e)
        assert result.reason == FailoverReason.rate_limit
        assert result.should_fallback is True

    def test_alibaba_rate_increased_too_quickly(self):
        """Alibaba/DashScope returns a unique throttling message.

        Port from anomalyco/opencode#21355.
        """
        msg = (
            "Upstream error from Alibaba: Request rate increased too quickly. "
            "To ensure system stability, please adjust your client logic to "
            "scale requests more smoothly over time."
        )
        e = MockAPIError(msg, status_code=400)
        result = classify_api_error(e)
        assert result.reason == FailoverReason.rate_limit
        assert result.retryable is True
        assert result.should_rotate_credential is True

    # ── Server errors ──

    def test_500_server_error(self):
        e = MockAPIError("Internal Server Error", status_code=500)
        result = classify_api_error(e)
        assert result.reason == FailoverReason.server_error
        assert result.retryable is True

    def test_502_server_error(self):
        e = MockAPIError("Bad Gateway", status_code=502)
        result = classify_api_error(e)
        assert result.reason == FailoverReason.server_error

    def test_503_overloaded(self):
        e = MockAPIError("Service Unavailable", status_code=503)
        result = classify_api_error(e)
        assert result.reason == FailoverReason.overloaded

    def test_529_anthropic_overloaded(self):
        e = MockAPIError("Overloaded", status_code=529)
        result = classify_api_error(e)
        assert result.reason == FailoverReason.overloaded

    # ── Model not found ──

    def test_404_model_not_found(self):
        e = MockAPIError("model not found", status_code=404)
        result = classify_api_error(e)
        assert result.reason == FailoverReason.model_not_found
        assert result.should_fallback is True
        assert result.retryable is False

    def test_404_generic(self):
        # Generic 404 with no "model not found" signal — common for local
        # llama.cpp/Ollama/vLLM endpoints with slightly wrong paths.  Treat
        # as unknown (retryable) so the real error surfaces, rather than
        # claiming the model is missing and silently falling back.
        e = MockAPIError("Not Found", status_code=404)
        result = classify_api_error(e)
        assert result.reason == FailoverReason.unknown
        assert result.retryable is True
        assert result.should_fallback is False

    # ── Provider policy-block (OpenRouter privacy/guardrail) ──

    def test_404_openrouter_policy_blocked(self):
        # Real OpenRouter error when the user's account privacy setting
        # excludes the only endpoint serving a model (e.g. DeepSeek V4 Pro
        # which is hosted only by DeepSeek, and their endpoint may log
        # inputs).  Must NOT classify as model_not_found — the model
        # exists, falling back won't help (same account setting applies),
        # and the error body already tells the user where to fix it.
        e = MockAPIError(
            "No endpoints available matching your guardrail restrictions "
            "and data policy. Configure: https://openrouter.ai/settings/privacy",
            status_code=404,
        )
        result = classify_api_error(e)
        assert result.reason == FailoverReason.provider_policy_blocked
        assert result.retryable is False
        assert result.should_fallback is False

    def test_400_openrouter_policy_blocked(self):
        # Defense-in-depth: if OpenRouter ever returns this as 400 instead
        # of 404, still classify it distinctly rather than as format_error
        # or model_not_found.
        e = MockAPIError(
            "No endpoints available matching your data policy",
            status_code=400,
        )
        result = classify_api_error(e)
        assert result.reason == FailoverReason.provider_policy_blocked
        assert result.retryable is False
        assert result.should_fallback is False

    def test_message_only_openrouter_policy_blocked(self):
        # No status code — classifier should still catch the fingerprint
        # via the message-pattern fallback.
        e = Exception(
            "No endpoints available matching your guardrail restrictions "
            "and data policy"
        )
        result = classify_api_error(e)
        assert result.reason == FailoverReason.provider_policy_blocked

    def test_404_model_not_found_still_works(self):
        # Regression guard: the new policy-block check must not swallow
        # genuine model_not_found 404s.
        e = MockAPIError(
            "openrouter/nonexistent-model is not a valid model ID",
            status_code=404,
        )
        result = classify_api_error(e)
        assert result.reason == FailoverReason.model_not_found
        assert result.should_fallback is True

    # ── Payload too large ──

    def test_413_payload_too_large(self):
        e = MockAPIError("Request Entity Too Large", status_code=413)
        result = classify_api_error(e)
        assert result.reason == FailoverReason.payload_too_large
        assert result.should_compress is True

    # ── Context overflow ──

    def test_400_context_length(self):
        e = MockAPIError("context length exceeded: 250000 > 200000", status_code=400)
        result = classify_api_error(e)
        assert result.reason == FailoverReason.context_overflow
        assert result.should_compress is True

    def test_400_too_many_tokens(self):
        e = MockAPIError("This model's maximum context is 128000 tokens, too many tokens", status_code=400)
        result = classify_api_error(e)
        assert result.reason == FailoverReason.context_overflow

    def test_400_prompt_too_long(self):
        e = MockAPIError("prompt is too long: 300000 tokens > 200000 maximum", status_code=400)
        result = classify_api_error(e)
        assert result.reason == FailoverReason.context_overflow

    def test_400_generic_large_session(self):
        """Generic 400 with large session → context overflow heuristic."""
        e = MockAPIError(
            "Error",
            status_code=400,
            body={"error": {"message": "Error"}},
        )
        result = classify_api_error(e, approx_tokens=100000, context_length=200000)
        assert result.reason == FailoverReason.context_overflow

    def test_400_generic_small_session_is_format_error(self):
        """Generic 400 with small session → format error, not context overflow."""
        e = MockAPIError(
            "Error",
            status_code=400,
            body={"error": {"message": "Error"}},
        )
        result = classify_api_error(e, approx_tokens=1000, context_length=200000)
        assert result.reason == FailoverReason.format_error

    def test_400_generic_many_messages_below_large_context_pressure_is_format_error(self):
        """Large-context sessions should not overflow solely due to message count."""
        e = MockAPIError(
            "Error",
            status_code=400,
            body={"error": {"message": "Error"}},
        )
        result = classify_api_error(
            e,
            provider="openai-codex",
            model="gpt-5.5",
            approx_tokens=74320,
            context_length=1_000_000,
            num_messages=432,
        )
        assert result.reason == FailoverReason.format_error
        assert result.should_compress is False

    # ── Server disconnect + large session ──

    def test_disconnect_large_session_context_overflow(self):
        """Server disconnect with large session → context overflow."""
        e = Exception("server disconnected without sending complete message")
        result = classify_api_error(e, approx_tokens=150000, context_length=200000)
        assert result.reason == FailoverReason.context_overflow
        assert result.should_compress is True

    def test_disconnect_small_session_timeout(self):
        """Server disconnect with small session → timeout."""
        e = Exception("server disconnected without sending complete message")
        result = classify_api_error(e, approx_tokens=5000, context_length=200000)
        assert result.reason == FailoverReason.timeout

    def test_disconnect_many_messages_below_large_context_pressure_is_timeout(self):
        """Large-context disconnects should not overflow solely due to message count."""
        e = Exception("server disconnected without sending complete message")
        result = classify_api_error(
            e,
            provider="openai-codex",
            model="gpt-5.5",
            approx_tokens=74320,
            context_length=1_000_000,
            num_messages=432,
        )
        assert result.reason == FailoverReason.timeout
        assert result.should_compress is False

    # ── Provider-specific: Anthropic thinking signature ──

    def test_anthropic_thinking_signature(self):
        e = MockAPIError(
            "thinking block has invalid signature",
            status_code=400,
        )
        result = classify_api_error(e, provider="anthropic")
        assert result.reason == FailoverReason.thinking_signature
        assert result.retryable is True

    def test_non_anthropic_400_with_signature_not_classified_as_thinking(self):
        """400 with 'signature' but from non-Anthropic → format error."""
        e = MockAPIError("invalid signature", status_code=400)
        result = classify_api_error(e, provider="openrouter", approx_tokens=0)
        # Without "thinking" in the message, it shouldn't be thinking_signature
        assert result.reason != FailoverReason.thinking_signature

    # ── Provider-specific: llama.cpp grammar-parse ──

    def test_llama_cpp_grammar_parse_error(self):
        """llama.cpp rejects regex escapes in JSON Schema `pattern`."""
        e = MockAPIError(
            "parse: error parsing grammar: unknown escape at \\d",
            status_code=400,
        )
        result = classify_api_error(e, provider="openai-compatible")
        assert result.reason == FailoverReason.llama_cpp_grammar_pattern
        assert result.retryable is True
        assert result.should_compress is False

    def test_llama_cpp_unable_to_generate_parser(self):
        """Older llama.cpp builds surface the error as 'unable to generate parser'."""
        e = MockAPIError(
            "Unable to generate parser for this template",
            status_code=400,
        )
        result = classify_api_error(e, provider="openai-compatible")
        assert result.reason == FailoverReason.llama_cpp_grammar_pattern

    def test_llama_cpp_json_schema_to_grammar_phrase(self):
        """Some builds mention the module name explicitly."""
        e = MockAPIError(
            "json-schema-to-grammar failed to convert schema",
            status_code=400,
        )
        result = classify_api_error(e, provider="openai-compatible")
        assert result.reason == FailoverReason.llama_cpp_grammar_pattern

    def test_llama_cpp_grammar_requires_400(self):
        """A 500 with the same phrase isn't the llama.cpp grammar case."""
        e = MockAPIError("error parsing grammar", status_code=500)
        result = classify_api_error(e, provider="openai-compatible")
        assert result.reason != FailoverReason.llama_cpp_grammar_pattern

    # ── Provider-specific: Anthropic long-context tier ──

    def test_anthropic_long_context_tier(self):
        e = MockAPIError(
            "Extra usage is required for long context requests over 200k tokens",
            status_code=429,
        )
        result = classify_api_error(e, provider="anthropic", model="claude-sonnet-4")
        assert result.reason == FailoverReason.long_context_tier
        assert result.should_compress is True

    def test_normal_429_not_long_context(self):
        """Normal 429 without 'extra usage' + 'long context' → rate_limit."""
        e = MockAPIError("Too Many Requests", status_code=429)
        result = classify_api_error(e, provider="anthropic")
        assert result.reason == FailoverReason.rate_limit

    # ── Provider-specific: Anthropic OAuth 1M-context beta forbidden ──

    def test_anthropic_oauth_1m_beta_forbidden(self):
        """400 + 'long context beta is not yet available for this subscription'
        → oauth_long_context_beta_forbidden (retryable, no compression)."""
        e = MockAPIError(
            "The long context beta is not yet available for this subscription.",
            status_code=400,
        )
        result = classify_api_error(e, provider="anthropic", model="claude-sonnet-4.6")
        assert result.reason == FailoverReason.oauth_long_context_beta_forbidden
        assert result.retryable is True
        assert result.should_compress is False

    def test_anthropic_oauth_1m_beta_forbidden_does_not_collide_with_tier_gate(self):
        """The 429 'extra usage' + 'long context' tier gate keeps its own
        classification even though its message mentions 'long context'."""
        e = MockAPIError(
            "Extra usage is required for long context requests over 200k tokens",
            status_code=429,
        )
        result = classify_api_error(e, provider="anthropic", model="claude-sonnet-4.6")
        assert result.reason == FailoverReason.long_context_tier

    def test_400_without_beta_phrase_is_not_1m_beta_forbidden(self):
        """A generic 400 that happens to mention 'long context' but not the
        exact beta-availability phrase should not be misclassified."""
        e = MockAPIError(
            "long context window exceeded",
            status_code=400,
        )
        result = classify_api_error(e, provider="anthropic")
        assert result.reason != FailoverReason.oauth_long_context_beta_forbidden

    # ── Transport errors ──

    def test_read_timeout(self):
        e = ReadTimeout("Read timed out")
        result = classify_api_error(e)
        assert result.reason == FailoverReason.timeout
        assert result.retryable is True

    def test_connect_error(self):
        e = ConnectError("Connection refused")
        result = classify_api_error(e)
        assert result.reason == FailoverReason.timeout

    def test_connection_error_builtin(self):
        e = ConnectionError("Connection reset by peer")
        result = classify_api_error(e)
        assert result.reason == FailoverReason.timeout

    def test_timeout_error_builtin(self):
        e = TimeoutError("timed out")
        result = classify_api_error(e)
        assert result.reason == FailoverReason.timeout

    def test_runtime_error_cli_turn_timed_out_classifies_as_timeout(self):
        # RuntimeError from a local claude-cli shim that wraps a subprocess
        # timeout must classify as FailoverReason.timeout, not unknown, so
        # the retry loop rebuilds the client instead of treating the turn as
        # an empty model response (#22548).
        e = RuntimeError("claude CLI turn timed out")
        result = classify_api_error(e)
        assert result.reason == FailoverReason.timeout
        assert result.retryable is True

    def test_runtime_error_request_timed_out_classifies_as_timeout(self):
        e = RuntimeError("request timed out after 120s")
        result = classify_api_error(e)
        assert result.reason == FailoverReason.timeout
        assert result.retryable is True

    def test_runtime_error_deadline_exceeded_classifies_as_timeout(self):
        e = RuntimeError("deadline exceeded")
        result = classify_api_error(e)
        assert result.reason == FailoverReason.timeout
        assert result.retryable is True

    # ── Error code classification ──

    def test_error_code_resource_exhausted(self):
        e = MockAPIError(
            "Resource exhausted",
            body={"error": {"code": "resource_exhausted", "message": "Too many requests"}},
        )
        result = classify_api_error(e)
        assert result.reason == FailoverReason.rate_limit

    def test_error_code_model_not_found(self):
        e = MockAPIError(
            "Model not available",
            body={"error": {"code": "model_not_found"}},
        )
        result = classify_api_error(e)
        assert result.reason == FailoverReason.model_not_found

    def test_error_code_context_length_exceeded(self):
        e = MockAPIError(
            "Context too large",
            body={"error": {"code": "context_length_exceeded"}},
        )
        result = classify_api_error(e)
        assert result.reason == FailoverReason.context_overflow

    # ── Message-only patterns (no status code) ──

    def test_message_billing_pattern(self):
        e = Exception("insufficient credits to complete this request")
        result = classify_api_error(e)
        assert result.reason == FailoverReason.billing

    def test_message_rate_limit_pattern(self):
        e = Exception("rate limit reached for this model")
        result = classify_api_error(e)
        assert result.reason == FailoverReason.rate_limit

    def test_message_auth_pattern(self):
        e = Exception("invalid api key provided")
        result = classify_api_error(e)
        assert result.reason == FailoverReason.auth

    def test_message_model_not_found_pattern(self):
        e = Exception("gpt-99 is not a valid model")
        result = classify_api_error(e)
        assert result.reason == FailoverReason.model_not_found

    def test_message_context_overflow_pattern(self):
        e = Exception("maximum context length exceeded")
        result = classify_api_error(e)
        assert result.reason == FailoverReason.context_overflow

    # ── Message-only usage limit disambiguation (no status code) ──

    def test_message_usage_limit_transient_is_rate_limit(self):
        """'usage limit' + 'try again' with no status code → rate_limit, not billing."""
        e = Exception("usage limit exceeded, try again in 5 minutes")
        result = classify_api_error(e)
        assert result.reason == FailoverReason.rate_limit
        assert result.retryable is True
        assert result.should_rotate_credential is True
        assert result.should_fallback is True

    def test_message_usage_limit_no_retry_signal_is_billing(self):
        """'usage limit' with no transient signal and no status code → billing."""
        e = Exception("usage limit reached")
        result = classify_api_error(e)
        assert result.reason == FailoverReason.billing
        assert result.retryable is False
        assert result.should_rotate_credential is True

    def test_message_quota_with_reset_window_is_rate_limit(self):
        """'quota' + 'resets at' with no status code → rate_limit."""
        e = Exception("quota exceeded, resets at midnight UTC")
        result = classify_api_error(e)
        assert result.reason == FailoverReason.rate_limit
        assert result.retryable is True

    def test_message_limit_exceeded_with_wait_is_rate_limit(self):
        """'limit exceeded' + 'wait' with no status code → rate_limit."""
        e = Exception("key limit exceeded, please wait before retrying")
        result = classify_api_error(e)
        assert result.reason == FailoverReason.rate_limit
        assert result.retryable is True

    # ── Unknown / fallback ──

    def test_generic_exception_is_unknown(self):
        e = Exception("something weird happened")
        result = classify_api_error(e)
        assert result.reason == FailoverReason.unknown
        assert result.retryable is True

    # ── Format error ──

    def test_400_descriptive_format_error(self):
        """400 with descriptive message (not context overflow) → format error."""
        e = MockAPIError(
            "Invalid value for parameter 'temperature': must be between 0 and 2",
            status_code=400,
            body={"error": {"message": "Invalid value for parameter 'temperature': must be between 0 and 2"}},
        )
        result = classify_api_error(e, approx_tokens=1000)
        assert result.reason == FailoverReason.format_error
        assert result.retryable is False

    def test_422_format_error(self):
        e = MockAPIError("Unprocessable Entity", status_code=422)
        result = classify_api_error(e)
        assert result.reason == FailoverReason.format_error
        assert result.retryable is False

    def test_400_flat_body_descriptive_not_context_overflow(self):
        """Responses API flat body with descriptive error + large session → format error.

        The Codex Responses API returns errors in flat body format:
        {"message": "...", "type": "..."} without an "error" wrapper.
        A descriptive 400 must NOT be misclassified as context overflow
        just because the session is large.
        """
        e = MockAPIError(
            "Invalid 'input[index].name': string does not match pattern.",
            status_code=400,
            body={"message": "Invalid 'input[index].name': string does not match pattern.",
                  "type": "invalid_request_error"},
        )
        result = classify_api_error(e, approx_tokens=200000, context_length=400000, num_messages=500)
        assert result.reason == FailoverReason.format_error
        assert result.retryable is False

    def test_400_flat_body_generic_large_session_still_context_overflow(self):
        """Flat body with generic 'Error' message + large session → context overflow.

        Regression: the flat-body fallback must not break the existing heuristic
        for genuinely generic errors from providers that use flat bodies.
        """
        e = MockAPIError(
            "Error",
            status_code=400,
            body={"message": "Error"},
        )
        result = classify_api_error(e, approx_tokens=100000, context_length=200000)
        assert result.reason == FailoverReason.context_overflow

    # ── Peer closed + large session ──

    def test_peer_closed_large_session(self):
        e = Exception("peer closed connection without sending complete message")
        result = classify_api_error(e, approx_tokens=130000, context_length=200000)
        assert result.reason == FailoverReason.context_overflow

    # ── Chinese error messages ──

    def test_chinese_context_overflow(self):
        e = MockAPIError("超过最大长度限制", status_code=400)
        result = classify_api_error(e)
        assert result.reason == FailoverReason.context_overflow

    # ── vLLM / local inference server error messages ──

    def test_vllm_max_model_len_overflow(self):
        """vLLM's 'exceeds the max_model_len' error → context_overflow."""
        e = MockAPIError(
            "The engine prompt length 1327246 exceeds the max_model_len 131072. "
            "Please reduce prompt.",
            status_code=400,
        )
        result = classify_api_error(e)
        assert result.reason == FailoverReason.context_overflow

    def test_vllm_prompt_length_exceeds(self):
        """vLLM prompt length error → context_overflow."""
        e = MockAPIError(
            "prompt length 200000 exceeds maximum model length 131072",
            status_code=400,
        )
        result = classify_api_error(e)
        assert result.reason == FailoverReason.context_overflow

    def test_vllm_input_too_long(self):
        """vLLM 'input is too long' error → context_overflow."""
        e = MockAPIError("input is too long for model", status_code=400)
        result = classify_api_error(e)
        assert result.reason == FailoverReason.context_overflow

    def test_ollama_context_length_exceeded(self):
        """Ollama 'context length exceeded' error → context_overflow."""
        e = MockAPIError("context length exceeded", status_code=400)
        result = classify_api_error(e)
        assert result.reason == FailoverReason.context_overflow

    def test_llamacpp_slot_context(self):
        """llama.cpp / llama-server 'slot context' error → context_overflow."""
        e = MockAPIError(
            "slot context: 4096 tokens, prompt 8192 tokens — not enough space",
            status_code=400,
        )
        result = classify_api_error(e)
        assert result.reason == FailoverReason.context_overflow

    # ── Result metadata ──

    def test_provider_and_model_in_result(self):
        e = MockAPIError("fail", status_code=500)
        result = classify_api_error(e, provider="openrouter", model="gpt-5")
        assert result.provider == "openrouter"
        assert result.model == "gpt-5"
        assert result.status_code == 500

    def test_message_extracted(self):
        e = MockAPIError(
            "outer",
            status_code=500,
            body={"error": {"message": "Internal server error occurred"}},
        )
        result = classify_api_error(e)
        assert result.message == "Internal server error occurred"


# ── Test: Adversarial / edge cases (from live testing) ─────────────────

class TestAdversarialEdgeCases:
    """Edge cases discovered during live testing with real SDK objects."""

    def test_empty_exception_message(self):
        result = classify_api_error(Exception(""))
        assert result.reason == FailoverReason.unknown
        assert result.retryable is True

    def test_500_with_none_body(self):
        e = MockAPIError("fail", status_code=500, body=None)
        result = classify_api_error(e)
        assert result.reason == FailoverReason.server_error

    def test_non_dict_body(self):
        """Some providers return strings instead of JSON."""
        class StringBodyError(Exception):
            status_code = 400
            body = "just a string"
        result = classify_api_error(StringBodyError("bad"))
        assert result.reason == FailoverReason.format_error

    def test_list_body(self):
        class ListBodyError(Exception):
            status_code = 500
            body = [{"error": "something"}]
        result = classify_api_error(ListBodyError("server error"))
        assert result.reason == FailoverReason.server_error

    def test_circular_cause_chain(self):
        """Must not infinite-loop on circular __cause__."""
        e = Exception("circular")
        e.__cause__ = e
        result = classify_api_error(e)
        assert result.reason == FailoverReason.unknown

    def test_three_level_cause_chain(self):
        inner = MockAPIError("inner", status_code=429)
        middle = Exception("middle")
        middle.__cause__ = inner
        outer = RuntimeError("outer")
        outer.__cause__ = middle
        result = classify_api_error(outer)
        assert result.status_code == 429
        assert result.reason == FailoverReason.rate_limit

    def test_400_with_rate_limit_text(self):
        """Some providers send rate limits as 400 instead of 429."""
        e = MockAPIError(
            "rate limit policy",
            status_code=400,
            body={"error": {"message": "rate limit exceeded on this model"}},
        )
        result = classify_api_error(e, provider="openrouter")
        assert result.reason == FailoverReason.rate_limit

    def test_400_with_billing_text(self):
        """Some providers send billing errors as 400."""
        e = MockAPIError(
            "billing",
            status_code=400,
            body={"error": {"message": "insufficient credits for this request"}},
        )
        result = classify_api_error(e)
        assert result.reason == FailoverReason.billing

    def test_200_with_error_body(self):
        """200 status with error in body — should be unknown, not crash."""
        class WeirdSuccess(Exception):
            status_code = 200
            body = {"error": {"message": "loading"}}
        result = classify_api_error(WeirdSuccess("model loading"))
        assert result.reason == FailoverReason.unknown

    def test_ollama_context_size_exceeded(self):
        e = MockAPIError(
            "Error",
            status_code=400,
            body={"error": {"message": "context size has been exceeded"}},
        )
        result = classify_api_error(e, provider="ollama")
        assert result.reason == FailoverReason.context_overflow

    def test_connection_refused_error(self):
        e = ConnectionRefusedError("Connection refused: localhost:11434")
        result = classify_api_error(e, provider="ollama")
        assert result.reason == FailoverReason.timeout

    def test_body_message_enrichment(self):
        """Body message must be included in pattern matching even when
        str(error) doesn't contain it (OpenAI SDK APIStatusError)."""
        e = MockAPIError(
            "Usage limit",  # str(e) = "usage limit"
            status_code=402,
            body={"error": {"message": "Usage limit reached, try again in 5 minutes"}},
        )
        result = classify_api_error(e)
        # "try again" is only in body, not in str(e)
        assert result.reason == FailoverReason.rate_limit

    def test_disconnect_pattern_ordering(self):
        """Disconnect + large session must beat generic transport catch."""
        class FakeRemoteProtocol(Exception):
            pass
        # Type name isn't in _TRANSPORT_ERROR_TYPES but message has disconnect pattern
        e = Exception("peer closed connection without sending complete message")
        result = classify_api_error(e, approx_tokens=150000, context_length=200000)
        assert result.reason == FailoverReason.context_overflow
        assert result.should_compress is True

    def test_credit_balance_too_low(self):
        e = MockAPIError(
            "Credits low",
            status_code=402,
            body={"error": {"message": "Your credit balance is too low"}},
        )
        result = classify_api_error(e, provider="anthropic")
        assert result.reason == FailoverReason.billing

    def test_deepseek_402_chinese(self):
        """Chinese billing message should still match billing patterns."""
        # "余额不足" doesn't match English billing patterns, but 402 defaults to billing
        e = MockAPIError("余额不足", status_code=402)
        result = classify_api_error(e, provider="deepseek")
        assert result.reason == FailoverReason.billing

    def test_openrouter_wrapped_context_overflow_in_metadata_raw(self):
        """OpenRouter wraps provider errors in metadata.raw JSON string."""
        e = MockAPIError(
            "Provider returned error",
            status_code=400,
            body={
                "error": {
                    "message": "Provider returned error",
                    "code": 400,
                    "metadata": {
                        "raw": '{"error":{"message":"context length exceeded: 50000 > 32768"}}'
                    }
                }
            },
        )
        result = classify_api_error(e, provider="openrouter", approx_tokens=10000)
        assert result.reason == FailoverReason.context_overflow
        assert result.should_compress is True

    def test_openrouter_wrapped_rate_limit_in_metadata_raw(self):
        e = MockAPIError(
            "Provider returned error",
            status_code=400,
            body={
                "error": {
                    "message": "Provider returned error",
                    "metadata": {
                        "raw": '{"error":{"message":"Rate limit exceeded. Please retry after 30s."}}'
                    }
                }
            },
        )
        result = classify_api_error(e, provider="openrouter")
        assert result.reason == FailoverReason.rate_limit

    def test_thinking_signature_via_openrouter(self):
        """Thinking signature errors proxied through OpenRouter must be caught."""
        e = MockAPIError(
            "thinking block has invalid signature",
            status_code=400,
        )
        # provider is openrouter, not anthropic — old code missed this
        result = classify_api_error(e, provider="openrouter", model="anthropic/claude-sonnet-4")
        assert result.reason == FailoverReason.thinking_signature

    def test_generic_400_large_by_message_count(self):
        """Many small messages (>80) should trigger context overflow heuristic."""
        e = MockAPIError(
            "Error",
            status_code=400,
            body={"error": {"message": "Error"}},
        )
        # Low token count but high message count
        result = classify_api_error(
            e, approx_tokens=5000, context_length=200000, num_messages=100,
        )
        assert result.reason == FailoverReason.context_overflow

    def test_disconnect_large_by_message_count(self):
        """Server disconnect with 200+ messages should trigger context overflow."""
        e = Exception("server disconnected without sending complete message")
        result = classify_api_error(
            e, approx_tokens=5000, context_length=200000, num_messages=250,
        )
        assert result.reason == FailoverReason.context_overflow

    def test_openrouter_wrapped_model_not_found_in_metadata_raw(self):
        e = MockAPIError(
            "Provider returned error",
            status_code=400,
            body={
                "error": {
                    "message": "Provider returned error",
                    "metadata": {
                        "raw": '{"error":{"message":"The model gpt-99 does not exist"}}'
                    }
                }
            },
        )
        result = classify_api_error(e, provider="openrouter")
        assert result.reason == FailoverReason.model_not_found

    # ── Regression: dict-typed message field (Issue #11233) ──

    def test_pydantic_dict_message_no_crash(self):
        """Pydantic validation errors return message as dict, not string.

        Regression: classify_api_error must not crash when body['message']
        is a dict (e.g. {"detail": [...]} from FastAPI/Pydantic). The
        'or ""' fallback only handles None/falsy values — a non-empty
        dict is truthy and passed to .lower(), causing AttributeError.
        """
        e = MockAPIError(
            "Unprocessable Entity",
            status_code=422,
            body={
                "object": "error",
                "message": {
                    "detail": [
                        {
                            "type": "extra_forbidden",
                            "loc": ["body", "think"],
                            "msg": "Extra inputs are not permitted",
                        }
                    ]
                },
            },
        )
        result = classify_api_error(e)
        assert result.reason == FailoverReason.format_error
        assert result.status_code == 422
        assert result.retryable is False

    def test_nested_error_dict_message_no_crash(self):
        """Nested body['error']['message'] as dict must not crash.

        Some providers wrap Pydantic errors in an 'error' object.
        """
        e = MockAPIError(
            "Validation error",
            status_code=400,
            body={
                "error": {
                    "message": {
                        "detail": [
                            {"type": "missing", "loc": ["body", "required"]}
                        ]
                    }
                }
            },
        )
        result = classify_api_error(e, approx_tokens=1000)
        assert result.reason == FailoverReason.format_error
        assert result.status_code == 400

    def test_metadata_raw_dict_message_no_crash(self):
        """OpenRouter metadata.raw with dict message must not crash."""
        e = MockAPIError(
            "Provider error",
            status_code=400,
            body={
                "error": {
                    "message": "Provider error",
                    "metadata": {
                        "raw": '{"error":{"message":{"detail":[{"type":"invalid"}]}}}'
                    }
                }
            },
        )
        result = classify_api_error(e)
        assert result.reason == FailoverReason.format_error

    # Broader non-string type guards — defense against other provider quirks.

    def test_list_message_no_crash(self):
        """Some providers return message as a list of error entries."""
        e = MockAPIError(
            "validation",
            status_code=400,
            body={"message": [{"msg": "field required"}]},
        )
        result = classify_api_error(e)
        assert result is not None

    def test_int_message_no_crash(self):
        """Any non-string type must be coerced safely."""
        e = MockAPIError("server error", status_code=500, body={"message": 42})
        result = classify_api_error(e)
        assert result is not None

    def test_none_message_still_works(self):
        """Regression: None fallback (the 'or \"\"' path) must still work."""
        e = MockAPIError("server error", status_code=500, body={"message": None})
        result = classify_api_error(e)
        assert result is not None


# ── Test: SSL/TLS transient errors ─────────────────────────────────────

class TestSSLTransientPatterns:
    """SSL/TLS alerts mid-stream should retry as timeout, not unknown, and
    should NOT trigger context compression even on a large session.

    Motivation: OpenSSL 3.x changed TLS alert error code format
    (`SSLV3_ALERT_BAD_RECORD_MAC` → `SSL/TLS_ALERT_BAD_RECORD_MAC`),
    breaking string-exact matching in downstream retry logic.  We match
    stable substrings instead.
    """

    def test_bad_record_mac_classifies_as_timeout(self):
        """OpenSSL 3.x mid-stream bad record mac alert."""
        e = Exception("[SSL: BAD_RECORD_MAC] sslv3 alert bad record mac (_ssl.c:2580)")
        result = classify_api_error(e)
        assert result.reason == FailoverReason.timeout
        assert result.retryable is True
        assert result.should_compress is False

    def test_openssl_3x_format_classifies_as_timeout(self):
        """New format `ERR_SSL_SSL/TLS_ALERT_BAD_RECORD_MAC` still matches
        because we key on both space- and underscore-separated forms of
        the stable `bad_record_mac` token."""
        e = Exception("ERR_SSL_SSL/TLS_ALERT_BAD_RECORD_MAC during streaming")
        result = classify_api_error(e)
        assert result.reason == FailoverReason.timeout
        assert result.retryable is True
        assert result.should_compress is False

    def test_tls_alert_internal_error_classifies_as_timeout(self):
        e = Exception("[SSL: TLSV1_ALERT_INTERNAL_ERROR] tlsv1 alert internal error")
        result = classify_api_error(e)
        assert result.reason == FailoverReason.timeout
        assert result.retryable is True
        assert result.should_compress is False

    def test_ssl_handshake_failure_classifies_as_timeout(self):
        e = Exception("ssl handshake failure during mid-stream")
        result = classify_api_error(e)
        assert result.reason == FailoverReason.timeout
        assert result.retryable is True

    def test_ssl_prefix_classifies_as_timeout(self):
        """Python's generic '[SSL: XYZ]' prefix from the ssl module."""
        e = Exception("[SSL: UNEXPECTED_EOF_WHILE_READING] EOF occurred in violation of protocol")
        result = classify_api_error(e)
        assert result.reason == FailoverReason.timeout
        assert result.retryable is True

    def test_ssl_alert_on_large_session_does_not_compress(self):
        """Critical: SSL alerts on big contexts must NOT trigger context
        compression — compression is expensive and won't fix a transport
        hiccup.  This is why _SSL_TRANSIENT_PATTERNS is separate from
        _SERVER_DISCONNECT_PATTERNS.
        """
        e = Exception("[SSL: BAD_RECORD_MAC] sslv3 alert bad record mac")
        result = classify_api_error(
            e,
            approx_tokens=180000,      # 90% of a 200k-context window
            context_length=200000,
            num_messages=300,
        )
        assert result.reason == FailoverReason.timeout
        assert result.should_compress is False

    def test_plain_disconnect_on_large_session_still_compresses(self):
        """Regression guard: the context-overflow-via-disconnect path
        (non-SSL disconnects on large sessions) must still trigger
        compression.  Only SSL-specific disconnects skip it.
        """
        e = Exception("Server disconnected without sending a response")
        result = classify_api_error(
            e,
            approx_tokens=180000,
            context_length=200000,
            num_messages=300,
        )
        assert result.reason == FailoverReason.context_overflow
        assert result.should_compress is True

    def test_real_ssl_error_type_classifies_as_timeout(self):
        """Real ssl.SSLError instance — the type name alone (not message)
        should route to the transport bucket."""
        import ssl
        e = ssl.SSLError("arbitrary ssl error")
        result = classify_api_error(e)
        assert result.reason == FailoverReason.timeout
        assert result.retryable is True

# ── Test: RateLimitError without status_code (Copilot/GitHub Models) ──────────

class TestRateLimitErrorWithoutStatusCode:
    """Regression tests for the Copilot/GitHub Models edge case where the
    OpenAI SDK raises RateLimitError but does not populate .status_code."""

    def _make_rate_limit_error(self, status_code=None):
        """Create an exception whose class name is 'RateLimitError' with
        an optionally missing status_code, mirroring the OpenAI SDK shape."""
        cls = type("RateLimitError", (Exception,), {})
        e = cls("You have exceeded your rate limit.")
        e.status_code = status_code  # None simulates the Copilot case
        return e

    def test_rate_limit_error_without_status_code_classified_as_rate_limit(self):
        """RateLimitError with status_code=None must classify as rate_limit."""
        e = self._make_rate_limit_error(status_code=None)
        result = classify_api_error(e, provider="copilot", model="gpt-4o")
        assert result.reason == FailoverReason.rate_limit

    def test_rate_limit_error_with_status_code_429_classified_as_rate_limit(self):
        """RateLimitError that does set status_code=429 still classifies correctly."""
        e = self._make_rate_limit_error(status_code=429)
        result = classify_api_error(e, provider="copilot", model="gpt-4o")
        assert result.reason == FailoverReason.rate_limit

    def test_other_error_without_status_code_not_forced_to_rate_limit(self):
        """A non-RateLimitError with missing status_code must NOT be forced to 429."""
        cls = type("APIError", (Exception,), {})
        e = cls("something went wrong")
        e.status_code = None
        result = classify_api_error(e, provider="copilot", model="gpt-4o")
        assert result.reason != FailoverReason.rate_limit
