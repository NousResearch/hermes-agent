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
            "upstream_rate_limit",
            "overloaded", "server_error", "timeout",
            "ssl_cert_verification",
            "context_overflow", "payload_too_large", "image_too_large",
            "model_not_found", "format_error",
            "invalid_encrypted_content",
            "multimodal_tool_content_unsupported",
            "provider_policy_blocked",
            "content_policy_blocked",
            "thinking_signature", "long_context_tier",
            "oauth_long_context_beta_forbidden",
            "llama_cpp_grammar_pattern",
            "upstream_provider_error",
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

    def test_from_status_attr(self):
        class ErrWithStatus(Exception):
            status = 503
        assert _extract_status_code(ErrWithStatus()) == 503

    def test_from_cause_chain(self):
        inner = MockAPIError("inner", status_code=401)
        outer = Exception("outer")
        outer.__cause__ = inner
        assert _extract_status_code(outer) == 401




# ── Test: Error body extraction ────────────────────────────────────────

class TestExtractErrorBody:
    def test_from_body_attr(self):
        e = MockAPIError("fail", body={"error": {"message": "bad"}})
        assert _extract_error_body(e) == {"error": {"message": "bad"}}

    def test_from_cause_chain_body_attr(self):
        inner = MockAPIError(
            "inner",
            status_code=402,
            body={"error": {"message": "Usage limit reached, try again in 5 minutes"}},
        )
        outer = Exception("outer")
        outer.__cause__ = inner
        assert _extract_error_body(outer) == {
            "error": {"message": "Usage limit reached, try again in 5 minutes"},
        }

    def test_empty_when_no_body(self):
        assert _extract_error_body(Exception("generic")) == {}


# ── Test: Error code extraction ────────────────────────────────────────

class TestExtractErrorCode:


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


    def test_quota_with_retry(self):
        """402 with 'quota' + 'retry' = rate limit."""
        result = _classify_402(
            "quota exceeded, please retry after the window resets",
            lambda reason, **kw: ClassifiedError(reason=reason, **kw),
        )
        assert result.reason == FailoverReason.rate_limit




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





    # ── Billing ──

    def test_402_plain_billing(self):
        e = MockAPIError("Payment Required", status_code=402)
        result = classify_api_error(e)
        assert result.reason == FailoverReason.billing
        assert result.retryable is False




    def test_404_free_tier_model_block_is_billing(self):
        e = MockAPIError(
            "Not Found",
            status_code=404,
            body={
                "status": 404,
                "message": (
                    "Model 'gpt-5' is not available on the Free Tier. "
                    "Upgrade at https://portal.nousresearch.com or pick a free model."
                ),
            },
        )
        result = classify_api_error(e, provider="nous", model="gpt-5")
        assert result.reason == FailoverReason.billing
        assert result.retryable is False
        assert result.should_fallback is True


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


    def test_408_request_timeout_is_retryable_timeout(self):
        """HTTP 408 Request Timeout is a transient timing failure the server
        itself flags as safe to retry (RFC 9110 §15.5.9) — commonly emitted by
        reverse proxies in front of self-hosted backends (llama.cpp / Ollama /
        vLLM) when a long generation outruns the proxy's request-read window.
        It must NOT fall into the generic 4xx bucket as a non-retryable
        format_error, which would abort the turn on a retry-safe error."""
        e = MockAPIError("Request Timeout", status_code=408)
        result = classify_api_error(e, provider="vllm")
        assert result.reason == FailoverReason.timeout
        assert result.retryable is True

    def test_400_bad_request_still_non_retryable_format_error(self):
        """Guard the boundary: a genuine 400 Bad Request must remain a
        non-retryable format_error and must not be swept up by the 408 branch."""
        e = MockAPIError("Bad Request", status_code=400)
        result = classify_api_error(e)
        assert result.reason == FailoverReason.format_error
        assert result.retryable is False

    def test_message_only_overloaded_without_status_is_overloaded(self):
        """Some Anthropic-compatible proxies surface 'overloaded' in the
        message with no 503/529 status_code. It must classify as overloaded
        (transient backoff+retry), not unknown / credential rotation. (#14261)"""
        e = MockAPIError(
            "Anthropic API error: Overloaded - the service is temporarily overloaded"
        )  # no status_code
        result = classify_api_error(e, provider="anthropic")
        assert result.reason == FailoverReason.overloaded
        assert result.retryable is True
        assert result.should_rotate_credential is False

    def test_429_with_overloaded_body_is_overloaded_not_rate_limit(self):
        """Z.AI / Zhipu reuse HTTP 429 for server-wide overload. The credential
        is valid — the server is just busy — so it must classify as overloaded
        (back off + retry the same key), NOT rate_limit (which would rotate and
        exhaust the pool, doing nothing for a single-key user). (#14038)"""
        e = MockAPIError(
            "The service may be temporarily overloaded, please try again later",
            status_code=429,
        )
        result = classify_api_error(e, provider="zai")
        assert result.reason == FailoverReason.overloaded
        assert result.retryable is True
        assert result.should_rotate_credential is False

    def test_429_normal_rate_limit_still_rotates(self):
        """Guard: a genuine 429 rate limit (no overload language) must still
        classify as rate_limit and rotate the credential. (#14038)"""
        e = MockAPIError(
            "Rate limit exceeded: too many requests", status_code=429
        )
        result = classify_api_error(e, provider="zai")
        assert result.reason == FailoverReason.rate_limit
        assert result.should_rotate_credential is True

    # ── 5xx that are actually request-validation errors ──
    # Some OpenAI-compatible gateways (e.g. codex.nekos.me) return
    # request-validation failures with a 5xx status. These are
    # deterministic, so they must NOT be retried — otherwise the retry
    # loop hammers the identical bad request into a flood.





    # ── 5xx that are actually context overflow ──
    # Some local inference servers (llama.cpp / llama-server, and vLLM/Ollama
    # behind a Cloudflare/Tailscale hop) report context overflow with a 5xx
    # status instead of the standard 400/413. These must route into the
    # compression-and-retry path, not the blind server_error/overloaded retry
    # that exhausts and drops the turn.




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




    # ── Provider content-policy block (per-prompt safety filter) ──
    #
    # Distinct from ``provider_policy_blocked`` above — these are upstream
    # model-provider safety refusals for THIS prompt, not OpenRouter
    # account-level data policy. Recovery is fallback model, not config fix.
    # See issue #18028 — OpenAI Codex was burning 3 retries on identical
    # refusals before users saw "API failed after 3 retries" on Telegram.

    def test_message_only_cyber_content_policy_blocked(self):
        # OpenAI Codex returns this without an HTTP status. Retrying the
        # same prompt three times only repeats the same policy decision, so
        # the classifier must jump straight to fallback / abort instead of
        # leaving it in the retryable ``unknown`` bucket.
        e = Exception(
            "This content was flagged for possible cybersecurity risk. If this "
            "seems wrong, try rephrasing your request. To get authorized for "
            "security work, join the Trusted Access for Cyber program."
        )
        result = classify_api_error(e, provider="openai-codex", model="gpt-5.5")
        assert result.reason == FailoverReason.content_policy_blocked
        assert result.retryable is False
        assert result.should_fallback is True
        assert result.should_compress is False






    # ── Payload too large ──

    def test_413_payload_too_large(self):
        e = MockAPIError("Request Entity Too Large", status_code=413)
        result = classify_api_error(e)
        assert result.reason == FailoverReason.payload_too_large
        assert result.should_compress is True

    # ── Context overflow ──







    # ── Server disconnect + large session ──




    # ── Provider-specific: Anthropic thinking signature ──









    @pytest.mark.parametrize("error_code", ["Invalid_Encrypted_Content", "INVALID_ENCRYPTED_CONTENT"])
    def test_invalid_encrypted_content_code_is_case_insensitive_for_400(self, error_code):
        e = MockAPIError(
            "Error code: 400 - bad request",
            status_code=400,
            body={"error": {"code": error_code, "message": "Bad request"}},
        )
        result = classify_api_error(e, provider="custom", model="gpt-5.4")
        assert result.reason == FailoverReason.invalid_encrypted_content
        assert result.retryable is True
        assert result.should_fallback is False

    # ── Provider-specific: llama.cpp grammar-parse ──





    # ── Provider-specific: Anthropic long-context tier ──

    def test_anthropic_long_context_tier(self):
        e = MockAPIError(
            "Extra usage is required for long context requests over 200k tokens",
            status_code=429,
        )
        result = classify_api_error(e, provider="anthropic", model="claude-sonnet-4")
        assert result.reason == FailoverReason.long_context_tier
        assert result.should_compress is True


    # ── Provider-specific: Anthropic OAuth 1M-context beta forbidden ──




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




    # ── Error code classification ──





    # ── Message-only patterns (no status code) ──







    # ── Message-only usage limit disambiguation (no status code) ──





    # ── Unknown / fallback ──


    # ── Format error ──











    def test_400_litellm_invalid_request_body_shape(self, caplog):
        """litellm/Bedrock proxy shape (errorMessage/errorCode) → format_error.

        The proxy in front of Anthropic surfaces the empty-content rejection
        as {"errorMessage": "...non-empty content...", "errorCode":
        "INVALID_REQUEST_BODY", "errorArgs": {"reason": "..."}}.  Those keys
        are not the standard error.message / message, so err_body_msg used to
        come back empty → is_generic=True → mis-routed into compression on a
        large session.  Both the message pattern and the errorCode must be
        recognized, and a distinct warning must be logged so the condition is
        observable in the field.
        """
        import logging
        proxy_msg = ("The provided request body is invalid: claude "
                     "messages.208: all messages must have non-empty content "
                     "except for the optional final assistant message")
        e = MockAPIError(
            proxy_msg,
            status_code=400,
            body={
                "errorMessage": proxy_msg,
                "errorCode": "INVALID_REQUEST_BODY",
                "statusCode": 400,
                "errorArgs": {"reason": "claude messages.208: ..."},
            },
        )
        with caplog.at_level(logging.WARNING, logger="agent.error_classifier"):
            result = classify_api_error(
                e, approx_tokens=66000, context_length=200000, num_messages=219,
            )
        assert result.reason == FailoverReason.format_error
        assert result.retryable is False
        assert result.should_compress is not True
        assert any(
            "Malformed message array 400" in r.getMessage()
            for r in caplog.records
        ), "Expected a distinct warning identifying the malformed-body 400"


    # ── Peer closed + large session ──


    # ── Chinese error messages ──


    # ── Z.AI / Zhipu GLM error messages ──

    def test_zai_glm_token_limit_overflow(self):
        """Z.AI GLM's 'tokens in request more than max tokens allowed'
        (error code 1210) → context_overflow, so the agent compresses
        instead of blindly retrying. Port of anomalyco/opencode#35671."""
        e = MockAPIError(
            '{"error": {"code": "1210", "message": '
            '"tokens in request more than max tokens allowed"}}',
            status_code=400,
        )
        result = classify_api_error(e, provider="zai")
        assert result.reason == FailoverReason.context_overflow

    # ── vLLM / local inference server error messages ──






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


    def test_400_anthropic_extra_usage_exhausted(self):
        """Anthropic returns 400 with 'out of extra usage' when the user's
        extra-usage allowance is depleted. Must classify as billing so the
        fallback chain engages (with credential rotation) instead of the
        generic format_error path, which never rotates. (#11736, #13170)"""
        e = MockAPIError(
            "You're out of extra usage. Add more at claude.ai/settings/usage and keep going.",
            status_code=400,
            body={"error": {
                "type": "invalid_request_error",
                "message": "You're out of extra usage. Add more at claude.ai/settings/usage and keep going.",
            }},
        )
        result = classify_api_error(e, provider="anthropic")
        assert result.reason == FailoverReason.billing
        assert result.should_fallback is True
        assert result.retryable is False
        assert result.should_rotate_credential is True

    def test_200_with_error_body(self):
        """200 status with error in body — should be unknown, not crash."""
        class WeirdSuccess(Exception):
            status_code = 200
            body = {"error": {"message": "loading"}}
        result = classify_api_error(WeirdSuccess("model loading"))
        assert result.reason == FailoverReason.unknown


    def test_connection_refused_error(self):
        e = ConnectionRefusedError("Connection refused: localhost:11434")
        result = classify_api_error(e, provider="ollama")
        assert result.reason == FailoverReason.timeout


    def test_disconnect_pattern_ordering(self):
        """Disconnect + large session must beat generic transport catch."""
        class FakeRemoteProtocol(Exception):
            pass
        # Type name isn't in _TRANSPORT_ERROR_TYPES but message has disconnect pattern
        e = Exception("peer closed connection without sending complete message")
        result = classify_api_error(e, approx_tokens=150000, context_length=200000)
        assert result.reason == FailoverReason.context_overflow
        assert result.should_compress is True


    def test_deepseek_402_chinese(self):
        """Chinese billing message should still match billing patterns."""
        # "余额不足" doesn't match English billing patterns, but 402 defaults to billing
        e = MockAPIError("余额不足", status_code=402)
        result = classify_api_error(e, provider="deepseek")
        assert result.reason == FailoverReason.billing







    # ── Regression: dict-typed message field (Issue #11233) ──




    # Broader non-string type guards — defense against other provider quirks.





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



# ── Test: SSL certificate verification failures (fail fast) ────────────

class TestSSLCertVerificationFailFast:
    """Certificate verification failures are deterministic for the host —
    a TLS-inspecting proxy, missing custom CA, expired or self-signed cert
    fails identically on every retry. They must classify as non-retryable
    ``ssl_cert_verification`` so the user sees the fix hint immediately,
    instead of matching the transient "[ssl:" pattern and retrying forever.

    Inspired by Claude Code v2.1.199 (July 2026).
    """

    def test_python_cert_verify_failed_is_non_retryable(self):
        import ssl
        e = ssl.SSLCertVerificationError(
            1,
            "[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: "
            "unable to get local issuer certificate (_ssl.c:1006)",
        )
        result = classify_api_error(e)
        assert result.reason == FailoverReason.ssl_cert_verification
        assert result.retryable is False
        assert result.should_compress is False





    def test_transient_ssl_alert_still_retries(self):
        """Regression guard: genuine transient alerts keep retrying."""
        e = Exception("[SSL: BAD_RECORD_MAC] sslv3 alert bad record mac")
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



# ── Test: multimodal_tool_content_unsupported pattern ───────────────────

class TestMultimodalToolContentUnsupported:
    """Issue #27344 — providers that reject list-type tool message content
    should be classified as ``multimodal_tool_content_unsupported`` so the
    retry loop can downgrade screenshots to text and try again.
    """

    def test_xiaomi_mimo_text_is_not_set_pattern(self):
        """The actual Xiaomi MiMo 400 wording from the bug report."""
        e = MockAPIError(
            "Error code: 400 - {'error': {'code': '400', 'message': 'Param Incorrect', 'param': 'text is not set', 'type': ''}}",
            status_code=400,
        )
        result = classify_api_error(e, provider="xiaomi", model="mimo-v2.5")
        assert result.reason == FailoverReason.multimodal_tool_content_unsupported
        assert result.retryable is True





    def test_unrelated_400_is_not_misclassified(self):
        """Make sure the patterns don't false-positive on normal 400s."""
        e = MockAPIError("bad request: missing field 'model'", status_code=400)
        result = classify_api_error(e, provider="openrouter", model="anthropic/claude-sonnet-4")


class TestUpstreamProviderError403:
    """Distinguish upstream-provider 403 from account-level 403 on aggregators.

    When an upstream model (Sakana, Poolside, etc.) returns a 403 through an
    aggregator (OpenRouter, Groq, Together, Fireworks), the aggregator wraps it
    with "Provider returned error". The user's aggregator key is healthy — we
    must fall back to a different model, NOT mark the credential exhausted.
    """

    def test_openrouter_upstream_403_classified_as_upstream_provider_error(self):
        """OpenRouter 403 with 'Provider returned error' -> upstream_provider_error."""
        e = MockAPIError(
            "Provider returned error",
            status_code=403,
            body={
                "error": {
                    "message": "Provider returned error",
                    "code": 403,
                    "metadata": {
                        "provider_name": "Sakana AI",
                        "raw": '{"error":{"message":"Access denied"}}',
                    },
                }
            },
        )
        result = classify_api_error(e, provider="openrouter", model="sakana/fugu-ultra")
        assert result.reason == FailoverReason.upstream_provider_error
        assert result.should_rotate_credential is False
        assert result.should_fallback is True
        assert result.error_context.get("upstream_provider") == "Sakana AI"

<<<<<<< HEAD
    def test_groq_upstream_403_classified_as_upstream_provider_error(self):
        """Groq 403 with upstream wrapper -> upstream_provider_error."""
        e = MockAPIError(
            "Provider returned error",
            status_code=403,
            body={
                "error": {
                    "message": "Provider returned error",
                    "metadata": {"provider_name": "Poolside", "raw": '{"error":{}}'},
                }
            },
        )
        result = classify_api_error(e, provider="groq", model="poolside/malibu")
        assert result.reason == FailoverReason.upstream_provider_error
        assert result.should_rotate_credential is False
        assert result.should_fallback is True
        assert result.error_context.get("upstream_provider") == "Poolside"
=======
>>>>>>> origin/main

    def test_together_upstream_403_classified_as_upstream_provider_error(self):
        """Together 403 with upstream wrapper -> upstream_provider_error."""
        e = MockAPIError(
            "Provider returned error",
            status_code=403,
            body={
                "error": {
                    "message": "Provider returned error",
                    "metadata": {"provider_name": "Together", "raw": '{"error":{}}'},
                }
            },
        )
        result = classify_api_error(e, provider="together", model="together/model")
        assert result.reason == FailoverReason.upstream_provider_error
        assert result.should_rotate_credential is False

    def test_fireworks_upstream_403_classified_as_upstream_provider_error(self):
        """Fireworks 403 with upstream wrapper -> upstream_provider_error."""
        e = MockAPIError(
            "Provider returned error",
            status_code=403,
            body={
                "error": {
                    "message": "Provider returned error",
                    "metadata": {"provider_name": "Fireworks", "raw": '{"error":{}}'},
                }
            },
        )
        result = classify_api_error(e, provider="fireworks", model="fireworks/model")
        assert result.reason == FailoverReason.upstream_provider_error
        assert result.should_rotate_credential is False

    def test_account_level_403_still_classified_as_auth_or_billing(self):
        """A real account-level 403 (no upstream wrapper) -> auth or billing."""
        # Plain 403 -> auth
        e = MockAPIError("Forbidden", status_code=403)
        result = classify_api_error(e, provider="openrouter", model="x")
        assert result.reason == FailoverReason.auth

        # 403 with billing language -> billing
        e = MockAPIError("spending limit reached", status_code=403)
        result = classify_api_error(e, provider="openrouter", model="x")
        assert result.reason == FailoverReason.billing
        assert result.should_rotate_credential is True

<<<<<<< HEAD
    def test_upstream_403_without_metadata_on_non_aggregator_not_matched(self):
        """'Provider returned error' alone on a non-aggregator provider -> plain auth."""
        e = MockAPIError(
            "Provider returned error",
            status_code=403,
            body={"error": {"message": "Provider returned error", "code": 403}},
        )
        result = classify_api_error(e, provider="anthropic", model="claude-sonnet-4")
        # Not an aggregator, so no upstream detection
        assert result.reason == FailoverReason.auth

    def test_upstream_provider_name_missing_yields_empty_context(self):
        """No provider_name in metadata -> upstream_provider_error with empty context."""
        e = MockAPIError(
            "Provider returned error",
            status_code=403,
            body={
                "error": {
                    "message": "Provider returned error",
                    "metadata": {"raw": '{"error":{"code":403}}'},
                }
            },
        )
        result = classify_api_error(e, provider="openrouter", model="x")
        assert result.reason == FailoverReason.upstream_provider_error
        assert result.error_context.get("upstream_provider") is None

    def test_overload_403_takes_precedence_over_upstream(self):
        """A 403 carrying overload language stays overloaded (retry same key)."""
        e = MockAPIError(
            "Provider returned error",
            status_code=403,
            body={
                "error": {
                    "message": "service is temporarily overloaded",
                    "metadata": {"provider_name": "Sakana AI"},
                }
            },
        )
        result = classify_api_error(e, provider="openrouter", model="x")
        # Overload disambiguation runs first; the outer message is the overload
        # phrase, so this is an overload, not an upstream provider error.
        assert result.reason == FailoverReason.overloaded
=======




# ── HTTP 408 request timeout ────────────────────────────────────────────

class Test408RequestTimeout:
    """HTTP 408 must never fall through to the non-retryable 'other 4xx'
    bucket (that abort persists an empty assistant turn — the "disappeared
    conversation" / blank-bubble symptom). ALL 408s are classified as a transient
    ``timeout``: retryable, and explicitly NOT should_compress.

    Design decision (field 2026-07-02): even the GitHub Copilot
    ``user_request_timeout`` / "Timed out reading request body ... use a
    smaller request size" case is a plain retry, NOT auto-compression. Real
    data showed the 408 is probabilistic jitter well below the hard prompt
    ceiling — the same ~785k-token request that 408'd once succeeded on the
    next attempt at ~786k — so retrying the same body usually works, and
    auto-compaction would silently delete conversation history for a merely
    transient timeout. Genuine over-window prompts surface as 413 /
    context_overflow (their own compression path); users compact 408-prone
    long sessions deliberately via ``/compress``.
    """

    def test_copilot_oversized_body_408_retries_as_timeout_not_compress(self):
        # The exact shape GitHub Copilot returns on a long session. It must
        # retry (timeout), and must NOT auto-compress.
        e = MockAPIError(
            "Error code: 408 - {'error': {'message': 'Timed out reading "
            "request body. Try again, or use a smaller request size.', "
            "'code': 'user_request_timeout'}}",
            status_code=408,
            body={"error": {"message": "Timed out reading request body. "
                            "Try again, or use a smaller request size.",
                            "code": "user_request_timeout"}},
        )
        result = classify_api_error(e, provider="copilot", model="claude-opus-4.8")
        assert result.reason == FailoverReason.timeout
        assert result.retryable is True
        assert result.should_compress is False




    def test_stale_breaker_runtime_error_triggers_fallback_not_retry(self):
        # The cross-turn stale-call circuit breaker (_check_stale_giveup in
        # chat_completion_helpers.py) raises a RuntimeError when the provider
        # has been unresponsive for N consecutive stale attempts.  This must
        # be classified as non-retryable + should_fallback so the retry loop
        # activates the fallback provider immediately instead of burning all
        # max_retries against the same dead provider (each retry hitting the
        # circuit breaker instantly with zero network overhead).
        e = RuntimeError(
            "Provider has been unresponsive (no response received) for "
            "6 consecutive stale attempts — aborting this call to "
            "avoid an indefinite stall. Switch models or start a new "
            "session, then retry."
        )
        result = classify_api_error(
            e, provider="openrouter", model="anthropic/claude-fable-5",
            approx_tokens=126327, context_length=200000, num_messages=274,
        )
        assert result.reason == FailoverReason.timeout
        assert result.retryable is False
        assert result.should_fallback is True
        assert result.should_compress is False


# ── Test: throttle vs overflow disambiguation + new overflow shapes ─────
# Port of anomalyco/opencode#37848 (expand context overflow patterns +
# rate-limit exclusion guard).

class TestThrottleVsOverflowDisambiguation:
    """Throttle messages that mention tokens must NOT route to compression."""

    def test_bedrock_throttling_too_many_tokens_is_rate_limit(self):
        # AWS Bedrock (and some proxies) surface throttling as
        # "Throttling error: Too many tokens, please wait before trying
        # again." — the "too many tokens" fragment sits in
        # _CONTEXT_OVERFLOW_PATTERNS, so before the "throttling" rate-limit
        # pattern this compressed a healthy session on every throttle.
        e = Exception(
            "Throttling error: Too many tokens, please wait before trying again."
        )
        result = classify_api_error(e, provider="bedrock", model="claude")
        assert result.reason == FailoverReason.rate_limit
        assert result.should_compress is False

    def test_plain_too_many_tokens_still_overflow(self):
        # Without any throttle wording, "Too many tokens" remains a
        # context-overflow signal (Z.AI / GLM family wording).
        e = Exception("Too many tokens")
        result = classify_api_error(e, provider="zai", model="glm-5")
        assert result.reason == FailoverReason.context_overflow
        assert result.should_compress is True


class TestExpandedOverflowPatterns:
    """New provider overflow wordings route into compression recovery."""

    def test_maximum_allowed_input_length_is_overflow(self):
        # Together/Fireworks-style wording — matched no pattern before.
        e = Exception(
            "Input length 131393 exceeds the maximum allowed input length "
            "of 131040 tokens."
        )
        result = classify_api_error(e, provider="together", model="m")
        assert result.reason == FailoverReason.context_overflow
        assert result.should_compress is True

    def test_request_too_large_message_only_is_payload_too_large(self):
        # Anthropic's structured 413 type re-wrapped by a proxy with no
        # status attribute — was falling through to `unknown`.
        e = Exception(
            '{"error":{"type":"request_too_large",'
            '"message":"Request exceeds the maximum size"}}'
        )
        result = classify_api_error(e, provider="anthropic", model="m")
        assert result.reason == FailoverReason.payload_too_large
        assert result.should_compress is True

    def test_longer_than_context_length_still_overflow(self):
        # Regression guard for wordings that already matched.
        e = Exception(
            "The input (516368 tokens) is longer than the model's context "
            "length (262144 tokens)."
        )
        result = classify_api_error(e, provider="openrouter", model="m")
        assert result.reason == FailoverReason.context_overflow



>>>>>>> origin/main
