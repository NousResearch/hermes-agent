"""Tests for agent.error_classifier — structured API error classification."""

from types import SimpleNamespace

import pytest
from agent.error_classifier import (
    ClassifiedError,
    FailoverReason,
    PROVIDER_STREAM_NON_JSON_ERROR_CODE,
    _CONTEXT_OVERFLOW_PATTERNS,
    _MEMORY_CEILING_PATTERNS,
    classify_api_error,
    _extract_status_code,
    _extract_error_body,
    _extract_error_code,
    _classify_402,
)


# ── Helper: mock API errors ────────────────────────────────────────────

class MockAPIError(Exception):
    """Simulates an OpenAI SDK APIStatusError."""
    def __init__(self, message, status_code=None, body=None, headers=None):
        super().__init__(message)
        self.status_code = status_code
        self.body = body or {}
        self.response = SimpleNamespace(headers=headers or {})


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
            "image_corrupt",
            "model_not_found", "format_error",
            "invalid_encrypted_content",
            "multimodal_tool_content_unsupported",
            "provider_policy_blocked",
            "content_policy_blocked",
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




# ── Captured oMLX prefill-memory-guard rejections (issue #52261) ───────
#
# Verbatim provider text from field reports on the issue thread, kept as
# module constants so every assertion below runs against exactly the same
# bytes the reporter captured.  Do not reflow or "tidy" these strings — the
# classifier matches substrings, so a reflow silently changes what is tested.

# oMLX 0.5.7, MID-STREAM abort.  Raised as a base ``openai.APIError`` from
# ``openai/_streaming.py:95``: that class carries no ``status_code`` at all,
# and the chat streaming generator's own ``except Exception`` handler builds
# ``{"error": {"message": str(e), "type": "server_error"}}`` without the
# ``isinstance(e, PrefillMemoryExceededError)`` discrimination the pre-stream
# path uses, so the structured ``prefill_memory_exceeded`` code is dropped
# too.  Classification therefore rests entirely on this message text.
# Reported by tkaufmann; the first code-less capture from a reporter other
# than the original 13.5 GB build, which is why it is pinned separately.
_OMLX_057_STREAM_ABORT = (
    "Prefill context too large for available memory (pre-chunk guard at 192 "
    "tokens, kv_len=37056): predicted peak would exceed prefill safety cap "
    "77.8GB (90% of metal_cap ceiling 86.4GB). Raise kernel "
    "iogpu.wired_limit_mb in Terminal (currently caps Metal at 86.40 GB), or "
    "reduce context length."
)

# oMLX 0.5.7, PRE-STREAM 400 — the same guard rejecting before the stream
# opens.  Here the prefill-aware body builder runs, so the wrapper prefix and
# the structured ``code``/``omlx_code`` survive.  Captured alongside the
# stream abort above (same reporter, same engine build), which is what makes
# the pair useful: it isolates the streaming exit as the only thing that
# strips the structure.
#
# COMPLETE capture.  An earlier revision of this fixture ended at "metal_cap
# ceiling 86.40 GB). ..." because the report it was transcribed from elided
# the remediation sentence; ``limit_bytes`` and the outer ``"type": "error"``
# sat behind a second elision.  All three are on the wire, and the first of
# them carries "context length", so this shape DOES reach the overflow branch
# — see the docstring on the 400 test below, which the elision had made
# wrong.  ``error.message`` is 562 characters.
#
# This is the 13 Aug 14:11 firing (``kv_len=83168``): the one that goes with
# the ``approx_tokens=63337`` the test below passes, and with the 174 -> 9
# message collapse the report is built on.  A 12 Aug 16:37 firing
# (``kv_len=51311``, ``estimated_bytes`` 84368206439, 560 characters) is
# equally real and identically worded — only the accounting numbers differ —
# so the self-consistent pairing is the one pinned here.
_OMLX_057_PREFILL_400_MESSAGE = (
    "oMLX prefill memory guard rejected this prompt: Prefill context too large for "
    "available memory (preflight safety guard, kv_len=83168, min_chunk=32): predicted "
    "peak would require ~78.31 GB (current 72.33 GB + KV 5.22 GB + min-chunk transient "
    "772.56 MB) but prefill safety cap is 77.76 GB (90% of metal_cap ceiling 86.40 GB). "
    "Raise kernel iogpu.wired_limit_mb in Terminal (currently caps Metal at 86.40 GB), "
    "or reduce context length. To continue, set Memory Guard to aggressive, raise the "
    "custom memory guard ceiling, free system memory, or compact/reduce context."
)

_OMLX_057_PREFILL_400_BODY = {
    "error": {
        "message": _OMLX_057_PREFILL_400_MESSAGE,
        "type": "invalid_request_error",
        "param": None,
        "code": "prefill_memory_exceeded",
        "omlx_code": "prefill_memory_exceeded",
        "estimated_bytes": 84082063701,
        "limit_bytes": 83493598003,
    },
    "type": "error",
}

# CONSTRUCTED — NOT a capture.  This is the truncated form the fixture above
# used to carry, kept deliberately because no real oMLX body omits the
# remediation sentence: the reporter went looking for a token-less capture and
# there is not one.  Its only job is to pin the LOWER BOUND — that the guard
# rests on the memory wording and the structured code, not on the presence of
# "context length" — so that a future engine copy-edit which drops the
# remediation hint cannot silently reopen this bug.  Do not describe it as
# captured output anywhere.
_SYNTHETIC_PREFILL_400_NO_OVERFLOW_TOKEN = (
    "oMLX prefill memory guard rejected this prompt: Prefill context too "
    "large for available memory (preflight safety guard, kv_len=51311, "
    "min_chunk=32): predicted peak would require ~78.57 GB (current 71.22 GB "
    "+ KV 3.28 GB + min-chunk transient 4.08 GB) but prefill safety cap is "
    "77.76 GB (90% of metal_cap ceiling 86.40 GB). ..."
)

# oMLX 0.5.7, THIRD shape — a different guard from the two above.  The prefill
# guard rejects a prompt before admitting it; this is the PROCESS MEMORY
# ENFORCER aborting a request already in flight because resident usage crossed
# a watermark.  Different guard, different trigger, same engine, and the same
# closing advice to reduce context length.
#
# Pinned because its unguarded failure mode is the worst of the three and is
# NOT the overflow misroute the other two suffer: "process memory limit
# exceeded" contains "limit exceeded", which is a _USAGE_LIMIT_PATTERNS entry
# checked ahead of the overflow branch, and the body carries none of the
# transient signals ("try again", "retry", "wait", …) that disambiguate a
# usage limit toward rate_limit.  So an unguarded classifier calls a local
# Metal watermark abort a BILLING failure — non-retryable, and reported to the
# user as an account problem.  Three shapes, two guards, one engine; a memory
# guard has to sit ahead of the overflow branch rather than beside it, and
# ahead of the usage-limit branch too.
#
# SEPARATOR: SETTLED — it is U+2192, and this fixture already carried the right
# one.  An earlier revision of this comment said the spelling "has not been
# established from the raw log"; it has been since.  ``grep`` over the raw logs
# returns twelve occurrences in ``agent.log`` and twelve in ``errors.log``,
# every one of them the bytes ``e2 86 92``, with zero occurrences of the ASCII
# "safe -> balanced" in either; and the engine holds the ladder in a single
# module-level constant, ``MEMORY_GUARD_TIER_LADDER = "safe → balanced →
# aggressive"``, with no ASCII sibling in the bundle.  Strictly the grep proves
# the decoded string rather than the wire encoding, and the constant settles the
# rest.
#
# The ASCII form below is therefore CONSTRUCTED, not captured — the same status
# as _SYNTHETIC_PREFILL_400_NO_OVERFLOW_TOKEN above, and it is labelled here so
# nobody reads it as a second real spelling.  It is kept because its job is a
# lower bound, not a claim: none of the matching keys ("memory limit exceeded",
# "memory_guard_tier", "context length") involve the separator, and the test
# below pins that, so a future transcoding hop — a proxy, a log shipper, a
# terminal — cannot quietly become load-bearing.  Do not add an ASCII spelling
# to any production pattern list.
_OMLX_057_PROCESS_MEMORY_ABORT = (
    "Request aborted: process memory limit exceeded (usage 49.8 GB, abort "
    "threshold (hard watermark) 49.2 GB, ceiling 51.8 GB). Reduce context "
    "length, free system memory, or loosen memory_guard_tier "
    "(safe → balanced → aggressive)."
)

_OMLX_057_PROCESS_MEMORY_ABORT_ASCII_ARROW = (
    _OMLX_057_PROCESS_MEMORY_ABORT.replace("→", "->")
)

# The SAME shape, CAPTURED AGAIN nine days later on a different host, with the
# remediation tail reworded.  Both halves are captures: 2 Aug above, 11 Aug
# here.
#
# Not a host difference.  engine_core.py on the first host carries an mtime one
# day after its own 2 Aug capture, and the tail now comes from
# describe_ceiling_binding(), which emits neither "loosen" nor "free system
# memory" in any branch — so that host cannot reproduce its own earlier wording
# today.  The engine reworded it.
#
# This is the same drift the memory-accounting entries in
# _MEMORY_CEILING_PATTERNS were added for, caught a second time on a different
# sentence: "loosen memory_guard_tier ... free system memory" became "Close
# other apps to free RAM ... raise memory_guard_tier", and the bound cap moved
# from a bare "ceiling" to a "dynamic ceiling" with the static cap reported
# alongside it.  Pinned so the pair is on the record rather than only the
# earlier half, and so a third copy-edit has something to fail against.
_OMLX_057_PROCESS_MEMORY_ABORT_REWORDED = (
    "Request aborted: process memory limit exceeded (usage 53.3 GB, abort "
    "threshold (hard watermark) 57.4 GB, dynamic ceiling 60.4 GB). Close other "
    "apps to free RAM (static cap is 90.00 GB but only 7.15 GB is reclaimable "
    "right now), raise memory_guard_tier (safe → balanced → aggressive), or "
    "reduce context length."
)

# oMLX 0.5.7, FOURTH shape — the model LOAD guard, reached before any prefill
# happens at all, and the only one of the four that arrives as HTTP 507.
# ``omlx/server.py`` maps both ``ModelTooLargeError`` and
# ``InsufficientMemoryError`` to 507, and ``/v1/chat/completions``,
# ``/v1/completions`` and ``/v1/messages`` all reach them through
# ``get_engine_for_model`` -> ``get_engine``, so an ordinary chat call against a
# model the host cannot seat comes back on a status the classifier had no
# branch for.
#
# CAPTURED: verbatim from a reporter's log, 11 Aug 04:03:30, a plain
# ``/v1/chat/completions`` call surfaced by the OpenAI SDK as
# ``openai.InternalServerError: Error code: 507``.  ``code`` is null in the
# body — the wrapper that carries the structured ``prefill_memory_*`` codes is
# on the prefill path, not this one — so classification rests entirely on the
# message, which carries "memory ceiling" and "memory_guard_tier".
#
# Unguarded, this lands in the generic "other 5xx" bucket: retryable=True but
# should_fallback=False.  ``should_fallback`` is the bit that matters here,
# because a model that does not fit under the ceiling does not begin to fit
# within the retry budget.  The ``InsufficientMemoryError`` sibling is not
# pinned: it is documented as sharing the 507 mapping, but no body for it has
# been captured, so there is nothing to assert against.
_OMLX_057_MODEL_LOAD_507 = (
    "Model 'Qwen3.6-27B-MLX-8bit' (33.95GB) does not fit under the dynamic "
    "memory ceiling (25.22GB). Close other apps to free RAM (static cap is "
    "90.00GB but only 7.15GB is reclaimable right now), raise memory_guard_tier "
    "(safe → balanced → aggressive), or use a smaller model."
)

_OMLX_057_MODEL_LOAD_507_BODY = {
    "error": {
        "message": _OMLX_057_MODEL_LOAD_507,
        "type": "server_error",
        "param": None,
        "code": None,
    },
}


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

    def test_404_requires_available_credits_is_billing(self):
        e = MockAPIError(
            "Not Found",
            status_code=404,
            body={
                "status": 404,
                "message": (
                    "Model 'openai/gpt-5.5-pro' requires available credits. "
                    "Your account balance is too low to use paid models — "
                    "add credits at https://portal.nousresearch.com or pick a free model."
                ),
            },
        )
        result = classify_api_error(e, provider="nous", model="openai/gpt-5.5-pro")
        assert result.reason == FailoverReason.billing
        assert result.retryable is False
        assert result.should_fallback is True

    def test_wrapped_402_uses_nested_body_message(self):
        inner = MockAPIError(
            "inner",
            status_code=402,
            body={"error": {"message": "Usage limit reached, try again in 5 minutes"}},
        )
        outer = Exception("outer")
        outer.__cause__ = inner

        result = classify_api_error(outer)

        assert result.reason == FailoverReason.rate_limit
        assert result.retryable is True
        assert result.message == "Usage limit reached, try again in 5 minutes"

    # ── Rate limit ──

    def test_429_rate_limit(self):
        e = MockAPIError("Too Many Requests", status_code=429)
        result = classify_api_error(e)
        assert result.reason == FailoverReason.rate_limit
        assert result.should_fallback is True

    def test_anthropic_429_usage_limit_without_reset_is_billing(self):
        e = MockAPIError(
            "usage limit reached",
            status_code=429,
            body={
                "error": {
                    "type": "usage_limit_reached",
                    "message": "Your account has reached its usage limit.",
                }
            },
        )

        result = classify_api_error(e, provider="anthropic", model="claude-opus-5")

        assert result.reason == FailoverReason.billing
        assert result.retryable is False
        assert result.should_fallback is True

    def test_anthropic_429_usage_limit_with_reset_stays_rate_limit(self):
        e = MockAPIError(
            "usage limit reached; resets at 2026-08-24T10:00:00Z",
            status_code=429,
        )

        result = classify_api_error(e, provider="anthropic", model="claude-opus-5")

        assert result.reason == FailoverReason.rate_limit
        assert result.retryable is True

    @pytest.mark.parametrize(
        ("reset_field", "reset_value"),
        [
            ("resets_in_seconds", 3600),
            ("resets_at", "2026-08-24T10:00:00Z"),
            ("reset_at", "2026-08-24T10:00:00Z"),
            ("retry_after", 3600),
        ],
    )
    def test_anthropic_429_usage_limit_with_structured_reset_stays_rate_limit(
        self,
        reset_field,
        reset_value,
    ):
        e = MockAPIError(
            "usage limit reached",
            status_code=429,
            body={
                "error": {
                    "type": "usage_limit_reached",
                    "message": "Your account has reached its usage limit.",
                    reset_field: reset_value,
                }
            },
        )

        result = classify_api_error(e, provider="anthropic", model="claude-opus-5")

        assert result.reason == FailoverReason.rate_limit
        assert result.retryable is True

    @pytest.mark.parametrize("header", ["Retry-After", "x-ratelimit-reset"])
    def test_anthropic_429_usage_limit_with_reset_header_stays_rate_limit(self, header):
        e = MockAPIError(
            "usage limit reached",
            status_code=429,
            body={
                "error": {
                    "type": "usage_limit_reached",
                    "message": "Your account has reached its usage limit.",
                }
            },
            headers={header: "3600"},
        )

        result = classify_api_error(e, provider="anthropic", model="claude-opus-5")

        assert result.reason == FailoverReason.rate_limit
        assert result.retryable is True

    def test_429_generic_quota_wall_is_billing(self):
        # Broadened from the narrow "usage limit" core to the full
        # _USAGE_LIMIT_PATTERNS: a bare "quota" / "limit exceeded" 429 with no
        # reset signal is a hard wall, not a retryable throttle. (credit #39441)
        for msg in ("Monthly quota reached.", "API key limit exceeded."):
            e = MockAPIError(msg, status_code=429)
            result = classify_api_error(e, provider="groq", model="llama-3")
            assert result.reason == FailoverReason.billing, msg
            assert result.retryable is False, msg

    def test_429_insufficient_credits_is_billing(self):
        e = MockAPIError("Insufficient credits remaining.", status_code=429)
        result = classify_api_error(e, provider="openrouter", model="x")
        assert result.reason == FailoverReason.billing
        assert result.retryable is False

    def test_429_rate_limit_phrase_never_promotes_to_billing(self):
        # The exclusion guard: "Rate limit exceeded" contains the
        # "limit exceeded" usage-limit substring, but an explicit rate-limit
        # phrase must stay a retryable rate limit. (guard credit #39441)
        for msg in (
            "Rate limit exceeded, please slow down.",
            "Too many requests; rate_limit hit.",
        ):
            e = MockAPIError(msg, status_code=429)
            result = classify_api_error(e, provider="anthropic", model="claude-opus-5")
            assert result.reason == FailoverReason.rate_limit, msg
            assert result.retryable is True, msg

    def test_codex_weekly_usage_limit_resets_in_stays_rate_limit(self):
        # Codex surfaces "Weekly usage limit reached. Resets in 6hr 29min."
        # "resets in" was NOT a transient signal before, so this wrongly read
        # as terminal billing. (transient-signal credit #63021)
        e = MockAPIError(
            "Weekly usage limit reached. Resets in 6hr 29min.",
            status_code=429,
        )
        result = classify_api_error(e, provider="openai-codex", model="gpt-5-codex")
        assert result.reason == FailoverReason.rate_limit
        assert result.retryable is True

    @pytest.mark.parametrize(
        "phrase",
        [
            "usage limit reached, reset after 3600s",
            "usage limit reached, available in 42 minutes",
            "usage limit reached; 20 requests per minute",
        ],
    )
    def test_429_usage_limit_with_extra_transient_phrases_stays_rate_limit(self, phrase):
        # Additional transient signals. (credit #74785)
        e = MockAPIError(phrase, status_code=429)
        result = classify_api_error(e, provider="anthropic", model="claude-opus-5")
        assert result.reason == FailoverReason.rate_limit
        assert result.retryable is True

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




    def test_non_json_stream_validation_error_is_non_retryable(self):
        e = MockAPIError(
            "Provider stream returned non-JSON SSE data",
            body={
                "error": {
                    "code": PROVIDER_STREAM_NON_JSON_ERROR_CODE,
                    "message": (
                        "request validation failed: unsupported reasoning_effort"
                    ),
                }
            },
        )

        result = classify_api_error(e)

        assert result.status_code is None
        assert result.reason == FailoverReason.format_error
        assert result.retryable is False
        assert result.should_fallback is True

    def test_non_json_stream_unknown_error_remains_retryable(self):
        e = MockAPIError(
            "Provider stream returned non-JSON SSE data",
            body={
                "error": {
                    "code": PROVIDER_STREAM_NON_JSON_ERROR_CODE,
                    "message": "upstream sent opaque plain-text stream data",
                }
            },
        )

        result = classify_api_error(e)

        assert result.status_code is None
        assert result.reason == FailoverReason.unknown
        assert result.retryable is True
        assert result.should_fallback is False

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

    def test_404_bare_model_id_missing_prefix_is_model_not_found(self):
        """A bare id the provider only serves as ``vendor/id`` is malformed.

        Regression for #78796: NVIDIA NIM answers a prefix-less
        ``nemotron-3-ultra-550b-a55b`` with a naked ``404 page not found``.
        Without the catalogue check this fell into the generic branch and
        burned three retries on a deterministic failure, reporting what
        looked like an outage.
        """
        e = MockAPIError("404 page not found", status_code=404)
        result = classify_api_error(
            e, provider="nvidia", model="nemotron-3-ultra-550b-a55b"
        )
        assert result.reason == FailoverReason.model_not_found
        assert result.retryable is False

    def test_404_correctly_prefixed_model_stays_generic(self):
        """A properly prefixed id hitting a 404 is a real endpoint problem —
        it must keep the retryable generic classification."""
        e = MockAPIError("404 page not found", status_code=404)
        result = classify_api_error(
            e, provider="nvidia", model="nvidia/nemotron-3-ultra-550b-a55b"
        )
        assert result.reason == FailoverReason.unknown
        assert result.retryable is True

    def test_404_unknown_bare_model_stays_generic(self):
        """A local NIM container isn't in the catalogue — no verdict invented."""
        e = MockAPIError("404 page not found", status_code=404)
        result = classify_api_error(e, provider="nvidia", model="my-local-nim")
        assert result.reason == FailoverReason.unknown
        assert result.retryable is True

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







    # ── Local-inference memory/resource-ceiling 400s (issue #52261) ──
    # A provider memory-guard / OOM rejection often suggests "reduce context
    # length", colliding with the context-overflow patterns.  Compressing
    # history cannot relieve a prefill memory peak, so these must classify as
    # transient ``overloaded`` (retry, no compression) — NOT context_overflow.

    def test_400_omlx_prefill_memory_guard_is_overloaded_not_context_overflow(self):
        # Verbatim oMLX memory-guard rejection on a TINY (~5.7k-token) prompt.
        e = MockAPIError(
            "oMLX prefill memory guard rejected this prompt: Prefill would "
            "require ~13.87 GB peak (current 13.46 GB + KV+SDPA 419.28 MB) but "
            "dynamic ceiling is 13.50 GB. ... or reduce context length.",
            status_code=400,
        )
        result = classify_api_error(
            e, provider="custom", model="omlx-chat",
            approx_tokens=5700, context_length=64000,
        )
        assert result.reason == FailoverReason.overloaded
        # Must NOT enter the compress-and-shrink loop.
        assert result.should_compress is False

    def test_400_process_memory_limit_exceeded_is_overloaded(self):
        e = MockAPIError(
            "Request aborted: process memory limit exceeded (usage 13.7 GB, "
            "ceiling 13.5 GB). Reduce context size or lower memory_guard_tier.",
            status_code=400,
        )
        result = classify_api_error(
            e, provider="custom", model="omlx-chat",
            approx_tokens=5745, context_length=64000,
        )
        assert result.reason == FailoverReason.overloaded
        assert result.should_compress is False

    def test_no_status_prefill_too_large_for_available_memory_is_overloaded(self):
        # Streaming / no-status path: APIError text, no HTTP code.
        e = Exception("Prefill context too large for available memory")
        result = classify_api_error(
            e, provider="custom", model="omlx-chat",
            approx_tokens=15000, context_length=64000,
        )
        assert result.reason == FailoverReason.overloaded
        assert result.should_compress is False

    def test_400_genuine_context_window_overflow_still_compresses(self):
        # NEGATIVE/invariant guard: a real window overflow must STILL route to
        # context_overflow + compression (proves the memory guard is precise
        # and does not swallow genuine "reduce the length" overflows).
        e = MockAPIError(
            "This model's maximum context length is 8192 tokens; "
            "reduce the length of the messages.",
            status_code=400,
        )
        result = classify_api_error(
            e, provider="custom", model="x",
            approx_tokens=9000, context_length=8192,
        )
        assert result.reason == FailoverReason.context_overflow
        assert result.should_compress is True

    def test_streaming_process_memory_limit_exceeded_is_overloaded_not_billing(self):
        # Status-LESS streaming abort (oMLX emits this on the streaming path
        # under memory pressure).  "process memory limit exceeded" contains the
        # substring "limit exceeded", a _USAGE_LIMIT_PATTERN — so before the
        # memory guard was moved ahead of the billing/usage checks in
        # _classify_by_message, this misclassified as BILLING (retryable=False,
        # rotate-credential), a different wrong bucket than context_overflow.
        e = Exception(
            "Request aborted: process memory limit exceeded (usage 13.7 GB, "
            "ceiling 13.5 GB). Reduce context size or lower memory_guard_tier."
        )
        result = classify_api_error(
            e, provider="custom", model="omlx-chat",
            approx_tokens=5745, context_length=64000,
        )
        assert result.reason == FailoverReason.overloaded
        assert result.retryable is True
        assert result.should_compress is False
        # Must NOT be routed into credential rotation like a billing error.
        assert result.should_rotate_credential is False
        # Failover-eligible: a wedged local memory wall recovers via a roomier
        # provider once retries are exhausted, not by hammering the same server.
        assert result.should_fallback is True

    def test_400_prefill_memory_code_reworded_message_is_overloaded(self):
        # Direct (non-proxied) connection: the message is reworded with NO
        # memory substring, but the structured body carries the unambiguous
        # ``code: "prefill_memory_exceeded"`` (+ limit_bytes in *bytes*).
        # Without the error-code guard this fell through to a non-retryable
        # ``format_error`` (no overflow wording to catch it either).
        body = {"error": {
            "message": "Prompt rejected by the prefill guard. Try a smaller request.",
            "type": "invalid_request_error",
            "code": "prefill_memory_exceeded",
            "omlx_code": "prefill_memory_exceeded",
            "limit_bytes": 14495514624,
        }, "type": "error"}
        e = MockAPIError(
            "Prompt rejected by the prefill guard. Try a smaller request.",
            status_code=400, body=body,
        )
        result = classify_api_error(
            e, provider="custom", model="omlx-chat",
            approx_tokens=5700, context_length=64000,
        )
        assert result.reason == FailoverReason.overloaded
        assert result.retryable is True
        assert result.should_compress is False
        assert result.should_fallback is True

    def test_no_status_prefill_memory_code_is_overloaded(self):
        # Streaming / no-status path carrying only the structured code (message
        # fully reworded).  Previously fell to the retryable ``unknown`` bucket;
        # the _classify_by_error_code memory-code guard now catches it.
        body = {"error": {
            "message": "Prompt rejected by the prefill guard. Try a smaller request.",
            "code": "prefill_memory_exceeded",
            "limit_bytes": 14495514624,
        }, "type": "error"}
        e = MockAPIError(
            "Prompt rejected by the prefill guard. Try a smaller request.",
            status_code=None, body=body,
        )
        result = classify_api_error(
            e, provider="custom", model="omlx-chat",
            approx_tokens=5700, context_length=64000,
        )
        assert result.reason == FailoverReason.overloaded
        assert result.retryable is True
        assert result.should_compress is False
        # Same rejection, detected via the structured code instead of the
        # message: the recovery annotation must be identical to the 400 and
        # message-pattern paths, not silently weaker.  Guards the shared
        # _memory_ceiling_result contract against per-route drift.
        assert result.should_fallback is True

    @pytest.mark.parametrize("status_code", [400, None])
    def test_prefill_memory_aborted_code_is_overloaded(self, status_code):
        # The SIBLING code.  oMLX's prefill-memory body builder picks between
        # ``prefill_memory_exceeded`` and ``prefill_memory_aborted`` by
        # exception type (``PrefillMemoryAbortedError``): "exceeded" is the
        # prompt turned away at admission, "aborted" is the prompt admitted and
        # then killed mid-prefill.  Same guard, same wall, same recovery — but
        # only "exceeded" was in _MEMORY_CEILING_ERROR_CODES, so with the
        # message stripped of memory wording (the scenario the code layer
        # exists for) the two diverged: exceeded -> overloaded/retryable,
        # aborted -> format_error/not-retryable on the 400 path and the
        # retryable ``unknown`` bucket on the status-less one.
        #
        # NOT a capture: the reporter's logs carry only
        # ``prefill_memory_exceeded``.  The code pairing is read from the
        # engine's body builder, and the reworded-message premise is the one
        # already established by the 0.5.6 -> 0.5.7 rewording documented below.
        body = {"error": {
            "message": "Prompt aborted by the prefill guard. Try a smaller request.",
            "type": "invalid_request_error",
            "code": "prefill_memory_aborted",
            "omlx_code": "prefill_memory_aborted",
            "limit_bytes": 14495514624,
        }, "type": "error"}
        e = MockAPIError(
            "Prompt aborted by the prefill guard. Try a smaller request.",
            status_code=status_code, body=body,
        )
        result = classify_api_error(
            e, provider="custom", model="omlx-chat",
            approx_tokens=5700, context_length=64000,
        )
        assert result.reason == FailoverReason.overloaded
        assert result.retryable is True
        assert result.should_compress is False
        assert result.should_fallback is True
        # The premise of this test: nothing in the message could have carried
        # the classification.
        assert not any(
            p in body["error"]["message"].lower() for p in _MEMORY_CEILING_PATTERNS
        )

    def test_400_litellm_proxy_flattened_memory_guard_is_overloaded(self):
        # Proxy-flattened transport shape (real oMLX-behind-LiteLLM capture from
        # issue #52261): LiteLLM collapses the structured body into a single
        # "OpenAIException - <message>" string and DROPS the ``code``, so the
        # error-code guard cannot fire — only the message substring survives.
        # On clean main this 400 matched the bare "reduce context length"
        # overflow pattern and misclassified as context_overflow (→ compress →
        # wedge-loop reset).  The message-pattern memory guard ("memory guard",
        # "dynamic ceiling", "prefill would require") must catch it even with no
        # status code or structured code to lean on.
        e = MockAPIError(
            "litellm.BadRequestError: OpenAIException - oMLX prefill memory "
            "guard rejected this prompt: Prefill would require ~13.87 GB peak "
            "(current 13.46 GB + KV+SDPA 419.28 MB) but dynamic ceiling is "
            "13.50 GB. Raise custom_ceiling_bytes in admin Memory settings, or "
            "reduce context length.",
            status_code=400,
        )
        result = classify_api_error(
            e, provider="custom", model="omlx-chat",
            approx_tokens=5700, context_length=64000,
        )
        assert result.reason == FailoverReason.overloaded
        assert result.retryable is True
        # Must NOT enter the compress-and-shrink loop a context_overflow would.
        assert result.should_compress is False
        assert result.should_fallback is True

    # ── Memory-ceiling rejections surfaced as 5xx (issue #52261) ──
    # main routes explicit context-overflow wording in a 500/502/503/529 body
    # into compression (llama.cpp/vLLM report overflow-as-5xx).  A memory abort
    # from those same servers carries the same "reduce context length" hint, so
    # without a guard it reaches the identical wedge-loop these tests exist to
    # prevent — the 400 fix alone does not cover it.

    @pytest.mark.parametrize("status_code", [500, 502, 503, 529])
    def test_5xx_memory_guard_is_overloaded_not_context_overflow(self, status_code):
        # Body carries BOTH memory wording and the colliding "reduce context
        # length" hint.  Fails before the 5xx guard: matches
        # _CONTEXT_OVERFLOW_PATTERNS → context_overflow + should_compress=True.
        e = MockAPIError(
            "oMLX prefill memory guard rejected this prompt: Prefill would "
            "require ~13.87 GB peak but dynamic ceiling is 13.50 GB. Raise "
            "custom_ceiling_bytes in admin Memory settings, or reduce "
            "context length.",
            status_code=status_code,
        )
        result = classify_api_error(
            e, provider="custom", model="omlx-chat",
            approx_tokens=5700, context_length=64000,
        )
        assert result.reason == FailoverReason.overloaded
        assert result.retryable is True
        # The whole point: compressing history cannot lower a prefill peak.
        assert result.should_compress is False
        # Same contract as the 400 / code / message routes.
        assert result.should_fallback is True

    @pytest.mark.parametrize("status_code", [500, 502, 503, 529])
    def test_5xx_memory_code_reworded_message_is_overloaded(self, status_code):
        # Direct (non-proxied) 5xx whose message was reworded to carry no memory
        # substring — only the structured code identifies it.
        body = {"error": {
            "message": "Prompt rejected by the prefill guard.",
            "code": "prefill_memory_exceeded",
            "limit_bytes": 14495514624,
        }, "type": "error"}
        e = MockAPIError(
            "Prompt rejected by the prefill guard.",
            status_code=status_code, body=body,
        )
        result = classify_api_error(
            e, provider="custom", model="omlx-chat",
            approx_tokens=5700, context_length=64000,
        )
        assert result.reason == FailoverReason.overloaded
        assert result.should_compress is False
        assert result.should_fallback is True

    @pytest.mark.parametrize("status_code", [500, 502, 503, 529])
    def test_5xx_genuine_context_overflow_still_compresses(self, status_code):
        # CONTROL for the guard above: a real overflow body (no memory wording)
        # must STILL take the compression path main added.  Proves the memory
        # guard is narrow and did not disable overflow-as-5xx handling.
        e = MockAPIError(
            "the request exceeds the available context size. try increasing "
            "the context size or enable context shift",
            status_code=status_code,
        )
        result = classify_api_error(
            e, provider="custom", model="local-llama",
            approx_tokens=150000, context_length=64000,
        )
        assert result.reason == FailoverReason.context_overflow
        assert result.retryable is True
        assert result.should_compress is True

    def test_503_generic_overload_unaffected_by_memory_guard(self):
        # CONTROL: a plain transient 503 with neither memory nor overflow
        # wording keeps its existing bare-``overloaded`` classification.
        e = MockAPIError("Service temporarily unavailable, please retry.",
                         status_code=503)
        result = classify_api_error(
            e, provider="custom", model="x",
            approx_tokens=5000, context_length=64000,
        )
        assert result.reason == FailoverReason.overloaded
        assert result.retryable is True
        assert result.should_compress is False

    def test_500_generic_server_error_unaffected_by_memory_guard(self):
        # CONTROL: a plain 500 with no memory/overflow wording must still be a
        # generic server_error, not swept into the memory bucket.
        e = MockAPIError("Internal server error", status_code=500)
        result = classify_api_error(
            e, provider="custom", model="x",
            approx_tokens=5000, context_length=64000,
        )
        assert result.reason == FailoverReason.server_error
        assert result.retryable is True

    # ── Memory-ceiling rejections surfaced as 507 (issue #52261) ──
    #
    # The model-LOAD guard, which is not the prefill guard and does not come
    # back as a 400.  See the _OMLX_057_MODEL_LOAD_507 fixture.

    def test_507_omlx_model_load_ceiling_is_overloaded_with_fallback(self):
        # Verbatim 507 capture.  Before the 507 branch this fell through to the
        # generic "other 5xx" rule: server_error, retryable=True, but
        # should_fallback=False — so once the retries were spent the turn died
        # on a host whose memory wall had not moved, with no failover to a
        # roomier provider.  Every other memory-ceiling route already sets
        # should_fallback; this one has to match them.
        e = MockAPIError(
            "Error code: 507 - " + repr(_OMLX_057_MODEL_LOAD_507_BODY),
            status_code=507,
            body=_OMLX_057_MODEL_LOAD_507_BODY,
        )
        result = classify_api_error(
            e, provider="custom", model="qwen3.6-27b-mlx-8bit",
            approx_tokens=63337, context_length=256000,
        )
        assert result.reason == FailoverReason.overloaded
        assert result.retryable is True
        # Nothing to compress: the model does not fit before a single token of
        # the conversation is read.
        assert result.should_compress is False
        # The regression this test exists for.
        assert result.should_fallback is True
        assert result.should_rotate_credential is False

    # ── Memory-ceiling rejections surfaced as 409 (issue #52261) ──
    #
    # NOT CAPTURED.  Read from the engine's source: oMLX's ``ModelLoadingError``
    # carries "Model 'X' load aborted: process memory limit exceeded" and
    # ``server.py`` maps it to 409.  No reporter has produced a 409 body, so
    # this fixture is constructed from that message, and the test is pinned as
    # a code reading rather than as evidence.

    def test_409_memory_abort_is_overloaded_not_format_error(self):
        # Before the 409 branch this reached the generic "other 4xx" bucket and
        # was reported as format_error / retryable=False: a transient memory
        # abort called a malformed request.  should_fallback was already set,
        # so the turn was not stranded — but it was never retried against the
        # primary either, even though the abort clears the moment the host
        # reclaims memory.
        e = MockAPIError(
            "Model 'Qwen3.6-27B-MLX-8bit' load aborted: process memory limit "
            "exceeded",
            status_code=409,
        )
        result = classify_api_error(
            e, provider="custom", model="qwen3.6-27b-mlx-8bit",
            approx_tokens=63337, context_length=256000,
        )
        assert result.reason == FailoverReason.overloaded
        assert result.retryable is True
        assert result.should_compress is False
        assert result.should_fallback is True

    def test_409_without_memory_wording_is_still_format_error(self):
        # CONTROL: 409 Conflict has ordinary uses (a model swap already in
        # flight, a duplicate request id).  Those must keep the existing 4xx
        # treatment, so the new branch cannot claim the whole status code.
        e = MockAPIError("A model swap is already in progress.", status_code=409)
        result = classify_api_error(
            e, provider="custom", model="x",
            approx_tokens=5000, context_length=64000,
        )
        assert result.reason == FailoverReason.format_error
        assert result.retryable is False
        assert result.should_fallback is True

    def test_507_without_memory_wording_is_generic_server_error(self):
        # CONTROL: 507 Insufficient Storage also has its literal meaning.  A
        # body with no memory wording must keep the generic 5xx treatment, so
        # the new branch cannot claim the whole status code.
        e = MockAPIError("Insufficient storage on device.", status_code=507)
        result = classify_api_error(
            e, provider="custom", model="x",
            approx_tokens=5000, context_length=64000,
        )
        assert result.reason == FailoverReason.server_error
        assert result.retryable is True
        assert result.should_compress is False

    def test_genuine_billing_credit_limit_still_billing(self):
        # NEGATIVE/invariant guard: a real billing exhaustion message must STILL
        # classify as billing — proves the memory-guard reorder in
        # _classify_by_message did not swallow legitimate billing errors.
        e = Exception("Your account has insufficient credits to complete this request.")
        result = classify_api_error(
            e, provider="openrouter", model="x",
            approx_tokens=5000, context_length=64000,
        )
        assert result.reason == FailoverReason.billing
        assert result.retryable is False

    def test_streaming_omlx_057_prefill_abort_without_status_or_code(self):
        # Verbatim mid-stream capture (issue #52261): base ``openai.APIError``
        # with NO http status and NO structured code, so neither the 400 route
        # nor the error-code route can fire — only _classify_by_message can.
        # The single overflow token in the text is "context length", from the
        # trailing remediation hint, so unguarded main reads a memory rejection
        # as a window overflow, compresses, and reports the session
        # uncompressible at 37,629 tokens two minutes later.
        #
        # ``context_length`` here only has to be a plausible window well above
        # 37,629; it is not a claim about what the compressor resolved.  (An
        # earlier revision of this comment asserted that Hermes could not read
        # the model's window because it was nested under ``text_config``, and
        # that 256,000 was therefore the number compression ran against.  Both
        # were wrong: oMLX reports ``{"max_model_len": 262144}`` on /v1/models,
        # which model_metadata already parses, and 256,000 is a dashboard
        # display value from a different code path.)  What the assertions below
        # rest on is only this: 37,629 tokens is nowhere near any plausible
        # ceiling, so the request was rejected on a GPU memory peak, not on
        # context.
        e = Exception(_OMLX_057_STREAM_ABORT)
        result = classify_api_error(
            e, provider="custom", model="qw36-27b-8bit-mtp:agent",
            approx_tokens=37629, context_length=256000,
        )
        assert result.reason == FailoverReason.overloaded
        assert result.retryable is True
        # The whole point: compression cannot lower a prefill memory peak.
        assert result.should_compress is False
        # Nor is it a credential problem.
        assert result.should_rotate_credential is False
        # A wedged local memory wall recovers via a roomier provider.
        assert result.should_fallback is True

    def test_400_omlx_057_prefill_memory_exceeded_is_overloaded(self):
        # Verbatim pre-stream capture (issue #52261), paired with the streaming
        # abort above from the same engine build.  ``str(error)`` is the OpenAI
        # SDK's rendering of an ``openai.BadRequestError``, i.e. the whole body
        # repr — which embeds ``'type': 'invalid_request_error'``.  That literal
        # is why _REQUEST_VALIDATION_PATTERNS excludes it from its own match;
        # without that exclusion this 400 would be a non-retryable format_error
        # before any memory or overflow check ran.
        #
        # On an unguarded classifier this body is read as context_overflow with
        # should_compress=True — the SAME failure direction as the mid-stream
        # shape above, from the same rejection: its remediation sentence ends
        # "or reduce context length", which is the single overflow token.
        #
        # (An earlier revision of this comment said the opposite — that the body
        # carried no overflow token and fell through to a non-retryable
        # format_error.  That described the elided transcription this fixture
        # used to carry, not oMLX; see the fixture comment.  The same claim is
        # in the message of commit 4db5a59, which is pushed and cannot be
        # amended, so it is corrected here instead.)
        e = MockAPIError(
            "Error code: 400 - " + repr(_OMLX_057_PREFILL_400_BODY),
            status_code=400,
            body=_OMLX_057_PREFILL_400_BODY,
        )
        result = classify_api_error(
            e, provider="custom", model="qw36-27b-8bit-mtp:agent",
            approx_tokens=63337, context_length=256000,
        )
        assert result.reason == FailoverReason.overloaded
        # A memory wall is transient; a format_error would strand the turn.
        assert result.retryable is True
        assert result.should_compress is False
        assert result.should_fallback is True

    def test_400_prefill_memory_code_without_overflow_token_is_overloaded(self):
        # LOWER BOUND, on a CONSTRUCTED body — see the fixture comment.  Every
        # real oMLX rejection ends with a "reduce context length" hint, so the
        # guard is never actually asked to work without one.  This pins that it
        # could: strip the remediation sentence and classification still rests
        # on the memory wording plus ``code: prefill_memory_exceeded``, so an
        # engine copy-edit that drops the hint cannot silently reopen #52261.
        #
        # This is also the one shape whose unguarded failure mode is
        # format_error rather than context_overflow: with no overflow token the
        # 400 falls past every check to the non-retryable default.
        body = {
            "error": {
                "message": _SYNTHETIC_PREFILL_400_NO_OVERFLOW_TOKEN,
                "type": "invalid_request_error",
                "param": None,
                "code": "prefill_memory_exceeded",
                "omlx_code": "prefill_memory_exceeded",
                "estimated_bytes": 84368206439,
                "limit_bytes": 83493598003,
            },
            "type": "error",
        }
        e = MockAPIError(
            "Error code: 400 - " + repr(body), status_code=400, body=body,
        )
        result = classify_api_error(
            e, provider="custom", model="qw36-27b-8bit-mtp:agent",
            approx_tokens=63337, context_length=256000,
        )
        assert result.reason == FailoverReason.overloaded
        assert result.retryable is True
        assert result.should_compress is False
        assert result.should_fallback is True
        # The premise of this fixture: no overflow token is present at all.
        assert not any(
            p in _SYNTHETIC_PREFILL_400_NO_OVERFLOW_TOKEN.lower()
            for p in _CONTEXT_OVERFLOW_PATTERNS
        )

    @pytest.mark.parametrize("arrow,message", [
        ("unicode", _OMLX_057_PROCESS_MEMORY_ABORT),
        ("ascii", _OMLX_057_PROCESS_MEMORY_ABORT_ASCII_ARROW),
    ])
    def test_omlx_057_process_memory_abort_is_overloaded(self, arrow, message):
        # The third shape (see fixture): the process memory enforcer, not the
        # prefill guard.  Arrives message-only, like the mid-stream abort.
        #
        # Unguarded, this is the worst-classified of the three — "limit
        # exceeded" routes it to billing, non-retryable, before the overflow
        # branch is ever reached — so a local Metal watermark abort surfaces as
        # an account problem and strands the turn outright.
        e = Exception(message)
        result = classify_api_error(
            e, provider="custom", model="qw36-27b-8bit-mtp:agent",
            approx_tokens=63337, context_length=256000,
        )
        assert result.reason == FailoverReason.overloaded
        assert result.retryable is True
        # Compression cannot lower resident process memory either.
        assert result.should_compress is False
        # Emphatically not a credential or billing problem.
        assert result.should_rotate_credential is False
        assert result.should_fallback is True

    def test_omlx_057_process_memory_abort_reworded_tail_is_overloaded(self):
        # The 11 Aug capture of the same guard, with the remediation tail
        # reworded by the engine (see fixture).  This one already classified
        # correctly before it was pinned — that is the point: it is a
        # characterisation test, not a regression test, and it exists because
        # the previous copy-edit to this engine's wording broke a pattern
        # silently and nothing failed.
        e = Exception(_OMLX_057_PROCESS_MEMORY_ABORT_REWORDED)
        result = classify_api_error(
            e, provider="custom", model="qw36-27b-8bit-mtp:agent",
            approx_tokens=63337, context_length=256000,
        )
        assert result.reason == FailoverReason.overloaded
        assert result.retryable is True
        assert result.should_compress is False
        assert result.should_rotate_credential is False
        assert result.should_fallback is True

    def test_omlx_057_process_memory_abort_wordings_actually_differ(self):
        # Guards the premise of the fixture pair: if a later tidy-up collapses
        # the two captures into one string, the drift they document disappears
        # and the test above becomes a duplicate.  The 2 Aug tail says "loosen"
        # and "free system memory"; the 11 Aug tail says neither.
        earlier = _OMLX_057_PROCESS_MEMORY_ABORT.lower()
        later = _OMLX_057_PROCESS_MEMORY_ABORT_REWORDED.lower()
        for token in ("loosen", "free system memory"):
            assert token in earlier
            assert token not in later
        # And both must still rest on more than one memory token, for the
        # reason spelled out in the 0.5.6 -> 0.5.7 rewording test below.
        for message in (earlier, later):
            matched = [p for p in _MEMORY_CEILING_PATTERNS if p in message]
            assert len(matched) >= 2, (
                f"process-memory shape rests on a single memory token {matched!r}"
            )

    def test_omlx_057_process_memory_abort_does_not_rest_on_the_arrow(self):
        # The tier separator is settled as U+2192 (see the fixture comment), so
        # this is no longer a test about an open question — it is a lower bound.
        # Nothing is allowed to depend on the separator surviving a transcoding
        # hop: the two spellings must be indistinguishable to the classifier,
        # and every matching token must live outside the separator.
        for message in (
            _OMLX_057_PROCESS_MEMORY_ABORT,
            _OMLX_057_PROCESS_MEMORY_ABORT_ASCII_ARROW,
        ):
            matched = [p for p in _MEMORY_CEILING_PATTERNS if p in message.lower()]
            assert len(matched) >= 2, (
                f"process-memory shape rests on a single memory token {matched!r}"
            )
            assert all("→" not in p and "->" not in p for p in matched)

    @pytest.mark.parametrize("shape,message", [
        ("mid-stream", _OMLX_057_STREAM_ABORT),
        ("pre-stream 400", _OMLX_057_PREFILL_400_MESSAGE),
    ])
    def test_omlx_057_wording_matches_more_than_one_memory_token(self, shape, message):
        # The oMLX 0.5.6 → 0.5.7 rewording is the evidence for this assertion,
        # not a hypothetical: "Prefill would require ~13.87 GB peak" became
        # "predicted peak would require ~78.57 GB", and the pattern written
        # against the first wording stopped matching the engine it was written
        # for without anything failing.  Before the memory-accounting entries
        # were added, the mid-stream shape matched exactly ONE token
        # ("available memory") while _CONTEXT_OVERFLOW_PATTERNS matched exactly
        # one of its own ("context length", from the remediation hint) — so a
        # single further copy-edit on either side flips the classification.
        #
        # Scoped to the two 0.5.7 captures because those are the shapes we have
        # two releases of wording for; it is not asserted as a property of every
        # message in the list.
        matched = [p for p in _MEMORY_CEILING_PATTERNS if p in message.lower()]
        assert len(matched) >= 2, (
            f"{shape} shape rests on a single memory token {matched!r}; "
            "one provider copy-edit would reclassify it"
        )

    def test_memory_ceiling_tokens_stay_disjoint_from_overflow_tokens(self):
        # The list's stated invariant: every memory-ceiling token names an
        # allocation ceiling, never a token or window count.  Guards the entries
        # added for the 0.5.7 wording against drifting into overflow language,
        # which would silently divert genuine window overflows out of
        # compression — the failure the whole guard exists to avoid causing.
        for mem in _MEMORY_CEILING_PATTERNS:
            for ovf in _CONTEXT_OVERFLOW_PATTERNS:
                assert mem not in ovf and ovf not in mem, (
                    f"memory token {mem!r} overlaps overflow token {ovf!r}"
                )

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

    def test_llama_cpp_unable_to_generate_parser_template(self):
        e = MockAPIError(
            "Unable to generate parser for this template. "
            "Automatic parser generation failed: error parsing grammar",
            status_code=400,
        )
        result = classify_api_error(e, provider="custom", model="local-llama")
        assert result.reason == FailoverReason.llama_cpp_grammar_pattern
        assert result.retryable is True
        assert result.should_compress is False

    def test_qwen_apply_prompt_template_no_user_query_not_llama_cpp_grammar(self):
        """Local engines wrap Qwen raise_exception as applyPromptTemplate 400.

        Must NOT classify as llama_cpp_grammar_pattern (which strips tool
        schema keywords and retries). Fail fast as format_error so the user
        sees a request-shape failure instead of a misleading template/parser
        loop — typical after context overflow + failed compression.
        """
        e = MockAPIError(
            "Engine protocol applyPromptTemplate request returned 400: "
            '{"error":{"code":400,"message":"Unable to generate parser for '
            "this template. Automatic parser generation failed: "
            "While executing CallExpression ... multi_step_tool %} "
            "{{- raise_exception('No user query found in messages')",
            status_code=400,
        )
        result = classify_api_error(
            e,
            provider="custom",
            model="qwen/qwen3.6-35b-a3b",
            approx_tokens=226_000,
            context_length=100_864,
        )
        assert result.reason == FailoverReason.format_error
        assert result.retryable is False
        assert result.should_compress is False
        assert result.should_fallback is True

    def test_bare_no_user_query_found_is_format_error_even_on_large_session(self):
        e = MockAPIError("No user query found in messages", status_code=400)
        result = classify_api_error(
            e,
            approx_tokens=226_000,
            context_length=100_864,
        )
        assert result.reason == FailoverReason.format_error
        assert result.retryable is False
        assert result.should_compress is False

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
        generic format_error path, which never rotates. (#11736, #13170)

        #82154: the identical body is ALSO returned when Anthropic's content
        filter rejects part of the request on a subscription OAuth token, so
        the billing verdict must be marked unverified — downstream surfaces
        hedge instead of asserting exhaustion, and the credential pool skips
        the one-hour billing bench."""
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
        assert result.billing_unverified is True
        assert result.error_context.get("possible_content_filter") is True

    def test_400_unambiguous_billing_body_is_not_marked_unverified(self):
        """A 400 whose billing evidence is NOT the ambiguous 'out of extra
        usage' body keeps a confirmed verdict (#82154)."""
        e = MockAPIError(
            "Your credit balance is too low to access the Anthropic API.",
            status_code=400,
            body={"error": {
                "type": "invalid_request_error",
                "message": "Your credit balance is too low to access the Anthropic API.",
            }},
        )
        result = classify_api_error(e, provider="anthropic")
        assert result.reason == FailoverReason.billing
        assert result.billing_unverified is False

    def test_statusless_extra_usage_is_marked_unverified(self):
        """Adapters can strip the HTTP status from the Anthropic 400; the
        message-only path must carry the same ambiguity marking (#82154)."""
        e = Exception(
            "You're out of extra usage. Add more at claude.ai/settings/usage and keep going."
        )
        result = classify_api_error(e, provider="anthropic")
        assert result.reason == FailoverReason.billing
        assert result.billing_unverified is True

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


class TestOpenRouterUpstreamRateLimit:
    """Distinguish upstream-provider 429 from account-level 429 on OpenRouter.

    When an upstream model (DeepSeek, Anthropic, etc.) rate-limits OpenRouter's
    aggregate traffic, OpenRouter returns 429 with the outer message "Provider
    returned error".  The user's key is healthy — we must fall back to a
    different model, NOT mark the credential exhausted.
    """

    def test_openrouter_upstream_429_classified_as_upstream_rate_limit(self):
        """OpenRouter 429 with 'Provider returned error' → upstream_rate_limit."""
        e = MockAPIError(
            "Provider returned error",
            status_code=429,
            body={
                "error": {
                    "message": "Provider returned error",
                    "code": 429,
                    "metadata": {
                        "provider_name": "DeepSeek",
                        "raw": '{"error":{"message":"Rate limit exceeded"}}',
                    },
                }
            },
        )
        result = classify_api_error(e, provider="openrouter", model="deepseek/deepseek-v4-flash")
        assert result.reason == FailoverReason.upstream_rate_limit
        assert result.should_rotate_credential is False
        assert result.should_fallback is True
        assert result.error_context.get("upstream_provider") == "DeepSeek"


    def test_account_level_429_still_rotates_credential(self):
        """A real account-level 429 (no upstream wrapper) → rate_limit, rotates."""
        e = MockAPIError(
            "Rate limit exceeded: 200 requests per minute",
            status_code=429,
            body={
                "error": {
                    "message": "Rate limit exceeded: 200 requests per minute",
                    "code": 429,
                }
            },
        )
        result = classify_api_error(e, provider="openrouter", model="deepseek/deepseek-v4-flash")
        assert result.reason == FailoverReason.rate_limit
        assert result.should_rotate_credential is True





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


# ── Test: connection/DNS failure message patterns on generic exception types ──
# Port of anomalyco/opencode#40707 (expand retryable error patterns): errors
# whose TYPE is generic (RuntimeError/Exception from local shims, MCP bridges,
# re-raising SDKs) but whose MESSAGE carries a connection-establishment or DNS
# failure must classify as retryable transport, not FailoverReason.unknown.

class TestConnectionMessagePatterns:
    """Generic-typed connect/DNS failures route to the transport bucket."""

    @pytest.mark.parametrize("message", [
        "connect ECONNREFUSED 127.0.0.1:11434",
        "Connection refused by proxy",
        "getaddrinfo failed",
        "getaddrinfo ENOTFOUND api.example.com",
        "[Errno -3] Temporary failure in name resolution",
        "[Errno 8] nodename nor servname provided, or not known",
        "getaddrinfo EAI_AGAIN openrouter.ai",
        "Name or service not known",
        "No route to host",
        "[Errno 101] Network is unreachable",
        "fetch failed",
        "TypeError: Failed to fetch",
        "upstream connect error or disconnect/reset before headers",
    ])
    def test_generic_exception_with_connect_failure_message_is_timeout(self, message):
        # RuntimeError — NOT in _TRANSPORT_ERROR_TYPES, not a ConnectionError
        # subclass, no status code. Without message matching this falls to
        # FailoverReason.unknown and misses the eager transport fallback.
        result = classify_api_error(RuntimeError(message))
        assert result.reason == FailoverReason.timeout, message
        assert result.retryable is True
        assert result.should_compress is False

    def test_connect_failure_never_routes_to_compression_on_large_session(self):
        # A connection that was never established is not an overflow signal,
        # even when the session is huge (the disconnect+large-session
        # heuristic must not apply to connect-phase failures).
        result = classify_api_error(
            RuntimeError("connect ECONNREFUSED 10.0.0.5:443"),
            approx_tokens=180000, context_length=200000, num_messages=400,
        )
        assert result.reason == FailoverReason.timeout
        assert result.should_compress is False

    def test_midstream_disconnect_patterns_still_use_disconnect_path(self):
        # "connection reset by peer" is deliberately NOT in the connect-phase
        # list — it stays on the _SERVER_DISCONNECT_PATTERNS path, which
        # routes large sessions to context-overflow compression.
        result = classify_api_error(
            RuntimeError("Connection reset by peer"),
            approx_tokens=180000, context_length=200000, num_messages=400,
        )
        assert result.reason == FailoverReason.context_overflow
        assert result.should_compress is True

    def test_plain_unknown_error_still_unknown(self):
        # Guard against over-matching: an unrelated message stays unknown.
        result = classify_api_error(RuntimeError("something exploded"))
        assert result.reason == FailoverReason.unknown


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


class TestServerInjectedParameterRejection:
    """A 400 blaming a parameter the client never sent is a server-side flake.

    The Codex backend (chatgpt.com/backend-api/codex) intermittently adds
    ``prompt_cache_retention`` to its own upstream call and then rejects it,
    so an identical request succeeds on retry ~80% of the time.  Hermes never
    sends that field on this route, so the 400 is not a deterministic
    request-shape error and must stay retryable instead of aborting the turn.
    """

    RETENTION_BODY = {
        "message": "prompt_cache_retention is not supported on this model",
        "type": "invalid_request_error",
        "param": "prompt_cache_retention",
        "code": "invalid_parameter",
    }

    def test_codex_retention_400_is_retryable_server_error(self):
        e = MockAPIError(
            "Error code: 400 - {'error': {'message': 'prompt_cache_retention "
            "is not supported on this model', 'type': 'invalid_request_error', "
            "'param': 'prompt_cache_retention', 'code': 'invalid_parameter'}}",
            status_code=400,
            body=dict(self.RETENTION_BODY),
        )
        result = classify_api_error(
            e,
            provider="openai-codex",
            model="gpt-5.6-sol",
            approx_tokens=546912,
            context_length=272000,
            num_messages=576,
        )
        assert result.reason == FailoverReason.server_error
        assert result.retryable is True
        # Retrying the identical request is the recovery — do NOT enter the
        # compression loop (the context was never the problem).
        assert result.should_compress is False

    def test_codex_retention_400_nested_error_body_is_retryable(self):
        """The same rejection arrives wrapped in an ``error`` envelope too."""
        e = MockAPIError(
            "prompt_cache_retention is not supported on this model",
            status_code=400,
            body={"error": dict(self.RETENTION_BODY)},
        )
        result = classify_api_error(
            e, provider="openai-codex", model="gpt-5.6-sol",
        )
        assert result.reason == FailoverReason.server_error
        assert result.retryable is True

    def test_codex_gateway_terse_retention_400_is_retryable(self):
        """The Codex gateway's own validator uses a bare ``detail`` body."""
        e = MockAPIError(
            "Unsupported parameter: prompt_cache_retention",
            status_code=400,
            body={"detail": "Unsupported parameter: prompt_cache_retention"},
        )
        result = classify_api_error(
            e, provider="openai-codex", model="gpt-5.6-sol",
        )
        assert result.reason == FailoverReason.server_error
        assert result.retryable is True

    def test_small_session_retention_400_is_still_retryable(self):
        """Must not depend on the context-size heuristic — a tiny request
        gets the identical spontaneous rejection (reproduced live)."""
        e = MockAPIError(
            "prompt_cache_retention is not supported on this model",
            status_code=400,
            body=dict(self.RETENTION_BODY),
        )
        result = classify_api_error(
            e,
            provider="openai-codex",
            model="gpt-5.6-sol",
            approx_tokens=50,
            num_messages=1,
        )
        assert result.reason == FailoverReason.server_error
        assert result.retryable is True

    def test_other_unsupported_parameter_400_stays_non_retryable(self):
        """Boundary: a genuine client-sent bad parameter is deterministic and
        must keep failing fast as a format_error (the existing behaviour)."""
        e = MockAPIError(
            "Unsupported parameter: 'max_tokens' is not supported with this "
            "model. Use 'max_completion_tokens' instead.",
            status_code=400,
            body={
                "message": "Unsupported parameter: 'max_tokens' is not supported.",
                "type": "invalid_request_error",
                "param": "max_tokens",
                "code": "unsupported_parameter",
            },
        )
        result = classify_api_error(
            e, provider="openai-codex", model="gpt-5.6-sol",
        )
        assert result.reason == FailoverReason.format_error
        assert result.retryable is False

    def test_retention_rejection_from_meta_host_stays_non_retryable(self):
        """Boundary: on api.meta.ai / Bedrock Mantle Hermes DOES send
        ``prompt_cache_retention`` deliberately, so a rejection there is a
        real client-side request error and must not be retried blindly."""
        e = MockAPIError(
            "prompt_cache_retention is not supported on this model",
            status_code=400,
            body=dict(self.RETENTION_BODY),
        )
        result = classify_api_error(
            e, provider="meta-ai", model="muse-spark-1.2",
        )
        assert result.reason == FailoverReason.format_error
        assert result.retryable is False

    @pytest.mark.parametrize("status_code", [500, 502])
    def test_retention_rejection_via_5xx_proxy_is_retryable(self, status_code):
        """Sibling path: a proxy in front of the route can surface the same
        injected-parameter rejection as 5xx, where the request-validation
        guard would also wrongly fail it fast as a format_error."""
        e = MockAPIError(
            "Unsupported parameter: prompt_cache_retention",
            status_code=status_code,
            body={"error": dict(self.RETENTION_BODY)},
        )
        result = classify_api_error(
            e, provider="openai-codex", model="gpt-5.6-sol",
        )
        assert result.reason == FailoverReason.server_error
        assert result.retryable is True

    @pytest.mark.parametrize("status_code", [500, 502])
    def test_other_bad_parameter_via_5xx_stays_non_retryable(self, status_code):
        """Boundary for the sibling path: the codex.nekos.me 502-on-bad-param
        behaviour must keep failing fast (regression guard for that fix)."""
        e = MockAPIError(
            "Unknown parameter: 'frequency_penalty'",
            status_code=status_code,
            body={"error": {"message": "Unknown parameter: 'frequency_penalty'",
                            "code": "unknown_parameter"}},
        )
        result = classify_api_error(e, provider="custom", model="m")
        assert result.reason == FailoverReason.format_error
        assert result.retryable is False


