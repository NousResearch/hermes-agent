"""Provider error bodies observed in real sessions, through the real SDK exception types.

Every row is a body Hermes actually received (relay traces + agent.log, scrubbed of request ids
and account data), raised as the exception class the OpenAI SDK / httpx raises for it, so a
classifier change is checked against what providers really send rather than hand-written
strings. Add a row whenever a new production body shows up in a bug report or trace.

OpenRouter's "can only afford N tokens" 402 (#49769) stays ``billing`` here; the main loop's
recovery chain retries it once with the lowered cap (see that test's docstring).
"""

from __future__ import annotations

import httpx
import openai
import pytest

from agent.error_classifier import FailoverReason, classify_api_error
from agent.message_sanitization import _looks_like_image_content_rejection

_REQUEST = httpx.Request("POST", "https://example.invalid/v1/chat/completions")


def _status(cls: type, code: int, body: dict) -> Exception:
    return cls(f"Error code: {code} - {body}", response=httpx.Response(code, request=_REQUEST, json=body), body=body)


AFFORD_402 = {"error": {"message": (
    "This request requires more credits, or fewer max_tokens. You requested up to 128000 tokens, but can only "
    "afford 81664. To increase, visit https://openrouter.ai/settings/credits and upgrade to a paid account"),
    "code": 402}}
NOT_MULTIMODAL_400 = {"error": {"message": "glm-5.2-fp8 is not a multimodal model", "type": "BadRequestError",
                                "param": None, "code": 400}}

# (id, provider, exception, expected reason, retryable, should_fallback)
OBSERVED = [
    ("connection-error", "custom", openai.APIConnectionError(request=_REQUEST), FailoverReason.timeout, True, False),
    ("broken-pipe", "nous", httpx.ReadError("[Errno 32] Broken pipe", request=_REQUEST),
     FailoverReason.timeout, True, False),
    ("peer-closed-mid-body", "custom", httpx.RemoteProtocolError(
        "peer closed connection without sending complete message body (incomplete chunked read)", request=_REQUEST),
     FailoverReason.timeout, True, False),
    ("request-timeout", "nous", openai.APITimeoutError(request=_REQUEST), FailoverReason.timeout, True, False),
    ("nous-404-no-credits", "nous", _status(openai.NotFoundError, 404, {"status": 404, "message": (
        "Model 'anthropic/claude-opus-5' requires available credits. Your account balance is too low to use paid "
        "models — add credits at https://portal.nousresearch.com or pick a free model.")}),
     FailoverReason.billing, False, True),
    ("nous-401-invalid-key", "nous", _status(openai.AuthenticationError, 401, {"status": 401, "message": (
        "Your API key is invalid, blocked or out of funds. Please go visit the portal to sort that out: "
        "https://portal.nousresearch.com ")}),
     FailoverReason.auth, False, True),
    ("cloudflare-502", "nous", _status(openai.InternalServerError, 502, {
        "type": "https://developers.cloudflare.com/support/troubleshooting/http-status-codes/cloudflare-5xx-errors/"
                "error-502/", "title": "Error 502: Bad gateway", "status": 502,
        "detail": "The origin web server returned an invalid or incomplete response to Cloudflare.",
        "error_code": 502, "error_name": "origin_bad_gateway", "error_category": "origin"}),
     FailoverReason.server_error, True, False),
    ("nous-500-unexpected", "nous", _status(openai.InternalServerError, 500, {"status": 500, "message": (
        "Something unexpected happened while processing your request. Please try again in a moment, or contact us "
        "if the issue persists.")}),
     FailoverReason.server_error, True, False),
    ("router-no-workers", "custom", _status(openai.InternalServerError, 500, {"error": {
        "message": "No available workers (all circuits open or unhealthy)"}}),
     FailoverReason.server_error, True, False),
    # 2026-11-16 agent.log: the Nous usage-window check itself was down, not the model's capacity.
    # Read as overloaded it backed off on the same model and never switched, so every turn that hit
    # it during the outage failed with no fallback.
    ("nous-503-usage-window-unavailable", "nous", _status(openai.InternalServerError, 503, {
        "status": 503, "code": "usage_window_unavailable",
        "message": "Hermes usage checks are temporarily unavailable. Please try again in a moment."}),
     FailoverReason.rate_limit, True, True),
]


@pytest.mark.parametrize("provider,error,reason,retryable,fallback", [r[1:] for r in OBSERVED],
                         ids=[r[0] for r in OBSERVED])
def test_observed_body_routes_to_its_recovery(provider, error, reason, retryable, fallback):
    result = classify_api_error(error, provider=provider, model="test-model")
    assert (result.reason, result.retryable, result.should_fallback) == (reason, retryable, fallback)


def test_affordable_402_classifies_as_billing_with_a_parseable_budget():
    """#49769 is recovered in the main loop, not by the classifier: the 402 stays ``billing`` and the
    recovery chain retries once with the cap lowered to the budget it names
    (``tests/agent/test_affordable_402_main_loop.py`` drives that retry end to end)."""
    from agent.auxiliary_client import _affordable_max_tokens_from_error

    error = _status(openai.APIStatusError, 402, AFFORD_402)
    result = classify_api_error(error, provider="openrouter", model="anthropic/claude-opus-5.5")
    assert result.reason == FailoverReason.billing
    assert _affordable_max_tokens_from_error(error) == 81664 - 64


def test_usage_window_check_outage_is_a_limit_hit_that_may_fall_back():
    """The 503 carries the ``usage_window`` tag (the credentials pool and the limit card read it)
    but keeps the fallback the 402 refuses: the allowance was never shown to be spent, so the
    requested model can still be served elsewhere. ``usage_window_outage`` is what the eager-fallback
    guard reads to tell this apart from a spent allowance."""
    body = {"status": 503, "code": "usage_window_unavailable",
            "message": "Hermes usage checks are temporarily unavailable. Please try again in a moment."}
    result = classify_api_error(_status(openai.InternalServerError, 503, body), provider="nous", model="m")
    assert result.reason == FailoverReason.rate_limit
    assert result.error_context["usage_window"] is True
    assert result.error_context["usage_window_outage"] is True
    assert result.retryable and result.should_fallback


def test_spent_usage_window_402_still_never_falls_back_on_its_own():
    """The other side of the guard: a usage-window *plan limit* keeps ``should_fallback`` False and
    carries no outage tag, so turn_recovery's plan-limit guard still blocks the model switch."""
    body = {"status": 402, "code": "usage_window_and_credits_exhausted",
            "message": "You've hit your daily limit."}
    result = classify_api_error(_status(openai.APIStatusError, 402, body), provider="nous", model="m")
    assert result.reason == FailoverReason.billing
    assert result.error_context["usage_window"] is True
    assert "usage_window_outage" not in result.error_context
    assert result.should_fallback is False


def test_usage_window_503_code_from_another_provider_keeps_the_overload_rules():
    """The code is Nous's. A non-Nous route sending the same 503 keeps main's classification, the
    same boundary ``_classify_402`` already enforces for the usage-window 402 code."""
    body = {"status": 503, "code": "usage_window_unavailable",
            "message": "Hermes usage checks are temporarily unavailable. Please try again in a moment."}
    result = classify_api_error(_status(openai.InternalServerError, 503, body), provider="openrouter", model="m")
    assert result.reason == FailoverReason.overloaded
    assert result.should_fallback is False
    assert "usage_window" not in result.error_context


def test_usage_window_503_outage_never_trips_the_nous_rate_breaker():
    """The breaker reads headers or a *last-known* state; a 503 carries neither, so without this
    exemption an exhausted stale snapshot would park every Nous model instead of falling back."""
    from types import SimpleNamespace
    from unittest.mock import patch

    from agent.turn_recovery import _is_genuine_nous_rate_limit

    agent = SimpleNamespace(provider="nous", _rate_limit_state=None, log_prefix="")
    outage = SimpleNamespace(error_context={"usage_window": True, "usage_window_outage": True})
    plan_limit = SimpleNamespace(error_context={"usage_window": True})
    with patch("agent.nous_rate_guard.is_genuine_nous_rate_limit", return_value=True):
        assert _is_genuine_nous_rate_limit(agent, RuntimeError("503"), {}, outage) is False
        # the plan-limit path keeps its existing behaviour
        assert _is_genuine_nous_rate_limit(agent, RuntimeError("402"), {}, plan_limit) is True


def test_usage_window_cooldown_is_model_scoped_on_a_pool_entry():
    import time
    from types import SimpleNamespace
    from typing import Any, cast

    from agent.credential_pool import PooledCredential
    from agent.credential_pool_model_cooldowns import (
        FAILURE_REASON_USAGE_WINDOW, CredentialPoolModelCooldownMixin, model_cooldown_until)

    mixin = cast(Any, CredentialPoolModelCooldownMixin())
    mixin.provider = "nous"
    assert mixin._is_model_scoped_failure(503, "model-a", FAILURE_REASON_USAGE_WINDOW) is True
    entry = cast(PooledCredential, SimpleNamespace(
        model_cooldowns={"model-a": time.time() + 600}, last_status="ok"))
    assert model_cooldown_until(entry, "model-a") is not None   # refused model is benched
    assert model_cooldown_until(entry, "model-b") is None       # sibling models stay usable


def test_not_a_multimodal_model_is_an_image_rejection():
    body = str(_status(openai.BadRequestError, 400, NOT_MULTIMODAL_400))
    assert _looks_like_image_content_rejection(body), f"image rejection not recognised: {body[:120]}"
