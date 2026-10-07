"""Provider error bodies observed in real sessions, through the real SDK exception types.

Every row is a body Hermes actually received (relay traces + agent.log, scrubbed of request ids
and account data), raised as the exception class the OpenAI SDK / httpx raises for it, so a
classifier change is checked against what providers really send rather than hand-written
strings. Add a row whenever a new production body shows up in a bug report or trace.

The last two tests are bodies main still mishandles; each is a run-time xfail keyed on the current
wrong outcome, so it turns into a passing regression test the moment its fix lands:

* #49769 — OpenRouter's "can only afford N tokens" 402 is recoverable by lowering max_tokens,
  but the main loop classifies it as terminal billing and abandons the provider.
* #85005 — "<model> is not a multimodal model" (vLLM/text-only endpoints) is not recognised as
  an image rejection, so the image is never stripped and every retry fails identically.
"""

from __future__ import annotations

import httpx
import openai
import pytest

from agent.error_classifier import FailoverReason, classify_api_error
from agent.message_sanitization import _looks_like_image_content_rejection
from tests.e2e.core._pending_fixes import known_failure

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
]


@pytest.mark.parametrize("provider,error,reason,retryable,fallback", [r[1:] for r in OBSERVED],
                         ids=[r[0] for r in OBSERVED])
def test_observed_body_routes_to_its_recovery(provider, error, reason, retryable, fallback):
    result = classify_api_error(error, provider=provider, model="test-model")
    assert (result.reason, result.retryable, result.should_fallback) == (reason, retryable, fallback)


def test_affordable_402_is_retried_on_the_same_provider():
    result = classify_api_error(_status(openai.APIStatusError, 402, AFFORD_402), provider="openrouter",
                                model="anthropic/claude-opus-5.5")
    outcome = f"reason={result.reason.name} retryable={result.retryable}"
    with known_failure(r"reason=billing retryable=False", "#49769: affordable-tokens 402 treated as terminal billing"):
        assert result.retryable and result.reason != FailoverReason.billing, (
            f"402 'can only afford N' must be retried on this provider, not abandoned as billing: {outcome}")


def test_not_a_multimodal_model_is_an_image_rejection():
    body = str(_status(openai.BadRequestError, 400, NOT_MULTIMODAL_400))
    with known_failure(r"not recognised", "#85005: text-only endpoint image rejection not recognised"):
        assert _looks_like_image_content_rejection(body), f"image rejection not recognised: {body[:120]}"
