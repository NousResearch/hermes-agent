"""Phrasings and error shapes the auxiliary client used to match privately, now read by the classifier."""

import httpx
import openai
import pytest

from hermes_agent.agent.error_classifier import FailoverReason, classify_api_error
from hermes_agent.agent.errors import InvalidProviderResponseError


_R = FailoverReason
_REQ = httpx.Request("POST", "https://example.invalid/v1/chat/completions")
_shaped = lambda name, msg, **attrs: type(name, (Exception,), attrs)(msg)  # SDK class-name shapes
_status_error = lambda msg, status: _shaped("APIStatusError", msg, status_code=status, body={})
_boto = lambda status: _shaped("ClientError", "x", response={"ResponseMetadata": {"HTTPStatusCode": status}})
_PHRASINGS = [  # (message or error, status, reason, rotates credential, error_context facts) the aux client matched privately
    ("Quota exceeded for this key", 403, _R.billing, True, {"quota_exhausted": True}),
    ("this model requires a subscription, upgrade for access: https://ollama.com/upgrade", 403, _R.billing, True, {}),
    ("RESOURCE_EXHAUSTED: project quota", 404, _R.billing, True, {}),
    ("Insufficient funds for this request", None, _R.billing, True, {}),
    ("This request requires more credits, but can only afford 7117", 402, _R.billing, True, {}),
    ("model_not_supported_on_free_tier", 400, _R.billing, True, {}),
    ("Rate limit exceeded: insufficient credits for burst", 429, _R.rate_limit, True, {}),
    ("RESOURCE_EXHAUSTED: Worker local total request limit reached (32/32)", 429, _R.rate_limit, True, {}),
    ("RESOURCE_EXHAUSTED: You have reached your daily limit.", 429, _R.billing, True, {"quota_exhausted": True}),
    ("You've hit your weekly usage limit.", None, _R.billing, True, {"quota_exhausted": True}),
    ("Too many tokens per day, please wait before retrying", 429, _R.rate_limit, True, {"quota_exhausted": True}),
    ("Quota exceeded for quota metric 'Generate Content API requests per day'", 429, _R.rate_limit, True, {"quota_exhausted": True}),
    ('event:error :HTTP_STATUS/429 data:{"code":"Throttling.AllocationQuota","message":"Allocated quota exceeded."}', 429, _R.rate_limit, True, {"quota_exhausted": True}),
    ("unauthenticated:bad-credentials", 403, _R.auth, True, {}),
    ("Error code: 401 - invalid key", None, _R.auth, True, {}),
    ("Model 'x' not found in our configuration or OpenRouter catalog.", 404, _R.model_not_found, False, {}),
    ("The model qwen3:8b is not supported when using this account", 400, _R.model_entitlement, True, {}),
    ("model kimi-k2-thinking is not supported when using this account", None, _R.model_entitlement, True, {}),
    ("Error code: 400 - {'error': {'message': \"The 'kimi-k2' model is not supported for this account.\"}}", None, _R.model_entitlement, True, {}),
    ("This model is not supported for image input", 400, _R.format_error, False, {"route_rejects_model": True}),
    ("Streaming is not supported for this account", 400, _R.format_error, False, {"route_rejects_model": True}),
    ("Tool use is not supported with this model", 400, _R.format_error, False, {"route_rejects_model": True}),
    ("The model parameter temperature is not supported for this account", 400, _R.format_error, False, {}),
    ("Vision: model is not supported for this account", None, _R.unknown, False, {}),
    ("RemoteProtocolError: response ended prematurely", None, _R.timeout, False, {"transport": "connection"}),
    (openai.APITimeoutError(request=_REQ), None, _R.timeout, False, {"transport": "timeout"}),
    (httpx.ConnectError("[Errno 61] Connection refused"), None, _R.timeout, False, {"transport": "connection"}),
    ("upstream: connection reset by peer", 502, _R.server_error, False, {"transport": "connection"}),
    (openai.APIError("x", _REQ, body={"type": "server_error"}), None, _R.unknown, False, {"statusless_structured": True}),
    (openai.APIError("Not Found", _REQ, body={"code": 404, "message": "Not Found"}), 404, _R.unknown, False, {"statusless_structured": True}),
    (_shaped("EndpointConnectionError", "Could not connect to the endpoint URL"), None, _R.timeout, False, {}),
    (_shaped("AuthenticationError", "token rejected"), 401, _R.auth, True, {}),
    (InvalidProviderResponseError("LLM returned invalid response"), None, _R.incomplete_response, False, {}),
    (_boto(429), 429, _R.rate_limit, True, {}),
    (_boto(503), 503, _R.overloaded, False, {}),
    (httpx.HTTPStatusError("x", request=_REQ, response=httpx.Response(429, request=_REQ)), 429, _R.rate_limit, True, {}),
    # The aux client's bare "credits" token is not an account wall: main's verdicts for these bodies hold.
    ("The model credits-v1 does not exist", 404, _R.model_not_found, False, {}),
    ("The model credits-v1 does not exist", None, _R.model_not_found, False, {}),
    ("Invalid API key. Manage API keys and credits at https://example.com/credits", 403, _R.auth, False, {}),
    ("Rate limit reached, see https://example.com/credits", 429, _R.rate_limit, True, {}),
]


@pytest.mark.parametrize("error,status,reason,rotates,facts", _PHRASINGS)
def test_auxiliary_phrasings_and_shapes(error, status, reason, rotates, facts):
    error = _status_error(error, status) if isinstance(error, str) else error
    verdict = classify_api_error(error, provider="openrouter")
    assert (verdict.reason, verdict.status_code, verdict.should_rotate_credential) == (reason, status, rotates)
    assert facts.items() <= verdict.error_context.items()
    assert verdict.should_fallback or reason not in (_R.billing, _R.rate_limit, _R.model_not_found, _R.model_entitlement, _R.incomplete_response)


@pytest.mark.parametrize("message,reason", [
    ("RemoteProtocolError: response ended prematurely", _R.timeout),
    ("RemoteProtocolError: stream reset", _R.timeout),
    ("LocalProtocolError: can't handle event type ConnectionClosed when role=SERVER", _R.timeout),
    ("upstream: connection reset", _R.timeout),
    ("peer closed connection without sending complete message body (incomplete chunked read)", _R.context_overflow),
])
def test_aux_premature_close_wording_is_never_a_large_sessions_overflow(message, reason):
    """Main's disconnect + large-session overflow guess keeps main's phrasings; these stay a transport retry."""
    verdict = classify_api_error(Exception(message), model="gpt-4o", approx_tokens=150_000, context_length=200_000,
                                 num_messages=250)
    assert (verdict.reason, verdict.should_compress) == (reason, reason is _R.context_overflow)
    assert verdict.error_context.get("transport") == "connection"
