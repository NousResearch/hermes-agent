"""Every free-tier refusal offers sign-in: any anonymous 429 (reason or not) and a 402 from the
welcome host classify as a welcome refusal; a signed-in account's identical response does not."""

from types import SimpleNamespace

import httpx
import openai
import pytest

from agent.error_classifier import classify_api_error
from agent.error_surface import build_error_surface_from_result
from agent.nous_rate_guard import is_long_welcome_rate_limit, welcome_refusal_from_headers
from agent.turn_recovery import max_retries_exhausted_result, nonretryable_client_error_result
from tests.hermes_cli.anon_portal import make_jwt

WELCOME = "https://welcome-api.nousresearch.com/v1"
MODEL = "nous/welcome"


def _agent(api_key):
    noop = lambda *a, **kw: None
    return SimpleNamespace(
        provider="nous", api_key=api_key, base_url=WELCOME, model=MODEL, log_prefix="",
        _rate_limit_state=None, _has_pending_fallback=lambda: False, _dump_api_request_debug=noop,
        _flush_status_buffer=noop, _summarize_api_error=lambda e: str(e), _emit_status=noop,
        _persist_session=noop, _plines=noop, _vprint=noop, _buffer_status=noop, _buffer_vprint=noop,
        _emit_diagnostic_status=noop, _buffer_diagnostic_status=noop, _try_activate_fallback=lambda: False)


def _error(status, body, headers=None):
    request = httpx.Request("POST", f"{WELCOME}/chat/completions")
    response = httpx.Response(status, json=body, headers=headers or {}, request=request)
    message = (body.get("error") or {}).get("message") if isinstance(body.get("error"), dict) else body.get("message")
    message = message or "refused"
    if status == 429:
        return openai.RateLimitError(message, response=response, body=body)
    return openai.APIStatusError(message, response=response, body=body)


# (id, status, body, headers, expected free-tier kind, breaker trips)
CASES = [
    ("reason-long", 429, {"reason": "rate_limited", "retry_after": 3600}, None, "rate_limited", True),
    ("reason-short", 429, {"reason": "rate_limited", "retry_after": 5}, None, "rate_limited", False),
    ("retry-after-only", 429, {}, {"Retry-After": "900"}, "rate_limited", True),
    ("bucket-empty", 429, {}, {"x-ratelimit-remaining-requests-1h": "0",
                              "x-ratelimit-reset-requests-1h": "3000"}, "rate_limited", True),
    ("healthy-hourly-short-header", 429, {}, {"Retry-After": "4", "x-ratelimit-remaining-requests-1h": "750",
                                              "x-ratelimit-reset-requests-1h": "2000"}, "at_capacity", False),
    ("short-header", 429, {}, {"Retry-After": "4"}, "at_capacity", False),
    ("bare", 429, {}, None, "at_capacity", False),
    ("unknown-reason", 429, {"reason": "something_else", "retry_after": 5}, None, "at_capacity", False),
    ("quota-words", 429, {"error": {"message": "quota exceeded"}}, None, "at_capacity", False),
    ("payment-402", 402, {"error": {"message": "payment required"}}, None, "refused", False),
]


def _result(agent, error, classified, status):
    common = dict(api_kwargs=None, api_messages=[], messages=[], conversation_history=[], api_call_count=1,
                  approx_tokens=10, provider="nous", base_url=WELCOME, model=MODEL)
    if status == 429:
        return max_retries_exhausted_result(agent, error, classified, attempts=3, is_rate_limited=True,
                                            error_msg=str(error), **common)
    return nonretryable_client_error_result(agent, error, classified, status_code=status, **common)


@pytest.mark.parametrize("case_id,status,body,headers,kind,breaker", CASES, ids=[c[0] for c in CASES])
def test_anonymous_refusals_all_offer_sign_in(case_id, status, body, headers, kind, breaker):
    agent = _agent(make_jwt())
    error = _error(status, body, headers)
    classified = classify_api_error(error, provider="nous", model=MODEL, base_url=WELCOME, api_key=agent.api_key)
    assert classified.error_context["welcome_refusal"]["reason"] == (
        "refused" if status == 402 else kind)
    # A guessed (missing) wait never trips the cross-session breaker.
    assert is_long_welcome_rate_limit(classified.error_context) is breaker
    result = _result(agent, error, classified, status)
    assert result["free_tier"]["kind"] == kind
    assert result["final_response"].endswith("To sign in: /login.")
    assert "fallback add" not in result["final_response"]
    surface = build_error_surface_from_result(result, provider="nous", model=MODEL)
    assert surface["code"] == f"free_tier_{kind}"


@pytest.mark.parametrize("case_id,status,body,headers,kind,breaker", CASES, ids=[c[0] for c in CASES])
def test_named_account_same_response_is_not_a_free_tier_refusal(case_id, status, body, headers, kind, breaker):
    agent = _agent(make_jwt(account_tier="free", client_id="hermes-cli"))
    error = _error(status, body, headers)
    classified = classify_api_error(error, provider="nous", model=MODEL, base_url=WELCOME, api_key=agent.api_key)
    assert "welcome_refusal" not in classified.error_context
    result = _result(agent, error, classified, status)
    assert "free_tier" not in result
    assert "To sign in: /login." not in result["final_response"]


def test_anonymous_402_off_the_welcome_host_keeps_ordinary_402_handling():
    paid = "https://inference-api.nousresearch.com/v1"
    error = _error(402, {"error": {"message": "payment required"}})
    classified = classify_api_error(error, provider="nous", model=MODEL, base_url=paid, api_key=make_jwt())
    assert "welcome_refusal" not in classified.error_context
    assert classified.reason.value == "billing"


@pytest.mark.parametrize("remaining,reason,retry_after", [("750", "at_capacity", 4), ("0", "rate_limited", 2000)])
def test_header_wait_is_the_exhausted_bucket_reset_else_retry_after(remaining, reason, retry_after):
    headers = {"Retry-After": "4", "x-ratelimit-remaining-requests-1h": remaining,
               "x-ratelimit-reset-requests-1h": "2000"}
    refusal = welcome_refusal_from_headers(headers)
    assert (refusal["reason"], refusal["retry_after"]) == (reason, retry_after)
