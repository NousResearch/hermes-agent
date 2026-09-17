"""The anonymous UI is a property of the request credential, not a free model or host (#111970)."""

from types import SimpleNamespace

import pytest

from agent.error_classifier import classify_api_error
from tests.hermes_cli.anon_portal import make_jwt
from agent.error_surface import build_error_surface_from_result
from agent.turn_recovery import max_retries_exhausted_result, nonretryable_client_error_result

WELCOME = "https://welcome-api.nousresearch.com/v1"
NAMED = "https://inference-api.nousresearch.com/v1"


def agent_for(api_key, base_url):
    return SimpleNamespace(
        provider="nous", api_key=api_key, base_url=base_url, model="stepfun/step-3.7-flash:free",
        log_prefix="", _rate_limit_state=None, _has_pending_fallback=lambda: False,
        _dump_api_request_debug=lambda *a, **kw: None, _flush_status_buffer=lambda: None,
        _summarize_api_error=lambda e: str(e), _emit_status=lambda *a, **kw: None,
        _persist_session=lambda *a, **kw: None, _plines=lambda *a, **kw: None, _vprint=lambda *a, **kw: None,
        _buffer_status=lambda *a, **kw: None, _buffer_vprint=lambda *a, **kw: None,
        _try_activate_fallback=lambda: False,
    )


@pytest.mark.parametrize("api_key", [
    make_jwt(account_tier="free", client_id="hermes-cli"),
    make_jwt(account_tier="paid", client_id="hermes-cli"), "sk-named", None,
], ids=["named-free", "named-paid", "api-key", "unknown"])
@pytest.mark.parametrize("base_url", [NAMED, WELCOME])
@pytest.mark.parametrize("case", ["rate_limited", "at_capacity", "admission_closed", "model_not_free", "feature_not_free", "403", "named_route", "503"])
def test_named_errors_do_not_offer_anonymous_recovery(api_key, base_url, case):
    """A shared fairshare body or wrong welcome URL cannot establish anonymous identity."""
    status = 403 if case in {"403", "named_route"} else 503 if case == "503" else 429
    message = ("This host serves anonymous Hermes Agent accounts only." if case == "named_route"
               else "You tried to access something that you don't have permissions for." if case == "403"
               else "The service refused this request.")
    body = {"status": status, "message": message, "reason": case, "retry_after": 158}
    error = Exception(message)
    error.status_code, error.body = status, body
    agent = agent_for(api_key, base_url)
    classified = classify_api_error(error, provider="nous", model=agent.model, base_url=base_url, api_key=api_key)
    common = dict(api_kwargs=None, api_messages=[], messages=[], conversation_history=[],
                  api_call_count=1, approx_tokens=10, provider="nous", base_url=base_url, model=agent.model)
    if status == 403:
        result = nonretryable_client_error_result(agent, error, classified, status_code=status, **common)
    else:
        result = max_retries_exhausted_result(agent, error, classified, max_retries=3,
                                             is_rate_limited=status == 429, error_msg=message, **common)
    assert "welcome_refusal" not in classified.error_context
    assert "welcome_route" not in classified.error_context
    assert "free_tier" not in result
    surface = build_error_surface_from_result(result, provider="nous", model=agent.model)
    assert surface["code"] == {403: "auth", 429: "rate_limit", 503: "overloaded"}[status]
    assert "without signing in" not in result["final_response"]


def guard_for(agent):
    from agent.turn_api_call import nous_rate_limit_guard
    return nous_rate_limit_guard(
        agent, _retry=None, api_messages=[], messages=[], conversation_history=[],
        active_system_prompt="system", retry_count=0, compression_attempts=0, api_call_count=0,
    )


@pytest.mark.parametrize("tier", ["free", "paid"])
def test_signing_in_does_not_inherit_anonymous_cooldown(tmp_path, monkeypatch, tier):
    from agent.agent_runtime_helpers import extract_api_error_context
    from agent.nous_rate_guard import clear_nous_rate_limit, nous_rate_limit_remaining, record_nous_rate_limit
    from agent.turn_recovery import _is_genuine_nous_rate_limit

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    guest = agent_for(make_jwt(), WELCOME)
    error = Exception("refused")
    error.status_code = 429
    error.body = {"status": 429, "message": "refused", "reason": "rate_limited", "retry_after": 600}
    classified = classify_api_error(error, provider="nous", base_url=WELCOME, api_key=guest.api_key)
    assert _is_genuine_nous_rate_limit(guest, error, extract_api_error_context(error), classified)
    blocked = guard_for(guest)
    assert blocked.action == "return"
    assert build_error_surface_from_result(blocked.result)["code"] == "free_tier_rate_limited"

    # Keep the old welcome URL deliberately: the changed credential owns the boundary.
    guest.api_key = make_jwt(account_tier=tier, client_id="hermes-cli")
    assert guard_for(guest).action == "fallthrough"
    record_nous_rate_limit(headers={"retry-after": "300"})
    named_blocked = guard_for(guest)
    assert named_blocked.action == "return"
    assert "free_tier" not in named_blocked.result
    assert "Your Nous account has hit its rate limit" in named_blocked.result["final_response"]
    clear_nous_rate_limit()
    assert nous_rate_limit_remaining() is None
    assert nous_rate_limit_remaining(anonymous=True) > 0
    record_nous_rate_limit(headers={"retry-after": "300"})
    clear_nous_rate_limit(anonymous=True)
    assert nous_rate_limit_remaining(anonymous=True) is None
    assert nous_rate_limit_remaining() > 0


def test_auxiliary_anonymous_cooldown_does_not_cache_named_provider_as_unhealthy(tmp_path, monkeypatch):
    import agent.auxiliary_client as aux
    from agent.nous_rate_guard import record_nous_rate_limit

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    record_nous_rate_limit(headers={"retry-after": "600"}, anonymous=True)
    runtime = [make_jwt(), WELCOME]
    monkeypatch.setattr(aux, "_read_nous_auth", lambda: {})
    monkeypatch.setattr(aux, "_resolve_nous_runtime_api", lambda **kw: tuple(runtime))
    unhealthy = []
    monkeypatch.setattr(aux, "_mark_provider_unhealthy", lambda *a, **kw: unhealthy.append(a))
    client = object()
    monkeypatch.setattr(aux, "_create_openai_client", lambda **kw: client)
    monkeypatch.setattr(aux, "_aux_probe_active", lambda: True)
    assert aux._try_nous() == (None, None)
    assert unhealthy == []
    runtime[:] = [make_jwt(account_tier="free", client_id="hermes-cli"), NAMED]
    assert aux._try_nous()[0] is client


@pytest.mark.parametrize("tier", ["anonymous", "free", "paid"])
def test_401_diagnostics_follow_request_identity_on_welcome_host(tier, capsys, monkeypatch):
    import agent.conversation_loop as loop
    from agent.turn_recovery import _print_nous_401_diagnostics

    monkeypatch.setattr(loop, "_print_nous_entitlement_guidance", lambda *a: False)
    _print_nous_401_diagnostics(agent_for(make_jwt(account_tier=tier), WELCOME), Exception("unauthorized"))
    output = capsys.readouterr().out
    assert ("Hermes couldn't start a new one" in output) == (tier == "anonymous")
    assert ("hermes auth add nous" in output) == (tier != "anonymous")


def test_anonymous_claim_does_not_classify_other_providers_as_nous():
    error = Exception("refused")
    error.status_code = 429
    error.body = {"reason": "rate_limited", "retry_after": 600}
    classified = classify_api_error(error, provider="custom", api_key=make_jwt(), base_url=WELCOME)
    assert "welcome_refusal" not in classified.error_context
