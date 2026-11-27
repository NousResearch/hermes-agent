"""A spent Claude plan window refuses every model on the account (#149112).

Anthropic names the window that refused a subscription request in its unified rate-limit headers.
``five_hour`` and ``seven_day`` cover the whole account: the pool must bench that credential for
every model until the window's unified reset. Any rejected window, a family claim
(``seven_day_opus``) included, rotates on its first 429 instead of retrying the spent seat after up to
``RETRY_AFTER_CAP_S``; a family claim and a 429 without the headers (API-key per-model limits) keep the
per-model cooldown of #111769, and the headerless 429 keeps its one retry.

Real ``agent.credential_pool`` against a temp auth store; the 429 is a real ``anthropic`` error.
"""
import json
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import anthropic
import httpx
import pytest

SPENT, SPARE = "sk-ant-api03-spent-seat", "sk-ant-api03-spare-seat"
FABLE, OPUS = "claude-fable-5-2", "claude-opus-5-5"
RATE_LIMIT_MESSAGE = "This request would exceed your account's rate limit. Please try again later."


def _plan_429(claim: str, reset: float, retry_after: float | None = None) -> anthropic.RateLimitError:
    """The 429 a spent subscription window returns (headers as captured in #147070 / #149112)."""
    body = {"type": "error", "error": {"type": "rate_limit_error", "message": RATE_LIMIT_MESSAGE}}
    response = httpx.Response(
        429, request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"), json=body,
        headers={
            "retry-after": str(int(reset - time.time() if retry_after is None else retry_after)),
            "anthropic-ratelimit-unified-status": "rejected",
            "anthropic-ratelimit-unified-representative-claim": claim,
            "anthropic-ratelimit-unified-reset": str(int(reset)),
            "anthropic-ratelimit-unified-5h-status": "allowed",
        },
    )
    return anthropic.RateLimitError(f"Error code: 429 - {body}", response=response, body=body)


def _isolated_home(tmp_path, monkeypatch):
    root = tmp_path / "hermes-root"
    root.mkdir()
    (tmp_path / "fakehome").mkdir()
    monkeypatch.setenv("HOME", str(tmp_path / "fakehome"))
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "fakehome"))
    for var in ("ANTHROPIC_TOKEN", "ANTHROPIC_API_KEY", "CLAUDE_CODE_OAUTH_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("HERMES_HOME", str(root))
    import hermes_constants
    monkeypatch.setattr(hermes_constants, "_default_hermes_root_memo", None, raising=False)
    return root


@pytest.fixture
def pool(tmp_path, monkeypatch):
    root = _isolated_home(tmp_path, monkeypatch)
    seats = [("spent", SPENT), ("spare", SPARE)]
    (root / "auth.json").write_text(json.dumps({"credential_pool": {"anthropic": [
        {"id": seat, "label": seat, "auth_type": "api_key", "priority": priority,
         "source": "manual", "access_token": key}
        for priority, (seat, key) in enumerate(seats)
    ]}}))
    from agent.credential_pool import load_pool
    return load_pool("anthropic")


def _bench_end(pool) -> float:
    from agent.credential_pool_timestamps import _parse_absolute_timestamp
    spent = next(e for e in pool.entries() if e.id == "spent")
    return _parse_absolute_timestamp(spent.last_error_reset_at)


@pytest.mark.parametrize("claim, account_wide", [
    ("seven_day", True), ("five_hour", True), ("seven_day_opus", False),
])
def test_spent_plan_window_bench_scope_follows_the_claim(pool, claim, account_wide):
    from agent.agent_runtime_helpers import extract_api_error_context
    from agent.credential_pool import load_pool

    reset = time.time() + 63 * 3600
    context = extract_api_error_context(_plan_429(claim, reset))
    rotated = pool.mark_exhausted_and_rotate(
        status_code=429, error_context=context, api_key_hint=SPENT, failure_reason="rate_limit", model=FABLE,
    )

    assert rotated.id == "spare"
    # In this pool and in a fresh load of the store: the refused model never goes back to the spent
    # seat; another model does only when the claim is a model family.
    for view in (pool, load_pool("anthropic")):
        assert view.select(model=FABLE).id == "spare"
        assert view.select(model=OPUS).id == ("spare" if account_wide else "spent")
    if account_wide:
        assert abs(_bench_end(pool) - reset) < 5  # benched until the window resets, not one hour


@pytest.mark.parametrize("claim, rotates_now", [("seven_day", True), ("seven_day_opus", True), (None, False)])
def test_first_spent_window_429_rotates_instead_of_waiting(pool, claim, rotates_now):
    from agent.agent_runtime_helpers import extract_api_error_context, recover_with_credential_pool

    context = (extract_api_error_context(_plan_429(claim, time.time() + 63 * 3600)) if claim
               else {"reason": "rate_limit_error", "message": RATE_LIMIT_MESSAGE})
    agent = SimpleNamespace(
        provider="anthropic", api_key=SPENT, model=FABLE, _credential_pool=pool,
        _credential_pool_entry_id="spent", _swap_credential=MagicMock(),
    )

    recovered, has_retried_429 = recover_with_credential_pool(
        agent, status_code=429, has_retried_429=False, error_context=context,
    )

    # A rejected plan window (account or family) stays shut until its reset; a plain per-minute 429 keeps its one retry.
    assert (recovered, has_retried_429) == ((True, False) if rotates_now else (False, True))
    if rotates_now:
        assert agent._swap_credential.call_args[0][0].id == "spare"


def test_auxiliary_429_keeps_the_plan_window_and_its_reset(pool, monkeypatch):
    """Titles and compression recover through the auxiliary client: a spent window benches the whole
    credential until its reset there too."""
    from agent import auxiliary_client as ac
    from agent.auxiliary_error_policy import classify_attempt

    reset = time.time() + 63 * 3600
    error = _plan_429("seven_day", reset)
    monkeypatch.setattr(ac, "load_pool", lambda provider: pool)
    monkeypatch.setattr(ac, "_evict_cached_clients", lambda provider: None)

    assert ac._recover_provider_pool(
        "anthropic", error, failed_api_key=SPENT, verdict=classify_attempt(error, "anthropic", model=FABLE),
    )

    assert pool.select(model=OPUS).id == "spare"
    assert abs(_bench_end(pool) - reset) < 5


def test_account_window_reset_beats_a_shorter_retry_after():
    """An account-wide window's future unified reset replaces a shorter retry-after; a family claim and a
    stale unified reset leave it."""
    from agent.agent_runtime_helpers import extract_api_error_context
    from agent.credential_pool_timestamps import _parse_absolute_timestamp

    now = time.time()
    reset, stale = now + 63 * 3600, now - 600
    for claim, window_reset, expected in (
        ("seven_day", reset, reset), ("seven_day_opus", reset, now + 2), ("seven_day", stale, now + 2),
    ):
        context = extract_api_error_context(_plan_429(claim, window_reset, retry_after=2))
        assert abs(_parse_absolute_timestamp(context["reset_at"]) - expected) < 5, (claim, window_reset)

    class _OddBody(Exception):
        body = {"error": {"type": "rate_limit_error", "reset_at": []}}

    assert extract_api_error_context(_OddBody("quota"))["reason"] == "rate_limit_error"


def test_spent_window_holds_env_tokens_and_is_reported_as_a_cooldown(tmp_path, monkeypatch):
    """An ``ANTHROPIC_API_KEY`` token skips pool selection: the bench must still hold it for every model,
    and a fully benched pool is a cooldown, not "No Anthropic credentials found"."""
    from agent.agent_runtime_helpers import extract_api_error_context
    from agent.anthropic_credentials import resolve_anthropic_token
    from agent.credential_pool import load_pool
    from hermes_cli.auth import AuthError, is_rate_limited_auth_error
    from hermes_cli.runtime_provider import _anthropic_token_or_raise

    _isolated_home(tmp_path, monkeypatch)
    monkeypatch.setenv("ANTHROPIC_API_KEY", SPENT)
    # Control: a 401 bench does not hold an env token; the user's own key is left to fail visibly.
    load_pool("anthropic").mark_exhausted_and_rotate(status_code=401, api_key_hint=SPENT, model=FABLE)
    assert resolve_anthropic_token(model=OPUS) == SPENT
    load_pool("anthropic").mark_exhausted_and_rotate(
        status_code=429, error_context=extract_api_error_context(_plan_429("seven_day", time.time() + 63 * 3600)),
        api_key_hint=SPENT, failure_reason="rate_limit", model=FABLE,
    )

    assert resolve_anthropic_token(model=FABLE) is None
    assert resolve_anthropic_token(model=OPUS) is None
    assert resolve_anthropic_token() == SPENT  # diagnostics without a model keep the token
    with pytest.raises(AuthError) as excinfo:
        _anthropic_token_or_raise(model=OPUS)
    assert excinfo.value.code == "model_rate_limited"
    assert is_rate_limited_auth_error(excinfo.value)  # gateway / Kanban / fallback treat it as quota, not login
    assert "cooling down" in str(excinfo.value) and "No Anthropic credentials" not in str(excinfo.value)


@pytest.mark.parametrize("status", [401, 402])
def test_an_auth_or_billing_bench_is_not_reported_as_a_rate_limit(pool, status):
    """Only a 429 bench is a cooldown; a fully benched pool after a 401/402 keeps the sign-in diagnostic."""
    from hermes_cli.auth import AuthError, is_rate_limited_auth_error
    from hermes_cli.runtime_provider import _anthropic_token_or_raise

    for key in (SPENT, SPARE):
        pool.mark_exhausted_and_rotate(status_code=status, api_key_hint=key, model=FABLE)
    with pytest.raises(AuthError) as excinfo:
        _anthropic_token_or_raise(model=OPUS)
    assert not is_rate_limited_auth_error(excinfo.value)
    assert "cooling down" not in str(excinfo.value)


def _env_key_beside_a_401_row(tmp_path, monkeypatch):
    from agent.credential_pool import load_pool

    root = _isolated_home(tmp_path, monkeypatch)
    (root / "auth.json").write_text(json.dumps({"credential_pool": {"anthropic": [
        {"id": "spare", "label": "spare", "auth_type": "api_key", "priority": 0, "source": "manual",
         "access_token": SPARE}]}}))
    monkeypatch.setenv("ANTHROPIC_API_KEY", SPENT)
    load_pool("anthropic").mark_exhausted_and_rotate(status_code=401, api_key_hint=SPARE, model=OPUS)


def _bench(key: str, claim: str, hours: float, model: str = OPUS) -> None:
    from agent.agent_runtime_helpers import extract_api_error_context
    from agent.credential_pool import load_pool

    load_pool("anthropic").mark_exhausted_and_rotate(
        status_code=429, error_context=extract_api_error_context(_plan_429(claim, time.time() + hours * 3600)),
        api_key_hint=key, failure_reason="rate_limit", model=model,
    )


@pytest.mark.parametrize("claim, account_wide", [("seven_day", True), ("seven_day_opus", False)])
def test_a_benched_env_token_beside_an_unusable_row_names_its_own_scope(tmp_path, monkeypatch, claim, account_wide):
    from hermes_cli.auth import AuthError, is_rate_limited_auth_error
    from hermes_cli.runtime_provider import _anthropic_token_or_raise

    _env_key_beside_a_401_row(tmp_path, monkeypatch)
    _bench(SPENT, claim, 63)
    with pytest.raises(AuthError) as excinfo:
        _anthropic_token_or_raise(model=OPUS)
    assert is_rate_limited_auth_error(excinfo.value)
    assert ("every Claude model" in str(excinfo.value)) is account_wide
    assert ("other Claude models remain available" in str(excinfo.value)) is not account_wide


@pytest.mark.parametrize("account_hours, model_hours", [(1, 2), (2, 1)])
def test_an_active_account_bench_is_named_whichever_cooldown_ends_last(tmp_path, monkeypatch, account_hours, model_hours):
    from hermes_cli.auth import AuthError
    from hermes_cli.runtime_provider import _anthropic_token_or_raise

    _env_key_beside_a_401_row(tmp_path, monkeypatch)
    _bench(SPENT, "seven_day_opus", model_hours)
    _bench(SPENT, "seven_day", account_hours, model=FABLE)
    with pytest.raises(AuthError) as excinfo:
        _anthropic_token_or_raise(model=OPUS)
    assert "every Claude model" in str(excinfo.value)
    assert "remain available" not in str(excinfo.value)


def test_auxiliary_client_skips_an_env_token_benched_for_the_requested_model(tmp_path, monkeypatch):
    from agent.auxiliary_client import _try_anthropic

    _isolated_home(tmp_path, monkeypatch)
    monkeypatch.setenv("ANTHROPIC_API_KEY", SPENT)
    _bench(SPENT, "seven_day_opus", 63)
    assert _try_anthropic(requested_model=FABLE)[0] is not None
    assert _try_anthropic(requested_model=f"anthropic/{OPUS}") == (None, None)
    _bench(SPENT, "seven_day", 63, model=FABLE)
    assert _try_anthropic(requested_model=FABLE) == (None, None)


def test_auxiliary_client_without_a_requested_model_is_not_gated_on_the_default_model(tmp_path, monkeypatch):
    import agent.auxiliary_client as aux

    _isolated_home(tmp_path, monkeypatch)
    monkeypatch.setenv("ANTHROPIC_API_KEY", SPENT)
    _bench(SPENT, "seven_day_opus", 63, model=aux._get_aux_model_for_provider("anthropic") or "claude-haiku-4-5-20251001")
    assert aux._try_anthropic()[0] is not None


def test_api_key_aux_chain_moves_past_a_refused_anthropic_client(monkeypatch):
    import agent.auxiliary_client as aux

    rows = [("anthropic", SimpleNamespace(auth_type="api_key")), ("next", SimpleNamespace(auth_type="api_key"))]
    monkeypatch.setattr("hermes_cli.auth_plugin_providers.canonical_registry_rows", lambda: rows)
    monkeypatch.setattr("hermes_cli.auth.is_provider_explicitly_configured", lambda pid: True)
    monkeypatch.setattr(aux, "_is_provider_unhealthy", lambda pid: False)
    served = MagicMock()
    for anthropic_result, expected in (((served, "claude-haiku"), (served, "claude-haiku")), ((None, None), (None, None))):
        monkeypatch.setattr(aux, "_try_anthropic", lambda result=anthropic_result: result)
        reached = []
        monkeypatch.setattr(aux, "_get_aux_model_for_provider", lambda pid: reached.append(pid))
        assert aux._resolve_api_key_provider() == expected
        assert reached == ([] if anthropic_result[0] else ["next"])
