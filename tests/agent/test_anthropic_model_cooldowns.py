"""A generic Anthropic 429 benches only the model that was rate-limited (#111769, #61451).

Real ``agent.credential_pool`` against a real temp auth store: one API-key credential,
one ``mark_exhausted_and_rotate(status_code=429, model=A)``.
"""
import json

import pytest

KEY = "sk-ant-api03-synthetic-test-key-0000"
MODEL_A = "claude-sonnet-4-5"
MODEL_B = "claude-haiku-4-5"


@pytest.fixture
def pool(tmp_path, monkeypatch):
    root = tmp_path / "hermes-root"
    root.mkdir()
    (tmp_path / "fakehome").mkdir()
    monkeypatch.setenv("HOME", str(tmp_path / "fakehome"))
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "fakehome"))
    for var in ("ANTHROPIC_TOKEN", "ANTHROPIC_API_KEY", "CLAUDE_CODE_OAUTH_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("HERMES_HOME", str(root))
    import hermes_constants
    hermes_constants._default_hermes_root_memo = None  # type: ignore[attr-defined]
    (root / "auth.json").write_text(json.dumps({"credential_pool": {"anthropic": [{
        "id": "seat", "label": "seat", "auth_type": "api_key", "priority": 0,
        "source": "manual", "access_token": KEY,
    }]}}))
    from agent.credential_pool import load_pool
    return load_pool("anthropic")


def test_generic_429_benches_only_the_rate_limited_model(pool, monkeypatch):
    from agent.anthropic_credentials import resolve_anthropic_token
    from agent.credential_pool import load_pool

    ctx = {"message": "This request would exceed your account's rate limit. Please try again later."}
    assert pool.mark_exhausted_and_rotate(
        status_code=429, error_context=ctx, api_key_hint=KEY, failure_reason="rate_limit", model=MODEL_A,
    ) is None  # a sole credential has nothing to rotate to for model A

    assert pool.entries()[0].last_status is None  # credential-wide state untouched
    assert pool.select(model=MODEL_A) is None
    assert pool.select(model=MODEL_B) is not None
    assert pool.select() is None  # a caller that names no model honours every active cooldown
    assert pool.next_available_at(model=MODEL_A) is not None
    assert pool.next_available_at(model=MODEL_B) is None

    # The cooldown is persisted, so another process (and the env/borrowed token resolver,
    # which reads the store fresh) sees the same per-model verdict.
    fresh = load_pool("anthropic")
    assert fresh.select(model=MODEL_A) is None and fresh.select(model=MODEL_B) is not None
    monkeypatch.setenv("ANTHROPIC_API_KEY", KEY)
    assert resolve_anthropic_token(model=MODEL_A) is None
    assert resolve_anthropic_token(model=MODEL_B) == KEY
    assert resolve_anthropic_token() == KEY  # model-less diagnostics keep the key


@pytest.mark.parametrize("status_code, failure_reason", [(401, None), (402, "billing"), (429, "billing")])
def test_auth_and_billing_failures_stay_credential_wide(pool, status_code, failure_reason):
    from agent.credential_pool import STATUS_EXHAUSTED

    pool.mark_exhausted_and_rotate(
        status_code=status_code, api_key_hint=KEY, failure_reason=failure_reason, model=MODEL_A,
    )
    assert pool.entries()[0].last_status == STATUS_EXHAUSTED
    assert not pool.entries()[0].model_cooldowns
    assert pool.select(model=MODEL_B) is None


def test_oauth_pool_discovery_keeps_model_specific_availability(pool, monkeypatch):
    from agent.anthropic_credentials import resolve_anthropic_token
    from agent.credential_pool import AUTH_TYPE_OAUTH

    entry = pool.entries()[0]
    entry.auth_type = AUTH_TYPE_OAUTH
    entry.refresh_token = "refresh-token"
    entry.model_cooldowns = {MODEL_A: 4102444800.0}
    pool._persist()

    assert resolve_anthropic_token(model=MODEL_A) is None
    assert resolve_anthropic_token(model=MODEL_B) == KEY
    assert resolve_anthropic_token() is None


@pytest.mark.parametrize("configured_model", [None, "claude-sonnet-4-5"])
@pytest.mark.parametrize("cool_aux_model", [True, False])
def test_auxiliary_resolver_honors_actual_model_cooldown(
    pool, monkeypatch, configured_model, cool_aux_model,
):
    from unittest.mock import Mock

    from agent import auxiliary_client
    from agent.credential_pool import AUTH_TYPE_OAUTH
    from hermes_constants import get_hermes_home

    aux_model = configured_model or "claude-haiku-4-5-20251001"
    cooled_model = aux_model if cool_aux_model else "claude-opus-4-6"
    entry = pool.entries()[0]
    entry.auth_type = AUTH_TYPE_OAUTH
    entry.refresh_token = "refresh-token"
    entry.model_cooldowns = {cooled_model: 4102444800.0}
    pool._persist()
    auth_path = get_hermes_home() / "auth.json"
    before = auth_path.read_bytes()

    # Keep real pool selection and token resolution. Substitute only model
    # configuration and SDK construction; no provider request is made.
    monkeypatch.setattr(auxiliary_client, "_get_aux_model_for_provider", lambda provider: configured_model)
    monkeypatch.setattr("agent.anthropic_credentials.read_claude_code_credentials", lambda: None)
    build = Mock(return_value=Mock())
    monkeypatch.setattr("agent.anthropic_adapter.build_anthropic_client", build)
    available, _ = pool._available_entries(model=aux_model)
    assert bool(available) is (not cool_aux_model)

    client, model = auxiliary_client._try_anthropic()
    if cool_aux_model:
        assert (client, model) == (None, None)
        build.assert_not_called()
    else:
        assert isinstance(client, auxiliary_client.AnthropicAuxiliaryClient)
        assert model == aux_model
        assert build.call_args.args[0] == KEY
    assert auth_path.read_bytes() == before
