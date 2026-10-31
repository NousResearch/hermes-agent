"""Session /model OAuth credentials follow the owning auth store, not a switch-time snapshot."""
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from gateway.run import GatewayRunner
from hermes_cli.auth import PROVIDER_REGISTRY
from hermes_cli.model_switch import ModelSwitchResult


def _runner(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    # Default-route fallback is an external dependency of this invariant, not its subject.
    monkeypatch.setattr("gateway.run._resolve_gateway_model", lambda *_: "default-model")
    monkeypatch.setattr("gateway.run._resolve_runtime_agent_kwargs", lambda: {
        "provider": "openrouter", "api_key": "default-key", "base_url": "https://openrouter.ai/api/v1"})
    runner = object.__new__(GatewayRunner)
    runner.config = None
    runner.session_store = None
    runner._session_model_overrides = {}
    return runner


def _write_auth(home, provider, token):
    state = {"tokens": {"access_token": token, "refresh_token": "fixture-refresh"}}
    (home / "auth.json").write_text(json.dumps({
        "version": 1, "providers": {provider: state} if token else {},
        "credential_pool": {},
    }), encoding="utf-8")


@pytest.mark.parametrize("provider", ["openai-codex", "xai-oauth"])
def test_managed_session_override_reresolves_each_turn_and_recovers_from_auth_loss(tmp_path, monkeypatch, provider):
    """Catches both the keyed fast return and later re-overlay of the stale key/pool."""
    runner = _runner(monkeypatch, tmp_path)
    endpoint = PROVIDER_REGISTRY[provider].inference_base_url
    (tmp_path / "config.yaml").write_text("model:\n  provider: openrouter\n  default: default-model\n")
    override = {
        "model": "selected-model", "provider": provider, "api_key": "fixture-A",
        "base_url": endpoint + "/", "api_mode": "codex_responses",
        "request_overrides": {"service_tier": "priority"}, "capabilities": {"vision": True},
        "credential_pool": object(),
    }
    runner._session_model_overrides["session"] = override
    signatures = set()
    for token in ("fixture-A", "fixture-B", "fixture-C"):
        _write_auth(tmp_path, provider, token)
        model, runtime = runner._resolve_session_agent_runtime(session_key="session")
        assert runtime["api_key"] == token
        assert (model, runtime["provider"], runtime["base_url"], runtime["api_mode"]) == (
            "selected-model", provider, endpoint + "/", "codex_responses")
        assert runtime["request_overrides"] == {"service_tier": "priority"}
        assert runtime["capabilities"] == {"vision": True}
        signatures.add(runner._agent_config_signature(model, runtime, [], ""))
        if token != "fixture-A":
            pool = runtime.get("credential_pool")
            if pool is not None:
                assert pool.entry_id_for_api_key(token) is not None
    assert len(signatures) == 3, "a new credential must rebuild, not reuse the old authenticated agent"

    _write_auth(tmp_path, provider, None)
    model, runtime = runner._resolve_session_agent_runtime(session_key="session")
    assert (model, runtime["provider"], runtime["api_key"]) == ("default-model", "openrouter", "default-key")
    assert provider in runner._pre_agent_fallback_notice
    assert runner._session_model_overrides["session"]["model"] == "selected-model"
    _write_auth(tmp_path, provider, "fixture-D")
    assert runner._resolve_session_agent_runtime(session_key="session")[1]["api_key"] == "fixture-D"
    assert runner._pre_agent_fallback_notice is None

    # A provider's newly configured proxy key must not be re-paired with the pinned official host.
    for fresh_route in (
        {"provider": provider, "api_key": "proxy-only", "base_url": "https://proxy.invalid/v1"},
        {"provider": "xai-oauth" if provider == "openai-codex" else "openai-codex",
         "api_key": "other-provider", "base_url": endpoint},
        {"provider": provider, "api_key": "", "base_url": endpoint},
    ):
        monkeypatch.setattr("gateway.run._resolve_runtime_agent_kwargs_for_provider", lambda *a, **k: dict(fresh_route))
        model, runtime = runner._resolve_session_agent_runtime(session_key="session")
        assert (model, runtime["provider"], runtime["api_key"]) == ("default-model", "openrouter", "default-key")
        assert provider in runner._pre_agent_fallback_notice


@pytest.mark.asyncio
@pytest.mark.parametrize("provider,alias,custom_endpoint", [
    ("custom:private", "", True),
    ("openai-codex", "", True),
    ("xai-oauth", "", True),
    ("openai-codex", "private-alias", False),
    ("xai-oauth", "private-alias", False),
])
async def test_recorded_explicit_routes_keep_their_own_key(tmp_path, monkeypatch, provider, alias, custom_endpoint):
    """Even an alias retaining a managed provider label and official URL owns its explicit key."""
    runner = _runner(monkeypatch, tmp_path)
    endpoint = "https://private.invalid/v1" if custom_endpoint else PROVIDER_REGISTRY[provider].inference_base_url
    runner._session_db = None
    runner._async_session_store = SimpleNamespace(_store=None, set_model_override=AsyncMock())
    runner._evict_cached_agent = Mock()
    ctx = SimpleNamespace(session_key="session", current_model="old-model", persist_global=False)
    from hermes_cli.model_switch import DirectAlias
    monkeypatch.setattr("hermes_cli.model_switch.DIRECT_ALIASES", {
        alias: DirectAlias("selected-model", provider, endpoint, api_key="explicit-key")
    } if alias else {})
    result = ModelSwitchResult(
        success=True, new_model="selected-model", target_provider=provider,
        api_key="explicit-key", base_url=endpoint, api_mode="codex_responses",
        resolved_via_alias=alias)
    await runner._record_model_switch(result, ctx, source=None, one_turn=False, picker=False)
    _write_auth(tmp_path, provider, "managed-key")
    # Pool discovery is not credential authority for a route with its own explicit key.
    monkeypatch.setattr("gateway.run._credential_pool_for_provider", lambda *_: None)
    for _ in range(2):
        model, runtime = runner._resolve_session_agent_runtime(session_key="session")
        assert (model, runtime["provider"], runtime["base_url"], runtime["api_key"]) == (
            "selected-model", provider, endpoint, "explicit-key")


@pytest.mark.asyncio
@pytest.mark.parametrize("provider,alias,selected", [
    ("openai-codex", "gpt", "gpt-fixture"),
    ("xai-oauth", "grok", "grok-fixture"),
])
async def test_catalog_alias_producer_keeps_managed_credential_ownership(tmp_path, monkeypatch, provider, alias, selected):
    from hermes_cli.model_switch import switch_model

    runner = _runner(monkeypatch, tmp_path)
    endpoint = PROVIDER_REGISTRY[provider].inference_base_url
    monkeypatch.setattr("hermes_cli.model_switch.DIRECT_ALIASES", {})
    monkeypatch.setattr("hermes_cli.model_switch.list_provider_models", lambda *_: [selected])
    monkeypatch.setattr("hermes_cli.models._PROVIDER_MODELS", {})
    monkeypatch.setattr("hermes_cli.model_switch.get_model_capabilities", lambda *a, **k: None)
    monkeypatch.setattr("hermes_cli.model_switch.get_model_info", lambda *a, **k: None)
    monkeypatch.setattr("hermes_cli.models_validate.validate_requested_model", lambda *a, **k: {"accepted": True})
    monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider", lambda **k: {
        "provider": provider, "api_key": "switch-key", "base_url": endpoint, "api_mode": "codex_responses"})
    result = switch_model(alias, provider, "old-model", endpoint, "switch-key")
    assert result.success, result.error_message
    assert result.resolved_via_alias == alias
    runner._session_db = None
    runner._async_session_store = SimpleNamespace(_store=None, set_model_override=AsyncMock())
    runner._evict_cached_agent = Mock()
    ctx = SimpleNamespace(session_key="session", current_model="old-model", persist_global=False)
    await runner._record_model_switch(result, ctx, source=None, one_turn=False, picker=False)
    monkeypatch.setattr("gateway.run._resolve_runtime_agent_kwargs_for_provider", lambda *a, **k: {
        "provider": provider, "api_key": "fresh-key", "base_url": endpoint, "credential_pool": None})
    monkeypatch.setattr("gateway.run._credential_pool_for_provider", lambda *_: None)
    model, runtime = runner._resolve_session_agent_runtime(session_key="session")
    assert (model, runtime["api_key"]) == (selected, "fresh-key")


@pytest.mark.parametrize("provider", ["openai-codex", "xai-oauth"])
@pytest.mark.parametrize("channel_failure", ["conflicting-pool", "provider-error", "lookup-error"])
def test_managed_session_route_bypasses_channel_resolution(tmp_path, monkeypatch, provider, channel_failure):
    from gateway.config import Platform
    from gateway.session import SessionSource

    runner = _runner(monkeypatch, tmp_path)
    runner.config = SimpleNamespace()
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="fixture-chat", user_id="fixture-user")
    endpoint = PROVIDER_REGISTRY[provider].inference_base_url
    override = {
        "model": "selected-model", "provider": provider, "requested_provider": provider,
        "api_key": "stale-key", "base_url": endpoint + "/", "api_mode": "codex_responses",
        "credential_pool": object(), "max_tokens": 123,
        "capabilities": {"vision": True}, "request_overrides": {"service_tier": "priority"},
    }
    runner._session_model_overrides["session"] = override
    channel_pool = object()

    def resolve(provider_name, **kwargs):
        if provider_name == provider:
            return {"provider": provider, "api_key": "fresh-key", "base_url": endpoint,
                    "credential_pool": None, "api_mode": "chat_completions"}
        if channel_failure == "provider-error":
            raise RuntimeError("channel provider unavailable")
        return {"provider": provider_name, "api_key": "channel-key", "base_url": "https://channel.invalid/v1",
                "credential_pool": channel_pool}

    channel_lookup = Mock(return_value=SimpleNamespace(model="channel-model", provider="openrouter"))
    if channel_failure == "lookup-error":
        channel_lookup.side_effect = RuntimeError("channel lookup unavailable")
    monkeypatch.setattr("gateway.run._get_channel_override", channel_lookup)
    monkeypatch.setattr("gateway.run._resolve_runtime_agent_kwargs_for_provider", resolve)
    model, runtime = runner._resolve_session_agent_runtime(source=source, session_key="session")
    assert model == "selected-model"
    assert runtime == {k: override.get(k) for k in (
        "provider", "requested_provider", "api_key", "base_url", "api_mode", "max_tokens",
        "credential_pool", "request_overrides", "capabilities")}
    assert runtime["api_key"] == "fresh-key"
    assert runtime["credential_pool"] is None
    assert override["credential_pool"] is None
    channel_lookup.assert_not_called()
