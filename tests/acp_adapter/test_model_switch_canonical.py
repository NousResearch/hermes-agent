"""5.8.6.5: ACP owns its switch selection and route, not the CLI coordinator."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from acp_adapter import model_switch_resolution as acp


@pytest.fixture(autouse=True)
def no_catalog_network(monkeypatch):
    monkeypatch.setattr(acp, "validate_model_switch", lambda result, config: "")


def test_qualified_choice_acquires_only_the_selected_provider_and_canonical_route(monkeypatch):
    calls = []
    pool = object()

    def acquire(**kw):
        calls.append(kw)
        return {"provider": "anthropic", "base_url": "https://api.anthropic.com",
                "api_mode": "chat_completions", "api_key": "acquired",
                "credential_pool": pool}

    monkeypatch.setattr(acp, "_acquire", acquire)
    route = acp.resolve_acp_model_switch(
        config={}, raw_model="anthropic:claude-sonnet-5",
        current_provider="openrouter", current_model="other",
        current_base_url="https://openrouter.ai/api/v1", current_api_key="foreign",
    )
    assert calls and calls[0]["requested"] == "anthropic"
    assert route.model == "claude-sonnet-5" and route.provider == "anthropic"
    assert route.api_mode == "anthropic_messages"
    assert route.runtime["credential_pool"] is pool
    assert route.runtime["api_key"] == "acquired"
    assert route.base_url == "https://api.anthropic.com"


def test_same_provider_keeps_endpoint_but_recomputes_model_wire(monkeypatch):
    monkeypatch.setattr(acp, "_acquire", lambda **kw: pytest.fail("must reuse live provider"))
    route = acp.resolve_acp_model_switch(
        config={}, raw_model="openrouter:gpt-5.6",
        current_provider="openrouter", current_model="legacy",
        current_base_url="https://openrouter.ai/api/v1",
        current_api_key="existing", keep_endpoint=True,
    )
    assert route.provider == "openrouter" and route.model == "openai/gpt-5.6"
    assert route.base_url == "https://openrouter.ai/api/v1"
    assert route.api_mode == "codex_responses"
    assert route.runtime["api_key"] == "existing"


def test_qualified_named_custom_provider_keeps_route_identity(monkeypatch):
    seen = []
    def acquire(**kw):
        seen.append(kw)
        return {"provider": "custom", "api_key": "named-secret",
                "base_url": "https://relay.example/v1"}
    monkeypatch.setattr(acp, "_acquire", acquire)
    cfg = {"providers": {"relay": {
        "name": "Relay", "base_url": "https://relay.example/v1",
        "default_model": "model-v1", "models": ["model-v1"],
    }}}
    route = acp.resolve_acp_model_switch(
        config=cfg, raw_model="custom:relay:model-v1",
        current_provider="anthropic", current_model="claude-sonnet-5",
    )
    assert route.provider == "custom:relay"
    assert route.model == "model-v1"
    assert route.base_url == "https://relay.example/v1"
    assert seen[0]["requested"] == "relay"


def test_explicit_unknown_provider_is_rejected_before_acquisition(monkeypatch):
    monkeypatch.setattr(acp, "_acquire", lambda **kw: pytest.fail("unknown provider acquired"))
    with pytest.raises(ValueError, match="Unknown provider"):
        acp.resolve_acp_model_switch(
            config={}, raw_model="unregistered:unknown",
            current_provider="unregistered", current_model="legacy",
        )


def test_same_provider_credential_does_not_cross_to_different_host(monkeypatch):
    seen = []
    def acquire(**kw):
        seen.append(kw)
        return {"provider": "custom", "api_key": "relay-key",
                "base_url": kw["explicit_base_url"]}
    monkeypatch.setattr(acp, "_acquire", acquire)
    cfg = {"model_aliases": {"relay": {
        "model": "private-model", "provider": "custom",
        "base_url": "https://other.example/v1",
    }}}
    route = acp.resolve_acp_model_switch(
        config=cfg, raw_model="relay",
        current_provider="custom", current_model="old-model",
        current_base_url="https://first.example/v1",
        current_api_key="do-not-leak", keep_endpoint=True,
    )
    assert route.base_url == "https://other.example/v1"
    assert route.runtime["api_key"] == "relay-key"
    assert seen and seen[0]["explicit_api_key"] is None

def test_same_host_different_path_does_not_reuse_live_credential(monkeypatch):
    calls = []

    def acquire(**kw):
        calls.append(kw)
        return {"provider": "custom", "api_key": "fresh",
                "base_url": kw["explicit_base_url"]}

    monkeypatch.setattr(acp, "_acquire", acquire)
    cfg = {"model_aliases": {"relay": {
        "model": "private-model", "provider": "custom",
        "base_url": "https://same.example/another-tenant/v1",
    }}}
    route = acp.resolve_acp_model_switch(
        config=cfg, raw_model="relay", current_provider="custom",
        current_model="old", current_base_url="https://same.example/first-tenant/v1",
        current_api_key="old-tenant-secret", keep_endpoint=True,
    )
    assert route.runtime["api_key"] == "fresh"
    assert calls and calls[0]["explicit_api_key"] is None


def test_auth_scoping_failure_is_not_disguised_as_invalid_model(monkeypatch):
    from agent.secret_scope import UnscopedSecretError

    def reject(**kw):
        raise UnscopedSecretError("SECRET")

    monkeypatch.setattr(acp, "_acquire", reject)
    with pytest.raises(UnscopedSecretError):
        acp.resolve_acp_model_switch(
            config={}, raw_model="anthropic:claude-sonnet-5",
            current_provider="openrouter", current_model="old",
        )

def test_explicit_provider_retains_alias_host_but_never_uses_alias_secret(monkeypatch):
    calls = []

    def acquire(**kw):
        calls.append(kw)
        return {"provider": "openrouter", "api_key": "openrouter-secret",
                "base_url": kw["explicit_base_url"]}

    monkeypatch.setattr(acp, "_acquire", acquire)
    route = acp.resolve_acp_model_switch(
        config={"model_aliases": {"relay": {
            "model": "private-model", "provider": "custom",
            "base_url": "https://relay.example/v1", "api_key": "wrong-alias-secret",
        }}},
        raw_model="openrouter:relay", current_provider="anthropic",
        current_model="old-model", keep_endpoint=True,
    )
    assert route.provider == "openrouter"
    assert route.base_url == "https://relay.example/v1"
    assert route.runtime["api_key"] == "openrouter-secret"
    assert calls[0]["explicit_base_url"] == "https://relay.example/v1"
    assert calls[0]["explicit_api_key"] is None


def test_same_provider_carries_live_credential_pool(monkeypatch):
    pool = object()
    monkeypatch.setattr(acp, "_acquire", lambda **kw: pytest.fail("unexpected acquisition"))
    route = acp.resolve_acp_model_switch(
        config={}, raw_model="anthropic:claude-sonnet-5",
        current_provider="anthropic", current_model="old",
        current_base_url="https://api.anthropic.com",
        current_api_key="live-secret",
        current_runtime={"credential_pool": pool, "runtime_kind": "http"},
        keep_endpoint=True,
    )
    assert route.runtime["credential_pool"] is pool
    assert route.api_mode == "anthropic_messages"
