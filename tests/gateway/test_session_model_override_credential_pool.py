"""Session /model overrides must attach credential_pool for 402 rotation."""

from __future__ import annotations

from unittest.mock import MagicMock

from gateway.run import GatewayRunner, _credential_pool_for_provider


def test_fast_session_override_includes_credential_pool(monkeypatch):
    runner = object.__new__(GatewayRunner)
    runner._session_model_overrides = {
        "sess-1": {
            "model": "kimi-k2.7",
            "provider": "custom:hyper",
            "api_key": "sk-test",
            "base_url": "https://hyper.charm.land/v1",
            "api_mode": "chat_completions",
        },
    }
    fake_pool = object()

    monkeypatch.setattr(
        "gateway.run._resolve_gateway_model",
        lambda _uc=None: "default-model",
    )
    monkeypatch.setattr(
        "gateway.run._credential_pool_for_provider",
        lambda provider: fake_pool if provider == "custom:hyper" else None,
    )

    model, runtime = runner._resolve_session_agent_runtime(session_key="sess-1")

    assert model == "kimi-k2.7"
    assert runtime.get("credential_pool") is fake_pool


def test_credentialless_route_override_clears_global_credentials():
    runner = object.__new__(GatewayRunner)
    runner._session_model_overrides = {
        "sess-1": {
            "model": "routed-model",
            "provider": "routed-provider",
            "base_url": "https://routed.example/v1",
        },
    }
    global_pool = object()

    model, runtime = runner._apply_session_model_override(
        "sess-1",
        "global-model",
        {
            "provider": "global-provider",
            "base_url": "https://global.example/v1",
            "api_key": "global-key",
            "api_mode": "global-mode",
            "credential_pool": global_pool,
        },
    )

    assert model == "routed-model"
    assert runtime == {
        "provider": "routed-provider",
        "base_url": "https://routed.example/v1",
        "api_key": None,
        "api_mode": None,
        "credential_pool": None,
    }


def test_model_only_override_keeps_active_route_credentials():
    runner = object.__new__(GatewayRunner)
    runner._session_model_overrides = {
        "sess-1": {"model": "alternate-global-model"},
    }
    global_pool = object()
    active_runtime = {
        "provider": "global-provider",
        "base_url": "https://global.example/v1",
        "api_key": "global-key",
        "api_mode": "global-mode",
        "credential_pool": global_pool,
    }

    model, runtime = runner._apply_session_model_override(
        "sess-1", "global-model", dict(active_runtime)
    )

    assert model == "alternate-global-model"
    assert runtime == active_runtime


def test_provider_override_without_endpoint_clears_global_endpoint():
    runner = object.__new__(GatewayRunner)
    runner._session_model_overrides = {
        "sess-1": {
            "model": "routed-model",
            "provider": "routed-provider",
        },
    }

    _model, runtime = runner._apply_session_model_override(
        "sess-1",
        "global-model",
        {
            "provider": "global-provider",
            "base_url": "https://global.example/v1",
            "api_key": "global-key",
            "api_mode": "global-mode",
            "credential_pool": object(),
        },
    )

    assert runtime == {
        "provider": "routed-provider",
        "base_url": None,
        "api_key": None,
        "api_mode": None,
        "credential_pool": None,
    }


