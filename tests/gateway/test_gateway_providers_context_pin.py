"""Regression coverage for a gateway context pin whose endpoint is in providers.<name>."""

from unittest.mock import patch

PINNED_CONTEXT = 1_048_576
PROXY_URL = "https://proxy.example.com/v1"


def test_gateway_context_pin_uses_named_provider_route(monkeypatch):
    from gateway import run as gateway_run

    config = {
        "model": {"default": "my-model", "provider": "my-proxy", "base_url": "", "context_length": PINNED_CONTEXT},
        "providers": {"my-proxy": {"base_url": PROXY_URL, "key_env": "MY_PROXY_KEY"}},
    }
    monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda: config)
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", lambda: {
        "provider": "custom", "base_url": PROXY_URL, "api_key": "test-key",
    })

    def fake_context_length(*_args, config_context_length=None, **_kwargs):
        return config_context_length or 256_000

    with patch("agent.model_metadata.get_model_context_length", side_effect=fake_context_length):
        context = gateway_run._resolve_gateway_model_context()

    assert context.context_length == PINNED_CONTEXT
    assert context.context_source == "config"


def test_hygiene_context_pin_uses_named_provider_route(monkeypatch):
    from gateway import run as gateway_run
    from gateway.run import GatewayRunner

    config = {
        "model": {"default": "my-model", "provider": "my-proxy", "base_url": "", "context_length": PINNED_CONTEXT},
        "providers": {"my-proxy": {"base_url": PROXY_URL, "key_env": "MY_PROXY_KEY"}},
    }
    monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda: config)
    runner = GatewayRunner.__new__(GatewayRunner)
    runner._resolve_session_agent_runtime = lambda **_kwargs: ("my-model", {
        "provider": "custom", "base_url": PROXY_URL, "api_key": "test-key",
    })

    import asyncio
    settings = asyncio.run(runner._hmwa_hygiene_settings(source=None, session_key=None))

    assert settings.config_context_length == PINNED_CONTEXT
