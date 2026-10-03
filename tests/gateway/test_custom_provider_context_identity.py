"""Regression coverage for provider identity in gateway context resolution."""

import asyncio
from types import MethodType

import gateway.run as gateway_run
from gateway.run import GatewayRunner


_CONFIG = {
    "model": {"default": "same-model", "provider": "custom:second-route"},
    "providers": {
        "first-route": {
            "name": "First Route",
            "base_url": "https://shared.invalid/v1",
            "models": {"same-model": {"context_length": 500_000}},
        },
        "second-route": {
            "name": "Second Route",
            "base_url": "https://shared.invalid/v1",
            "models": {"same-model": {"context_length": 1_050_000}},
        },
    },
}

_RUNTIME = {
    "provider": "custom",
    "requested_provider": "custom:second-route",
    "base_url": "https://shared.invalid/v1",
    "api_key": "test-key",
    "api_mode": "chat_completions",
}


def test_gateway_context_resolution_uses_requested_provider_identity(monkeypatch):
    monkeypatch.setattr(gateway_run, "_resolve_gateway_model", lambda: "same-model")
    monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda: _CONFIG)
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", lambda: dict(_RUNTIME))

    resolved = gateway_run._resolve_gateway_model_context()

    assert resolved.context_length == 1_050_000
    assert resolved.provider == "custom"


def test_hygiene_settings_uses_requested_provider_identity(monkeypatch):
    runner = object.__new__(GatewayRunner)
    monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda: _CONFIG)
    def _resolve(_self, **_kwargs):
        return "same-model", dict(_RUNTIME)

    runner._resolve_session_agent_runtime = MethodType(_resolve, runner)

    async def _run():
        return await runner._hmwa_hygiene_settings(None, "agent:main")

    settings = asyncio.run(_run())

    assert settings.config_context_length == 1_050_000
    assert settings.provider == "custom"
    assert settings.requested_provider == "custom:second-route"
