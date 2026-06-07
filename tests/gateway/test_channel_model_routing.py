"""Per-route channel_models override in _resolve_session_agent_runtime.

Precedence contract: session /model override > channel_models > config default.
"""

from unittest.mock import AsyncMock, MagicMock

import gateway.run as gateway_run
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.session import SessionSource


def _make_runner(channel_models=None):
    runner = object.__new__(gateway_run.GatewayRunner)
    runner._session_model_overrides = {}
    runner._session_reasoning_overrides = {}
    extra = {"channel_models": channel_models} if channel_models else {}
    runner.config = GatewayConfig(platforms={Platform.TELEGRAM: PlatformConfig(extra=extra)})
    runner.hooks = MagicMock()
    runner.hooks.emit = AsyncMock()
    return runner


def _source(chat_id="100", **kw):
    return SessionSource(platform=Platform.TELEGRAM, chat_id=chat_id, chat_type="group", user_id="u1", **kw)


def test_channel_model_used_when_no_session_override(monkeypatch):
    monkeypatch.setattr(gateway_run, "_resolve_gateway_model", lambda *_: "config/default")
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", dict)
    runner = _make_runner({"100": "anthropic/claude-opus-4-8"})
    model, _ = runner._resolve_session_agent_runtime(source=_source(), user_config={})
    assert model == "anthropic/claude-opus-4-8"


def test_config_default_used_when_no_route(monkeypatch):
    monkeypatch.setattr(gateway_run, "_resolve_gateway_model", lambda *_: "config/default")
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", dict)
    runner = _make_runner({"999": "anthropic/claude-opus-4-8"})
    model, _ = runner._resolve_session_agent_runtime(source=_source(), user_config={})
    assert model == "config/default"


def test_session_override_beats_channel_model(monkeypatch):
    monkeypatch.setattr(gateway_run, "_resolve_gateway_model", lambda *_: "config/default")
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", dict)
    runner = _make_runner({"100": "anthropic/claude-opus-4-8"})
    source = _source()
    key = runner._session_key_for_source(source)
    runner._session_model_overrides[key] = {
        "model": "openai/gpt-5",
        "provider": "openai",
        "api_key": "***",
        "base_url": None,
        "api_mode": None,
    }
    model, _ = runner._resolve_session_agent_runtime(source=source, user_config={})
    assert model == "openai/gpt-5"
