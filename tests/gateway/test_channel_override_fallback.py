"""Unavailable static routes fall back as a whole, without losing the override."""
from copy import deepcopy
from unittest.mock import patch

import pytest

from gateway.config import ChannelOverride, GatewayConfig, Platform, PlatformConfig
from gateway.run import GatewayRunner
from gateway.session import SessionSource
from hermes_cli.auth import AuthError


@pytest.mark.parametrize("channel_model", ["channel-model", None])
@pytest.mark.parametrize("fallback_model", [None, "fallback-model"])
def test_failed_channel_provider_preserves_default_route_and_retries(channel_model, fallback_model):
    runner = object.__new__(GatewayRunner)
    channel = ChannelOverride(model=channel_model, provider="openai-codex")
    runner.config = GatewayConfig(platforms={Platform.TELEGRAM: PlatformConfig(
        enabled=True, channel_overrides={"chat": channel})})
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="chat", user_id="u")
    default = {"provider": "anthropic", "api_key": "fixture", "base_url": "https://default.invalid",
               "api_mode": "anthropic", "request_overrides": {"max_tokens": 321}}
    if fallback_model:
        default.update(model=fallback_model, _fallback_notice="existing default fallback notice")
    original = deepcopy(channel)
    recovered = {"provider": "openai-codex", "api_key": "recovered", "model": "provider-model"}
    with patch("gateway.run._resolve_gateway_model", return_value="default-model"), \
         patch("gateway.run._resolve_runtime_agent_kwargs", side_effect=lambda: deepcopy(default)), \
         patch("gateway.run._resolve_runtime_agent_kwargs_for_provider", side_effect=[
             AuthError("quota exhausted", code="codex_rate_limited"), deepcopy(recovered)]) as resolve:
        model, runtime = runner._resolve_session_agent_runtime(source=source)
        assert model == (fallback_model or "default-model")
        assert runtime == {k: v for k, v in default.items() if k not in {"model", "_fallback_notice"}}
        notice = runner._pre_agent_fallback_notice
        if fallback_model:
            assert notice == default["_fallback_notice"]
        else:
            assert "openai-codex" in notice and "anthropic/default-model" in notice
        assert channel == original
        model, runtime = runner._resolve_session_agent_runtime(source=source)
        assert model == (channel_model or "provider-model")
        assert runtime["provider"] == "openai-codex"
        assert resolve.call_count == 2
        assert channel == original


def test_model_only_channel_does_not_resolve_a_provider():
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(platforms={Platform.TELEGRAM: PlatformConfig(
        channel_overrides={"chat": ChannelOverride(model="channel-model")})})
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="chat", user_id="u")
    with patch("gateway.run._resolve_gateway_model", return_value="default-model"), \
         patch("gateway.run._resolve_runtime_agent_kwargs", return_value={"provider": "local"}), \
         patch("gateway.run._resolve_runtime_agent_kwargs_for_provider") as resolve:
        assert runner._resolve_session_agent_runtime(source=source) == ("channel-model", {"provider": "local"})
        resolve.assert_not_called()
