"""Route and failure contracts using real profile-scoped configuration."""
from unittest.mock import patch

import pytest
import yaml

from gateway.config import ChannelOverride, GatewayConfig, Platform, PlatformConfig
from gateway.run import GatewayRunner
from gateway.session import SessionSource


@pytest.fixture
def banner(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr("gateway.run._hermes_home", tmp_path)
    (tmp_path / "config.yaml").write_text(yaml.safe_dump({
        "model": {"default": "base", "context_length": 2000},
        "agent": {"reasoning_effort": "low", "service_tier": "fast",
                  "reasoning_overrides": {"bundled": "high", "channel": False}},
    }))
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig()
    return runner, SessionSource(platform=Platform.MATRIX, chat_id="room")


@pytest.mark.parametrize("explicit", [False, True])
def test_channel_provider_route(banner, explicit):
    runner, source = banner
    runner.config.platforms[Platform.MATRIX] = PlatformConfig(channel_overrides={
        "room": ChannelOverride(provider="custom:channel", model="channel" if explicit else None),
    })
    with patch("gateway.run._resolve_runtime_agent_kwargs", return_value={"provider": "custom:base"}), \
         patch("gateway.run._resolve_runtime_agent_kwargs_for_provider", return_value={
             "provider": "custom:channel", "model": "bundled"}), \
         patch("agent.model_metadata.get_model_context_length", return_value=2000):
        text = runner._reset_notice_session_info(source)
    assert f"Model: `{'channel' if explicit else 'bundled'}`" in text
    assert "Provider: custom:channel" in text
    assert f"Main reasoning: {'off' if explicit else 'high'}" in text
    assert "Service tier (requested): priority" in text


def test_route_failure_does_not_claim_global_route(banner):
    runner, source = banner
    with patch.object(runner, "_resolve_session_agent_runtime", side_effect=RuntimeError("private detail")):
        text = runner._reset_notice_session_info(source, "new-session-identifier")
    assert "Model: unknown" in text
    assert "Main reasoning: unknown (route unavailable)" in text
    assert "new-session-identifier" in text
    assert "private detail" not in text


def test_context_failure_retains_settings_and_id(banner):
    runner, source = banner
    with patch("gateway.run._resolve_runtime_agent_kwargs", return_value={}), \
         patch("agent.model_metadata.get_model_context_length", side_effect=RuntimeError("private detail")):
        text = runner._reset_notice_session_info(source, "new-session-identifier")
    assert "Main reasoning: low" in text
    assert "Context: unknown" in text
    assert "new-session-identifier" in text
    assert "private detail" not in text


def test_empty_route_is_unknown(banner):
    runner, source = banner
    with patch.object(runner, "_resolve_session_agent_runtime", return_value=("", {})):
        text = runner._reset_notice_session_info(source)
    assert "Model: unknown" in text
    assert "Main reasoning: unknown (route unavailable)" in text
