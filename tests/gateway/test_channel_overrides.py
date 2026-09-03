"""Tests for per-channel model and system prompt overrides (Fixes #1955)."""

from unittest.mock import patch

import pytest

from gateway.config import (
    ChannelOverride,
    GatewayConfig,
    Platform,
    PlatformConfig,
)
from gateway.run import _get_channel_override, GatewayRunner
from gateway.session import SessionSource
from types import SimpleNamespace


class TestGetChannelOverride:


    def test_no_override_when_channel_not_in_overrides(self):
        config = GatewayConfig(
            platforms={
                Platform.DISCORD: PlatformConfig(
                    enabled=True,
                    channel_overrides={
                        "999": ChannelOverride(model="openrouter/healer-alpha"),
                    },
                ),
            },
        )
        assert _get_channel_override(config, Platform.DISCORD, "123") is None

    def test_returns_override_when_channel_matches(self):
        ov = ChannelOverride(
            model="openrouter/healer-alpha",
            provider="openrouter",
            system_prompt="You are a summarizer.",
        )
        config = GatewayConfig(
            platforms={
                Platform.DISCORD: PlatformConfig(
                    enabled=True,
                    channel_overrides={"1234567890": ov},
                ),
            },
        )
        result = _get_channel_override(config, Platform.DISCORD, "1234567890")
        assert result is not None
        assert result.model == "openrouter/healer-alpha"
        assert result.provider == "openrouter"
        assert result.system_prompt == "You are a summarizer."


    def test_thread_id_lookup_when_chat_id_misses(self):
        config = GatewayConfig(
            platforms={
                Platform.DISCORD: PlatformConfig(
                    enabled=True,
                    channel_overrides={
                        "thread_99": ChannelOverride(model="topic-model"),
                    },
                ),
            },
        )
        result = _get_channel_override(
            config, Platform.DISCORD, "parent_chan", thread_id="thread_99"
        )
        assert result is not None
        assert result.model == "topic-model"


class TestResolveModelForChannel:
    def test_uses_channel_override_when_present(self):
        config = GatewayConfig(
            platforms={
                Platform.DISCORD: PlatformConfig(
                    enabled=True,
                    channel_overrides={
                        "chan_1": ChannelOverride(model="anthropic/claude-opus-4.6"),
                    },
                ),
            },
        )
        runner = object.__new__(GatewayRunner)
        runner.config = config
        model = runner._resolve_model_for_channel(Platform.DISCORD, "chan_1")
        assert model == "anthropic/claude-opus-4.6"


class TestGetSystemPromptForChannel:
    def test_uses_channel_override_when_present(self):
        config = GatewayConfig(
            platforms={
                Platform.DISCORD: PlatformConfig(
                    enabled=True,
                    channel_overrides={
                        "chan_1": ChannelOverride(system_prompt="You are a coding assistant."),
                    },
                ),
            },
        )
        runner = object.__new__(GatewayRunner)
        runner.config = config
        runner._ephemeral_system_prompt = "Global prompt"
        prompt = runner._get_system_prompt_for_channel(Platform.DISCORD, "chan_1")
        assert prompt == "You are a coding assistant."


class TestResolveSessionAgentRuntimePriority:
    """Model/runtime priority: session /model → channel_overrides → global."""

    def test_channel_override_beats_global(self):
        runner = object.__new__(GatewayRunner)
        runner._session_model_overrides = {}
        runner.config = GatewayConfig(
            platforms={
                Platform.DISCORD: PlatformConfig(
                    enabled=True,
                    channel_overrides={
                        "chan_1": ChannelOverride(
                            model="channel/model",
                            provider="openrouter",
                        ),
                    },
                ),
            },
        )
        source = SessionSource(
            platform=Platform.DISCORD,
            chat_id="chan_1",
            user_id="u1",
        )
        with patch("gateway.run._resolve_gateway_model", return_value="global/model"), \
             patch("gateway.run._resolve_runtime_agent_kwargs", return_value={
                 "provider": "anthropic",
                 "api_key": "k",
                 "base_url": "https://api.anthropic.com",
                 "api_mode": "chat_completions",
             }), \
             patch(
                 "gateway.run._resolve_runtime_agent_kwargs_for_provider",
                 return_value={
                     "provider": "openrouter",
                     "api_key": "k2",
                     "base_url": "https://openrouter.ai/api/v1",
                     "api_mode": "chat_completions",
                 },
             ):
            model, runtime = runner._resolve_session_agent_runtime(
                source=source,
                user_config={"model": {"default": "global/model"}},
            )
        assert model == "channel/model"
        assert runtime["provider"] == "openrouter"


class TestChannelOverrideReasoningEffort:
    """ChannelOverride carries reasoning_effort and warns on unknown keys (#79468)."""

    def test_parses_reasoning_effort(self):
        ov = ChannelOverride.from_dict(
            {"model": "m", "provider": "p", "reasoning_effort": "high"}
        )
        assert ov.reasoning_effort == "high"
        assert ov.to_dict()["reasoning_effort"] == "high"

    def test_from_dict_warns_on_unknown_key(self, caplog):
        import logging
        with caplog.at_level(logging.WARNING, logger="gateway.config"):
            ov = ChannelOverride.from_dict(
                {"model": "m", "reasoning_effort": "low", "typo_key": 1}
            )
        assert ov.reasoning_effort == "low"
        assert ov.model == "m"
        assert "typo_key" in caplog.text

    def test_channel_override_used_when_session_has_no_reasoning_override(self):
        """channel_overrides.reasoning_effort wins over global config (#79468)."""
        runner = object.__new__(GatewayRunner)
        runner.config = GatewayConfig(
            platforms={
                Platform.DISCORD: PlatformConfig(
                    enabled=True,
                    channel_overrides={
                        "chan_1": ChannelOverride(reasoning_effort="xhigh"),
                    },
                ),
            },
        )
        runner._peek_session_state = lambda key: SimpleNamespace(
            conversation=SimpleNamespace(reasoning_override=None)
        )
        runner._session_key_for_source = lambda source: "discord:chan_1"
        source = SessionSource(
            platform=Platform.DISCORD, chat_id="chan_1", user_id="u1"
        )
        with patch.object(GatewayRunner, "_load_reasoning_config", return_value=None) as mock_load:
            result = runner._resolve_session_reasoning_config(source=source)
        assert result == {"enabled": True, "effort": "xhigh"}
        mock_load.assert_not_called()

    def test_session_reasoning_override_still_wins_over_channel(self):
        """Session-scoped /reasoning --session remains the top priority (#79468)."""
        runner = object.__new__(GatewayRunner)
        runner.config = GatewayConfig(
            platforms={
                Platform.DISCORD: PlatformConfig(
                    enabled=True,
                    channel_overrides={
                        "chan_1": ChannelOverride(reasoning_effort="high"),
                    },
                ),
            },
        )
        runner._peek_session_state = lambda key: SimpleNamespace(
            conversation=SimpleNamespace(
                reasoning_override={"enabled": False}
            )
        )
        runner._session_key_for_source = lambda source: "discord:chan_1"
        source = SessionSource(
            platform=Platform.DISCORD, chat_id="chan_1", user_id="u1"
        )
        with patch.object(GatewayRunner, "_load_reasoning_config", return_value=None):
            result = runner._resolve_session_reasoning_config(source=source)
        assert result == {"enabled": False}


class TestFormatSessionInfoChannelModel:
    """_format_session_info advertises the channel-override model (#79468)."""

    def test_banner_uses_channel_override_model(self):
        runner = object.__new__(GatewayRunner)
        runner.config = GatewayConfig(
            platforms={
                Platform.DISCORD: PlatformConfig(
                    enabled=True,
                    channel_overrides={
                        "chan_1": ChannelOverride(model="channel/model"),
                    },
                ),
            },
        )
        source = SessionSource(
            platform=Platform.DISCORD, chat_id="chan_1", user_id="u1"
        )
        with patch("gateway.run._resolve_gateway_model", return_value="global/model"), \
             patch(
                 "gateway.run._resolve_runtime_agent_kwargs",
                 return_value={
                     "provider": "anthropic",
                     "api_key": "k",
                     "base_url": "https://api.anthropic.com",
                 },
             ), \
             patch("agent.model_metadata.get_model_context_length", return_value=128000):
            info = runner._format_session_info(source)
        assert "channel/model" in info
        assert "global/model" not in info

    def test_banner_falls_back_to_global_without_override(self):
        runner = object.__new__(GatewayRunner)
        runner.config = GatewayConfig(platforms={})
        source = SessionSource(
            platform=Platform.DISCORD, chat_id="chan_1", user_id="u1"
        )
        with patch("gateway.run._resolve_gateway_model", return_value="global/model"), \
             patch(
                 "gateway.run._resolve_runtime_agent_kwargs",
                 return_value={
                     "provider": "anthropic",
                     "api_key": "k",
                     "base_url": "https://api.anthropic.com",
                 },
             ), \
             patch("agent.model_metadata.get_model_context_length", return_value=128000):
            info = runner._format_session_info(source)
        assert "global/model" in info


