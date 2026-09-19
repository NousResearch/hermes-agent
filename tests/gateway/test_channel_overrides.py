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

    def test_telegram_topic_key_overrides_chat_default(self):
        config = GatewayConfig(
            platforms={
                Platform.TELEGRAM: PlatformConfig(
                    enabled=True,
                    channel_overrides={
                        "-100123": ChannelOverride(model="chat-model"),
                        "-100123:188": ChannelOverride(model="topic-model"),
                    },
                ),
            },
        )
        result = _get_channel_override(
            config,
            Platform.TELEGRAM,
            "-100123",
            thread_id="188",
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

    def test_telegram_topic_override_beats_chat_default(self):
        runner = object.__new__(GatewayRunner)
        runner._session_model_overrides = {}
        runner.config = GatewayConfig(
            platforms={
                Platform.TELEGRAM: PlatformConfig(
                    enabled=True,
                    channel_overrides={
                        "-100123": ChannelOverride(
                            model="chat-model",
                            provider="chat-provider",
                        ),
                        "-100123:188": ChannelOverride(
                            model="topic-model",
                            provider="topic-provider",
                        ),
                    },
                ),
            },
        )
        source = SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="-100123",
            chat_type="forum",
            thread_id="188",
            user_id="u1",
        )
        with patch("gateway.run._resolve_gateway_model", return_value="global/model"), \
             patch("gateway.run._resolve_runtime_agent_kwargs", return_value={
                 "provider": "global-provider",
                 "api_key": "global-key",
             }), \
             patch(
                 "gateway.run._resolve_runtime_agent_kwargs_for_provider",
                 return_value={
                     "provider": "topic-provider",
                     "api_key": "topic-key",
                 },
             ):
            model, runtime = runner._resolve_session_agent_runtime(
                source=source,
                user_config={"model": {"default": "global/model"}},
            )
        assert model == "topic-model"
        assert runtime["provider"] == "topic-provider"
        assert runtime["api_key"] == "topic-key"


class TestResolveSessionReasoningPriority:
    """Reasoning priority: session -> channel -> per-model -> global."""

    def test_channel_reasoning_effort_beats_global(self):
        runner = object.__new__(GatewayRunner)
        runner.config = GatewayConfig(
            platforms={
                Platform.TELEGRAM: PlatformConfig(
                    enabled=True,
                    channel_overrides={
                        "-100123:188": ChannelOverride(reasoning_effort="high"),
                    },
                ),
            },
        )
        source = SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="-100123",
            chat_type="forum",
            thread_id="188",
            user_id="u1",
        )
        with patch.object(
            GatewayRunner,
            "_load_reasoning_config",
            staticmethod(lambda _model="": {"enabled": True, "effort": "low"}),
        ):
            reasoning = runner._resolve_session_reasoning_config(source=source)
        assert reasoning == {"enabled": True, "effort": "high"}

    def test_channel_reasoning_effort_beats_per_model_override(self):
        runner = object.__new__(GatewayRunner)
        runner.config = GatewayConfig(
            platforms={
                Platform.TELEGRAM: PlatformConfig(
                    enabled=True,
                    channel_overrides={
                        "-100123:188": ChannelOverride(reasoning_effort="high"),
                    },
                ),
            },
        )
        source = SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="-100123",
            chat_type="forum",
            thread_id="188",
            user_id="u1",
        )
        with patch.object(
            GatewayRunner,
            "_load_reasoning_config",
            staticmethod(lambda _model="": {"enabled": True, "effort": "xhigh"}),
        ):
            reasoning = runner._resolve_session_reasoning_config(
                source=source,
                model="openai/gpt-5",
            )
        assert reasoning == {"enabled": True, "effort": "high"}

    def test_fallback_uses_effective_model(self):
        runner = object.__new__(GatewayRunner)
        runner.config = GatewayConfig()

        with patch.object(
            GatewayRunner,
            "_load_reasoning_config",
            return_value={"enabled": True, "effort": "xhigh"},
        ) as load_reasoning:
            reasoning = runner._resolve_session_reasoning_config(
                session_key="agent:main:telegram:private:1",
                model="openai/gpt-5",
            )

        assert reasoning == {"enabled": True, "effort": "xhigh"}
        load_reasoning.assert_called_once_with("openai/gpt-5")

    def test_session_reasoning_beats_channel_override(self):
        runner = object.__new__(GatewayRunner)
        runner.config = GatewayConfig(
            platforms={
                Platform.TELEGRAM: PlatformConfig(
                    enabled=True,
                    channel_overrides={
                        "-100123:188": ChannelOverride(reasoning_effort="high"),
                    },
                ),
            },
        )
        source = SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="-100123",
            chat_type="forum",
            thread_id="188",
            user_id="u1",
        )
        session_key = runner._session_key_for_source(source)
        runner._set_session_reasoning_override(
            session_key,
            {"enabled": True, "effort": "minimal"},
        )
        with patch.object(
            GatewayRunner,
            "_load_reasoning_config",
            staticmethod(lambda _model="": {"enabled": True, "effort": "low"}),
        ):
            reasoning = runner._resolve_session_reasoning_config(source=source)
        assert reasoning == {"enabled": True, "effort": "minimal"}

