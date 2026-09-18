"""Tests for per-channel model and system prompt overrides (Fixes #1955)."""

from types import SimpleNamespace
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

    def test_model_switch_preserves_existing_system_prompt(self):
        runner = object.__new__(GatewayRunner)
        runner.config = GatewayConfig(
            platforms={
                Platform.TELEGRAM: PlatformConfig(
                    enabled=True,
                    channel_overrides={
                        "group-1": ChannelOverride(
                            model="old/model",
                            provider="old-provider",
                            system_prompt="Keep this prompt",
                        ),
                    },
                ),
            },
        )
        source = SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="group-1",
            chat_type="group",
        )

        runner._apply_channel_model_override(
            source,
            SimpleNamespace(new_model="new/model", target_provider="new-provider"),
        )

        override = runner.config.platforms[Platform.TELEGRAM].channel_overrides["group-1"]
        assert override.model == "new/model"
        assert override.provider == "new-provider"
        assert override.system_prompt == "Keep this prompt"
        assert override.enforce is True

    def test_model_switch_creates_enum_key_when_platform_is_absent(self):
        runner = object.__new__(GatewayRunner)
        runner.config = GatewayConfig(platforms={})
        source = SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="group-1",
            chat_type="group",
        )

        runner._apply_channel_model_override(
            source,
            SimpleNamespace(new_model="new/model", target_provider="new-provider"),
        )

        assert Platform.TELEGRAM in runner.config.platforms
        assert "telegram" not in runner.config.platforms
        override = runner.config.platforms[Platform.TELEGRAM].channel_overrides["group-1"]
        assert override.model == "new/model"
        assert override.provider == "new-provider"
        assert override.enforce is True

    def test_model_switch_mutates_the_routed_profile_config(self):
        runner = object.__new__(GatewayRunner)
        runner.config = GatewayConfig(
            platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, channel_overrides={})},
            multiplex_profiles=True,
        )
        secondary_config = GatewayConfig(
            platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, channel_overrides={})},
            multiplex_profiles=True,
        )
        runner._profile_configs = {"secondary": secondary_config}
        source = SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="group-1",
            chat_type="group",
            profile="secondary",
        )

        runner._apply_channel_model_override(
            source,
            SimpleNamespace(new_model="secondary/model", target_provider="secondary-provider"),
        )

        assert secondary_config.platforms[Platform.TELEGRAM].channel_overrides["group-1"].model == "secondary/model"
        assert runner.config.platforms[Platform.TELEGRAM].channel_overrides == {}


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

    def test_channel_override_beats_enforced_session_override(self):
        runner = object.__new__(GatewayRunner)
        runner._session_model_overrides = {
            "agent:main:telegram:group:chan_1:u1": {
                "model": "sender/model",
                "provider": "deepseek",
                "api_key": "sender-key",
            },
        }
        runner._peek_session_state = lambda _key: type(
            "State", (), {"conversation": type(
                "Conversation", (), {"model_override": runner._session_model_overrides.get(
                    "agent:main:telegram:group:chan_1:u1"
                )}
            )()}
        )()
        runner._rehydrate_session_model_override = lambda _key: None
        runner.config = GatewayConfig(
            platforms={
                Platform.TELEGRAM: PlatformConfig(
                    enabled=True,
                    channel_overrides={
                        "chan_1": ChannelOverride(
                            model="channel/model",
                            provider="openrouter",
                            enforce=True,
                        ),
                    },
                ),
            },
        )
        source = SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="chan_1",
            chat_type="group",
            user_id="u1",
        )
        with patch("gateway.run._resolve_gateway_model", return_value="global/model"), \
             patch("gateway.run._resolve_runtime_agent_kwargs", return_value={
                 "provider": "anthropic",
                 "api_key": "global-key",
                 "base_url": "https://api.anthropic.com",
                 "api_mode": "chat_completions",
             }), \
             patch(
                 "gateway.run._resolve_runtime_agent_kwargs_for_provider",
                 return_value={
                     "provider": "openrouter",
                     "api_key": "channel-key",
                     "base_url": "https://openrouter.ai/api/v1",
                     "api_mode": "chat_completions",
                 },
             ):
            model, runtime = runner._resolve_session_agent_runtime(
                source=source,
                session_key="agent:main:telegram:group:chan_1:u1",
                user_config={"model": {"default": "global/model"}},
            )
        assert model == "channel/model"
        assert runtime["provider"] == "openrouter"
        assert runtime["api_key"] == "channel-key"


