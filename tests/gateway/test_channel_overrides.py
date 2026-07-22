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
    def test_no_override_when_empty_config(self):
        config = GatewayConfig()
        assert _get_channel_override(config, Platform.DISCORD, "123") is None

    def test_no_override_when_platform_not_configured(self):
        config = GatewayConfig(platforms={})
        assert _get_channel_override(config, Platform.DISCORD, "123") is None

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

    def test_returns_override_when_chat_id_is_int_like(self):
        """Caller may pass str(chat_id); override keys are normalized to str."""
        config = GatewayConfig(
            platforms={
                Platform.DISCORD: PlatformConfig(
                    enabled=True,
                    channel_overrides={"123": ChannelOverride(model="gpt-4")},
                ),
            },
        )
        assert _get_channel_override(config, Platform.DISCORD, "123").model == "gpt-4"

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

    def test_parent_id_fallback_when_thread_has_no_entry(self):
        config = GatewayConfig(
            platforms={
                Platform.DISCORD: PlatformConfig(
                    enabled=True,
                    channel_overrides={
                        "parent_chan": ChannelOverride(model="parent-model"),
                    },
                ),
            },
        )
        result = _get_channel_override(
            config,
            Platform.DISCORD,
            "thread_only",
            parent_id="parent_chan",
        )
        assert result is not None
        assert result.model == "parent-model"

    def test_exact_thread_overrides_parent(self):
        config = GatewayConfig(
            platforms={
                Platform.DISCORD: PlatformConfig(
                    enabled=True,
                    channel_overrides={
                        "thread_1": ChannelOverride(model="thread-model"),
                        "parent_chan": ChannelOverride(model="parent-model"),
                    },
                ),
            },
        )
        result = _get_channel_override(
            config, Platform.DISCORD, "thread_1", parent_id="parent_chan"
        )
        assert result.model == "thread-model"

    def test_telegram_composite_topic_key_precedes_chat_and_thread(self):
        config = GatewayConfig(
            platforms={
                Platform.TELEGRAM: PlatformConfig(
                    enabled=True,
                    channel_overrides={
                        "123:42": ChannelOverride(model="topic-model"),
                        "123": ChannelOverride(model="chat-model"),
                        "42": ChannelOverride(model="bare-thread-model"),
                    },
                ),
            },
        )

        result = _get_channel_override(
            config, Platform.TELEGRAM, "123", thread_id="42"
        )

        assert result is not None
        assert result.model == "topic-model"

    def test_telegram_composite_topic_keys_do_not_collide_across_chats(self):
        config = GatewayConfig(
            platforms={
                Platform.TELEGRAM: PlatformConfig(
                    enabled=True,
                    channel_overrides={
                        "123:42": ChannelOverride(model="first-chat-model"),
                        "456:42": ChannelOverride(model="second-chat-model"),
                    },
                ),
            },
        )

        first = _get_channel_override(
            config, Platform.TELEGRAM, "123", thread_id="42"
        )
        second = _get_channel_override(
            config, Platform.TELEGRAM, "456", thread_id="42"
        )

        assert first is not None and first.model == "first-chat-model"
        assert second is not None and second.model == "second-chat-model"


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

    def test_falls_back_to_global_when_no_override(self, monkeypatch):
        monkeypatch.setattr(
            "gateway.run._resolve_gateway_model",
            lambda _cfg=None: "global-model/default",
        )
        config = GatewayConfig(
            platforms={
                Platform.DISCORD: PlatformConfig(enabled=True, channel_overrides={}),
            },
        )
        runner = object.__new__(GatewayRunner)
        runner.config = config
        model = runner._resolve_model_for_channel(Platform.DISCORD, "unknown_channel")
        assert model == "global-model/default"


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

    def test_falls_back_to_global_when_no_override(self):
        config = GatewayConfig(
            platforms={Platform.DISCORD: PlatformConfig(enabled=True)},
        )
        runner = object.__new__(GatewayRunner)
        runner.config = config
        runner._ephemeral_system_prompt = "Global prompt"
        prompt = runner._get_system_prompt_for_channel(Platform.DISCORD, "other")
        assert prompt == "Global prompt"


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

    def test_telegram_provider_only_topic_adopts_provider_model(self):
        runner = object.__new__(GatewayRunner)
        runner._session_model_overrides = {}
        runner.config = GatewayConfig(
            platforms={
                Platform.TELEGRAM: PlatformConfig(
                    enabled=True,
                    channel_overrides={
                        "123:42": ChannelOverride(provider="openrouter"),
                    },
                ),
            },
        )
        source = SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="123",
            thread_id="42",
            user_id="u1",
        )
        with patch("gateway.run._resolve_gateway_model", return_value="global/model"), \
             patch("gateway.run._resolve_runtime_agent_kwargs", return_value={
                 "provider": "anthropic",
                 "api_key": "k",
                 "model": "global/runtime-model",
             }), \
             patch(
                 "gateway.run._resolve_runtime_agent_kwargs_for_provider",
                 return_value={
                     "provider": "openrouter",
                     "api_key": "k2",
                     "base_url": "https://openrouter.ai/api/v1",
                     "api_mode": "chat_completions",
                     "model": "openrouter/provider-default",
                 },
             ):
            model, runtime = runner._resolve_session_agent_runtime(source=source)

        assert model == "openrouter/provider-default"
        assert runtime["provider"] == "openrouter"


class TestResolveEnabledToolsets:
    def _runner(self, override: ChannelOverride) -> tuple[GatewayRunner, SessionSource]:
        runner = object.__new__(GatewayRunner)
        runner.config = GatewayConfig(
            platforms={
                Platform.TELEGRAM: PlatformConfig(
                    enabled=True,
                    channel_overrides={"123:42": override},
                ),
            },
        )
        source = SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="123",
            thread_id="42",
            user_id="u1",
        )
        return runner, source

    def test_topic_toolsets_use_standard_platform_resolver(self):
        runner, source = self._runner(ChannelOverride(toolsets=["web", "file", "no_mcp"]))

        enabled = runner._resolve_enabled_toolsets(
            {"platform_toolsets": {"telegram": ["hermes-telegram"]}},
            "telegram",
            source,
        )

        assert "web" in enabled
        assert "file" in enabled

    def test_empty_topic_toolsets_preserve_platform_surface(self):
        runner, source = self._runner(ChannelOverride(toolsets=[]))
        user_config = {"platform_toolsets": {"telegram": ["hermes-telegram"]}}

        enabled = runner._resolve_enabled_toolsets(user_config, "telegram", source)

        assert enabled

    def test_session_model_beats_channel_override(self):
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
        session_key = "agent:main:discord:channel:chan_1"
        runner._session_model_overrides = {
            session_key: {
                "model": "session/model",
                "provider": "anthropic",
            },
        }
        source = SessionSource(
            platform=Platform.DISCORD,
            chat_id="chan_1",
            chat_type="channel",
            user_id="u1",
        )
        with patch("gateway.run._resolve_gateway_model", return_value="global/model"), \
             patch("gateway.run._resolve_runtime_agent_kwargs", return_value={
                 "provider": "openrouter",
                 "api_key": "k",
                 "base_url": "https://openrouter.ai/api/v1",
                 "api_mode": "chat_completions",
             }):
            model, runtime = runner._resolve_session_agent_runtime(
                source=source,
                session_key=session_key,
            )
        assert model == "session/model"
        assert runtime["provider"] == "anthropic"

    def test_parent_channel_model_inherited_in_thread(self):
        runner = object.__new__(GatewayRunner)
        runner._session_model_overrides = {}
        runner.config = GatewayConfig(
            platforms={
                Platform.DISCORD: PlatformConfig(
                    enabled=True,
                    channel_overrides={
                        "parent_chan": ChannelOverride(model="parent/model"),
                    },
                ),
            },
        )
        source = SessionSource(
            platform=Platform.DISCORD,
            chat_id="thread_1",
            chat_type="thread",
            parent_chat_id="parent_chan",
            user_id="u1",
        )
        with patch("gateway.run._resolve_gateway_model", return_value="global/model"), \
             patch("gateway.run._resolve_runtime_agent_kwargs", return_value={
                 "provider": "anthropic",
                 "api_key": "k",
                 "base_url": "https://api.anthropic.com",
                 "api_mode": "chat_completions",
             }):
            model, _runtime = runner._resolve_session_agent_runtime(source=source)
        assert model == "parent/model"
