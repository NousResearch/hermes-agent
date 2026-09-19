"""Tests for per-channel model and system prompt overrides (Fixes #1955)."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

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


class TestChannelOverrideFallbackProviders:
    """Channel fallback chains must preserve absent-versus-empty semantics."""

    def test_config_round_trip_preserves_explicit_empty_fallback_chain(self):
        unset = ChannelOverride.from_dict({})
        disabled = ChannelOverride.from_dict({"fallback_providers": []})

        assert unset.fallback_providers is None
        assert disabled.fallback_providers == []
        assert disabled.to_dict() == {"fallback_providers": []}

    def test_source_fallback_uses_global_chain_only_when_override_is_unset(self):
        global_chain = [{"provider": "global", "model": "global/model"}]
        source = SessionSource(platform=Platform.DISCORD, chat_id="chan_1", user_id="u1")
        runner = object.__new__(GatewayRunner)
        runner.config = GatewayConfig(
            platforms={
                Platform.DISCORD: PlatformConfig(
                    enabled=True,
                    channel_overrides={"chan_1": ChannelOverride()},
                ),
            },
        )
        refresh = Mock(return_value=global_chain)
        runner._refresh_fallback_model = refresh

        assert runner._resolve_fallback_model_for_source(source) == global_chain
        refresh.assert_called_once_with()

        runner.config.platforms[Platform.DISCORD].channel_overrides["chan_1"] = ChannelOverride(
            fallback_providers=[],
        )
        assert runner._resolve_fallback_model_for_source(source) == []
        refresh.assert_called_once_with()

    def test_fresh_agent_receives_source_resolved_fallback_chain(self):
        """The fresh interactive construction path preserves the resolved route chain."""
        from gateway.run_turn_runner import TurnRunner

        expected = [{"provider": "channel", "model": "channel/model"}]
        captured = {}

        class CapturingAgent:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        source = SessionSource(platform=Platform.DISCORD, chat_id="chan_1", user_id="u1")
        runner = SimpleNamespace(
            _prefill_messages=None,
            _service_tier=None,
            _session_db=None,
            _resolve_fallback_model_for_source=Mock(return_value=expected),
        )
        context = SimpleNamespace(
            AIAgent=CapturingAgent,
            user_config={},
            enabled_toolsets=[],
            disabled_toolsets=None,
            session_id="session",
            session_key="agent:main:discord:group:chan_1",
            source=source,
        )
        turn_runner = TurnRunner(runner, context)

        with patch("gateway.run._checkpoint_agent_kwargs", return_value={}):
            turn_runner._build_fresh_agent(
                {"model": "primary", "runtime": {}},
                "discord", "", 10, None, {}, False,
            )

        assert captured["fallback_model"] == expected

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
