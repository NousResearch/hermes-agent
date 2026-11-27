"""Tests for per-channel model and system prompt overrides (Fixes #1955)."""

from unittest.mock import patch

import pytest

from gateway.config import (
    ChannelOverride,
    GatewayConfig,
    Platform,
    PlatformConfig,
)
from gateway.run import _channel_override_lookup_keys, _get_channel_override, GatewayRunner
from gateway.run_config_loaders import _effective_allow_silence
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


def _forum_config(overrides: dict) -> GatewayConfig:
    return GatewayConfig(
        platforms={
            Platform.TELEGRAM: PlatformConfig(enabled=True, channel_overrides=overrides),
        },
    )


class TestCompositeChatThreadKey:
    """``<chat_id>:<thread_id>`` addresses one forum topic exactly.

    On a forum platform ``chat_id`` is the GROUP and ``thread_id`` the topic (Telegram topics),
    the opposite of Discord, where ``chat_id`` is already the thread's own channel id. Without a
    composite key the group entry shadows every topic entry, and a bare topic id (``"4"``) is not
    unique across groups.
    """

    def test_composite_key_beats_the_group_entry(self):
        config = _forum_config({
            "-1001234567890": ChannelOverride(model="group/model"),
            "-1001234567890:4": ChannelOverride(model="topic/model"),
        })
        override = _get_channel_override(
            config, Platform.TELEGRAM, "-1001234567890", thread_id="4")
        assert override is not None
        assert override.model == "topic/model"

    def test_group_entry_still_serves_the_other_topics(self):
        config = _forum_config({
            "-1001234567890": ChannelOverride(model="group/model"),
            "-1001234567890:4": ChannelOverride(model="topic/model"),
        })
        override = _get_channel_override(
            config, Platform.TELEGRAM, "-1001234567890", thread_id="9")
        assert override is not None
        assert override.model == "group/model"

    def test_composite_key_does_not_leak_into_another_group(self):
        config = _forum_config({"-1009999999999:4": ChannelOverride(model="other/topic")})
        assert _get_channel_override(
            config, Platform.TELEGRAM, "-1001234567890", thread_id="4") is None

    def test_bare_thread_key_still_matches(self):
        """Backwards compatibility: the documented bare thread/topic key keeps working."""
        config = _forum_config({"4": ChannelOverride(model="legacy/topic")})
        override = _get_channel_override(
            config, Platform.TELEGRAM, "-1001234567890", thread_id="4")
        assert override is not None
        assert override.model == "legacy/topic"

    def test_discord_parent_inheritance_unchanged(self):
        config = GatewayConfig(
            platforms={
                Platform.DISCORD: PlatformConfig(
                    enabled=True,
                    channel_overrides={"parent_chan": ChannelOverride(model="parent/model")},
                ),
            },
        )
        override = _get_channel_override(
            config, Platform.DISCORD, "thread_chan", parent_id="parent_chan")
        assert override is not None
        assert override.model == "parent/model"

    def test_lookup_key_order(self):
        assert _channel_override_lookup_keys("c", thread_id="t", parent_id="p") == [
            "c:t", "c", "t", "p"]
        assert _channel_override_lookup_keys("c") == ["c"]
        assert _channel_override_lookup_keys("c", parent_id="p") == ["c", "p"]
        # A thread id equal to the chat id (Discord auto-thread continuity) adds no junk key.
        assert _channel_override_lookup_keys("c", thread_id="c") == ["c"]

    def test_room_allow_silence_prefers_the_composite_entry(self):
        platform_config = PlatformConfig(
            enabled=True,
            channel_overrides={
                "-1001234567890": ChannelOverride(allow_silence=False),
                "-1001234567890:4": ChannelOverride(allow_silence=True),
            },
            extra={},
        )
        source = SessionSource(
            platform=Platform.TELEGRAM, chat_id="-1001234567890", chat_type="thread",
            thread_id="4", user_id="u1")
        assert _effective_allow_silence(platform_config, source) is True

    def test_idle_compaction_prefers_the_composite_entry(self):
        runner = object.__new__(GatewayRunner)
        runner.config = _forum_config({
            "-1001234567890": ChannelOverride(idle_compact_after_seconds=0),
            "-1001234567890:4": ChannelOverride(idle_compact_after_seconds=900),
        })
        source = SessionSource(
            platform=Platform.TELEGRAM, chat_id="-1001234567890", chat_type="thread",
            thread_id="4", user_id="u1")
        assert runner._idle_compact_after_for_source(source, {}) == 900


