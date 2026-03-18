"""Tests for per-channel model and system prompt overrides (Fixes #1955)."""

import pytest

from gateway.config import (
    ChannelOverride,
    GatewayConfig,
    Platform,
    PlatformConfig,
)
from gateway.run import _get_channel_override, GatewayRunner


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
            lambda: "global-model/default",
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
