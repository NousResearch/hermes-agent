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
from gateway.run import (
    _resolve_runtime_agent_kwargs_for_provider,
    _try_resolve_fallback_provider,
)
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
        with (
            patch("gateway.run._resolve_gateway_model", return_value="global/model"),
            patch(
                "gateway.run._resolve_runtime_agent_kwargs",
                return_value={
                    "provider": "anthropic",
                    "api_key": "k",
                    "base_url": "https://api.anthropic.com",
                    "api_mode": "chat_completions",
                },
            ),
            patch(
                "gateway.run._resolve_runtime_agent_kwargs_for_provider",
                return_value={
                    "provider": "openrouter",
                    "api_key": "k2",
                    "base_url": "https://openrouter.ai/api/v1",
                    "api_mode": "chat_completions",
                },
            ) as resolve_runtime,
        ):
            model, runtime = runner._resolve_session_agent_runtime(
                source=source,
                user_config={"model": {"default": "global/model"}},
            )
        assert model == "channel/model"
        assert runtime["provider"] == "openrouter"
        resolve_runtime.assert_called_once_with(
            "openrouter", target_model="channel/model"
        )


def test_provider_helper_passes_target_model_to_runtime_resolver(monkeypatch):
    captured = {}

    def fake_resolve_runtime_provider(**kwargs):
        captured.update(kwargs)
        return {
            "provider": "copilot",
            "api_key": "token",
            "base_url": "https://api.githubcopilot.com",
            "api_mode": "codex_responses",
        }

    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        fake_resolve_runtime_provider,
    )

    runtime = _resolve_runtime_agent_kwargs_for_provider(
        "copilot", target_model="gpt-5.6-sol"
    )

    assert captured == {"requested": "copilot", "target_model": "gpt-5.6-sol"}
    assert runtime["api_mode"] == "codex_responses"


def test_fallback_provider_passes_its_model_to_runtime_resolver(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        "gateway.run._load_gateway_runtime_config",
        lambda: {
            "fallback_providers": [{"provider": "copilot", "model": "claude-opus-5"}]
        },
    )
    monkeypatch.setattr(
        "hermes_cli.fallback_config.resolve_entry_api_key", lambda entry: None
    )

    def fake_resolve_runtime_provider(**kwargs):
        captured.update(kwargs)
        return {
            "provider": "copilot",
            "api_key": "token",
            "base_url": "https://api.githubcopilot.com",
            "api_mode": "chat_completions",
        }

    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        fake_resolve_runtime_provider,
    )
    runtime = _try_resolve_fallback_provider()

    assert runtime is not None
    assert captured["target_model"] == "claude-opus-5"
    assert runtime["model"] == "claude-opus-5"
    assert runtime["api_mode"] == "chat_completions"


