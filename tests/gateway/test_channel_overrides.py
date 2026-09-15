"""Tests for per-channel model and system prompt overrides (Fixes #1955)."""

import json
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


def _write_directory(tmp_path, platforms):
    """Fake ``channel_directory.json`` (the gateway rebuilds it every 5 min)."""
    path = tmp_path / "channel_directory.json"
    path.write_text(json.dumps({"updated_at": "2026-01-01T00:00:00", "platforms": platforms}))
    return path


def _discord_config(overrides):
    return GatewayConfig(
        platforms={Platform.DISCORD: PlatformConfig(enabled=True, channel_overrides=overrides)},
    )


@pytest.fixture(autouse=True)
def _isolate_channel_directory(tmp_path_factory):
    """Never read the real ~/.hermes channel directory / alias overlay from a lookup test."""
    missing = tmp_path_factory.mktemp("directory") / "none.json"
    with patch("gateway.channel_directory.DIRECTORY_PATH", missing), \
         patch("gateway.channel_directory.CHANNEL_ALIASES_PATH", missing):
        yield


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


class TestChannelNameAndPatternKeys:
    """#109676: ``channel_overrides`` keys may be channel NAMES, or regex patterns over names.

    Specificity: exact id → exact name (resolved only after an id miss) → first matching pattern.
    """

    CHANNEL_ID = "1543849479231246416"

    def _directory(self, tmp_path, channels):
        return patch(
            "gateway.channel_directory.DIRECTORY_PATH",
            _write_directory(tmp_path, {"discord": channels}),
        )

    def test_exact_id_hit_never_reads_the_directory(self):
        config = _discord_config({self.CHANNEL_ID: ChannelOverride(model="by-id")})
        with patch("gateway.channel_directory.load_directory") as load_directory:
            ov = _get_channel_override(config, Platform.DISCORD, self.CHANNEL_ID)
        assert ov is not None and ov.model == "by-id"
        load_directory.assert_not_called()

    def test_name_key_matches_after_id_miss(self, tmp_path):
        config = _discord_config({"work-evs_root": ChannelOverride(model="by-name")})
        channels = [{"id": self.CHANNEL_ID, "name": "work-evs_root", "guild": "EVS", "type": "channel"}]
        with self._directory(tmp_path, channels):
            ov = _get_channel_override(config, Platform.DISCORD, self.CHANNEL_ID)
        assert ov is not None and ov.model == "by-name"

    def test_id_and_name_keys_coexist_with_id_winning(self, tmp_path):
        config = _discord_config({
            self.CHANNEL_ID: ChannelOverride(model="by-id"),
            "work-evs_root": ChannelOverride(model="by-name"),
        })
        channels = [{"id": self.CHANNEL_ID, "name": "work-evs_root", "guild": "EVS", "type": "channel"}]
        with self._directory(tmp_path, channels):
            ov = _get_channel_override(config, Platform.DISCORD, self.CHANNEL_ID)
        assert ov is not None and ov.model == "by-id"

    def test_name_key_resolves_for_thread_and_parent_slots(self, tmp_path):
        config = _discord_config({"work-evs_root": ChannelOverride(model="parent-name")})
        channels = [{"id": "111", "name": "work-evs_root", "guild": "EVS", "type": "channel"}]
        with self._directory(tmp_path, channels):
            ov = _get_channel_override(
                config, Platform.DISCORD, "thread-1",
                thread_id="thread-9", parent_id="111",
            )
        assert ov is not None and ov.model == "parent-name"

    def test_unresolved_name_keeps_the_original_miss(self, tmp_path):
        """A name key is inert when the id resolves to a different channel (or to nothing)."""
        config = _discord_config({"work-evs_root": ChannelOverride(model="by-name")})
        channels = [{"id": "999", "name": "other-channel", "guild": "EVS", "type": "channel"}]
        with self._directory(tmp_path, channels):
            assert _get_channel_override(config, Platform.DISCORD, self.CHANNEL_ID) is None

    def test_missing_directory_file_keeps_the_original_miss(self):
        """No directory (fresh home, DMs, unwritten cache) → only id keys apply, never an error."""
        config = _discord_config({"work-evs_root": ChannelOverride(model="by-name")})
        assert _get_channel_override(config, Platform.DISCORD, self.CHANNEL_ID) is None

    def test_pattern_key_matches_channel_name(self, tmp_path):
        config = _discord_config({"^work-.*_project-": ChannelOverride(model="by-pattern")})
        channels = [
            {"id": "111", "name": "work-evs_project-idea-engine", "guild": "EVS", "type": "channel"},
            {"id": "222", "name": "work-evs_general", "guild": "EVS", "type": "channel"},
        ]
        with self._directory(tmp_path, channels):
            assert _get_channel_override(config, Platform.DISCORD, "111").model == "by-pattern"
            assert _get_channel_override(config, Platform.DISCORD, "222") is None

    def test_first_matching_pattern_in_config_order_wins(self, tmp_path):
        channels = [{"id": "111", "name": "work-evs_project-core", "guild": "EVS", "type": "channel"}]
        first = _discord_config({
            "^work-.*_project-": ChannelOverride(model="project"),
            "^work-": ChannelOverride(model="all-work"),
        })
        with self._directory(tmp_path, channels):
            assert _get_channel_override(first, Platform.DISCORD, "111").model == "project"
            assert list(first.platforms[Platform.DISCORD].channel_overrides) == [
                "^work-.*_project-", "^work-",
            ]
        second = _discord_config({
            "^work-": ChannelOverride(model="all-work"),
            "^work-.*_project-": ChannelOverride(model="project"),
        })
        with self._directory(tmp_path, channels):
            assert _get_channel_override(second, Platform.DISCORD, "111").model == "all-work"

    def test_exact_name_beats_pattern(self, tmp_path):
        config = _discord_config({
            "^work-": ChannelOverride(model="by-pattern"),
            "work-evs_root": ChannelOverride(model="by-name"),
        })
        channels = [{"id": self.CHANNEL_ID, "name": "work-evs_root", "guild": "EVS", "type": "channel"}]
        with self._directory(tmp_path, channels):
            assert _get_channel_override(config, Platform.DISCORD, self.CHANNEL_ID).model == "by-name"

    def test_invalid_pattern_key_is_inert(self, tmp_path):
        config = _discord_config({"^work-(": ChannelOverride(model="broken")})
        channels = [{"id": "111", "name": "work-evs_project", "guild": "EVS", "type": "channel"}]
        with self._directory(tmp_path, channels):
            assert _get_channel_override(config, Platform.DISCORD, "111") is None


