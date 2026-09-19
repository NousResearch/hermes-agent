"""Regression tests for #115216: DISCORD_HOME_CHANNEL accepts a pasted channel link.

Discord's client puts "Copy Link" next to "Copy Channel ID", so users paste
``https://discord.com/channels/<guild>/<channel>`` into ``DISCORD_HOME_CHANNEL``.
The value used to be stored verbatim and every home-channel delivery then died
inside the adapter on ``int(chat_id)`` — logged as a generic send failure while
the platform still reported ``connected``. These tests pin the normalization
(the link reduces to the numeric channel id) plus the diagnostic warning for a
value that is neither an id nor a link.
"""

import logging

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.config_env import (
    _apply_env_overrides,
    _env_home_channel,
    _normalize_discord_channel_ref,
)


class TestNormalizeDiscordChannelRef:
    def test_full_channel_link_reduces_to_channel_id(self):
        assert _normalize_discord_channel_ref(
            "https://discord.com/channels/202523857364451329/202523857364451329"
        ) == "202523857364451329"

    def test_message_link_reduces_to_its_channel_id(self):
        assert _normalize_discord_channel_ref(
            "https://discord.com/channels/202523857364451329/202523857364451329/111222333444555666"
        ) == "202523857364451329"

    def test_dm_link_with_at_me_guild_segment_reduces(self):
        assert _normalize_discord_channel_ref(
            "https://discord.com/channels/@me/202523857364451329"
        ) == "202523857364451329"

    def test_subdomain_and_legacy_hosts_reduce(self):
        for ref in (
            "https://canary.discord.com/channels/1/202523857364451329",
            "https://ptb.discord.com/channels/1/202523857364451329",
            "https://discordapp.com/channels/1/202523857364451329",
            "discord.com/channels/1/202523857364451329",
        ):
            assert _normalize_discord_channel_ref(ref) == "202523857364451329", ref

    def test_guild_channel_pair_reduces_to_channel_id(self):
        assert _normalize_discord_channel_ref("202523857364451329/202523857364451329") == "202523857364451329"

    def test_numeric_id_passes_through_unchanged(self):
        assert _normalize_discord_channel_ref("202523857364451329") == "202523857364451329"

    def test_non_numeric_non_link_value_passes_through_with_warning(self, caplog):
        with caplog.at_level(logging.WARNING, logger="gateway.config"):
            assert _normalize_discord_channel_ref("general") == "general"
        assert "DISCORD_HOME_CHANNEL" in caplog.text
        assert "Copy Channel ID" in caplog.text


class TestDiscordHomeChannelEnvStep:
    def _config(self) -> GatewayConfig:
        config = GatewayConfig()
        config.platforms[Platform.DISCORD] = PlatformConfig()
        return config

    def _apply(self, monkeypatch, **env) -> GatewayConfig:
        for key, value in env.items():
            monkeypatch.setenv(key, value)
        config = self._config()
        _env_home_channel(
            config, Platform.DISCORD, "DISCORD_HOME_CHANNEL",
            strip=True, normalize=_normalize_discord_channel_ref,
        )
        return config

    def test_link_value_lands_as_numeric_chat_id(self, monkeypatch):
        config = self._apply(
            monkeypatch,
            DISCORD_HOME_CHANNEL="https://discord.com/channels/202523857364451329/202523857364451329\n",
        )
        home = config.platforms[Platform.DISCORD].home_channel
        assert home is not None
        assert home.chat_id == "202523857364451329"  # never reaches int() as a URL
        int(home.chat_id)

    def test_name_and_thread_id_envs_still_apply(self, monkeypatch):
        config = self._apply(
            monkeypatch,
            DISCORD_HOME_CHANNEL="202523857364451329",
            DISCORD_HOME_CHANNEL_NAME="Ops",
            DISCORD_HOME_CHANNEL_THREAD_ID="111222333444555666",
        )
        home = config.platforms[Platform.DISCORD].home_channel
        assert home.chat_id == "202523857364451329"
        assert home.name == "Ops"
        assert home.thread_id == "111222333444555666"

    def test_blank_env_leaves_home_channel_unset(self, monkeypatch):
        monkeypatch.setenv("DISCORD_HOME_CHANNEL", "  ")
        config = self._config()
        _env_home_channel(
            config, Platform.DISCORD, "DISCORD_HOME_CHANNEL",
            strip=True, normalize=_normalize_discord_channel_ref,
        )
        assert config.platforms[Platform.DISCORD].home_channel is None

    def test_discord_home_step_wired_into_env_overrides(self, monkeypatch):
        """The wired _ENV_STEPS entry normalizes too, not just the bare helper."""
        monkeypatch.setenv(
            "DISCORD_HOME_CHANNEL",
            "https://discord.com/channels/202523857364451329/202523857364451329",
        )
        config = GatewayConfig()
        config.platforms[Platform.DISCORD] = PlatformConfig()
        _apply_env_overrides(config)
        home = config.platforms[Platform.DISCORD].home_channel
        assert home is not None
        assert home.chat_id == "202523857364451329"
