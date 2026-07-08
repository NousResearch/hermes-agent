"""Tests for the ``"*"`` platform-wide channel override catch-all.

The wildcard lets a whole platform default to a model/provider (e.g. route
all WhatsApp chats through the Model Router virtual provider) while exact
chat/thread/parent entries still win, and other platforms stay untouched.
"""

from gateway.config import (
    ChannelOverride,
    GatewayConfig,
    Platform,
    PlatformConfig,
)
from gateway.run import _get_channel_override


def _config(overrides, platform=Platform.WHATSAPP):
    return GatewayConfig(
        platforms={
            platform: PlatformConfig(enabled=True, channel_overrides=overrides),
        },
    )


class TestWildcardChannelOverride:
    def test_wildcard_applies_to_any_channel(self):
        config = _config({"*": ChannelOverride(provider="router", model="default")})
        result = _get_channel_override(config, Platform.WHATSAPP, "31633042449")
        assert result is not None
        assert result.provider == "router"
        assert result.model == "default"

    def test_exact_key_beats_wildcard(self):
        config = _config(
            {
                "*": ChannelOverride(provider="router", model="default"),
                "12345": ChannelOverride(provider="openai-codex", model="gpt-5.5"),
            }
        )
        exact = _get_channel_override(config, Platform.WHATSAPP, "12345")
        assert exact.provider == "openai-codex"
        other = _get_channel_override(config, Platform.WHATSAPP, "99999")
        assert other.provider == "router"

    def test_thread_and_parent_keys_beat_wildcard(self):
        config = _config(
            {
                "*": ChannelOverride(model="wildcard-model"),
                "parent_1": ChannelOverride(model="parent-model"),
            }
        )
        result = _get_channel_override(
            config, Platform.WHATSAPP, "chat_x", parent_id="parent_1"
        )
        assert result.model == "parent-model"

    def test_wildcard_scoped_to_its_platform(self):
        config = GatewayConfig(
            platforms={
                Platform.WHATSAPP: PlatformConfig(
                    enabled=True,
                    channel_overrides={"*": ChannelOverride(provider="router", model="default")},
                ),
                Platform.TELEGRAM: PlatformConfig(enabled=True, channel_overrides={}),
            },
        )
        assert _get_channel_override(config, Platform.WHATSAPP, "1").provider == "router"
        assert _get_channel_override(config, Platform.TELEGRAM, "1") is None

    def test_no_wildcard_no_match_returns_none(self):
        config = _config({"12345": ChannelOverride(model="x")})
        assert _get_channel_override(config, Platform.WHATSAPP, "99999") is None

    def test_empty_overrides_unchanged(self):
        config = _config({})
        assert _get_channel_override(config, Platform.WHATSAPP, "1") is None
