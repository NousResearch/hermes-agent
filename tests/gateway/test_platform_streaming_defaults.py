"""Registered platform defaults gate streaming without overriding explicit settings."""

import pytest

from gateway.config import StreamingConfig
from gateway.display_config import resolve_display_setting
from hermes_cli.config_defaults import DEFAULT_CONFIG
from hermes_cli.plugins import discover_plugins


@pytest.fixture(autouse=True)
def _plugins():
    discover_plugins()


def test_discord_and_telegram_streaming_defaults():
    discord = resolve_display_setting({}, "discord", "streaming")
    telegram = resolve_display_setting({}, "telegram", "streaming")

    assert discord is False
    assert telegram is True
    enabled = StreamingConfig.from_dict({"enabled": True})
    assert enabled.enabled_for(discord) is False
    assert enabled.enabled_for(telegram) is True
    assert StreamingConfig().enabled_for(telegram) is False


def test_explicit_discord_streaming_overrides_default():
    config = {"display": {"platforms": {"discord": {"streaming": True}}}}
    override = resolve_display_setting(config, "discord", "streaming")

    assert override is True
    assert StreamingConfig.from_dict({"enabled": True}).enabled_for(override) is True


def test_slack_native_streaming_follows_the_master_switch():
    slack = resolve_display_setting({}, "slack", "streaming")

    assert StreamingConfig.from_dict({"enabled": True}).enabled_for(slack) is True
    assert StreamingConfig().enabled_for(slack) is False


def test_documented_platform_defaults_match_runtime():
    enabled = StreamingConfig.from_dict({"enabled": True})
    for platform, entry in DEFAULT_CONFIG["display"]["platforms"].items():
        if "streaming" in entry:
            resolved = resolve_display_setting({}, platform, "streaming")
            assert enabled.enabled_for(resolved) is entry["streaming"], platform
