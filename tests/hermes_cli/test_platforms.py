"""Tests for hermes_cli.platforms — platform_label and get_all_platforms."""

from __future__ import annotations

from collections import OrderedDict
from unittest.mock import patch

from hermes_cli.platforms import PLATFORMS, PlatformInfo, get_all_platforms, platform_label


class TestPlatformLabel:
    def test_known_platform_returns_label(self):
        assert platform_label("telegram") == "📱 Telegram"

    def test_known_cli_platform(self):
        assert platform_label("cli") == "🖥️  CLI"

    def test_known_discord(self):
        assert platform_label("discord") == "💬 Discord"

    def test_unknown_platform_returns_default(self):
        assert platform_label("nonexistent") == ""

    def test_unknown_platform_custom_default(self):
        assert platform_label("unknown", default="N/A") == "N/A"

    def test_plugin_registry_fallback(self):
        with patch("gateway.platform_registry.platform_registry") as mock_reg:
            mock_entry = type("Entry", (), {"emoji": "🔌", "label": "CustomPlugin"})()
            mock_reg.get.return_value = mock_entry
            result = platform_label("custom_plugin")
            assert result == "🔌  CustomPlugin"

    def test_plugin_registry_no_emoji(self):
        with patch("gateway.platform_registry.platform_registry") as mock_reg:
            mock_entry = type("Entry", (), {"emoji": None, "label": "NoEmoji"})()
            mock_reg.get.return_value = mock_entry
            result = platform_label("plain")
            assert result == "NoEmoji"

    def test_plugin_registry_not_found_returns_default(self):
        with patch("gateway.platform_registry.platform_registry") as mock_reg:
            mock_reg.get.return_value = None
            result = platform_label("missing", default="???")
            assert result == "???"


class TestGetAllPlatforms:
    def test_returns_builtin_platforms(self):
        result = get_all_platforms()
        assert isinstance(result, OrderedDict)
        assert "telegram" in result
        assert "discord" in result
        assert "cli" in result

    def test_no_plugin_platforms(self):
        with patch("gateway.platform_registry.platform_registry") as mock_reg:
            mock_reg.plugin_entries.return_value = []
            result = get_all_platforms()
            assert len(result) == len(PLATFORMS)

    def test_plugin_platforms_appended(self):
        with patch("gateway.platform_registry.platform_registry") as mock_reg:
            mock_entry = type("Entry", (), {"name": "custom", "emoji": "🔧", "label": "Custom"})()
            mock_reg.plugin_entries.return_value = [mock_entry]
            result = get_all_platforms()
            assert "custom" in result
            assert result["custom"].label == "🔧  Custom"
            assert result["custom"].default_toolset == "hermes-custom"

    def test_plugin_without_emoji(self):
        with patch("gateway.platform_registry.platform_registry") as mock_reg:
            mock_entry = type("Entry", (), {"name": "plain", "emoji": None, "label": "Plain"})()
            mock_reg.plugin_entries.return_value = [mock_entry]
            result = get_all_platforms()
            assert result["plain"].label == "Plain"

    def test_duplicate_plugin_not_appended(self):
        with patch("gateway.platform_registry.platform_registry") as mock_reg:
            mock_entry = type("Entry", (), {"name": "telegram", "emoji": "X", "label": "Override"})()
            mock_reg.plugin_entries.return_value = [mock_entry]
            result = get_all_platforms()
            assert result["telegram"].label == "📱 Telegram"


class TestPlatformsDict:
    def test_all_platforms_have_required_fields(self):
        for key, info in PLATFORMS.items():
            assert isinstance(info, PlatformInfo)
            assert isinstance(info.label, str)
            assert isinstance(info.default_toolset, str)
            assert len(info.label) > 0
            assert info.default_toolset.startswith("hermes-")

    def test_platform_keys_are_strings(self):
        for key in PLATFORMS:
            assert isinstance(key, str)

    def test_minimum_platform_count(self):
        assert len(PLATFORMS) >= 10
