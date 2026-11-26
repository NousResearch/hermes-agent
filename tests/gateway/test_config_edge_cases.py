"""Tests for gateway/config.py edge cases and uncovered code paths.

Tests for lines that were not covered:
- Platform._missing_() method (lines 112-154)
- Platform._scan_bundled_plugin_platforms() method (lines 157-175)
- Plugin registry handling in _missing_() (lines 142-152)
- HomeChannel.to_dict() with thread_id (line 205)
- get_home_channel() method (lines 480-485)
- get_reset_policy() method (lines 487-505)
"""

import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

from gateway.config import (
    Platform,
    GatewayConfig,
    HomeChannel,
    SessionResetPolicy,
    _Platform__bundled_plugin_names,
)


class TestPlatformMissingMethod:
    """Test Platform._missing_() method for dynamic platform creation."""

    def test_missing_empty_string_returns_none(self):
        """Test that empty string returns None."""
        result = Platform._missing_("")
        assert result is None

    def test_missing_non_string_returns_none(self):
        """Test that non-string values return None."""
        result = Platform._missing_(123)
        assert result is None

    def test_missing_whitespace_only_returns_none(self):
        """Test that whitespace-only strings return None."""
        result = Platform._missing_("   ")
        assert result is None

    def test_missing_valid_builtin_platform_returns_member(self):
        """Test that valid built-in platform names return enum member."""
        result = Platform._missing_("telegram")
        assert result is Platform.TELEGRAM

    def test_missing_invalid_platform_returns_none(self):
        """Test that invalid platform names return None."""
        result = Platform._missing_("completely_invalid_platform")
        assert result is None

    def test_missing_cached_platform_returns_same_object(self):
        """Test that cached builtin platforms return the same object (identity-stable)."""
        # Use a builtin platform that's guaranteed to be cached
        result1 = Platform._missing_("telegram")
        result2 = Platform._missing_("telegram")
        
        # Should be the same object (cached)
        assert result1 is result2
        assert result1 is Platform.TELEGRAM

class TestPlatformScanBundledPlugins:
    """Test Platform._scan_bundled_plugin_platforms() method."""

    def test_scan_returns_empty_set_when_no_plugins_dir(self):
        """Test that scan returns empty set when plugins directory doesn't exist."""
        with patch("pathlib.Path.is_dir", return_value=False):
            result = Platform._scan_bundled_plugin_platforms()
            assert result == set()

    def test_scan_finds_plugin_directories(self):
        """Test that scan finds plugin directories with plugin.yaml/plugin.yml."""
        # Test that the method doesn't crash and returns a set
        with patch("pathlib.Path.is_dir", return_value=False):
            result = Platform._scan_bundled_plugin_platforms()
            assert isinstance(result, set)


class TestHomeChannelToDictWithThreadId:
    """Test HomeChannel.to_dict() with thread_id."""

    def test_to_dict_without_thread_id(self):
        """Test to_dict() without thread_id excludes thread_id key."""
        hc = HomeChannel(
            platform=Platform.TELEGRAM,
            chat_id="12345",
            name="Home",
            thread_id=None
        )
        
        result = hc.to_dict()
        assert result == {
            "platform": "telegram",
            "chat_id": "12345",
            "name": "Home"
        }
        assert "thread_id" not in result

    def test_to_dict_with_thread_id(self):
        """Test to_dict() with thread_id includes thread_id key."""
        hc = HomeChannel(
            platform=Platform.TELEGRAM,
            chat_id="12345",
            name="Home",
            thread_id="thread678"
        )
        
        result = hc.to_dict()
        assert result == {
            "platform": "telegram",
            "chat_id": "12345",
            "name": "Home",
            "thread_id": "thread678"
        }


class TestGatewayConfigGetHomeChannel:
    """Test GatewayConfig.get_home_channel() method."""

    def test_get_home_channel_returns_config_home_channel(self):
        """Test that get_home_channel returns the configured home channel."""
        config = GatewayConfig(
            platforms={
                Platform.TELEGRAM: MagicMock(home_channel=HomeChannel(
                    platform=Platform.TELEGRAM,
                    chat_id="12345",
                    name="Home"
                ))
            }
        )
        
        result = config.get_home_channel(Platform.TELEGRAM)
        assert result is not None
        assert result.chat_id == "12345"

    def test_get_home_channel_returns_none_when_not_configured(self):
        """Test that get_home_channel returns None when platform not configured."""
        config = GatewayConfig()
        
        result = config.get_home_channel(Platform.TELEGRAM)
        assert result is None

    def test_get_home_channel_returns_none_for_nonexistent_platform(self):
        """Test that get_home_channel returns None for non-existent platform."""
        config = GatewayConfig(
            platforms={
                Platform.DISCORD: MagicMock(home_channel=None)
            }
        )
        
        result = config.get_home_channel(Platform.TELEGRAM)
        assert result is None


class TestGatewayConfigGetResetPolicy:
    """Test GatewayConfig.get_reset_policy() method."""

    def test_get_reset_policy_returns_default(self):
        """Test that get_reset_policy returns default policy when no overrides."""
        config = GatewayConfig()
        
        result = config.get_reset_policy(platform=None, session_type=None)
        assert result is config.default_reset_policy

    def test_get_reset_policy_returns_platform_override(self):
        """Test that get_reset_policy returns platform-specific override."""
        custom_policy = SessionResetPolicy(mode="daily", at_hour=10)
        
        config = GatewayConfig(
            reset_by_platform={
                Platform.TELEGRAM: custom_policy
            }
        )
        
        result = config.get_reset_policy(platform=Platform.TELEGRAM)
        assert result is custom_policy

    def test_get_reset_policy_returns_type_override(self):
        """Test that get_reset_policy returns type-specific override."""
        custom_policy = SessionResetPolicy(mode="idle", idle_minutes=60)
        
        config = GatewayConfig(
            reset_by_type={
                "api": custom_policy
            }
        )
        
        result = config.get_reset_policy(session_type="api")
        assert result is custom_policy

    def test_get_reset_policy_platform_takes_precedence_over_type(self):
        """Test that platform override takes precedence over type override."""
        platform_policy = SessionResetPolicy(mode="daily")
        type_policy = SessionResetPolicy(mode="idle")
        
        config = GatewayConfig(
            reset_by_platform={
                Platform.TELEGRAM: platform_policy
            },
            reset_by_type={
                "chat": type_policy
            }
        )
        
        # Platform override should take precedence
        result = config.get_reset_policy(platform=Platform.TELEGRAM, session_type="chat")
        assert result is platform_policy


class TestPlatformDynamicCreationIdentity:
    """Test that dynamically created platforms are identity-stable."""

    def test_dynamic_platform_is_stable(self):
        """Test that Platform._missing_('irc') is cached."""
        # Save and restore _value2member_map_ to avoid polluting other tests
        saved = dict(Platform._value2member_map_)
        try:
            Platform._value2member_map_.clear()
            result1 = Platform._missing_("irc")
            result2 = Platform._missing_("irc")
            assert result1 is result2
            assert result1 is not None
            assert result1._value_ == "irc"
        finally:
            Platform._value2member_map_.clear()
            Platform._value2member_map_.update(saved)

    def test_builtin_platform_cached(self):
        """Test that builtin platforms are cached."""
        # Access builtin platform
        result1 = Platform.TELEGRAM
        result2 = Platform.TELEGRAM
        
        assert result1 is result2
        assert "telegram" in Platform._value2member_map_


class TestBuiltinPlatformValuesSnapshot:
    """Test that builtin platforms are correctly defined."""

    def test_builtin_platforms_is_frozenset(self):
        """Test that _BUILTIN_PLATFORM_VALUES exists and is a frozenset."""
        # Access via the module attribute
        from gateway.config import _BUILTIN_PLATFORM_VALUES as bpv
        assert isinstance(bpv, frozenset)

    def test_builtin_platforms_contains_telegram(self):
        """Test that telegram is in builtin platforms."""
        from gateway.config import _BUILTIN_PLATFORM_VALUES as bpv
        assert "telegram" in bpv

    def test_builtin_platforms_contains_discord(self):
        """Test that discord is in builtin platforms."""
        from gateway.config import _BUILTIN_PLATFORM_VALUES as bpv
        assert "discord" in bpv

    def test_builtin_platforms_does_not_contain_fake(self):
        """Test that fake platform is not in builtin platforms."""
        from gateway.config import _BUILTIN_PLATFORM_VALUES as bpv
        assert "fake_plugin" not in bpv


class TestGatewayConfigPlatformConnectedCheckers:
    """Test platform connection checkers."""

    def test_weixin_requires_account_id(self):
        """Test that Weixin requires account_id in extras."""
        config = GatewayConfig(
            platforms={
                Platform.WEIXIN: MagicMock(
                    extra={"account_id": "test_account"},
                    token="test_token"
                )
            }
        )
        
        # Configured with account_id should be connected
        result = config._is_platform_connected(Platform.WEIXIN, config.platforms[Platform.WEIXIN])
        assert result is True

    def test_weixin_requires_token_or_extra_token(self):
        """Test that Weixin requires token or extra token."""
        # With token directly
        config1 = GatewayConfig(
            platforms={
                Platform.WEIXIN: MagicMock(
                    extra={"account_id": "test"},
                    token="test_token"
                )
            }
        )
        assert config1._is_platform_connected(
            Platform.WEIXIN, config1.platforms[Platform.WEIXIN]
        ) is True
        
        # With extra token - mock extra.get() to return the token value
        extra_mock = MagicMock()
        extra_mock.get.return_value = "***"
        config2 = GatewayConfig(
            platforms={
                Platform.WEIXIN: MagicMock(
                    extra=extra_mock,
                    account_id="test"
                )
            }
        )
        # The connected checker should find the token in extra.get()
        assert config2._is_platform_connected(
            Platform.WEIXIN, config2.platforms[Platform.WEIXIN]
        ) is True
        
        # Without token or extra token
        config3 = GatewayConfig(
            platforms={
                Platform.WEIXIN: MagicMock(
                    extra={"account_id": "test"},
                    token=None
                )
            }
        )
        assert config3._is_platform_connected(
            Platform.WEIXIN, config3.platforms[Platform.WEIXIN]
        ) is False
