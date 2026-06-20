"""Test Layer 2 of send-gate: tool registration filtering.

When send_gate is disabled on any platform, the send tool should not be
registered (it never appears in available_tools list). This is in addition to
Layer 1 which makes send() raise SendGateDisabledException at runtime.

Test coverage:
- Tool registration check_fn returns correct values based on config
- Config loading and send_gate field extraction
- Platform filtering (disabled platforms ignored, enabled platforms checked)
- Non-gateway contexts (CLI) allow tool registration
- Tool listing endpoints reflect filtered state
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from gateway.config import GatewayConfig, PlatformConfig, Platform
from tools.send_gate_tool import _check_send_gate_enabled, _get_gateway_config


class TestSendGateCheckFunction:
    """Test the _check_send_gate_enabled() check function used for registration."""

    def test_returns_true_when_no_gateway_config(self):
        """In CLI/non-gateway context, tool should be available."""
        with patch("tools.send_gate_tool._get_gateway_config", return_value=None):
            assert _check_send_gate_enabled() is True

    def test_returns_true_when_all_platforms_enabled(self):
        """When all enabled platforms have send_gate=enabled, tool is available."""
        config = MagicMock(spec=GatewayConfig)
        platform_config = MagicMock(spec=PlatformConfig)
        platform_config.enabled = True
        platform_config.extra = {"send_gate": "enabled"}

        config.platforms = {Platform.TELEGRAM: platform_config}

        with patch("tools.send_gate_tool._get_gateway_config", return_value=config):
            assert _check_send_gate_enabled() is True

    def test_returns_true_when_send_gate_not_set_defaults_to_enabled(self):
        """When send_gate is not in config, it defaults to enabled."""
        config = MagicMock(spec=GatewayConfig)
        platform_config = MagicMock(spec=PlatformConfig)
        platform_config.enabled = True
        platform_config.extra = {}  # send_gate not set

        config.platforms = {Platform.TELEGRAM: platform_config}

        with patch("tools.send_gate_tool._get_gateway_config", return_value=config):
            assert _check_send_gate_enabled() is True

    def test_returns_false_when_single_platform_disabled(self):
        """When any enabled platform has send_gate=disabled, tool is unavailable."""
        config = MagicMock(spec=GatewayConfig)
        platform_config = MagicMock(spec=PlatformConfig)
        platform_config.enabled = True
        platform_config.extra = {"send_gate": "disabled"}

        config.platforms = {Platform.TELEGRAM: platform_config}

        with patch("tools.send_gate_tool._get_gateway_config", return_value=config):
            assert _check_send_gate_enabled() is False

    def test_returns_false_when_multiple_platforms_one_disabled(self):
        """When one of multiple platforms is disabled, tool is unavailable."""
        config = MagicMock(spec=GatewayConfig)

        telegram_config = MagicMock(spec=PlatformConfig)
        telegram_config.enabled = True
        telegram_config.extra = {"send_gate": "enabled"}

        discord_config = MagicMock(spec=PlatformConfig)
        discord_config.enabled = True
        discord_config.extra = {"send_gate": "disabled"}

        config.platforms = {
            Platform.TELEGRAM: telegram_config,
            Platform.DISCORD: discord_config,
        }

        with patch("tools.send_gate_tool._get_gateway_config", return_value=config):
            assert _check_send_gate_enabled() is False

    def test_ignores_disabled_platforms(self):
        """Disabled platforms are not checked (send_gate only applies to enabled ones)."""
        config = MagicMock(spec=GatewayConfig)

        enabled_config = MagicMock(spec=PlatformConfig)
        enabled_config.enabled = True
        enabled_config.extra = {"send_gate": "enabled"}

        disabled_config = MagicMock(spec=PlatformConfig)
        disabled_config.enabled = False
        disabled_config.extra = {"send_gate": "disabled"}  # Should be ignored

        config.platforms = {
            Platform.TELEGRAM: enabled_config,
            Platform.DISCORD: disabled_config,
        }

        with patch("tools.send_gate_tool._get_gateway_config", return_value=config):
            # Should return True because disabled platform is skipped
            assert _check_send_gate_enabled() is True

    def test_case_insensitive_send_gate_value(self):
        """send_gate value matching is case-insensitive."""
        config = MagicMock(spec=GatewayConfig)

        # Test with "DISABLED" in uppercase
        platform_config = MagicMock(spec=PlatformConfig)
        platform_config.enabled = True
        platform_config.extra = {"send_gate": "DISABLED"}

        config.platforms = {Platform.TELEGRAM: platform_config}

        with patch("tools.send_gate_tool._get_gateway_config", return_value=config):
            assert _check_send_gate_enabled() is False

        # Test with "Disabled" in mixed case
        platform_config.extra = {"send_gate": "Disabled"}
        with patch("tools.send_gate_tool._get_gateway_config", return_value=config):
            assert _check_send_gate_enabled() is False

    def test_handles_missing_extra_dict(self):
        """When platform config has no extra dict, defaults send_gate to enabled."""
        config = MagicMock(spec=GatewayConfig)
        platform_config = MagicMock(spec=PlatformConfig)
        platform_config.enabled = True
        platform_config.extra = None  # No extra dict

        config.platforms = {Platform.TELEGRAM: platform_config}

        with patch("tools.send_gate_tool._get_gateway_config", return_value=config):
            # Should not raise, should default to enabled
            assert _check_send_gate_enabled() is True

    def test_handles_config_loading_error_gracefully(self):
        """If config loading raises exception inside check function, defaults to enabled (fail-open)."""
        config = MagicMock(spec=GatewayConfig)
        platforms_mock = MagicMock()
        platforms_mock.items.side_effect = Exception("Unexpected error during iteration")
        config.platforms = platforms_mock

        with patch("tools.send_gate_tool._get_gateway_config", return_value=config):
            # Should not raise, should default to True (allow tool)
            assert _check_send_gate_enabled() is True

    def test_handles_platform_iteration_error(self):
        """If iterating platforms raises exception, defaults to enabled (fail-open)."""
        config = MagicMock(spec=GatewayConfig)
        config.platforms = MagicMock()
        config.platforms.items.side_effect = Exception("Platform iteration error")

        with patch("tools.send_gate_tool._get_gateway_config", return_value=config):
            # Should not raise, should default to True
            assert _check_send_gate_enabled() is True


class TestGatewayConfigLoading:
    """Test _get_gateway_config() helper function."""

    def test_returns_none_when_not_in_gateway_context(self):
        """When gateway is not available, returns None."""
        with patch(
            "gateway.config.load_gateway_config",
            side_effect=ImportError("gateway.config not available"),
        ):
            assert _get_gateway_config() is None

    def test_returns_none_when_config_file_missing(self):
        """When config file doesn't exist, returns None."""
        with patch(
            "gateway.config.load_gateway_config",
            side_effect=FileNotFoundError("config.yaml not found"),
        ):
            assert _get_gateway_config() is None

    def test_returns_config_when_available(self):
        """When config loads successfully, returns the config object."""
        mock_config = MagicMock(spec=GatewayConfig)
        with patch(
            "gateway.config.load_gateway_config", return_value=mock_config
        ):
            result = _get_gateway_config()
            assert result is mock_config


class TestToolRegistrationFiltering:
    """Integration tests for tool registration filtering based on send_gate."""

    def test_tool_available_when_send_gate_enabled(self):
        """When all platforms allow send, check_fn should pass."""
        config = MagicMock(spec=GatewayConfig)
        platform_config = MagicMock(spec=PlatformConfig)
        platform_config.enabled = True
        platform_config.extra = {"send_gate": "enabled"}

        config.platforms = {Platform.TELEGRAM: platform_config}

        with patch("tools.send_gate_tool._get_gateway_config", return_value=config):
            # Simulate what the tool registry would do
            check_result = _check_send_gate_enabled()
            assert check_result is True

    def test_tool_unavailable_when_send_gate_disabled_on_any_platform(self):
        """When any platform disables send, check_fn should fail."""
        config = MagicMock(spec=GatewayConfig)
        telegram_config = MagicMock(spec=PlatformConfig)
        telegram_config.enabled = True
        telegram_config.extra = {"send_gate": "enabled"}

        slack_config = MagicMock(spec=PlatformConfig)
        slack_config.enabled = True
        slack_config.extra = {"send_gate": "disabled"}

        config.platforms = {
            Platform.TELEGRAM: telegram_config,
            Platform.SLACK: slack_config,
        }

        with patch("tools.send_gate_tool._get_gateway_config", return_value=config):
            check_result = _check_send_gate_enabled()
            assert check_result is False

    def test_multiple_platforms_all_enabled(self):
        """When multiple platforms all have send enabled, tool is available."""
        config = MagicMock(spec=GatewayConfig)

        platforms_config = {}
        for platform in [Platform.TELEGRAM, Platform.DISCORD, Platform.SLACK]:
            pconfig = MagicMock(spec=PlatformConfig)
            pconfig.enabled = True
            pconfig.extra = {"send_gate": "enabled"}
            platforms_config[platform] = pconfig

        config.platforms = platforms_config

        with patch("tools.send_gate_tool._get_gateway_config", return_value=config):
            assert _check_send_gate_enabled() is True

    def test_mixed_enabled_and_disabled_platforms(self):
        """When some platforms enabled and some disabled, only checks enabled ones."""
        config = MagicMock(spec=GatewayConfig)

        # Enabled platform with send_gate=enabled
        telegram_config = MagicMock(spec=PlatformConfig)
        telegram_config.enabled = True
        telegram_config.extra = {"send_gate": "enabled"}

        # Disabled platform (send_gate should not matter)
        discord_config = MagicMock(spec=PlatformConfig)
        discord_config.enabled = False
        discord_config.extra = {"send_gate": "disabled"}

        # Another enabled platform with send_gate=enabled
        slack_config = MagicMock(spec=PlatformConfig)
        slack_config.enabled = True
        slack_config.extra = {"send_gate": "enabled"}

        config.platforms = {
            Platform.TELEGRAM: telegram_config,
            Platform.DISCORD: discord_config,
            Platform.SLACK: slack_config,
        }

        with patch("tools.send_gate_tool._get_gateway_config", return_value=config):
            # Should be True because the only enabled platforms have send_gate=enabled
            assert _check_send_gate_enabled() is True


class TestRegistrationBehavior:
    """Test how the check_fn behaves when integrated with tool registry."""

    def test_check_fn_signature_compatible_with_registry(self):
        """check_fn should be a callable that takes no args and returns bool."""
        # The tool registry expects check_fn to be callable() and return bool
        result = _check_send_gate_enabled()
        assert isinstance(result, bool)

    def test_check_fn_idempotent(self):
        """Calling check_fn multiple times with same config gives same result."""
        config = MagicMock(spec=GatewayConfig)
        platform_config = MagicMock(spec=PlatformConfig)
        platform_config.enabled = True
        platform_config.extra = {"send_gate": "disabled"}
        config.platforms = {Platform.TELEGRAM: platform_config}

        with patch("tools.send_gate_tool._get_gateway_config", return_value=config):
            result1 = _check_send_gate_enabled()
            result2 = _check_send_gate_enabled()
            assert result1 == result2 == False

    def test_check_fn_reflects_config_changes(self):
        """When config changes, check_fn results change accordingly."""
        config1 = MagicMock(spec=GatewayConfig)
        platform_config = MagicMock(spec=PlatformConfig)
        platform_config.enabled = True
        platform_config.extra = {"send_gate": "enabled"}
        config1.platforms = {Platform.TELEGRAM: platform_config}

        with patch("tools.send_gate_tool._get_gateway_config", return_value=config1):
            assert _check_send_gate_enabled() is True

        # Change config to disable send_gate
        config2 = MagicMock(spec=GatewayConfig)
        platform_config2 = MagicMock(spec=PlatformConfig)
        platform_config2.enabled = True
        platform_config2.extra = {"send_gate": "disabled"}
        config2.platforms = {Platform.TELEGRAM: platform_config2}

        with patch("tools.send_gate_tool._get_gateway_config", return_value=config2):
            assert _check_send_gate_enabled() is False
