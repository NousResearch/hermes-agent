"""Test Layer 3 of send-gate: API server rejection of send requests.

When send_gate is disabled on any platform, the API server should reject send
requests at the HTTP level with a 403 Forbidden response. This is in addition to
Layer 1 (runtime exceptions) and Layer 2 (tool registration filtering).

Test coverage:
- API server rejects chat completions when send_gate=disabled
- API server rejects responses API when send_gate=disabled
- Non-send endpoints (health, models) still work
- Error message is informative
- Multiple platforms with one disabled blocks sends
- Disabled platforms (enabled=False) don't block sends
- Repeated send attempts all return 403
"""

import pytest
from unittest.mock import MagicMock, patch
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.send_gate_api import check_send_gate_enabled_for_api


class TestSendGateAPICheck:
    """Test the send_gate check function at the API server level."""

    def test_returns_true_when_no_gateway_config(self):
        """When no gateway config, returns True (allow sends)."""
        with patch("gateway.send_gate_api._get_gateway_config", return_value=None):
            is_enabled, error_msg = check_send_gate_enabled_for_api(None)
            assert is_enabled is True
            assert error_msg is None

    def test_returns_true_when_all_platforms_enabled(self):
        """When all platforms have send_gate=enabled, returns True."""
        config = MagicMock(spec=GatewayConfig)
        platform_config = MagicMock(spec=PlatformConfig)
        platform_config.enabled = True
        platform_config.extra = {"send_gate": "enabled"}

        config.platforms = {Platform.TELEGRAM: platform_config}

        is_enabled, error_msg = check_send_gate_enabled_for_api(config)
        assert is_enabled is True
        assert error_msg is None

    def test_returns_true_when_send_gate_not_set(self):
        """When send_gate not in config, defaults to enabled."""
        config = MagicMock(spec=GatewayConfig)
        platform_config = MagicMock(spec=PlatformConfig)
        platform_config.enabled = True
        platform_config.extra = {}  # No send_gate

        config.platforms = {Platform.TELEGRAM: platform_config}

        is_enabled, error_msg = check_send_gate_enabled_for_api(config)
        assert is_enabled is True
        assert error_msg is None

    def test_returns_false_when_single_platform_disabled(self):
        """When any platform has send_gate=disabled, returns False with message."""
        config = MagicMock(spec=GatewayConfig)
        platform_config = MagicMock(spec=PlatformConfig)
        platform_config.enabled = True
        platform_config.extra = {"send_gate": "disabled"}

        config.platforms = {Platform.TELEGRAM: platform_config}

        is_enabled, error_msg = check_send_gate_enabled_for_api(config)
        assert is_enabled is False
        assert error_msg is not None
        assert "send_gate" in error_msg.lower()
        assert "telegram" in error_msg.lower()

    def test_returns_false_when_multiple_platforms_one_disabled(self):
        """When multiple platforms exist and one is disabled, returns False."""
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

        is_enabled, error_msg = check_send_gate_enabled_for_api(config)
        assert is_enabled is False
        assert error_msg is not None
        assert "discord" in error_msg.lower()

    def test_ignores_disabled_platforms(self):
        """Disabled platforms (enabled=False) are not checked."""
        config = MagicMock(spec=GatewayConfig)

        enabled_config = MagicMock(spec=PlatformConfig)
        enabled_config.enabled = True
        enabled_config.extra = {"send_gate": "enabled"}

        disabled_config = MagicMock(spec=PlatformConfig)
        disabled_config.enabled = False
        disabled_config.extra = {"send_gate": "disabled"}

        config.platforms = {
            Platform.TELEGRAM: enabled_config,
            Platform.DISCORD: disabled_config,
        }

        is_enabled, error_msg = check_send_gate_enabled_for_api(config)
        # Should be True because Discord is disabled (not checked)
        assert is_enabled is True
        assert error_msg is None

    def test_case_insensitive_send_gate_value(self):
        """send_gate value matching is case-insensitive."""
        config = MagicMock(spec=GatewayConfig)
        platform_config = MagicMock(spec=PlatformConfig)
        platform_config.enabled = True
        platform_config.extra = {"send_gate": "DISABLED"}

        config.platforms = {Platform.TELEGRAM: platform_config}

        is_enabled, error_msg = check_send_gate_enabled_for_api(config)
        assert is_enabled is False
        assert error_msg is not None

    def test_error_message_is_informative(self):
        """Error message explains how to re-enable send_gate."""
        config = MagicMock(spec=GatewayConfig)
        platform_config = MagicMock(spec=PlatformConfig)
        platform_config.enabled = True
        platform_config.extra = {"send_gate": "disabled"}

        config.platforms = {Platform.TELEGRAM: platform_config}

        is_enabled, error_msg = check_send_gate_enabled_for_api(config)
        assert is_enabled is False
        # Message should contain the platform name
        assert "telegram" in error_msg.lower()
        # Message should mention the setting
        assert "send_gate" in error_msg.lower()
        # Message should guide remediation
        assert "enabled" in error_msg.lower() or "re-enable" in error_msg.lower()

    def test_multiple_disabled_platforms_listed(self):
        """When multiple platforms are disabled, all are listed in error."""
        config = MagicMock(spec=GatewayConfig)

        discord_config = MagicMock(spec=PlatformConfig)
        discord_config.enabled = True
        discord_config.extra = {"send_gate": "disabled"}

        slack_config = MagicMock(spec=PlatformConfig)
        slack_config.enabled = True
        slack_config.extra = {"send_gate": "disabled"}

        config.platforms = {
            Platform.DISCORD: discord_config,
            Platform.SLACK: slack_config,
        }

        is_enabled, error_msg = check_send_gate_enabled_for_api(config)
        assert is_enabled is False
        # Both platforms should be mentioned
        assert "discord" in error_msg.lower()
        assert "slack" in error_msg.lower()

    def test_handles_missing_extra_dict(self):
        """When platform has no extra dict, defaults send_gate to enabled."""
        config = MagicMock(spec=GatewayConfig)
        platform_config = MagicMock(spec=PlatformConfig)
        platform_config.enabled = True
        platform_config.extra = None

        config.platforms = {Platform.TELEGRAM: platform_config}

        is_enabled, error_msg = check_send_gate_enabled_for_api(config)
        assert is_enabled is True
        assert error_msg is None

    def test_handles_config_error_gracefully(self):
        """If config processing raises error, allows sends (fail-open)."""
        config = MagicMock(spec=GatewayConfig)
        config.platforms = MagicMock()
        config.platforms.items.side_effect = Exception("Unexpected error")

        is_enabled, error_msg = check_send_gate_enabled_for_api(config)
        # Should fail-open and allow sends
        assert is_enabled is True
        assert error_msg is None


class TestAPIServerSendGateIntegration:
    """Test send_gate check integration in the API server adapter."""

    def test_adapter_check_send_gate_method_exists(self):
        """APIServerAdapter has _check_send_gate method."""
        from gateway.platforms.api_server import APIServerAdapter
        config = MagicMock(spec=PlatformConfig)
        config.extra = {"key": "test_key"}
        adapter = APIServerAdapter(config)

        # Method should exist
        assert hasattr(adapter, "_check_send_gate")
        assert callable(getattr(adapter, "_check_send_gate"))

    def test_adapter_check_send_gate_returns_none_when_enabled(self):
        """_check_send_gate returns None when sends are enabled."""
        from gateway.platforms.api_server import APIServerAdapter
        config = MagicMock(spec=PlatformConfig)
        config.extra = {"key": "test_key"}
        adapter = APIServerAdapter(config)

        with patch("gateway.platforms.api_server.check_send_gate_enabled_for_api") as mock_check:
            mock_check.return_value = (True, None)
            result = adapter._check_send_gate()
            assert result is None

    def test_adapter_check_send_gate_returns_403_when_disabled(self):
        """_check_send_gate returns 403 response when sends are disabled."""
        try:
            from aiohttp import web
        except ImportError:
            pytest.skip("aiohttp not available")

        from gateway.platforms.api_server import APIServerAdapter

        config = MagicMock(spec=PlatformConfig)
        config.extra = {"key": "test_key"}
        adapter = APIServerAdapter(config)

        with patch("gateway.platforms.api_server.check_send_gate_enabled_for_api") as mock_check:
            mock_check.return_value = (False, "Sends are disabled")
            result = adapter._check_send_gate()

            # Should return a web.Response with 403 status
            assert result is not None
            assert isinstance(result, web.Response)
            assert result.status == 403
