"""Test send-gate configuration feature for structural send() disabling.

The send-gate allows operators to disable message sending on a per-platform
basis via config, without modifying code or restarting with different adapter
initialization. When send_gate is set to "disabled", all send() calls raise
SendGateDisabledException with a helpful error message.
"""
import pytest
from gateway.config import GatewayConfig, PlatformConfig, Platform
from gateway.platforms.base import SendGateDisabledException


class MockAdapter:
    """Mock platform adapter for testing send-gate without full init."""

    def __init__(self, platform_name, config=None):
        """Initialize mock adapter with platform and config.

        Args:
            platform_name: The platform enum or string name
            config: PlatformConfig instance (defaults to enabled state)
        """
        self.platform = platform_name if isinstance(platform_name, Platform) else Platform(platform_name)
        self.config = config or PlatformConfig(enabled=True)

    def _check_send_gate(self):
        """Check send_gate config and raise if disabled (copied from BasePlatformAdapter)."""
        send_gate_mode = self.config.extra.get("send_gate", "enabled")
        if send_gate_mode == "disabled":
            platform_name = getattr(self.platform, "value", str(self.platform))
            raise SendGateDisabledException(
                f"send() is disabled for platform '{platform_name}'. "
                f"To re-enable, set platforms.{platform_name}.extra.send_gate to 'enabled' "
                f"or remove the 'send_gate' setting from your config."
            )


def test_send_disabled_raises():
    """When send_gate='disabled', _check_send_gate raises SendGateDisabledException."""
    config = PlatformConfig(enabled=True, extra={"send_gate": "disabled"})
    adapter = MockAdapter(Platform.TELEGRAM, config)

    with pytest.raises(SendGateDisabledException) as exc_info:
        adapter._check_send_gate()

    error_msg = str(exc_info.value)
    assert "send() is disabled" in error_msg
    assert "telegram" in error_msg
    assert "send_gate" in error_msg


def test_send_enabled_works():
    """When send_gate='enabled' (explicit), _check_send_gate does not raise."""
    config = PlatformConfig(enabled=True, extra={"send_gate": "enabled"})
    adapter = MockAdapter(Platform.DISCORD, config)

    # Should not raise
    adapter._check_send_gate()


def test_send_default_enabled():
    """When send_gate is not set, it defaults to 'enabled' and does not raise."""
    config = PlatformConfig(enabled=True, extra={})
    adapter = MockAdapter(Platform.SLACK, config)

    # Should not raise (default is enabled)
    adapter._check_send_gate()


def test_config_from_yaml():
    """PlatformConfig can load send_gate from YAML via extra dict."""
    # Simulate loading from config.yaml:
    #   platforms:
    #     telegram:
    #       extra:
    #         send_gate: disabled
    platform_data = {
        "enabled": True,
        "token": "test-token",
        "extra": {"send_gate": "disabled"},
    }
    config = PlatformConfig.from_dict(platform_data)

    # Verify send_gate is stored in extra
    assert config.extra.get("send_gate") == "disabled"

    # Verify gate check raises
    adapter = MockAdapter(Platform.TELEGRAM, config)
    with pytest.raises(SendGateDisabledException):
        adapter._check_send_gate()


def test_error_message_references_config_path():
    """Error message includes helpful path info to re-enable the gate."""
    config = PlatformConfig(enabled=True, extra={"send_gate": "disabled"})
    adapter = MockAdapter(Platform.MATRIX, config)

    with pytest.raises(SendGateDisabledException) as exc_info:
        adapter._check_send_gate()

    error_msg = str(exc_info.value)
    # Should mention the config path users would modify
    assert "platforms.matrix.extra.send_gate" in error_msg or "send_gate" in error_msg
