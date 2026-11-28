"""A2A "configured" status versus the env auto-enable gate. Regression for #160508."""

from __future__ import annotations

from types import SimpleNamespace

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.config_env import _plugin_is_configured
from plugins.platforms import a2a


def test_yaml_enabled_platform_is_configured_without_a2a_port(monkeypatch):
    """``platforms.a2a.enabled: true`` with the default port is a complete setup, so status surfaces
    must not report "Needs setup" for a platform the gateway is running."""
    monkeypatch.delenv("A2A_PORT", raising=False)
    platform = Platform("a2a")
    config = GatewayConfig(platforms={platform: PlatformConfig(enabled=True)})

    assert config._is_platform_connected(platform, config.platforms[platform]) is True
    assert platform in config.get_connected_platforms()
    assert config._is_platform_connected(platform, PlatformConfig(enabled=False)) is False


def test_env_enablement_probe_still_needs_the_profiles_own_port(monkeypatch):
    """The auto-enable sweep probes ``is_connected`` with a transient ``enabled=True`` view; that flag
    must not count as configuration, or every profile would start an inbound server (#122126)."""
    monkeypatch.delenv("A2A_PORT", raising=False)

    assert _plugin_is_configured(SimpleNamespace(name="a2a", is_connected=a2a.is_connected), {}, None) is False
    assert a2a.is_connected(PlatformConfig(enabled=True)) is False
