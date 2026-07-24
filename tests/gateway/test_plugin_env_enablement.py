from types import SimpleNamespace
from unittest.mock import patch

from gateway.config import GatewayConfig, Platform, PlatformConfig, _apply_env_overrides


def _entry(name="photon"):
    return SimpleNamespace(
        name=name,
        env_enablement_fn=lambda: {"project_id": "pid", "project_secret": "secret"},
        is_connected=lambda cfg: True,
        check_fn=lambda: True,
    )


def test_plugin_env_enablement_respects_existing_disabled_config():
    platform = Platform("photon")
    cfg = GatewayConfig(platforms={platform: PlatformConfig(enabled=False, extra={})})

    with patch("hermes_cli.plugins.discover_plugins", lambda: None), patch(
        "gateway.platform_registry.platform_registry.plugin_entries",
        return_value=[_entry()],
    ):
        _apply_env_overrides(cfg)

    assert cfg.platforms[platform].enabled is False


def test_plugin_env_enablement_still_enables_env_only_platform():
    platform = Platform("photon")
    cfg = GatewayConfig(platforms={})

    with patch("hermes_cli.plugins.discover_plugins", lambda: None), patch(
        "gateway.platform_registry.platform_registry.plugin_entries",
        return_value=[_entry()],
    ):
        _apply_env_overrides(cfg)

    assert cfg.platforms[platform].enabled is True
    assert cfg.platforms[platform].extra["project_id"] == "pid"
