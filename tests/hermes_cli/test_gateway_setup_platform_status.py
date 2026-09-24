"""Configuration status coverage for plugin entries in the gateway setup wizard."""

import pytest

import hermes_cli.gateway as gateway
from gateway.config import GatewayConfig, Platform, PlatformConfig, load_gateway_config
from gateway.platform_registry import PlatformEntry


@pytest.mark.parametrize(
    ("platform", "is_connected", "env", "extra"),
    [
        (Platform.FEISHU, "plugins.platforms.feishu.adapter._is_connected", {"FEISHU_APP_ID": "app-id", "FEISHU_APP_SECRET": "secret"}, {"app_id": "yaml-app"}),
        (Platform.WECOM, "plugins.platforms.wecom.adapter._is_connected", {"WECOM_BOT_ID": "bot-id", "WECOM_SECRET": "secret"}, {"bot_id": "yaml-bot"}),
        (Platform.WECOM_CALLBACK, "plugins.platforms.wecom.adapter._callback_is_connected", {"WECOM_CALLBACK_CORP_ID": "corp-id", "WECOM_CALLBACK_CORP_SECRET": "secret"}, {"corp_id": "yaml-corp"}),
    ],
)
@pytest.mark.parametrize("credential_source", ("environment", "config_extra"))
def test_plugin_platform_status_uses_effective_config_credentials(
    monkeypatch, tmp_path, platform, is_connected, env, extra, credential_source,
):
    """Environment-only and config-extra credentials reach a plugin status checker."""
    module_name, function_name = is_connected.rsplit(".", 1)
    module = __import__(module_name, fromlist=[function_name])
    entry = PlatformEntry(
        name=platform.value,
        label=platform.value,
        adapter_factory=lambda config: None,
        check_fn=lambda: True,
        is_connected=getattr(module, function_name),
    )
    menu_entry = {"key": platform.value, "_registry_entry": entry}

    if credential_source == "environment":
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "env-only"))
        for key, value in env.items():
            monkeypatch.setenv(key, value)
        monkeypatch.setattr(gateway, "load_gateway_config", load_gateway_config)
    else:
        yaml_config = GatewayConfig(platforms={platform: PlatformConfig(enabled=True, extra=extra)})
        monkeypatch.setattr(gateway, "load_gateway_config", lambda: yaml_config)

    assert gateway._platform_status(menu_entry) == "configured"
