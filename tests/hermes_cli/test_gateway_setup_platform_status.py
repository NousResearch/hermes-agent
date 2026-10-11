"""Plugin platform status uses the same credential sources as runtime setup."""

import pytest

import hermes_cli.gateway as gateway
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platform_registry import PlatformEntry


@pytest.mark.parametrize(
    ("platform", "is_connected", "env"),
    [
        (Platform.FEISHU, "plugins.platforms.feishu.adapter._is_connected", {"FEISHU_APP_ID": "id", "FEISHU_APP_SECRET": "secret"}),
        (Platform.WECOM, "plugins.platforms.wecom.adapter._is_connected", {"WECOM_BOT_ID": "id", "WECOM_SECRET": "secret"}),
        (Platform.WECOM_CALLBACK, "plugins.platforms.wecom.adapter._callback_is_connected", {"WECOM_CALLBACK_CORP_ID": "id", "WECOM_CALLBACK_CORP_SECRET": "secret"}),
    ],
)
@pytest.mark.parametrize("credential_source", ("environment", "config_extra"))
def test_plugin_platform_status_uses_runtime_credentials(monkeypatch, platform, is_connected, env, credential_source):
    module_name, function_name = is_connected.rsplit(".", 1)
    module = __import__(module_name, fromlist=[function_name])
    entry = PlatformEntry(name=platform.value, label=platform.value, adapter_factory=lambda config: None,
                          check_fn=lambda: True, is_connected=getattr(module, function_name))
    menu_entry = {"key": platform.value, "_registry_entry": entry}
    if credential_source == "environment":
        for key, value in env.items():
            monkeypatch.setenv(key, value)
    else:
        extra = {"app_id": "id", "app_secret": "secret"} if platform is Platform.FEISHU else (
            {"bot_id": "id", "secret": "secret"} if platform is Platform.WECOM else {"corp_id": "id", "corp_secret": "secret"})
        monkeypatch.setattr(gateway, "load_gateway_config", lambda: GatewayConfig(platforms={platform: PlatformConfig(enabled=True, extra=extra)}))
    assert gateway._platform_status(menu_entry) == "configured"
