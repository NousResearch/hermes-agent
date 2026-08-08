"""Regression coverage for platform env auto-enablement precedence (#73289)."""

from __future__ import annotations

from typing import Any

import pytest

from gateway.config import Platform, load_gateway_config
from gateway.platform_registry import PlatformEntry, platform_registry


_PLATFORM_CASES: tuple[
    tuple[str, dict[str, str], tuple[tuple[str, str, Any], ...]], ...
] = (
    (
        "whatsapp",
        {"WHATSAPP_ENABLED": "true"},
        (),
    ),
    (
        "whatsapp_cloud",
        {
            "WHATSAPP_CLOUD_PHONE_NUMBER_ID": "phone-123",
            "WHATSAPP_CLOUD_ACCESS_TOKEN": "cloud-token",
        },
        (
            ("extra", "phone_number_id", "phone-123"),
            ("extra", "access_token", "cloud-token"),
        ),
    ),
    (
        "homeassistant",
        {"HASS_TOKEN": "hass-token", "HASS_URL": "http://ha.local:8123"},
        (
            ("attr", "token", "hass-token"),
            ("extra", "url", "http://ha.local:8123"),
        ),
    ),
    (
        "email",
        {
            "EMAIL_ADDRESS": "hermes@example.com",
            "EMAIL_PASSWORD": "email-password",
            "EMAIL_IMAP_HOST": "imap.example.com",
            "EMAIL_SMTP_HOST": "smtp.example.com",
        },
        (
            ("extra", "address", "hermes@example.com"),
            ("extra", "imap_host", "imap.example.com"),
            ("extra", "smtp_host", "smtp.example.com"),
        ),
    ),
    (
        "sms",
        {
            "TWILIO_ACCOUNT_SID": "AC123",
            "TWILIO_AUTH_TOKEN": "twilio-token",
        },
        (("attr", "api_key", "twilio-token"),),
    ),
    (
        "api_server",
        {"API_SERVER_ENABLED": "true", "API_SERVER_KEY": "a" * 32},
        (("extra", "key", "a" * 32),),
    ),
    (
        "webhook",
        {"WEBHOOK_ENABLED": "true", "WEBHOOK_SECRET": "webhook-secret"},
        (("extra", "secret", "webhook-secret"),),
    ),
    (
        "msgraph_webhook",
        {
            "MSGRAPH_WEBHOOK_ENABLED": "true",
            "MSGRAPH_WEBHOOK_CLIENT_STATE": "client-state",
        },
        (("extra", "client_state", "client-state"),),
    ),
    (
        "dingtalk",
        {
            "DINGTALK_CLIENT_ID": "ding-id",
            "DINGTALK_CLIENT_SECRET": "ding-secret",
        },
        (
            ("extra", "client_id", "ding-id"),
            ("extra", "client_secret", "ding-secret"),
        ),
    ),
    (
        "feishu",
        {"FEISHU_APP_ID": "feishu-id", "FEISHU_APP_SECRET": "feishu-secret"},
        (
            ("extra", "app_id", "feishu-id"),
            ("extra", "app_secret", "feishu-secret"),
        ),
    ),
    (
        "wecom",
        {"WECOM_BOT_ID": "wecom-id", "WECOM_SECRET": "wecom-secret"},
        (
            ("extra", "bot_id", "wecom-id"),
            ("extra", "secret", "wecom-secret"),
        ),
    ),
    (
        "wecom_callback",
        {
            "WECOM_CALLBACK_CORP_ID": "corp-id",
            "WECOM_CALLBACK_CORP_SECRET": "corp-secret",
        },
        (
            ("extra", "corp_id", "corp-id"),
            ("extra", "corp_secret", "corp-secret"),
        ),
    ),
    (
        "weixin",
        {"WEIXIN_TOKEN": "weixin-token", "WEIXIN_ACCOUNT_ID": "wx-account"},
        (
            ("attr", "token", "weixin-token"),
            ("extra", "account_id", "wx-account"),
        ),
    ),
    (
        "bluebubbles",
        {
            "BLUEBUBBLES_SERVER_URL": "https://bluebubbles.local/",
            "BLUEBUBBLES_PASSWORD": "blue-password",
        },
        (
            ("extra", "server_url", "https://bluebubbles.local"),
            ("extra", "password", "blue-password"),
        ),
    ),
    (
        "qqbot",
        {"QQ_APP_ID": "qq-id", "QQ_CLIENT_SECRET": "qq-secret"},
        (
            ("extra", "app_id", "qq-id"),
            ("extra", "client_secret", "qq-secret"),
        ),
    ),
    (
        "yuanbao",
        {"YUANBAO_APP_ID": "yuanbao-id", "YUANBAO_APP_SECRET": "yuanbao-secret"},
        (
            ("extra", "app_id", "yuanbao-id"),
            ("extra", "app_secret", "yuanbao-secret"),
        ),
    ),
)


def _write_platform_config(tmp_path, body: str) -> None:
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(body, encoding="utf-8")


def _assert_seeded(config, platform: Platform, expected) -> None:
    platform_config = config.platforms[platform]
    for source, key, value in expected:
        actual = (
            getattr(platform_config, key)
            if source == "attr"
            else platform_config.extra.get(key)
        )
        assert actual == value


@pytest.mark.parametrize(
    ("platform_name", "env", "expected"),
    _PLATFORM_CASES,
    ids=[case[0] for case in _PLATFORM_CASES],
)
def test_explicit_platform_disable_wins_over_env_auto_enable(
    tmp_path,
    monkeypatch,
    platform_name: str,
    env: dict[str, str],
    expected: tuple[tuple[str, str, Any], ...],
):
    """Credentials may be seeded, but an explicit disable stays authoritative."""
    _write_platform_config(
        tmp_path,
        f"platforms:\n  {platform_name}:\n    enabled: false\n",
    )
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    for name, value in env.items():
        monkeypatch.setenv(name, value)

    config = load_gateway_config()

    platform = Platform(platform_name)
    platform_config = config.platforms[platform]
    assert platform_config.enabled is False
    assert "_enabled_explicit" not in platform_config.extra
    _assert_seeded(config, platform, expected)


@pytest.mark.parametrize(
    "yaml_body",
    (
        "weixin:\n  enabled: false\n",
        "gateway:\n  platforms:\n    weixin:\n      enabled: false\n",
    ),
    ids=("top-level-platform-block", "nested-gateway-platforms"),
)
def test_explicit_disable_is_preserved_across_supported_yaml_shapes(
    tmp_path, monkeypatch, yaml_body: str
):
    _write_platform_config(tmp_path, yaml_body)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setenv("WEIXIN_TOKEN", "weixin-token")

    config = load_gateway_config()

    assert config.platforms[Platform.WEIXIN].enabled is False
    assert config.platforms[Platform.WEIXIN].token == "weixin-token"


def test_env_only_platform_still_auto_enables(tmp_path, monkeypatch):
    _write_platform_config(tmp_path, "{}\n")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setenv("WEIXIN_TOKEN", "weixin-token")

    config = load_gateway_config()

    assert config.platforms[Platform.WEIXIN].enabled is True
    assert config.platforms[Platform.WEIXIN].token == "weixin-token"


def test_platform_with_unspecified_enabled_still_auto_enables(tmp_path, monkeypatch):
    _write_platform_config(
        tmp_path,
        "platforms:\n  weixin:\n    extra:\n      dm_policy: allowlist\n",
    )
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setenv("WEIXIN_TOKEN", "weixin-token")

    config = load_gateway_config()

    weixin = config.platforms[Platform.WEIXIN]
    assert weixin.enabled is True
    assert weixin.token == "weixin-token"
    assert weixin.extra["dm_policy"] == "allowlist"


def test_plugin_registry_does_not_probe_or_enable_explicitly_disabled_platform(
    tmp_path, monkeypatch
):
    calls = {"check": 0, "connected": 0, "seed": 0}
    name = "precedence-test-plugin"
    entry = PlatformEntry(
        name=name,
        label="Precedence Test Plugin",
        adapter_factory=lambda cfg: object(),
        check_fn=lambda: calls.__setitem__("check", calls["check"] + 1) or True,
        is_connected=lambda cfg: calls.__setitem__("connected", calls["connected"] + 1) or True,
        env_enablement_fn=lambda: (
            calls.__setitem__("seed", calls["seed"] + 1)
            or {"credential": "seeded"}
        ),
        source="plugin",
    )
    platform_registry.register(entry)
    try:
        _write_platform_config(
            tmp_path,
            f"platforms:\n  {name}:\n    enabled: false\n",
        )
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))

        config = load_gateway_config()

        platform = Platform(name)
        assert config.platforms[platform].enabled is False
        assert calls == {"check": 0, "connected": 0, "seed": 0}
        assert "_enabled_explicit" not in config.platforms[platform].extra
    finally:
        platform_registry.unregister(name)


def test_plugin_registry_env_only_auto_enable_is_unchanged(tmp_path, monkeypatch):
    calls = {"check": 0, "connected": 0, "seed": 0}
    name = "precedence-test-plugin-env-only"
    entry = PlatformEntry(
        name=name,
        label="Precedence Test Plugin Env Only",
        adapter_factory=lambda cfg: object(),
        check_fn=lambda: calls.__setitem__("check", calls["check"] + 1) or True,
        is_connected=lambda cfg: calls.__setitem__("connected", calls["connected"] + 1) or bool(
            cfg.extra.get("credential")
        ),
        env_enablement_fn=lambda: (
            calls.__setitem__("seed", calls["seed"] + 1)
            or {"credential": "seeded"}
        ),
        source="plugin",
    )
    platform_registry.register(entry)
    try:
        _write_platform_config(tmp_path, "{}\n")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))

        config = load_gateway_config()

        platform = Platform(name)
        platform_config = config.platforms[platform]
        assert platform_config.enabled is True
        assert platform_config.extra["credential"] == "seeded"
        assert calls == {"check": 1, "connected": 1, "seed": 1}
    finally:
        platform_registry.unregister(name)
