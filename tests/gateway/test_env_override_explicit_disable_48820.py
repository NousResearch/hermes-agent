"""Regression tests for #48820 Bug 2: an explicit ``platforms.<x>.enabled: false``
in config.yaml must survive ``_apply_env_overrides`` when that platform's
credentials are present in the environment.

Before the fix, twelve credential-presence branches (weixin, whatsapp_cloud,
homeassistant, email, sms, dingtalk, feishu, wecom, wecom_callback, bluebubbles,
qqbot, yuanbao) force-set ``enabled = True`` unconditionally, while Telegram /
Discord / Slack routed through ``_enable_from_env`` and honored the
``_enabled_explicit`` marker.  These tests drive the real ``load_gateway_config``
against a temp HERMES_HOME — real YAML I/O, no mocks of the code under test.
"""

import logging

import pytest

from gateway import config as gateway_config
from gateway.config import Platform, load_gateway_config


# platform -> env credentials that trigger its env-enable branch
CRED_ENV = {
    "weixin": {
        "WEIXIN_TOKEN": "wx_9f8e7d6c5b4a3f2e1d0c9b8a7f6e5d4c3b2a1f0e",
        "WEIXIN_ACCOUNT_ID": "acct_12345",
    },
    "whatsapp_cloud": {
        "WHATSAPP_CLOUD_PHONE_NUMBER_ID": "1234567890",
        "WHATSAPP_CLOUD_ACCESS_TOKEN": "EAAB-test-access-token",
    },
    "homeassistant": {"HASS_TOKEN": "hass-long-lived-token"},
    "email": {
        "EMAIL_ADDRESS": "bot@example.com",
        "EMAIL_PASSWORD": "app-password",
        "EMAIL_IMAP_HOST": "imap.example.com",
        "EMAIL_SMTP_HOST": "smtp.example.com",
    },
    "sms": {"TWILIO_ACCOUNT_SID": "ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"},
    "dingtalk": {"DINGTALK_CLIENT_ID": "ding-id", "DINGTALK_CLIENT_SECRET": "ding-secret"},
    "feishu": {"FEISHU_APP_ID": "cli_feishu", "FEISHU_APP_SECRET": "feishu-secret"},
    "wecom": {"WECOM_BOT_ID": "wecom-bot", "WECOM_SECRET": "wecom-secret"},
    "wecom_callback": {
        "WECOM_CALLBACK_CORP_ID": "corp-id",
        "WECOM_CALLBACK_CORP_SECRET": "corp-secret",
    },
    "bluebubbles": {
        "BLUEBUBBLES_SERVER_URL": "http://127.0.0.1:1234",
        "BLUEBUBBLES_PASSWORD": "bb-password",
    },
    "qqbot": {"QQ_APP_ID": "qq-app", "QQ_CLIENT_SECRET": "qq-secret"},
    "yuanbao": {"YUANBAO_APP_ID": "yb-app", "YUANBAO_APP_SECRET": "yb-secret"},
    # control: the pattern that always honored the explicit disable
    "telegram": {"TELEGRAM_BOT_TOKEN": "123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11"},
}

_PLATFORM_ENV_PREFIXES = (
    "TELEGRAM_", "DISCORD_", "SLACK_", "WEIXIN_", "WHATSAPP_", "HASS_", "EMAIL_",
    "TWILIO_", "DINGTALK_", "FEISHU_", "WECOM_", "BLUEBUBBLES_", "QQ_", "QQBOT_",
    "YUANBAO_", "GATEWAY_RELAY", "SIGNAL_", "MATTERMOST_", "MATRIX_",
)


def _isolate(monkeypatch, tmp_path, env):
    import os

    for key in list(os.environ):
        if key.startswith(_PLATFORM_ENV_PREFIXES):
            monkeypatch.delenv(key, raising=False)
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    return hermes_home


@pytest.mark.parametrize("platform", sorted(CRED_ENV))
def test_yaml_explicit_disable_survives_env_credentials(platform, tmp_path, monkeypatch):
    """``platforms.<x>.enabled: false`` + credentials in env -> stays disabled."""
    hermes_home = _isolate(monkeypatch, tmp_path, CRED_ENV[platform])
    (hermes_home / "config.yaml").write_text(
        f"platforms:\n  {platform}:\n    enabled: false\n", encoding="utf-8"
    )

    config = load_gateway_config()

    cfg = config.platforms.get(Platform(platform))
    assert cfg is not None
    assert cfg.enabled is False, (
        f"{platform}: env credentials re-enabled a platform the user explicitly "
        "disabled in config.yaml (#48820 Bug 2)"
    )


@pytest.mark.parametrize("platform", sorted(CRED_ENV))
def test_env_credentials_still_enable_without_yaml_opinion(platform, tmp_path, monkeypatch):
    """No ``enabled`` key in YAML + credentials in env -> env-only setup still works."""
    hermes_home = _isolate(monkeypatch, tmp_path, CRED_ENV[platform])
    (hermes_home / "config.yaml").write_text("platforms: {}\n", encoding="utf-8")

    config = load_gateway_config()

    cfg = config.platforms.get(Platform(platform))
    assert cfg is not None and cfg.enabled is True, (
        f"{platform}: env-only configuration must still enable the platform"
    )


def test_env_credentials_still_populate_extra_when_yaml_disables(tmp_path, monkeypatch):
    """The disable only gates ``enabled``; credentials are still wired through
    (mirrors the Slack/API-server contract so send-only tooling keeps working)."""
    hermes_home = _isolate(monkeypatch, tmp_path, CRED_ENV["weixin"])
    (hermes_home / "config.yaml").write_text(
        "platforms:\n  weixin:\n    enabled: false\n", encoding="utf-8"
    )

    config = load_gateway_config()

    cfg = config.platforms[Platform.WEIXIN]
    assert cfg.enabled is False
    assert cfg.token == CRED_ENV["weixin"]["WEIXIN_TOKEN"]
    assert cfg.extra.get("account_id") == "acct_12345"
    # marker never leaks out of config load
    assert "_enabled_explicit" not in cfg.extra


@pytest.fixture()
def _fresh_warn_dedup(monkeypatch):
    """The explicit-disable notice is one-time per process; start each test clean."""
    monkeypatch.setattr(gateway_config, "_EXPLICIT_DISABLE_WARNED", set())


@pytest.mark.usefixtures("_fresh_warn_dedup")
@pytest.mark.parametrize("platform", sorted(CRED_ENV))
def test_explicit_disable_with_env_credentials_warns_once(platform, tmp_path, monkeypatch, caplog):
    """Users who relied on 'creds in .env = platform on' must be told why it went
    dark: one WARNING naming the platform, the winning config key, and the env
    credential(s) — emitted once per process, not on every config reload."""
    hermes_home = _isolate(monkeypatch, tmp_path, CRED_ENV[platform])
    (hermes_home / "config.yaml").write_text(
        f"platforms:\n  {platform}:\n    enabled: false\n", encoding="utf-8"
    )

    with caplog.at_level(logging.WARNING, logger="gateway.config"):
        load_gateway_config()
        load_gateway_config()  # reload: must not repeat

    hits = [
        r for r in caplog.records
        if r.levelno == logging.WARNING and f"platforms.{platform}.enabled: false" in r.getMessage()
    ]
    assert len(hits) == 1, [r.getMessage() for r in caplog.records]
    msg = hits[0].getMessage()
    assert f"Platform '{platform}'" in msg
    for env_name in CRED_ENV[platform]:
        assert env_name in msg
    assert f"platforms.{platform}.enabled: true" in msg  # the remedy


@pytest.mark.usefixtures("_fresh_warn_dedup")
def test_no_warning_when_yaml_has_no_opinion_or_is_enabled(tmp_path, monkeypatch, caplog):
    hermes_home = _isolate(monkeypatch, tmp_path, {**CRED_ENV["weixin"], **CRED_ENV["homeassistant"]})
    (hermes_home / "config.yaml").write_text(
        "platforms:\n  homeassistant:\n    enabled: true\n", encoding="utf-8"
    )

    with caplog.at_level(logging.WARNING, logger="gateway.config"):
        config = load_gateway_config()

    assert config.platforms[Platform.WEIXIN].enabled is True
    assert config.platforms[Platform.HOMEASSISTANT].enabled is True
    assert not [r for r in caplog.records if "explicitly disabled" in r.getMessage()]


@pytest.mark.usefixtures("_fresh_warn_dedup")
def test_no_warning_when_disabled_and_no_env_credentials(tmp_path, monkeypatch, caplog):
    """The notice is about credentials being IGNORED; a plain disable is silent."""
    hermes_home = _isolate(monkeypatch, tmp_path, {})
    (hermes_home / "config.yaml").write_text(
        "platforms:\n  weixin:\n    enabled: false\n", encoding="utf-8"
    )

    with caplog.at_level(logging.WARNING, logger="gateway.config"):
        config = load_gateway_config()

    assert config.platforms[Platform.WEIXIN].enabled is False
    assert not [r for r in caplog.records if "explicitly disabled" in r.getMessage()]


def test_every_env_enable_branch_is_named_for_the_warning():
    """Each platform routed through ``_enable_from_env`` needs a credential
    entry so the WARNING can name what is being ignored."""
    import inspect, re

    src = inspect.getsource(gateway_config._apply_env_overrides)
    routed = {Platform[name] for name in re.findall(r"_enable_from_env\(Platform\.([A-Z_]+)\)", src)}
    routed.add(Platform.SLACK)  # Slack has its own inline copy of the logic
    missing = {p.value for p in routed} - {p.value for p in gateway_config._ENV_ENABLE_CREDENTIALS}
    assert not missing, f"platforms without a credential entry for the explicit-disable warning: {missing}"
