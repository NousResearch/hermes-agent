"""Credential env vars must not wipe platform settings that live in config.yaml.

When a platform's credentials are in ``.env``, ``_apply_env_overrides`` copies
env values into ``PlatformConfig.extra``. A setting whose env var is unset must
keep the config.yaml value instead of being reset to "" or a hardcoded default.
Drives the real ``load_gateway_config`` against a temp HERMES_HOME.
"""

import os

import pytest

from gateway.config import Platform, load_gateway_config

# platform, credential env, extra key, config.yaml value, env var for that key, env value, expected env result
CASES = [
    (
        "feishu", {"FEISHU_APP_ID": "cli_feishu", "FEISHU_APP_SECRET": "feishu-secret"},
        "connection_mode", "webhook", "FEISHU_CONNECTION_MODE", "websocket", "websocket",
    ),
    (
        "wecom_callback", {"WECOM_CALLBACK_CORP_ID": "corp-id", "WECOM_CALLBACK_CORP_SECRET": "corp-secret"},
        "token", "yaml-callback-token", "WECOM_CALLBACK_TOKEN", "env-token", "env-token",
    ),
    (
        "bluebubbles", {"BLUEBUBBLES_SERVER_URL": "http://127.0.0.1:1234", "BLUEBUBBLES_PASSWORD": "bb-pw"},
        "send_read_receipts", False, "BLUEBUBBLES_SEND_READ_RECEIPTS", "true", True,
    ),
    (
        "matrix", {"MATRIX_ACCESS_TOKEN": "syt_token", "MATRIX_HOMESERVER": "https://matrix.example.org"},
        "encryption", True, "MATRIX_ENCRYPTION", "false", False,
    ),
]

_PREFIXES = ("FEISHU_", "WECOM_", "BLUEBUBBLES_", "MATRIX_", "GATEWAY_RELAY")


def _load(monkeypatch, tmp_path, platform, cred_env, key, yaml_value, extra_env):
    for name in list(os.environ):
        if name.startswith(_PREFIXES):
            monkeypatch.delenv(name, raising=False)
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    for name, value in {**cred_env, **extra_env}.items():
        monkeypatch.setenv(name, value)
    yaml_literal = str(yaml_value).lower() if isinstance(yaml_value, bool) else f'"{yaml_value}"'
    (home / "config.yaml").write_text(
        f"platforms:\n  {platform}:\n    extra:\n      {key}: {yaml_literal}\n", encoding="utf-8"
    )
    return load_gateway_config().platforms[Platform(platform)]


@pytest.mark.parametrize("platform,cred_env,key,yaml_value,env_name,env_value,env_result", CASES)
def test_yaml_setting_survives_when_its_env_var_is_unset(
    platform, cred_env, key, yaml_value, env_name, env_value, env_result, tmp_path, monkeypatch
):
    cfg = _load(monkeypatch, tmp_path, platform, cred_env, key, yaml_value, {})

    assert cfg.enabled is True
    assert cfg.extra[key] == yaml_value


@pytest.mark.parametrize("platform,cred_env,key,yaml_value,env_name,env_value,env_result", CASES)
def test_env_value_still_beats_yaml_when_set(
    platform, cred_env, key, yaml_value, env_name, env_value, env_result, tmp_path, monkeypatch
):
    cfg = _load(monkeypatch, tmp_path, platform, cred_env, key, yaml_value, {env_name: env_value})

    assert cfg.extra[key] == env_result
    assert cfg.extra[key] != yaml_value


def test_yaml_port_string_is_converted_when_env_is_unset(tmp_path, monkeypatch):
    cfg = _load(
        monkeypatch, tmp_path, "bluebubbles",
        {"BLUEBUBBLES_SERVER_URL": "http://127.0.0.1:1234", "BLUEBUBBLES_PASSWORD": "bb-pw"},
        "webhook_port", "9999", {},
    )

    assert cfg.extra["webhook_port"] == 9999


def test_yaml_port_garbage_uses_the_converter_default(tmp_path, monkeypatch):
    cfg = _load(
        monkeypatch, tmp_path, "bluebubbles",
        {"BLUEBUBBLES_SERVER_URL": "http://127.0.0.1:1234", "BLUEBUBBLES_PASSWORD": "bb-pw"},
        "webhook_port", "not-a-port", {},
    )

    assert cfg.extra["webhook_port"] == 8645


def test_yaml_quoted_bool_is_converted_when_env_is_unset(tmp_path, monkeypatch):
    cfg = _load(
        monkeypatch, tmp_path, "bluebubbles",
        {"BLUEBUBBLES_SERVER_URL": "http://127.0.0.1:1234", "BLUEBUBBLES_PASSWORD": "bb-pw"},
        "send_read_receipts", "true", {},
    )

    assert cfg.extra["send_read_receipts"] is True


def test_yaml_native_bool_survives_a_string_converter(tmp_path, monkeypatch):
    cfg = _load(
        monkeypatch, tmp_path, "bluebubbles",
        {"BLUEBUBBLES_SERVER_URL": "http://127.0.0.1:1234", "BLUEBUBBLES_PASSWORD": "bb-pw"},
        "require_mention", False, {},
    )

    assert cfg.extra["require_mention"] is False
