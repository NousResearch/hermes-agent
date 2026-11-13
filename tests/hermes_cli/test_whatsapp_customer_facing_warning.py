"""``hermes whatsapp`` warns once when a bot-mode number is open to everyone with no home channel."""

from __future__ import annotations

import io
import os
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace

import pytest

_DOCS_ANCHOR = "whatsapp/#customer-facing-bot"


@pytest.fixture
def isolated_home(tmp_path, monkeypatch):
    hermes = tmp_path / "home" / ".hermes"
    hermes.mkdir(parents=True)
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    monkeypatch.setenv("HERMES_HOME", str(hermes))
    for key in list(os.environ):
        if key.startswith("WHATSAPP_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr("builtins.input", lambda _prompt="": "n")
    monkeypatch.setattr("hermes_cli.main._require_tty", lambda *_a, **_kw: None)
    # Stop right after the access-control step: no bridge script on disk.
    monkeypatch.setattr("gateway.platforms.whatsapp_common.resolve_whatsapp_bridge_dir", lambda: tmp_path / "nobridge")
    return hermes


def _run_wizard() -> str:
    from hermes_cli.main_platform_setup import cmd_whatsapp

    buf = io.StringIO()
    with redirect_stdout(buf):
        cmd_whatsapp(SimpleNamespace())
    return buf.getvalue()


@pytest.mark.parametrize("allow_env", [
    ("WHATSAPP_ALLOWED_USERS", "*"),
    ("WHATSAPP_ALLOWED_USERS", "15551234567, *"),
    ("WHATSAPP_ALLOW_ALL_USERS", "true"),
])
def test_open_bot_without_home_channel_warns_once(isolated_home, monkeypatch, allow_env):
    monkeypatch.setenv("WHATSAPP_MODE", "bot")
    monkeypatch.setenv(*allow_env)
    assert _run_wizard().count(_DOCS_ANCHOR) == 1


@pytest.mark.parametrize("env", [
    {"WHATSAPP_MODE": "bot", "WHATSAPP_ALLOWED_USERS": "*", "WHATSAPP_HOME_CHANNEL": "15551234567@s.whatsapp.net"},
    {"WHATSAPP_MODE": "bot", "WHATSAPP_ALLOWED_USERS": "15551234567"},
    {"WHATSAPP_MODE": "self-chat", "WHATSAPP_ALLOWED_USERS": "15551234567"},
    {"WHATSAPP_MODE": "self-chat", "WHATSAPP_ALLOWED_USERS": "*"},
])
def test_no_warning_when_home_set_or_allowlist_is_explicit(isolated_home, monkeypatch, env):
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    assert _DOCS_ANCHOR not in _run_wizard()


def test_no_warning_when_home_channel_is_set_in_config_yaml(isolated_home, monkeypatch):
    (isolated_home / "config.yaml").write_text(
        "platforms:\n  whatsapp:\n    home_channel:\n      platform: whatsapp\n"
        "      chat_id: \"15551234567@s.whatsapp.net\"\n", encoding="utf-8")
    monkeypatch.setenv("WHATSAPP_MODE", "bot")
    monkeypatch.setenv("WHATSAPP_ALLOWED_USERS", "*")
    assert _DOCS_ANCHOR not in _run_wizard()


@pytest.mark.parametrize("yaml_text, env_allowed", [
    ('whatsapp:\n  allow_from: "*"\n', None),
    ('platforms:\n  whatsapp:\n    extra:\n      allow_from: ["*"]\n', None),
    ('whatsapp:\n  allow_from: "*"\n', "15551234567"),  # config.yaml beats .env, as in the adapter
])
def test_open_bot_from_config_yaml_allowlist_warns(isolated_home, monkeypatch, yaml_text, env_allowed):
    (isolated_home / "config.yaml").write_text(yaml_text, encoding="utf-8")
    monkeypatch.setenv("WHATSAPP_MODE", "bot")
    if env_allowed:
        monkeypatch.setenv("WHATSAPP_ALLOWED_USERS", env_allowed)
    # Enter at the allowlist prompt keeps the config.yaml allowlist.
    monkeypatch.setattr("builtins.input", lambda prompt="": "" if "Phone" in str(prompt) else "n")
    assert _run_wizard().count(_DOCS_ANCHOR) == 1
