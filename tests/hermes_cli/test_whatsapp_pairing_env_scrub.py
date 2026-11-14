"""Pairing launches of bridge.js (dashboard and `hermes whatsapp`) never inherit Hermes' credentials (#38079)."""

from types import SimpleNamespace

import pytest

_OPERATOR_SECRETS = {
    "OPENAI_API_KEY": "sk-operator-openai",
    "TELEGRAM_BOT_TOKEN": "operator-telegram-token",
    "GATEWAY_ALLOWED_USERS": "operator-gateway-allowlist",
    "MODAL_TOKEN_SECRET": "operator-modal-secret",
}


@pytest.fixture
def bridge_dir(tmp_path, monkeypatch):
    import hermes_constants
    from gateway.platforms import whatsapp_common

    for key, value in _OPERATOR_SECRETS.items():
        monkeypatch.setenv(key, value)
    bridge = tmp_path / "bridge"
    bridge.mkdir()
    (bridge / "bridge.js").write_text("", encoding="utf-8")
    monkeypatch.setattr(whatsapp_common, "resolve_whatsapp_bridge_dir", lambda: bridge)
    monkeypatch.setattr(hermes_constants, "find_node_executable", lambda name: "/usr/bin/node")
    return bridge


def test_dashboard_pairing_bridge_env_is_scrubbed(bridge_dir, tmp_path, monkeypatch):
    from hermes_cli.web_routers import messaging

    monkeypatch.setattr(messaging, "_ensure_whatsapp_bridge_dependencies", lambda _dir: None)
    captured = {}
    monkeypatch.setattr(messaging, "spawn_owned", lambda argv, **kw: captured.update(kw) or SimpleNamespace())

    messaging._spawn_whatsapp_pairing_process(tmp_path / "session", "bot")

    env = captured["env"]
    assert sorted(k for k in _OPERATOR_SECRETS if k in env) == []
    assert env["WHATSAPP_MODE"] == "bot" and env["WHATSAPP_DM_POLICY"] == "pairing"


def test_cli_pairing_bridge_env_is_scrubbed(bridge_dir, tmp_path, monkeypatch):
    import subprocess

    from hermes_cli import main, main_platform_setup as setup

    monkeypatch.setattr(main, "_require_tty", lambda _name: None)
    monkeypatch.setattr(main, "get_hermes_home", lambda: tmp_path / "home")
    monkeypatch.setattr(setup, "_whatsapp_choose_mode", lambda *_a: "bot")
    monkeypatch.setattr(setup, "_whatsapp_allowed_users", lambda *_a: None)
    monkeypatch.setattr(setup, "_whatsapp_install_bridge", lambda _dir: True)
    captured = {}

    def fake_run(argv, **kw):
        captured.update(kw)
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(subprocess, "run", fake_run)
    setup.cmd_whatsapp(SimpleNamespace(passkey_browser=None))

    env = captured["env"]
    assert sorted(k for k in _OPERATOR_SECRETS if k in env) == []
    assert env.get("PATH")
