import asyncio
import json
import time
from types import SimpleNamespace

import pytest
import hermes_cli.config as _cfg_mod
import hermes_cli.web_models as _web_models
import hermes_cli.web_routers.messaging as _rt_messaging
import hermes_cli.web_server_messaging as _web_server_messaging


class _FakeProc:
    def __init__(self, lines=None, returncode=0):
        self.stdout = iter(lines or [])
        self._returncode = returncode
        self.terminated = False
        self.killed = False
        self.pid = 12345

    def poll(self):
        return None if not self.terminated and not self.killed else self._returncode

    def wait(self, timeout=None):
        return self._returncode

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True








def test_apply_whatsapp_onboarding_saves_pairing_policy(monkeypatch):
    from hermes_cli import web_server as ws

    saved = {}
    removed = []
    enabled = []

    monkeypatch.setattr(_cfg_mod, "save_env_value", lambda key, value: saved.setdefault(key, value))
    monkeypatch.setattr(_cfg_mod, "remove_env_value", lambda key: removed.append(key))
    monkeypatch.setattr(_web_server_messaging, "_write_platform_enabled", lambda platform, value: enabled.append((platform, value)))
    monkeypatch.setattr(
        _web_server_messaging,
        "_activate_gateway_whatsapp_after_onboarding",
        lambda profile=None: {"activation_started": True, "needs_restart": False, "active": True},
    )

    record = _web_server_messaging._WhatsAppOnboardingSession(
        proc=None,
        mode="bot",
        allowed_users="",
        session_path="/tmp/session",
        expires_at="2099-01-01T00:00:00Z",
        expires_at_ts=time.time() + 600,
        status="connected",
    )
    _web_server_messaging._whatsapp_onboarding_sessions.clear()
    _web_server_messaging._whatsapp_onboarding_sessions["pairing"] = record

    result = asyncio.run(
        _rt_messaging.apply_whatsapp_onboarding(
            "pairing",
            _web_models.WhatsAppOnboardingApply(mode="bot", allowed_users=""),
        )
    )

    assert result["ok"] is True
    assert result["active"] is True
    assert result["needs_restart"] is False
    assert saved["WHATSAPP_MODE"] == "bot"
    assert saved["WHATSAPP_DM_POLICY"] == "pairing"
    assert saved["WHATSAPP_ENABLED"] == "true"
    assert "WHATSAPP_ALLOWED_USERS" not in removed
    assert enabled == [("whatsapp", True)]
    assert "pairing" not in _web_server_messaging._whatsapp_onboarding_sessions


def test_start_whatsapp_onboarding_existing_creds_returns_linked_account(monkeypatch, tmp_path):
    from hermes_cli import web_server as ws

    session_dir = tmp_path / "session"
    session_dir.mkdir()
    (session_dir / "creds.json").write_text(
        '{"me":{"id":"15551234567:1@s.whatsapp.net","name":"Hermes Bot"}}',
        encoding="utf-8",
    )

    old_proc = _FakeProc(returncode=1)
    old_record = _web_server_messaging._WhatsAppOnboardingSession(
        proc=old_proc,
        mode="bot",
        allowed_users="",
        session_path=str(session_dir),
        expires_at="2099-01-01T00:00:00Z",
        expires_at_ts=time.time() + 600,
    )
    _web_server_messaging._whatsapp_onboarding_sessions.clear()
    _web_server_messaging._whatsapp_onboarding_sessions["old"] = old_record
    monkeypatch.setattr(_web_server_messaging, "_whatsapp_session_path", lambda: session_dir)
    monkeypatch.setattr(ws.secrets, "token_urlsafe", lambda size: "existing-creds")

    result = asyncio.run(
        _rt_messaging.start_whatsapp_onboarding(
            _web_models.WhatsAppOnboardingStart(mode="self-chat", allowed_users="")
        )
    )

    assert result["pairing_id"] == "existing-creds"
    assert result["status"] == "connected"
    assert result["qr_payload"] is None
    assert result["account_id"] == "15551234567:1@s.whatsapp.net"
    assert result["account_name"] == "Hermes Bot"
    assert result["account_phone"] == "15551234567"
    assert old_record.status == "cancelled"
    assert old_proc.terminated is True
    assert _web_server_messaging._whatsapp_onboarding_sessions["existing-creds"].account_phone == "15551234567"
    _web_server_messaging._whatsapp_onboarding_sessions.clear()


@pytest.mark.parametrize("setup", ["cli", "dashboard"])
def test_pairing_install_is_ready_for_gateway_without_second_npm(tmp_path, monkeypatch, setup):
    from gateway.config import PlatformConfig
    from hermes_cli.main_platform_setup import _whatsapp_install_bridge
    from hermes_cli.web_routers.messaging import _ensure_whatsapp_bridge_dependencies
    from plugins.platforms.whatsapp.adapter import WhatsAppAdapter
    from plugins.platforms.whatsapp.bridge_dependencies import bridge_dependencies_ready

    bridge = tmp_path / "bridge"
    bridge.mkdir()
    (bridge / "package.json").write_text('{"name":"whatsapp-test"}', encoding="utf-8")
    monkeypatch.setattr("hermes_constants.find_node_executable", lambda name: "npm")
    monkeypatch.setattr("hermes_constants.with_hermes_node_path", dict)
    calls = []

    def install(*args, **kwargs):
        calls.append(args)
        (bridge / "node_modules").mkdir(exist_ok=True)
        return SimpleNamespace(returncode=0, stderr="", stdout="")

    monkeypatch.setattr("subprocess.run", install)
    if setup == "cli":
        assert _whatsapp_install_bridge(bridge) is True
    else:
        _ensure_whatsapp_bridge_dependencies(bridge)
    assert len(calls) == 1
    assert bridge_dependencies_ready(bridge)

    adapter = WhatsAppAdapter(PlatformConfig(extra={"bridge_script": str(bridge / "bridge.js")}))
    assert adapter._ensure_bridge_deps(bridge) is True
    assert len(calls) == 1


def test_preexisting_bridge_install_without_stamp_recovers_without_npm(tmp_path):
    from plugins.platforms.whatsapp.bridge_dependencies import bridge_dependencies_ready

    bridge = tmp_path / "bridge"
    package = {"dependencies": {"core": "1.0.0"}}
    lock = {"lockfileVersion": 3, "packages": {
        "": package,
        "node_modules/core": {"version": "1.0.0", "integrity": "sha512-core"},
        "node_modules/platform-optional": {"version": "2.0.0", "optional": True},
    }}
    installed = {"lockfileVersion": 3, "packages": {
        "node_modules/core": {"version": "1.0.0", "integrity": "sha512-core"},
    }}
    (bridge / "node_modules" / "core").mkdir(parents=True)
    for path, data in ((bridge / "package.json", package), (bridge / "package-lock.json", lock),
                       (bridge / "node_modules" / ".package-lock.json", installed)):
        path.write_text(json.dumps(data), encoding="utf-8")
    assert bridge_dependencies_ready(bridge) is False  # interrupted install left an empty package directory
    (bridge / "node_modules" / "core" / "package.json").write_text(
        json.dumps({"version": "1.0.0"}), encoding="utf-8")
    assert bridge_dependencies_ready(bridge) is True
    assert (bridge / "node_modules" / ".hermes-pkg-hash").exists()




