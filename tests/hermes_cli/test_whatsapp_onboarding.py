import asyncio
import time


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


def test_whatsapp_pairing_watcher_records_qr_and_connected():
    from hermes_cli import web_server as ws

    proc = _FakeProc([
        '{"event":"started","session":"/tmp/session"}\n',
        '{"event":"qr","qr":"qr-payload"}\n',
        '{"event":"connected","user":{"id":"15551234567:1@s.whatsapp.net","name":"Hermes Bot"}}\n',
    ])
    record = ws._WhatsAppOnboardingSession(
        proc=proc,
        mode="bot",
        allowed_users="",
        session_path="/tmp/session",
        expires_at="2099-01-01T00:00:00Z",
        expires_at_ts=time.time() + 600,
    )
    ws._whatsapp_onboarding_sessions.clear()
    ws._whatsapp_onboarding_sessions["pairing"] = record

    ws._watch_whatsapp_pairing("pairing", proc)

    assert record.status == "connected"
    assert record.qr_payload == "qr-payload"
    assert record.account_id == "15551234567:1@s.whatsapp.net"
    assert record.account_name == "Hermes Bot"
    assert record.account_phone == "15551234567"
    assert record.error is None
    ws._whatsapp_onboarding_sessions.clear()


def test_whatsapp_pairing_payload_includes_linked_account():
    from hermes_cli import web_server as ws

    record = ws._WhatsAppOnboardingSession(
        proc=None,
        mode="bot",
        allowed_users="",
        session_path="/tmp/session",
        expires_at="2099-01-01T00:00:00Z",
        expires_at_ts=time.time() + 600,
        status="connected",
        account_id="15551234567@s.whatsapp.net",
        account_name="Hermes Bot",
        account_phone="15551234567",
    )

    payload = ws._whatsapp_onboarding_payload("pairing", record)

    assert payload["account_id"] == "15551234567@s.whatsapp.net"
    assert payload["account_name"] == "Hermes Bot"
    assert payload["account_phone"] == "15551234567"


def test_messaging_payload_includes_safe_whatsapp_setup(monkeypatch):
    from hermes_cli import web_server as ws

    entry = {
        "id": "whatsapp",
        "name": "WhatsApp",
        "description": "WhatsApp bridge",
        "docs_url": "",
        "env_vars": ("WHATSAPP_MODE", "WHATSAPP_ALLOWED_USERS", "WHATSAPP_ENABLED"),
        "required_env": (),
    }
    monkeypatch.setattr(ws, "get_running_pid", lambda: None)
    monkeypatch.setattr(ws, "get_runtime_status_running_pid", lambda runtime: None)
    monkeypatch.setattr(
        ws,
        "load_config",
        lambda: {
            "platforms": {
                "whatsapp": {
                    "enabled": True,
                    "home_channel": {
                        "platform": "whatsapp",
                        "chat_id": "280912570925281@lid",
                        "name": "Home",
                    },
                }
            }
        },
    )

    payload = ws._messaging_platform_payload(
        entry,
        {
            "WHATSAPP_MODE": "self-chat",
            "WHATSAPP_ALLOWED_USERS": "61405484224",
            "WHATSAPP_ENABLED": "true",
        },
        runtime=None,
        scoped=True,
    )

    assert payload["whatsapp_setup"] == {
        "mode": "self-chat",
        "allowed_users_set": True,
        "home_channel_set": True,
    }
    assert "61405484224" not in str(payload["whatsapp_setup"])


def test_apply_whatsapp_onboarding_saves_pairing_policy(monkeypatch):
    from hermes_cli import web_server as ws

    saved = {}
    removed = []
    enabled = []

    monkeypatch.setattr(ws, "save_env_value", lambda key, value: saved.setdefault(key, value))
    monkeypatch.setattr(ws, "remove_env_value", lambda key: removed.append(key))
    monkeypatch.setattr(ws, "_write_platform_enabled", lambda platform, value: enabled.append((platform, value)))
    monkeypatch.setattr(
        ws,
        "_restart_gateway_after_whatsapp_onboarding",
        lambda profile=None: {"restart_started": True, "restart_pid": 12345},
    )

    record = ws._WhatsAppOnboardingSession(
        proc=None,
        mode="bot",
        allowed_users="",
        session_path="/tmp/session",
        expires_at="2099-01-01T00:00:00Z",
        expires_at_ts=time.time() + 600,
        status="connected",
    )
    ws._whatsapp_onboarding_sessions.clear()
    ws._whatsapp_onboarding_sessions["pairing"] = record

    result = asyncio.run(
        ws.apply_whatsapp_onboarding(
            "pairing",
            ws.WhatsAppOnboardingApply(mode="bot", allowed_users=""),
        )
    )

    assert result["ok"] is True
    assert saved["WHATSAPP_MODE"] == "bot"
    assert saved["WHATSAPP_DM_POLICY"] == "pairing"
    assert saved["WHATSAPP_ENABLED"] == "true"
    assert "WHATSAPP_ALLOWED_USERS" not in removed
    assert enabled == [("whatsapp", True)]
    assert "pairing" not in ws._whatsapp_onboarding_sessions


def test_apply_whatsapp_onboarding_self_chat_defaults_to_linked_account(monkeypatch):
    from hermes_cli import web_server as ws

    saved = {}
    removed = []

    monkeypatch.setattr(ws, "save_env_value", lambda key, value: saved.setdefault(key, value))
    monkeypatch.setattr(ws, "remove_env_value", lambda key: removed.append(key))
    monkeypatch.setattr(ws, "_write_platform_enabled", lambda platform, value: None)
    monkeypatch.setattr(
        ws,
        "_restart_gateway_after_whatsapp_onboarding",
        lambda profile=None: {"restart_started": True, "restart_pid": 12345},
    )

    record = ws._WhatsAppOnboardingSession(
        proc=None,
        mode="self-chat",
        allowed_users="",
        session_path="/tmp/session",
        expires_at="2099-01-01T00:00:00Z",
        expires_at_ts=time.time() + 600,
        status="connected",
        account_id="15551234567:1@s.whatsapp.net",
        account_phone="15551234567",
    )
    ws._whatsapp_onboarding_sessions.clear()
    ws._whatsapp_onboarding_sessions["pairing"] = record

    result = asyncio.run(
        ws.apply_whatsapp_onboarding(
            "pairing",
            ws.WhatsAppOnboardingApply(mode="self-chat", allowed_users=""),
        )
    )

    assert result["ok"] is True
    assert saved["WHATSAPP_MODE"] == "self-chat"
    assert saved["WHATSAPP_ALLOWED_USERS"] == "15551234567"
    assert "WHATSAPP_ALLOWED_USERS" not in removed
    assert "pairing" not in ws._whatsapp_onboarding_sessions


def test_start_whatsapp_onboarding_existing_creds_returns_linked_account(monkeypatch, tmp_path):
    from hermes_cli import web_server as ws

    session_dir = tmp_path / "session"
    session_dir.mkdir()
    (session_dir / "creds.json").write_text(
        '{"me":{"id":"15551234567:1@s.whatsapp.net","name":"Hermes Bot"}}',
        encoding="utf-8",
    )

    old_proc = _FakeProc(returncode=1)
    old_record = ws._WhatsAppOnboardingSession(
        proc=old_proc,
        mode="bot",
        allowed_users="",
        session_path=str(session_dir),
        expires_at="2099-01-01T00:00:00Z",
        expires_at_ts=time.time() + 600,
    )
    ws._whatsapp_onboarding_sessions.clear()
    ws._whatsapp_onboarding_sessions["old"] = old_record
    monkeypatch.setattr(ws, "_whatsapp_session_path", lambda: session_dir)
    monkeypatch.setattr(ws.secrets, "token_urlsafe", lambda size: "existing-creds")

    result = asyncio.run(
        ws.start_whatsapp_onboarding(
            ws.WhatsAppOnboardingStart(mode="self-chat", allowed_users="")
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
    assert ws._whatsapp_onboarding_sessions["existing-creds"].account_phone == "15551234567"
    ws._whatsapp_onboarding_sessions.clear()


def test_start_whatsapp_onboarding_returns_before_bridge_spawn(monkeypatch, tmp_path):
    from hermes_cli import web_server as ws

    captured = {}

    class FakeThread:
        def __init__(self, *, target, args, daemon):
            captured["target"] = target
            captured["args"] = args
            captured["daemon"] = daemon

        def start(self):
            captured["started"] = True

    ws._whatsapp_onboarding_sessions.clear()
    monkeypatch.setattr(ws, "_whatsapp_session_path", lambda: tmp_path / "session")
    monkeypatch.setattr(ws.secrets, "token_urlsafe", lambda size: "pairing-start")
    monkeypatch.setattr(ws.threading, "Thread", FakeThread)

    result = asyncio.run(
        ws.start_whatsapp_onboarding(
            ws.WhatsAppOnboardingStart(mode="bot", allowed_users="")
        )
    )

    assert result["pairing_id"] == "pairing-start"
    assert result["status"] == "starting"
    assert result["qr_payload"] is None
    assert captured["target"] is ws._run_whatsapp_pairing
    assert captured["args"] == ("pairing-start", tmp_path / "session", "bot")
    assert captured["daemon"] is True
    assert captured["started"] is True
    assert ws._whatsapp_onboarding_sessions["pairing-start"].proc is None
    ws._whatsapp_onboarding_sessions.clear()


def test_spawn_whatsapp_pairing_process_uses_json_mode(monkeypatch, tmp_path):
    from gateway.platforms import whatsapp_common
    from hermes_cli import web_server as ws
    import hermes_constants

    bridge_dir = tmp_path / "bridge"
    bridge_dir.mkdir()
    (bridge_dir / "bridge.js").write_text("// bridge", encoding="utf-8")
    session_dir = tmp_path / "session"
    captured = {}

    monkeypatch.setattr(whatsapp_common, "resolve_whatsapp_bridge_dir", lambda: bridge_dir)
    monkeypatch.setattr(hermes_constants, "find_node_executable", lambda command: "/usr/bin/node")
    monkeypatch.setattr(hermes_constants, "with_hermes_node_path", lambda env=None: {})
    monkeypatch.setattr(ws, "_ensure_whatsapp_bridge_dependencies", lambda bridge_dir: None)

    def fake_popen(args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return _FakeProc()

    monkeypatch.setattr(ws.subprocess, "Popen", fake_popen)

    proc = ws._spawn_whatsapp_pairing_process(session_dir, "bot")

    assert isinstance(proc, _FakeProc)
    assert "--pair-only" in captured["args"]
    assert "--pair-json" in captured["args"]
    assert str(session_dir) in captured["args"]
    assert captured["kwargs"]["env"]["WHATSAPP_MODE"] == "bot"
    assert captured["kwargs"]["env"]["WHATSAPP_DM_POLICY"] == "pairing"
