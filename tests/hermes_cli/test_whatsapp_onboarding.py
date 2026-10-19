import asyncio
import time
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
        "_restart_gateway_after_whatsapp_onboarding",
        lambda profile=None: {"restart_started": True, "restart_pid": 12345},
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
    monkeypatch.setattr(
        _rt_messaging,
        "fetch_whatsapp_bridge_health",
        lambda port=3000, timeout=2: {"status": "connected", "linked": True},
    )

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


def test_start_whatsapp_onboarding_needs_qr_shows_bridge_qr(monkeypatch, tmp_path):
    from hermes_cli import web_server as ws

    session_dir = tmp_path / "session"
    session_dir.mkdir()
    (session_dir / "creds.json").write_text("{}", encoding="utf-8")
    followed = []

    monkeypatch.setattr(_web_server_messaging, "_whatsapp_session_path", lambda: session_dir)
    monkeypatch.setattr(ws.secrets, "token_urlsafe", lambda size: "needs-qr")
    monkeypatch.setattr(
        _rt_messaging,
        "fetch_whatsapp_bridge_health",
        lambda port=3000, timeout=2: {"status": "needs_qr", "linked": False, "qr": "qr-from-bridge"},
    )
    monkeypatch.setattr(
        _rt_messaging,
        "_start_whatsapp_link_followup",
        lambda decision, pairing_id, session_path, mode: followed.append(decision),
    )

    result = asyncio.run(
        _rt_messaging.start_whatsapp_onboarding(
            _web_models.WhatsAppOnboardingStart(mode="bot", allowed_users="")
        )
    )

    assert result["status"] == "waiting"
    assert result["qr_payload"] == "qr-from-bridge"
    assert result["account_id"] is None
    assert followed == ["watch"]
    assert (session_dir / "creds.json").exists()
    _web_server_messaging._whatsapp_onboarding_sessions.clear()


def test_start_whatsapp_onboarding_logged_out_clears_session(monkeypatch, tmp_path):
    from hermes_cli import web_server as ws

    session_dir = tmp_path / "session"
    session_dir.mkdir()
    (session_dir / "creds.json").write_text('{"me":{"id":"dead"}}', encoding="utf-8")
    followed = []

    monkeypatch.setattr(_web_server_messaging, "_whatsapp_session_path", lambda: session_dir)
    monkeypatch.setattr(ws.secrets, "token_urlsafe", lambda size: "logged-out")
    monkeypatch.setattr(
        _rt_messaging,
        "fetch_whatsapp_bridge_health",
        lambda port=3000, timeout=2: {"status": "logged_out", "linked": False},
    )
    monkeypatch.setattr(
        _rt_messaging,
        "_start_whatsapp_link_followup",
        lambda decision, pairing_id, session_path, mode: followed.append((decision, session_path.exists())),
    )

    result = asyncio.run(
        _rt_messaging.start_whatsapp_onboarding(
            _web_models.WhatsAppOnboardingStart(mode="bot", allowed_users="")
        )
    )

    assert result["status"] == "starting"
    assert result["qr_payload"] is None
    assert followed == [("clear_and_pair", False)]
    assert not session_dir.exists()
    assert list(tmp_path.iterdir()) == []
    _web_server_messaging._whatsapp_onboarding_sessions.clear()


def test_start_whatsapp_onboarding_creds_without_bridge_are_not_linked(monkeypatch, tmp_path):
    from hermes_cli import web_server as ws

    session_dir = tmp_path / "session"
    session_dir.mkdir()
    (session_dir / "creds.json").write_text("{}", encoding="utf-8")
    followed = []

    monkeypatch.setattr(_web_server_messaging, "_whatsapp_session_path", lambda: session_dir)
    monkeypatch.setattr(ws.secrets, "token_urlsafe", lambda size: "stale-creds")
    monkeypatch.setattr(_rt_messaging, "fetch_whatsapp_bridge_health", lambda port=3000, timeout=2: None)
    monkeypatch.setattr(
        _rt_messaging,
        "_start_whatsapp_link_followup",
        lambda decision, pairing_id, session_path, mode: followed.append(decision),
    )

    result = asyncio.run(
        _rt_messaging.start_whatsapp_onboarding(
            _web_models.WhatsAppOnboardingStart(mode="bot", allowed_users="")
        )
    )

    assert result["status"] != "connected"
    assert result["qr_payload"] is None
    assert followed == ["pair"]
    assert (session_dir / "creds.json").exists()
    _web_server_messaging._whatsapp_onboarding_sessions.clear()


def test_pairing_logged_out_event_waits_for_qr():
    record = _web_server_messaging._WhatsAppOnboardingSession(
        proc=None,
        mode="bot",
        allowed_users="",
        session_path="/tmp/session",
        expires_at="2099-01-01T00:00:00Z",
        expires_at_ts=time.time() + 600,
        status="starting",
        account_id="15551234567:1@s.whatsapp.net",
        qr_payload="stale",
    )

    _rt_messaging._apply_pairing_event(record, {"event": "logged_out", "reason": 401})
    assert record.status == "waiting"
    assert record.qr_payload is None
    assert record.account_id is None
    assert record.error is None

    _rt_messaging._apply_pairing_event(record, {"event": "qr", "qr": "fresh-qr"})
    assert record.status == "waiting"
    assert record.qr_payload == "fresh-qr"




