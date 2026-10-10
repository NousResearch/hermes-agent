"""Exercise the protected read-only route with real profile/session/config reads."""
import json
from types import SimpleNamespace

import pytest
from starlette.testclient import TestClient

from hermes_cli.web_routers import messaging

PATH = "/api/messaging/whatsapp/identity"
CLOSED = dict(connected=False, account_id=None, account_name=None, account_phone=None)


@pytest.fixture
def identity_http(monkeypatch, _isolate_hermes_home):
    from hermes_constants import get_hermes_home
    from hermes_cli import profiles, web_server

    root = get_hermes_home()
    homes = {"default": root, "worker": root / "profiles" / "worker"}
    for name, home in homes.items():
        session = home / "platforms" / "whatsapp" / "session"
        session.mkdir(parents=True)
        (home / "config.yaml").write_text("platforms:\n  whatsapp:\n    enabled: true\n")
        (home / ".env").write_text("WHATSAPP_ENABLED=true\nWHATSAPP_MODE=self-chat\n")
        (session / "creds.json").write_text(json.dumps({"me": {
            "id": ("15551234567" if name == "default" else "15557654321") + ":1@s.whatsapp.net",
            "name": name,
        }}))
        (home / "gateway_state.json").write_text(json.dumps({
            "platforms": {"whatsapp": {"state": "connected"}},
        }))
    monkeypatch.setattr(profiles, "_get_default_hermes_home", lambda: root)
    monkeypatch.setattr(profiles, "_get_profiles_root", lambda: root / "profiles")
    # Only the external process-liveness boundary is simulated; profile resolution,
    # .env/config, runtime-state, credentials and the payload builder remain real.
    live = {"running": True}
    monkeypatch.setattr(messaging, "get_runtime_status_running_pid", lambda *a, **kw: 12345)
    monkeypatch.setattr(messaging, "resolve_gateway_liveness", lambda **kw: SimpleNamespace(running=live["running"]))
    client = TestClient(web_server.app)
    headers = {web_server._SESSION_HEADER_NAME: web_server._SESSION_TOKEN}
    # Initialize ordinary config scaffolding before measuring probe side effects.
    from hermes_cli.config import load_config
    from hermes_cli.web_server_profiles import _config_profile_scope
    for profile in homes:
        with _config_profile_scope(profile):
            load_config()
    return client, headers, homes, live


def test_identity_http_requires_session_token(identity_http):
    client, headers, homes, live = identity_http
    assert client.get(PATH).status_code == 401
    assert client.get(PATH, headers={next(iter(headers)): "wrong"}).status_code == 401
    assert client.get(PATH, headers=headers).status_code == 200
    assert client.get(PATH, headers={"Authorization": f"Bearer {next(iter(headers.values()))}"}).status_code == 200


def test_identity_http_profile_a_b_a_read_only(identity_http):
    client, headers, homes, live = identity_http
    from hermes_cli import web_server_messaging as onboarding
    sessions_before = dict(onboarding._whatsapp_onboarding_sessions)
    before = {p: p.read_bytes() for home in homes.values() for p in home.rglob("*") if p.is_file()}
    for profile, phone in [("default", "15551234567"), ("worker", "15557654321"), ("default", "15551234567")]:
        response = client.get(PATH, params={"profile": profile}, headers=headers)
        assert response.status_code == 200
        assert response.json() == dict(connected=True, account_id=phone + ":1@s.whatsapp.net", account_name=profile, account_phone=phone)
    assert client.get(PATH, headers=headers).json()["account_name"] == "default"
    assert {p: p.read_bytes() for home in homes.values() for p in home.rglob("*") if p.is_file()} == before
    assert onboarding._whatsapp_onboarding_sessions == sessions_before


def test_identity_http_rejects_unknown_profile(identity_http):
    client, headers, homes, live = identity_http
    response = client.get(PATH, params={"profile": "missing_profile"}, headers=headers)
    assert response.status_code == 404


@pytest.mark.parametrize("case", ["stopped", "disconnected", "bot", "cloud", "lid", "missing", "malformed"])
def test_identity_http_fails_closed(identity_http, case):
    client, headers, homes, live = identity_http
    home = homes["worker"]
    creds = home / "platforms" / "whatsapp" / "session" / "creds.json"
    if case == "stopped":
        live["running"] = False
    elif case == "disconnected":
        (home / "gateway_state.json").write_text(json.dumps({"platforms": {"whatsapp": {"state": "disconnected"}}}))
    elif case in {"bot", "cloud"}:
        (home / ".env").write_text(f"WHATSAPP_ENABLED=true\nWHATSAPP_MODE={case}\n")
    elif case == "lid":
        creds.write_text(json.dumps({"me": {"id": "15551234567890:1@lid", "name": "worker"}}))
    elif case == "missing":
        creds.unlink()
    elif case == "malformed":
        creds.write_text("not json")
    response = client.get(PATH, params={"profile": "worker"}, headers=headers)
    assert response.status_code == 200
    assert response.json() == CLOSED


def test_onboarding_lid_retains_id_but_not_phone(identity_http):
    client, headers, homes, live = identity_http
    jid = "15551234567890:1@lid"
    creds = homes["worker"] / "platforms" / "whatsapp" / "session" / "creds.json"
    creds.write_text(json.dumps({"me": {"id": jid, "name": "worker"}}))
    from hermes_cli import web_server_messaging as onboarding
    previous = dict(onboarding._whatsapp_onboarding_sessions)
    try:
        response = client.post("/api/messaging/whatsapp/onboarding/start", headers=headers, json={"profile": "worker", "mode": "self-chat", "allowed_users": ""})
        assert response.status_code == 200
        assert response.json()["account_id"] == jid
        assert response.json()["account_phone"] is None
        record = onboarding._WhatsAppOnboardingSession(proc=None, mode="self-chat", allowed_users="", session_path=str(creds.parent), expires_at="2099-01-01T00:00:00Z", expires_at_ts=4070908800)
        messaging._apply_pairing_event(record, {"event": "connected", "user": {"id": jid, "name": "worker"}})
        assert record.account_id == jid
        assert record.account_phone is None
    finally:
        onboarding._whatsapp_onboarding_sessions.clear()
        onboarding._whatsapp_onboarding_sessions.update(previous)


@pytest.mark.parametrize("jid", ["1２3456789@s.whatsapp.net", "15551234567:１@s.whatsapp.net"])
def test_phone_jid_requires_ascii_digits(jid):
    assert messaging._whatsapp_phone_from_identifier(jid) is None
