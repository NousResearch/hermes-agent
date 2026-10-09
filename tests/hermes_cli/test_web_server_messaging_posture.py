"""The Channels "Test" button must answer the auth-posture question too (#66030).

``POST /api/messaging/platforms/{platform_id}/test`` used to stop at connectivity:
a Telegram bot whose gateway denies every sender (no allowlists, no allow-all flag,
no pairing grants) still answered ``ok: true, "Telegram is connected."`` — a green
checkmark on a bot that ignores everyone. The test result now carries the platform's
access posture for the ACTIVE profile (allowlist / open / pairing / deny_all),
counting pairing grants, evaluated against the profile's own env/config/pairing
store — never the dashboard process's environment.
"""
import json
import os
from pathlib import Path

import pytest


_VALID_BOT_TOKEN = "123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZ_1234"


def _gateway_state(home: Path) -> None:
    """A live gateway record: this pytest process's PID wearing a gateway command
    line, with the Telegram adapter reported ``connected``."""
    import gateway.status as _gw_status

    start_time = _gw_status._get_process_start_time(os.getpid())
    (home / "gateway.pid").write_text(
        json.dumps({"pid": os.getpid(), "hermes_home": str(home)}), encoding="utf-8"
    )
    (home / "gateway_state.json").write_text(json.dumps({
        "pid": os.getpid(), "hermes_home": str(home), "gateway_state": "running",
        "kind": "hermes-gateway", "start_time": start_time,
        "updated_at": "2026-01-01T00:00:00+00:00",
        "platforms": {"telegram": {"state": "connected", "writer_pid": os.getpid()}},
    }), encoding="utf-8")


@pytest.fixture
def client(monkeypatch, _isolate_hermes_home):
    from starlette.testclient import TestClient

    import hermes_state
    import hermes_constants
    import gateway.status as _gw_status
    from hermes_constants import get_hermes_home
    from hermes_cli.web_server import app, _SESSION_HEADER_NAME, _SESSION_TOKEN

    home = get_hermes_home()
    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", home / "state.db")
    monkeypatch.setattr(hermes_constants, "_default_hermes_root_memo", None)
    # Liveness is a verified identity: this pytest process passes as the gateway
    # only while its probed command line matches the home being asked about
    # (``_command_line_belongs_to_profile``). The worker test below re-points the
    # faked command line at ``-p worker`` before its request.
    cmdline = {"current": "hermes gateway run"}
    monkeypatch.setattr(
        _gw_status, "_read_process_cmdline", lambda pid: cmdline["current"]
    )
    # The dashboard process's own env must never authorize anyone here.
    for var in ("TELEGRAM_BOT_TOKEN", "TELEGRAM_ALLOWED_USERS", "GATEWAY_ALLOWED_USERS",
                "GATEWAY_ALLOW_ALL_USERS", "TELEGRAM_GROUP_ALLOWED_USERS",
                "TELEGRAM_GROUP_ALLOWED_CHATS"):
        monkeypatch.delenv(var, raising=False)

    (home / ".env").write_text(f"TELEGRAM_BOT_TOKEN={_VALID_BOT_TOKEN}\n", encoding="utf-8")
    (home / "config.yaml").write_text("platforms:\n  telegram:\n    enabled: true\n", encoding="utf-8")
    _gateway_state(home)
    c = TestClient(app)
    c.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN
    return c


def _test(client, **params):
    resp = client.post("/api/messaging/platforms/telegram/test", params=params)
    assert resp.status_code == 200, resp.text
    return resp.json()


def test_connected_deny_all_reports_posture(client):
    """Connected, but nothing a user can do would reach the agent — the exact
    incident from the report (#66030): green "connected" for three days."""
    payload = _test(client)
    assert payload["ok"] is True
    assert payload["state"] == "connected"
    assert payload["auth_posture"] == "deny_all"
    assert "TELEGRAM_ALLOWED_USERS" in payload["message"]


def test_connected_with_allowlist_reports_allowlist(client, monkeypatch):
    from hermes_constants import get_hermes_home

    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "42, 43")
    payload = _test(client)
    assert payload["ok"] is True
    assert payload["auth_posture"] == "allowlist"
    assert "2 users allowlisted via TELEGRAM_ALLOWED_USERS" in payload["message"]


def test_connected_with_global_allow_all_reports_open(client, monkeypatch):
    monkeypatch.setenv("GATEWAY_ALLOW_ALL_USERS", "true")
    payload = _test(client)
    assert payload["ok"] is True
    assert payload["auth_posture"] == "open"


def test_connected_with_gateway_allowed_users_reports_allowlist(client, monkeypatch):
    monkeypatch.setenv("GATEWAY_ALLOWED_USERS", "42")
    payload = _test(client)
    assert payload["ok"] is True
    assert payload["auth_posture"] == "allowlist"
    assert "GATEWAY_ALLOWED_USERS" in payload["message"]


def test_connected_with_pairing_grant_counts_it(client):
    """An approved pairing request is a first-class grant (authz union): a store
    with one approved user is reachable, not deny-all."""
    from gateway.pairing import PairingStore

    store = PairingStore()
    store._approve_user("telegram", "42", "operator")
    try:
        payload = _test(client)
        assert payload["auth_posture"] == "pairing"
        assert "1 user paired" in payload["message"]
    finally:
        store.revoke("telegram", "42")


def test_connected_deny_all_mentions_pending_pairing_requests(client):
    import time as _time

    from gateway.pairing import PairingStore

    store = PairingStore()
    with store._lock:
        # A pending (unapproved) request is NOT a grant; the posture stays deny_all
        # and the message points the operator at the waiting request (fresh enough
        # to survive the store's TTL cleanup).
        pending = {"r1": {"user_id": "42", "user_name": "operator", "created_at": _time.time(),
                          "hash": "0" * 64, "salt": "00" * 16}}
        store._save_json(store._pending_path("telegram"), pending)
    try:
        payload = _test(client)
        assert payload["auth_posture"] == "deny_all"
        assert "1 pairing request pending approval" in payload["message"]
    finally:
        store.clear_pending("telegram")


def test_profile_scoped_posture_ignores_root_env_and_reports_profile_grants(
    client, monkeypatch, tmp_path
):
    """The dashboard process env belongs to the ROOT install: a named profile's
    Test must neither borrow the root's allowlist nor miss the profile's own.
    Regression for the false deny-all/false green in the #66055 review."""
    import gateway.status as _gw_status
    from hermes_cli import profiles as profiles_mod
    from hermes_constants import get_hermes_home

    default_home = get_hermes_home()
    profiles_root = default_home / "profiles"
    worker_home = profiles_root / "worker"
    worker_home.mkdir(parents=True)
    (worker_home / "config.yaml").write_text("platforms:\n  telegram:\n    enabled: true\n", encoding="utf-8")
    (worker_home / ".env").write_text(f"TELEGRAM_BOT_TOKEN={_VALID_BOT_TOKEN}\n", encoding="utf-8")
    _gateway_state(worker_home)
    monkeypatch.setattr("profiles.paths._get_default_hermes_home", lambda: default_home)
    monkeypatch.setattr("profiles.paths._get_profiles_root", lambda: profiles_root)
    # ROOT-env allowlist must not authorize the worker profile's bot.
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "111,222")
    # The verified gateway identity must belong to the WORKER home: a bare "gateway
    # run" belongs to the default profile, so the worker's liveness probe needs the
    # ``-p worker`` command line (same identity check ``hermes -p worker status`` runs).
    monkeypatch.setattr(
        _gw_status, "_read_process_cmdline", lambda pid: "hermes gateway run -p worker"
    )

    payload = _test(client, profile="worker")
    assert payload["auth_posture"] == "deny_all"

    # The profile's own allowlist wins when present.
    (worker_home / ".env").write_text(
        f"TELEGRAM_BOT_TOKEN={_VALID_BOT_TOKEN}\nTELEGRAM_ALLOWED_USERS=7\n", encoding="utf-8"
    )
    payload = _test(client, profile="worker")
    assert payload["auth_posture"] == "allowlist"
    assert "TELEGRAM_ALLOWED_USERS" in payload["message"]
