"""Desktop (dashboard API) Nous sign-in: the plain device-code login, then the free tier's handoff.

``POST /api/providers/oauth/nous/start`` hands the renderer the device page and code; the poller
takes the token grant, persists the account, moves a replaced free tier's apps, and settles the
default model. Driven through the real FastAPI routes against a fake portal (the same fake
``hermes auth upgrade`` is tested with).
"""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from hermes_cli import anon_auth
from hermes_cli.auth import _load_auth_store
from hermes_cli.web_server import _SESSION_TOKEN, app
from tests.hermes_cli.anon_portal import as_local_desktop_backend, claimed_trial
from tests.hermes_cli.test_anon_upgrade import (
    EMAIL, FREE_PICK, INFERENCE, PORTAL, WELCOME, _model_config, _write_model_config, free_account, portal)

client = TestClient(app)
HEADERS = {"X-Hermes-Session-Token": _SESSION_TOKEN}

__all__ = ["free_account", "portal"]  # fixtures imported from the CLI test module


def _wait_for_terminal(session_id: str, timeout: float = 10.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        body = client.get(f"/api/providers/oauth/nous/poll/{session_id}", headers=HEADERS).json()
        if body["status"] != "pending":
            return body
        time.sleep(0.05)
    pytest.fail("poller never left pending")


def test_start_shows_the_device_code_and_completion_reports_what_moved(portal, free_account):
    anon_auth.ensure_portal_identity(explicit=True)
    _write_model_config({"provider": "nous", "default": anon_auth.GUEST_MODEL, "base_url": WELCOME})

    resp = client.post("/api/providers/oauth/nous/start", headers=HEADERS)
    assert resp.status_code == 200, resp.text
    start = resp.json()
    assert start["user_code"] == portal.user_code
    assert start["verification_url"] == f"{PORTAL}/device?user_code={portal.user_code}"

    body = _wait_for_terminal(start["session_id"])
    assert body["status"] == "approved"
    assert body["account_email"] == EMAIL
    assert body["model"] == FREE_PICK
    assert body["handoff"] == [f"Your apps moved to {EMAIL}: gmail."]
    assert not [p for _, p in portal.calls if "promotion" in p]
    state = _load_auth_store()["providers"]["nous"]
    assert "anon_token" not in state and not anon_auth.is_guest_state(state)
    model_cfg = _model_config()
    assert model_cfg["default"] == FREE_PICK
    assert model_cfg["base_url"] == INFERENCE.rstrip("/")
    assert not anon_auth.route_is_welcome_host(model_cfg["base_url"])


def test_an_expired_code_reports_the_timeout_reason_and_keeps_the_free_tier(portal):
    guest = anon_auth.ensure_portal_identity(explicit=True)
    portal.token_pending, portal.expires_in = True, 1

    start = client.post("/api/providers/oauth/nous/start", headers=HEADERS).json()
    body = _wait_for_terminal(start["session_id"])
    assert body["status"] == "error"
    assert body["reason"] == "timeout"
    assert _load_auth_store()["providers"]["nous"]["anon_token"] == guest["anon_token"]


def test_status_routes_report_the_free_tier(portal, monkeypatch):
    claimed_trial(monkeypatch)
    portal_status = client.get("/api/portal", headers=HEADERS).json()
    assert portal_status["free_tier"] is True
    assert portal_status["account_tier"] == "anonymous"
    providers = client.get("/api/providers/oauth", headers=HEADERS).json()["providers"]
    nous = next(p for p in providers if p["id"] == "nous")
    assert nous["status"]["free_tier"] is True


def test_a_connector_only_guest_is_not_reported_as_the_free_tier(portal, monkeypatch):
    as_local_desktop_backend(monkeypatch)
    anon_auth.ensure_portal_identity(explicit=True)
    assert client.get("/api/portal", headers=HEADERS).json()["free_tier"] is False
    providers = client.get("/api/providers/oauth", headers=HEADERS).json()["providers"]
    nous = next(p for p in providers if p["id"] == "nous")
    assert not nous["status"].get("free_tier")


def test_a_sign_in_cancelled_while_waiting_stops_the_token_poll_and_persists_nothing(portal, free_account):
    """The poller is blocked on the token endpoint when the user cancels: the poll stops, and
    nothing reaches the auth store."""
    import threading
    from hermes_cli import web_server_oauth
    guest = anon_auth.ensure_portal_identity(explicit=True)
    portal.token_pending = True

    threads_before = set(threading.enumerate())
    start = client.post("/api/providers/oauth/nous/start", headers=HEADERS).json()
    assert client.delete(f"/api/providers/oauth/sessions/{start['session_id']}", headers=HEADERS).json()["ok"] is True
    for t in set(threading.enumerate()) - threads_before:
        t.join(timeout=5)
        assert not t.is_alive()
    portal.token_pending = False
    assert portal.token_grants == 0
    state = _load_auth_store()["providers"]["nous"]
    assert state["anon_token"] == guest["anon_token"] and anon_auth.is_guest_state(state)
    assert web_server_oauth._oauth_sessions.get(start["session_id"]) is None
