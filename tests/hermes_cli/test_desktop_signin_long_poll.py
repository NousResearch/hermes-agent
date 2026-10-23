"""The desktop's sign-in and handoff reads are long-polls.

``GET /api/providers/oauth/{provider}/poll/{session}?wait=`` holds the request until the sign-in
leaves ``pending``; ``GET /api/providers/oauth/nous/handoff?since=&wait=`` holds it until the free
tier's handoff record changes. Both carry the handoff as fields (``guest_handoff.wire_status``) so
the desktop can name the apps that moved and the ones to reconnect. Driven through the real FastAPI
routes against the fake portal ``hermes auth upgrade`` is tested with.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from hermes_cli import anon_auth, guest_handoff
from hermes_cli.web_server import _SESSION_TOKEN, app
from tests.hermes_cli.test_anon_upgrade import EMAIL, free_account, portal

client = TestClient(app)
HEADERS = {"X-Hermes-Session-Token": _SESSION_TOKEN}

__all__ = ["free_account", "portal"]  # fixtures imported from the CLI test module


def _poll(session_id: str, wait: float) -> dict:
    resp = client.get(f"/api/providers/oauth/nous/poll/{session_id}", params={"wait": wait}, headers=HEADERS)
    assert resp.status_code == 200, resp.text
    return resp.json()


def test_one_long_poll_returns_the_finished_sign_in_with_the_handoff_as_fields(portal, free_account):
    anon_auth.ensure_portal_identity(explicit=True)
    portal.claim_reply["connectors"] = {"kind": "moved", "moved_toolkits": ["gmail"], "dropped_toolkits": ["slack"]}

    start = client.post("/api/providers/oauth/nous/start", headers=HEADERS).json()
    body = _poll(start["session_id"], wait=25)

    assert body["status"] == "approved"
    handoff = body["handoff_status"]
    assert handoff["state"] == "claimed"
    assert handoff["account_email"] == EMAIL
    assert handoff["moved_toolkits"] == ["gmail"] and handoff["dropped_toolkits"] == ["slack"]
    # The fields' copy is the same lines every other sign-in surface prints.
    assert handoff["lines"] == body["handoff"] == guest_handoff.result_lines(handoff)
    assert "anon_token" not in handoff


def test_a_sign_in_that_replaced_no_free_tier_reports_no_handoff(portal, free_account):
    start = client.post("/api/providers/oauth/nous/start", headers=HEADERS).json()
    body = _poll(start["session_id"], wait=25)
    assert body["status"] == "approved"
    assert body["handoff_status"] is None and body["handoff"] == []


def test_a_move_no_attempt_reached_still_reports_the_waiting_record(portal, free_account, monkeypatch):
    """Another process holds the claim, so this sign-in's own run attempts nothing: the poll still
    reports the replaced free tier's record (still moving) instead of no handoff at all."""
    anon_auth.ensure_portal_identity(explicit=True)
    monkeypatch.setattr(guest_handoff, "_run_one", lambda *a, **kw: None)

    start = client.post("/api/providers/oauth/nous/start", headers=HEADERS).json()
    body = _poll(start["session_id"], wait=25)

    assert body["status"] == "approved"
    # A move bound to this account and not yet attempted reads as retrying (handoff_status's rule).
    assert body["handoff_status"]["state"] == "retrying"
    assert body["handoff_status"]["lines"] == guest_handoff.result_lines({"state": "retrying"})
    assert not portal.claims


def test_a_long_poll_answers_pending_once_its_wait_runs_out(portal):
    anon_auth.ensure_portal_identity(explicit=True)
    portal.token_pending = True
    start = client.post("/api/providers/oauth/nous/start", headers=HEADERS).json()
    try:
        assert _poll(start["session_id"], wait=0.3)["status"] == "pending"
    finally:
        client.delete(f"/api/providers/oauth/sessions/{start['session_id']}", headers=HEADERS)


def _handoff(**params) -> dict:
    resp = client.get("/api/providers/oauth/nous/handoff", params=params, headers=HEADERS)
    assert resp.status_code == 200, resp.text
    return resp.json()


def test_the_handoff_route_reads_the_record_the_sign_in_left(portal, free_account):
    anon_auth.ensure_portal_identity(explicit=True)
    start = client.post("/api/providers/oauth/nous/start", headers=HEADERS).json()
    signed_in = _poll(start["session_id"], wait=25)["handoff_status"]

    status = _handoff()
    assert status["state"] == "claimed" and status["moved_toolkits"] == ["gmail"]
    assert status["updated_at"] == signed_in["updated_at"]
    assert status["lines"] == [f"Your apps moved to {EMAIL}: gmail."]


def test_the_handoff_long_poll_answers_when_the_record_changes(monkeypatch):
    retrying = {"state": "retrying", "updated_at": "t1", "moved_toolkits": [], "dropped_toolkits": []}
    claimed = {"state": "claimed", "updated_at": "t2", "account_email": EMAIL, "connectors_kind": "moved",
               "moved_toolkits": ["gmail"], "dropped_toolkits": []}
    reads = iter([retrying, retrying, claimed])
    monkeypatch.setattr(guest_handoff, "handoff_status", lambda: next(reads, claimed))

    status = _handoff(since="t1", wait=25)

    assert status["state"] == "claimed" and status["updated_at"] == "t2"
    assert status["lines"] == [f"Your apps moved to {EMAIL}: gmail."]


def test_the_handoff_long_poll_returns_the_unchanged_record_after_its_wait(monkeypatch):
    retrying = {"state": "retrying", "updated_at": "t1", "moved_toolkits": [], "dropped_toolkits": []}
    monkeypatch.setattr(guest_handoff, "handoff_status", lambda: retrying)

    status = _handoff(since="t1", wait=0.2)

    assert status["state"] == "retrying"
    assert status["lines"] == guest_handoff.result_lines(retrying)


def test_the_handoff_route_needs_the_session_token():
    assert client.get("/api/providers/oauth/nous/handoff").status_code == 401


@pytest.mark.parametrize("since", [None, "anything"])
def test_with_no_record_the_handoff_is_idle(portal, since):
    params = {"wait": 0} if since is None else {"since": since, "wait": 0}
    assert _handoff(**params)["state"] == "idle"
