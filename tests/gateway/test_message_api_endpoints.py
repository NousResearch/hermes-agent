"""Hub-authenticated peer-send message API (§1/§3/§4) — the three new routes.

POST /v1/messages                    send + admission (receiver's Bot Chat resolved
                                     server-side; idempotency key derived when absent)
GET  /v1/messages/{delivery_id}      sender-visible status read
POST /v1/messages/{delivery_id}/ack  receiver-written acknowledgement

Driven through the REAL handlers (``_handle_message_send`` / ``_handle_message_status`` /
``_handle_message_ack``) against the REAL on-disk delivery queue and the REAL
idempotency registry under a tmp HERMES_HOME. aiohttp is not installed in this venv,
so the response seam is a minimal ``web`` stand-in (same technique as
test_peer_delivery_acceptance.py) and every assertion is on the returned JSON/status.
"""

from __future__ import annotations

import asyncio
import json
import os
from types import SimpleNamespace

import pytest

from gateway.platforms import api_server as api
from gateway.platforms.api_server_run_idempotency import RunIdempotencyStore
from tools import bot_delivery_queue as q


API_KEY = "test-hub-key"


# ── response seam ──────────────────────────────────────────────────────────────

class _FakeResponse:
    def __init__(self, payload, status=200, headers=None):
        self.status = status
        self.body = json.dumps(payload).encode()
        self.headers = headers or {}


class _FakeWeb:
    @staticmethod
    def json_response(payload, status=200, headers=None):
        return _FakeResponse(payload, status, headers)


class _FakeRequest:
    """A request shaped like the fields the three handlers touch."""

    def __init__(self, *, headers=None, body=None, query=None, match_info=None,
                 content_length=0):
        self.headers = headers or {}
        self._body = body if body is not None else {}
        self.query = query or {}
        self.match_info = match_info or {}
        self.content_length = content_length
        self.method = "POST"
        self.path_qs = "/v1/messages"
        self.transport = None
        self.remote = ""

    async def json(self):
        return self._body


# ── adapter seam (same shape as test_peer_delivery_acceptance.py) ──────────────

@pytest.fixture()
def home(tmp_path, monkeypatch):
    path = tmp_path / "home"
    path.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(path))
    return path


@pytest.fixture()
def adapter(home, monkeypatch):
    monkeypatch.setattr(api, "web", _FakeWeb)
    a = api.APIServerAdapter.__new__(api.APIServerAdapter)
    a._run_idempotency_store = RunIdempotencyStore(":memory:")
    a._run_statuses = {}
    a._run_idempotency_ids = set()
    a._run_owners = {}
    a._run_owner_pid = os.getpid()
    a._run_owner_started = 0
    a._model_name = "test-model"
    a._api_key = API_KEY                 # hub auth expects the gateway's own key
    a._pending_agent_requests = 0
    a._run_idempotency_scope = lambda request: "peer-scope"
    a.calls = []

    async def _history(session_id):
        return []

    async def _run_agent(conversation_history=None, **kwargs):
        a.calls.append(kwargs)
        return {"final_response": "pong"}, {}

    a._conversation_history_for_session = _history
    a._run_agent = _run_agent
    # Keep the probe loop honest but fast.
    monkeypatch.setattr(q, "receipt_after_seconds", lambda: 0.05)
    monkeypatch.setattr(q, "probe_delay", lambda n, **kw: 0.01)
    yield a
    a._run_idempotency_store.close()


def _auth_headers(**extra):
    headers = {"Authorization": f"Bearer {API_KEY}",
               "X-Hermes-Sender-Profile": "sender"}
    headers.update(extra)
    return headers


def _send(adapter, *, body, headers=None):
    req = _FakeRequest(headers=headers or _auth_headers(),
                       body=body, content_length=len(json.dumps(body)))
    return asyncio.run(adapter._handle_message_send(req))


def _status(adapter, delivery_id, *, profile=None):
    req = _FakeRequest(headers=_auth_headers(),
                       query={"profile": profile} if profile else {},
                       match_info={"delivery_id": delivery_id})
    return asyncio.run(adapter._handle_message_status(req))


def _ack(adapter, delivery_id, *, profile=None):
    body = {"profile": profile} if profile else {}
    req = _FakeRequest(headers=_auth_headers(), body=body,
                       match_info={"delivery_id": delivery_id},
                       content_length=len(json.dumps(body)))
    return asyncio.run(adapter._handle_message_ack(req))


def _envelope(resp):
    return json.loads(resp.body), resp.status


# ── route registration ─────────────────────────────────────────────────────────

def test_routes_are_registered(adapter):
    rows = adapter._http_route_table()
    registered = {(m, p) for m, p, _ in rows}
    assert ("POST", "/v1/messages") in registered
    assert ("GET", "/v1/messages/{delivery_id}") in registered
    assert ("POST", "/v1/messages/{delivery_id}/ack") in registered


# ── hub auth ───────────────────────────────────────────────────────────────────

def test_send_rejects_a_bad_bearer_token(adapter):
    req = _FakeRequest(
        headers={"Authorization": "Bearer wrong",
                 "X-Hermes-Sender-Profile": "sender"},
        body={"target_profile": "default", "message": "hi", "session_id": "sess-1"})
    resp = asyncio.run(adapter._handle_message_send(req))
    assert resp.status == 401
    assert adapter.calls == []


def test_status_rejects_a_bad_bearer_token(adapter):
    req = _FakeRequest(headers={"Authorization": "Bearer wrong"},
                       match_info={"delivery_id": "d_0001"})
    resp = asyncio.run(adapter._handle_message_status(req))
    assert resp.status == 401


def test_send_rejects_missing_sender_header(adapter):
    req = _FakeRequest(headers={"Authorization": f"Bearer {API_KEY}"},
                       body={"target_profile": "default", "message": "hi",
                             "session_id": "sess-1"})
    resp = asyncio.run(adapter._handle_message_send(req))
    env, status = _envelope(resp)
    assert status == 400
    assert env["error"]["code"] == "missing_sender"


def test_send_rejects_an_invalid_target_profile(adapter):
    req = _FakeRequest(headers=_auth_headers(),
                       body={"target_profile": "has spaces!", "message": "hi",
                             "session_id": "sess-1"})
    resp = asyncio.run(adapter._handle_message_send(req))
    env, status = _envelope(resp)
    assert status == 400
    assert env["error"]["code"] == "invalid_target_profile"


# ── send -> status -> ack -> status lifecycle ──────────────────────────────────

def test_send_status_ack_lifecycle(adapter, home):
    key = q.validate_idempotency_key("auto:msgapi:lifecycle")

    # (1) SEND: slot is free, so the turn runs in-call and settles DELIVERED.
    resp = _send(adapter, body={"target_profile": "default", "message": "hello",
                                "session_id": "sess-1"},
                 headers=_auth_headers(**{"Idempotency-Key": key}))
    env, status = _envelope(resp)
    assert status == 200
    assert env["object"] == q.OBJECT_NAME
    assert env["status"] == q.STATUS_DELIVERED
    assert env["result"] == q.RESULT_DELIVERED
    assert env["reply"] == "pong"
    delivery_id = env["delivery_id"]
    assert env["idempotency_key"] == key

    # (2) STATUS READ reflects the delivered record.
    env, status = _envelope(_status(adapter, delivery_id))
    assert status == 200
    assert env["delivery_id"] == delivery_id
    assert env["status"] == q.STATUS_DELIVERED

    # (3) receiver-written ACK moves delivered -> acknowledged.
    env, status = _envelope(_ack(adapter, delivery_id))
    assert status == 200
    assert env["status"] == q.STATUS_ACKNOWLEDGED
    assert q.read_record(home, delivery_id)["status"] == q.STATUS_ACKNOWLEDGED

    # (4) STATUS READ now reports acknowledged (a sender can PROVE delivery).
    env, status = _envelope(_status(adapter, delivery_id))
    assert status == 200
    assert env["status"] == q.STATUS_ACKNOWLEDGED


def test_ack_is_idempotent(adapter, home):
    key = q.validate_idempotency_key("auto:msgapi:ack-idem")
    env, _ = _envelope(_send(adapter, body={"target_profile": "default",
                                            "message": "hello", "session_id": "sess-1"},
                             headers=_auth_headers(**{"Idempotency-Key": key})))
    delivery_id = env["delivery_id"]

    first, status1 = _envelope(_ack(adapter, delivery_id))
    second, status2 = _envelope(_ack(adapter, delivery_id))
    assert status1 == status2 == 200
    assert first["status"] == second["status"] == q.STATUS_ACKNOWLEDGED
    assert q.read_record(home, delivery_id)["status"] == q.STATUS_ACKNOWLEDGED


def test_ack_refuses_a_non_delivered_record(adapter, home):
    key = q.validate_idempotency_key("auto:msgapi:ack-not-delivered")
    # Admit into the queue while the turn lock is held -> status stays QUEUED.
    from tools.bot_mode_probe import _hermes_root
    from tools.bot_relay import acquire_turn_lock

    root = _hermes_root(home)
    with acquire_turn_lock(root, "default", timeout_seconds=0):
        env, status = _envelope(_send(
            adapter, body={"target_profile": "default", "message": "hello",
                           "session_id": "sess-1"},
            headers=_auth_headers(**{"Idempotency-Key": key})))
    assert status == 200
    assert env["status"] == q.STATUS_QUEUED
    delivery_id = env["delivery_id"]

    ack_env, ack_status = _envelope(_ack(adapter, delivery_id))
    assert ack_status == 409
    assert ack_env["error"]["code"] == "not_delivered"
    assert q.read_record(home, delivery_id)["status"] == q.STATUS_QUEUED


def test_send_derives_the_key_when_the_header_is_absent(adapter, home):
    # No Idempotency-Key: the server derives it from (sender, target, session,
    # message) and the envelope returns it so the sender can poll with it.
    env, status = _envelope(_send(
        adapter, body={"target_profile": "default", "message": "derive me",
                       "session_id": "sess-1"},
        headers=_auth_headers()))
    assert status == 200
    assert env["status"] == q.STATUS_DELIVERED
    assert env["idempotency_key"]
    # The same message re-sent without a header re-derives the SAME key -> dedup.
    env2, status2 = _envelope(_send(
        adapter, body={"target_profile": "default", "message": "derive me",
                       "session_id": "sess-1"},
        headers=_auth_headers()))
    assert status2 == 200
    assert env2["idempotency_key"] == env["idempotency_key"]
    assert env2["delivery_id"] == env["delivery_id"]


def test_status_read_404s_for_an_unknown_delivery(adapter):
    env, status = _envelope(_status(adapter, "0" * 32))
    assert status == 404
    assert env["error"]["code"] == "delivery_not_found"


def test_status_read_400s_for_a_malformed_delivery_id(adapter):
    env, status = _envelope(_status(adapter, "not-a-delivery-id"))
    assert status == 400
    assert env["error"]["code"] == "invalid_delivery_id"
