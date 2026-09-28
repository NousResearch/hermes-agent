"""LitKitClient against a fake LitKit: headers, assertion MAC, retries, honest errors, streaming."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging

import pytest

from litco.assertion import verify_user_assertion
from litco.litkit.client import (LitKitClient, LitKitConfig, LitKitError, LitKitPermissionError,
                                 PERMISSION_MESSAGE)
from litco.litkit.context import TurnIdentity, turn_scope
from tests.litco._litkit_fake import HOST_SECRET, MATTER_ID, TOKEN, USER_ID, FakeLitKit, ndjson


@pytest.fixture
def fake():
    server = FakeLitKit().start()
    yield server
    server.stop()


@pytest.fixture
def client(fake):
    sleeps = []
    c = LitKitClient(LitKitConfig(instance_url=fake.url, token=TOKEN, host_secret=HOST_SECRET, matter_id=MATTER_ID),
                     sleep=sleeps.append, max_retries=3)
    c.sleeps = sleeps
    yield c
    c.close()


def _mac(payload: str) -> str:
    raw = hmac.new(HOST_SECRET.encode(), payload.encode(), hashlib.sha256).digest()
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


def test_headers_carry_token_and_fresh_assertion_for_the_turn_user(fake, client):
    fake.route("GET", r"/api/matters/.*/docs", {"total": 3})
    with turn_scope(TurnIdentity(turn_id="t1", matter_id=MATTER_ID, acting_user=USER_ID)):
        body = client.get(f"/api/matters/{MATTER_ID}/docs", params={"countOnly": "1", "skip": None})
    assert body == {"total": 3}
    req = fake.requests[-1]
    assert req.headers["authorization"] == f"Bearer {TOKEN}"
    assert req.headers["x-litkit-acting-user"] == USER_ID
    assertion = req.headers["x-litkit-user-assertion"]
    version, user, matter, iat, ttl, mac = assertion.split(".")
    assert (version, user, matter, ttl) == ("v1", USER_ID, MATTER_ID, "300000")
    # recompute the MAC with the host secret, exactly as LitKit does
    assert mac == _mac(f"v1.{user}.{matter}.{iat}.{ttl}")
    assert verify_user_assertion(assertion, matter_id=MATTER_ID, secret=HOST_SECRET).ok
    assert "origin" not in req.headers
    assert req.query == {"countOnly": ["1"]}


def test_no_assertion_outside_a_turn(fake, client):
    fake.route("GET", r"/api/matters/.*", {"id": MATTER_ID})
    client.get(f"/api/matters/{MATTER_ID}")
    req = fake.requests[-1]
    assert req.headers["authorization"] == f"Bearer {TOKEN}"
    assert "x-litkit-acting-user" not in req.headers and "x-litkit-user-assertion" not in req.headers


def test_turn_without_verified_user_sends_no_assertion(fake, client):
    fake.route("GET", r"/api/x", {})
    with turn_scope(TurnIdentity(turn_id="t2", acting_user=None)):
        client.get("/api/x")
    assert "x-litkit-user-assertion" not in fake.requests[-1].headers


def test_each_retry_mints_a_new_assertion_and_retries_5xx_reads(fake, client):
    attempts = []

    def flaky(req):
        attempts.append(req.headers.get("x-litkit-user-assertion"))
        return (503, {"error": "busy"}) if len(attempts) < 3 else (200, {"ok": True})

    fake.route("GET", r"/api/flaky", flaky)
    with turn_scope(TurnIdentity(acting_user=USER_ID)):
        assert client.get("/api/flaky") == {"ok": True}
    assert len(attempts) == 3 and all(attempts)
    assert len(client.sleeps) == 2


def test_retry_after_is_honored_on_429(fake, client):
    calls = []
    fake.route("POST", r"/api/p", lambda r: (calls.append(1) and None) or
               ((429, {"error": "slow"}, {"Retry-After": "2"}) if len(calls) == 1 else (201, {"id": 1})))
    assert client.post("/api/p", {"a": 1}) == {"id": 1}
    assert client.sleeps == [2.0]


def test_writes_are_not_retried_on_500(fake, client):
    fake.route("POST", r"/api/w", (500, {"error": "boom"}))
    with pytest.raises(LitKitError) as info:
        client.post("/api/w", {})
    assert info.value.status == 500
    assert len(fake.calls("POST", r"/api/w")) == 1


def test_retries_exhaust_to_an_honest_error(fake, client):
    fake.route("GET", r"/api/down", (502, "bad gateway"))
    with pytest.raises(LitKitError) as info:
        client.get("/api/down")
    assert info.value.status == 502 and len(fake.calls("GET", r"/api/down")) == 4


@pytest.mark.parametrize("status,code", [(401, "unauthorized"), (403, "forbidden"),
                                         (403, "agent_token_matter_mismatch")])
def test_401_403_surface_as_permission_errors_and_are_never_retried(fake, client, status, code):
    fake.route("GET", r"/api/secret", (status, {"error": code}))
    with pytest.raises(LitKitPermissionError) as info:
        client.get("/api/secret")
    err = info.value
    assert PERMISSION_MESSAGE in str(err) and err.status == status and err.code == code
    assert err.to_dict()["permission_denied"] is True
    assert len(fake.calls("GET", r"/api/secret")) == 1
    assert client.sleeps == []


def test_token_and_secret_never_leak(fake, client, caplog):
    caplog.set_level(logging.DEBUG)
    fake.route("GET", r"/api/flaky", (503, {"error": "busy"}))
    with turn_scope(TurnIdentity(acting_user=USER_ID)):
        with pytest.raises(LitKitError) as info:
            client.get("/api/flaky")
    text = caplog.text + repr(client) + str(client.config) + json.dumps(info.value.to_dict()) + str(info.value)
    assert TOKEN not in text and HOST_SECRET not in text


def test_matter_id_resolves_from_the_single_visible_matter(fake):
    fake.route("GET", r"/api/matters", {"matters": [{"id": MATTER_ID, "name": "M"}]})
    c = LitKitClient(LitKitConfig(instance_url=fake.url, token=TOKEN, host_secret=HOST_SECRET))
    assert c.matter_id == MATTER_ID
    assert "x-litkit-user-assertion" not in fake.requests[-1].headers
    c.close()


def test_stream_ndjson(fake, client):
    fake.route("POST", r"/api/matters/.*/export/text",
               lambda r: ndjson([{"docId": d, "text": f"t{d}"} for d in r.json()["docIds"]] + [{"error": "x"}]))
    rows = list(client.stream_ndjson("POST", f"/api/matters/{MATTER_ID}/export/text", json_body={"docIds": ["a", "b"]}))
    assert rows == [{"docId": "a", "text": "ta"}, {"docId": "b", "text": "tb"}, {"error": "x"}]


def test_download_and_upload(fake, client, tmp_path):
    fake.route("GET", r"/api/documents/.*/pdf", (200, b"%PDF-1.7 bytes", {"Content-Type": "application/pdf"}))
    info = client.download("/api/documents/d1/pdf", tmp_path / "out" / "x.pdf")
    assert (tmp_path / "out" / "x.pdf").read_bytes() == b"%PDF-1.7 bytes"
    assert info["sha256"] == hashlib.sha256(b"%PDF-1.7 bytes").hexdigest() and info["head"].startswith(b"%PDF")
    assert not list((tmp_path / "out").glob("*.part"))

    src = tmp_path / "memo.docx"
    src.write_bytes(b"PK docx")
    fake.route("POST", r"/api/up", (201, {"ok": True}))
    status, body = client.upload("/api/up", src, fields={"deliverableClass": "memo", "provenance": {"a": 1}})
    assert status == 201 and body == {"ok": True}
    form = fake.requests[-1].form()
    assert form["deliverableClass"] == "memo" and json.loads(form["provenance"]) == {"a": 1}
    assert form["file"] == ("memo.docx", b"PK docx")


def test_unconfigured_client_refuses():
    c = LitKitClient(LitKitConfig(instance_url="", token=""))
    with pytest.raises(LitKitError):
        c.get("/api/matters")
