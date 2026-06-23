"""Photon webhook inbound tests: signature verification + payload mapping.

These exercise ``PhotonWebhookServer`` without binding a socket — ``_verify``
is called with a fake request and ``_process`` is driven directly, so the
HMAC scheme, the staleness/replay guards, the webhook→sidecar event mapping,
and the dedup/echo gating are all covered offline.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import time
from typing import Any, Dict, List

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.photon.adapter import PhotonAdapter
from plugins.platforms.photon.webhook import PhotonWebhookServer

_SECRET = "test_signing_secret"


def _make_server(monkeypatch: pytest.MonkeyPatch) -> tuple[PhotonWebhookServer, List[Dict[str, Any]]]:
    monkeypatch.setenv("PHOTON_PROJECT_ID", "test-project-id")
    monkeypatch.setenv("PHOTON_PROJECT_SECRET", "test-project-secret")
    adapter = PhotonAdapter(PlatformConfig(enabled=True, token="", extra={}))

    dispatched: List[Dict[str, Any]] = []

    async def fake_dispatch(event: Dict[str, Any]) -> None:
        dispatched.append(event)

    monkeypatch.setattr(adapter, "_dispatch_inbound", fake_dispatch)
    server = PhotonWebhookServer(
        adapter, host="127.0.0.1", port=0, path="/photon/webhook", secret=_SECRET
    )
    return server, dispatched


class _FakeReq:
    def __init__(self, headers: Dict[str, str]) -> None:
        self.headers = headers


def _sign(ts: str, raw: bytes) -> str:
    base = b"v0:" + ts.encode() + b":" + raw
    return "v0=" + hmac.new(_SECRET.encode(), base, hashlib.sha256).hexdigest()


def _text_body(msg_id: str = "spc-msg-1", direction: str = "inbound") -> bytes:
    return json.dumps(
        {
            "event": "messages",
            "message": {
                "id": msg_id,
                "direction": direction,
                "timestamp": "2026-06-23T10:00:00.000Z",
                "sender": {"id": "+15550100"},
                "space": {"id": "any;-;+15550100", "type": "dm", "phone": "+15551234567"},
                "content": {"type": "text", "text": "hello via webhook"},
            },
        }
    ).encode()


def test_verify_accepts_valid_signature(monkeypatch: pytest.MonkeyPatch) -> None:
    server, _ = _make_server(monkeypatch)
    raw = _text_body()
    ts = str(int(time.time()))
    req = _FakeReq({"X-Spectrum-Timestamp": ts, "X-Spectrum-Signature": _sign(ts, raw)})
    assert server._verify(req, raw) is True


def test_verify_rejects_tampered_body(monkeypatch: pytest.MonkeyPatch) -> None:
    server, _ = _make_server(monkeypatch)
    raw = _text_body()
    ts = str(int(time.time()))
    req = _FakeReq({"X-Spectrum-Timestamp": ts, "X-Spectrum-Signature": _sign(ts, raw)})
    assert server._verify(req, raw + b"x") is False


def test_verify_rejects_stale_timestamp(monkeypatch: pytest.MonkeyPatch) -> None:
    server, _ = _make_server(monkeypatch)
    raw = _text_body()
    old = str(int(time.time()) - 9999)
    req = _FakeReq({"X-Spectrum-Timestamp": old, "X-Spectrum-Signature": _sign(old, raw)})
    assert server._verify(req, raw) is False


def test_verify_rejects_missing_headers(monkeypatch: pytest.MonkeyPatch) -> None:
    server, _ = _make_server(monkeypatch)
    assert server._verify(_FakeReq({}), _text_body()) is False


def test_verify_rejects_when_secret_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    server, _ = _make_server(monkeypatch)
    server._secret = ""
    raw = _text_body()
    ts = str(int(time.time()))
    req = _FakeReq({"X-Spectrum-Timestamp": ts, "X-Spectrum-Signature": _sign(ts, raw)})
    assert server._verify(req, raw) is False


@pytest.mark.asyncio
async def test_process_maps_to_sidecar_event_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    server, dispatched = _make_server(monkeypatch)
    await server._process(json.loads(_text_body()))
    assert len(dispatched) == 1
    ev = dispatched[0]
    assert ev["messageId"] == "spc-msg-1"
    assert ev["space"] == {"id": "any;-;+15550100", "type": "dm", "phone": "+15551234567"}
    assert ev["sender"] == {"id": "+15550100"}
    assert ev["content"] == {"type": "text", "text": "hello via webhook"}


@pytest.mark.asyncio
async def test_process_dedups_replay(monkeypatch: pytest.MonkeyPatch) -> None:
    server, dispatched = _make_server(monkeypatch)
    body = json.loads(_text_body())
    await server._process(body)
    await server._process(body)  # at-least-once replay
    assert len(dispatched) == 1


@pytest.mark.asyncio
async def test_process_ignores_outbound_echo(monkeypatch: pytest.MonkeyPatch) -> None:
    server, dispatched = _make_server(monkeypatch)
    await server._process(json.loads(_text_body(msg_id="spc-msg-2", direction="outbound")))
    assert dispatched == []


@pytest.mark.asyncio
async def test_process_ignores_non_message_event(monkeypatch: pytest.MonkeyPatch) -> None:
    server, dispatched = _make_server(monkeypatch)
    await server._process({"event": "something-else", "message": {}})
    assert dispatched == []
