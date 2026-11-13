"""Unauthenticated Feishu webhook traffic must not consume the authenticated delivery quota."""

from __future__ import annotations

import asyncio
import hashlib
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

pytest.importorskip("aiohttp")

from hermes_agent.gateway.config import PlatformConfig
from plugins.platforms.feishu.adapter import _FEISHU_WEBHOOK_RATE_LIMIT_MAX, FeishuAdapter

_SOURCE_IP = "203.0.113.7"
_EVENT = {"header": {"event_type": "im.message.receive_v1", "token": "tok"}, "event": {"message": {"message_id": "om_1"}}}


class _Content:
    def __init__(self, body: bytes):
        self.body = body

    async def readexactly(self, size: int) -> bytes:
        if len(self.body) < size:
            raise asyncio.IncompleteReadError(self.body, size)
        return self.body[:size]


def _request(payload: dict, headers: dict | None = None) -> SimpleNamespace:
    body = json.dumps(payload).encode()
    return SimpleNamespace(remote=_SOURCE_IP, content_length=None, headers=headers or {}, content=_Content(body))


def _signed(payload: dict, key: str) -> SimpleNamespace:
    body = json.dumps(payload).encode()
    ts, nonce = "1700000000", "n1"
    sig = hashlib.sha256(f"{ts}{nonce}{key}".encode() + body).hexdigest()
    headers = {"x-lark-request-timestamp": ts, "x-lark-request-nonce": nonce, "x-lark-signature": sig}
    return SimpleNamespace(remote=_SOURCE_IP, content_length=None, headers=headers, content=_Content(body))


def _adapter(monkeypatch, **env) -> FeishuAdapter:
    for name in ("FEISHU_VERIFICATION_TOKEN", "FEISHU_ENCRYPT_KEY"):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    adapter = FeishuAdapter(PlatformConfig())
    adapter._on_message_event = Mock()
    return adapter


def _send(adapter: FeishuAdapter, request: SimpleNamespace) -> int:
    return asyncio.run(adapter._handle_webhook_request(request)).status


def test_bad_token_flood_does_not_block_authenticated_delivery(monkeypatch):
    adapter = _adapter(monkeypatch, FEISHU_VERIFICATION_TOKEN="tok")
    forged = {**_EVENT, "header": {**_EVENT["header"], "token": "wrong"}}
    for _ in range(_FEISHU_WEBHOOK_RATE_LIMIT_MAX + 5):
        assert _send(adapter, _request(forged)) == 401

    assert _send(adapter, _request(_EVENT)) == 200
    adapter._on_message_event.assert_called_once()


def test_bad_signature_flood_does_not_block_authenticated_delivery(monkeypatch):
    adapter = _adapter(monkeypatch, FEISHU_ENCRYPT_KEY="key")
    bad_sig = {"x-lark-request-timestamp": "1", "x-lark-request-nonce": "n", "x-lark-signature": "0" * 64}
    for _ in range(_FEISHU_WEBHOOK_RATE_LIMIT_MAX + 5):
        assert _send(adapter, _request(_EVENT, bad_sig)) == 401

    assert _send(adapter, _signed(_EVENT, "key")) == 200
    adapter._on_message_event.assert_called_once()


def test_authenticated_deliveries_are_still_rate_limited(monkeypatch):
    adapter = _adapter(monkeypatch, FEISHU_VERIFICATION_TOKEN="tok")
    for _ in range(_FEISHU_WEBHOOK_RATE_LIMIT_MAX):
        assert _send(adapter, _request(_EVENT)) == 200
    assert _send(adapter, _request(_EVENT)) == 429


def test_unauthenticated_url_verification_flood_does_not_block_signed_delivery(monkeypatch):
    adapter = _adapter(monkeypatch, FEISHU_ENCRYPT_KEY="key")
    for _ in range(_FEISHU_WEBHOOK_RATE_LIMIT_MAX + 5):
        assert _send(adapter, _request({"type": "url_verification", "challenge": "x"})) == 200

    assert _send(adapter, _signed(_EVENT, "key")) == 200
    adapter._on_message_event.assert_called_once()


def test_token_authenticated_url_verification_is_rate_limited(monkeypatch):
    adapter = _adapter(monkeypatch, FEISHU_VERIFICATION_TOKEN="tok")
    challenge = {"type": "url_verification", "challenge": "x", "token": "tok"}
    for _ in range(_FEISHU_WEBHOOK_RATE_LIMIT_MAX):
        assert _send(adapter, _request(challenge)) == 200
    assert _send(adapter, _request(challenge)) == 429
