"""REST send route for the A2A plugin: ``POST /api/v1/messages/send`` (#128747).

Split out of ``test_a2a_plugin.py``, which is at the 2000-line target; the live
adapter helpers it drives stay in the parent module.
"""

from __future__ import annotations

import asyncio
import json
import urllib.error
import urllib.request

import pytest

from plugins.platforms.a2a import protocol
from tests.plugins.test_a2a_plugin import _get_json, _make_live_adapter

def _post_status(url, body, headers=None):
    data = body if isinstance(body, bytes) else json.dumps(body).encode()
    req = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json", **(headers or {})}, method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        try:
            payload = json.loads(e.read().decode())
        except Exception:
            payload = {}
        return e.code, payload


@pytest.mark.integration
class TestRestMessageSend:
    """POST /api/v1/messages/send — REST send route (#128747)."""

    def test_rest_send_delivers_and_returns_200(self, monkeypatch):
        monkeypatch.delenv("A2A_BEARER_TOKEN", raising=False)
        monkeypatch.delenv("A2A_PEER_TOKENS", raising=False)
        adapter, base = _make_live_adapter(monkeypatch)

        async def run():
            assert await adapter.connect() is True
            status, resp = await asyncio.to_thread(
                _post_status, base + "/api/v1/messages/send", {"text": "hello rest"})
            assert status == 200
            assert resp["ok"] is True
            task = resp["task"]
            assert task["status"]["state"] == "TASK_STATE_COMPLETED"
            assert "ECHO:" in protocol.extract_text(task["artifacts"][0])
            await adapter.disconnect()

        asyncio.run(run())

    def test_rest_send_bad_body_400(self, monkeypatch):
        monkeypatch.delenv("A2A_BEARER_TOKEN", raising=False)
        monkeypatch.delenv("A2A_PEER_TOKENS", raising=False)
        adapter, base = _make_live_adapter(monkeypatch)

        async def run():
            assert await adapter.connect() is True
            status, _ = await asyncio.to_thread(
                _post_status, base + "/api/v1/messages/send", b"{not json")
            assert status == 400
            status, _ = await asyncio.to_thread(
                _post_status, base + "/api/v1/messages/send", {"nonsense": 1})
            assert status == 400
            await adapter.disconnect()

        asyncio.run(run())

    def test_rest_send_unauthorized_401(self, monkeypatch):
        monkeypatch.setenv("A2A_BEARER_TOKEN", "rest-secret")
        monkeypatch.delenv("A2A_PEER_TOKENS", raising=False)
        adapter, base = _make_live_adapter(monkeypatch)

        async def run():
            assert await adapter.connect() is True
            status, _ = await asyncio.to_thread(
                _post_status, base + "/api/v1/messages/send", {"text": "hi"})
            assert status == 401
            await adapter.disconnect()

        asyncio.run(run())

    def test_rest_send_get_405(self, monkeypatch):
        monkeypatch.delenv("A2A_BEARER_TOKEN", raising=False)
        monkeypatch.delenv("A2A_PEER_TOKENS", raising=False)
        adapter, base = _make_live_adapter(monkeypatch)

        async def run():
            assert await adapter.connect() is True
            try:
                await asyncio.to_thread(_get_json, base + "/api/v1/messages/send")
                raised = None
            except urllib.error.HTTPError as e:
                raised = e.code
            assert raised == 405
            await adapter.disconnect()

        asyncio.run(run())
