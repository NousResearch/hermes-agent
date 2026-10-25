"""Transport-free adapter construction and webhook requests for BlueBubbles tests."""

import asyncio
import json

from gateway.config import PlatformConfig


def _make_adapter(monkeypatch, **extra):
    monkeypatch.setenv("BLUEBUBBLES_SERVER_URL", "http://localhost:1234")
    monkeypatch.setenv("BLUEBUBBLES_PASSWORD", "secret")
    from plugins.platforms.bluebubbles.adapter import BlueBubblesAdapter

    cfg = PlatformConfig(
        enabled=True,
        extra={
            "server_url": "http://localhost:1234",
            "password": "secret",
            **extra,
        },
    )
    adapter = BlueBubblesAdapter(cfg)
    return adapter


class _FakeBlueBubblesRequest:
    def __init__(self, payload, password="secret"):
        self.query = {"password": password}
        self.headers = {}
        self._body = json.dumps(payload).encode("utf-8")

    async def read(self):
        return self._body


class Request:
    query = {"password": "synthetic-secret"}
    headers = {}

    def __init__(self, *, text="caption", attachments=None, event="new-message"):
        self.body = json.dumps({"type": event, "data": {
            "guid": "same-message", "text": text,
            "handle": {"address": "sender@example.test"},
            "chats": [{"guid": "iMessage;-;sender@example.test", "style": 45}],
            "attachments": attachments or [],
        }}).encode()

    async def read(self):
        return self.body

def adapter():
    from plugins.platforms.bluebubbles.adapter import BlueBubblesAdapter

    return BlueBubblesAdapter(PlatformConfig(enabled=True, extra={
        "server_url": "http://example.invalid", "password": "synthetic-secret",
        "send_read_receipts": False, "require_mention": False,
    }))

async def settle(instance):
    while instance._background_tasks:
        await asyncio.wait_for(asyncio.gather(*list(instance._background_tasks)), 5)
        await asyncio.sleep(0)
