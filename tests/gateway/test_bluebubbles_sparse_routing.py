"""Metadata completion releases retained text/media; pure receipts start no new turn."""
import base64
import json
from pathlib import Path

import httpx
import pytest

from gateway.config import PlatformConfig
from gateway.platforms.bluebubbles import BlueBubblesAdapter

_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jR7kAAAAASUVORK5CYII=")
_GROUP = [{"guid": "iMessage;+;actual-group", "style": 43}]
_ATTACHMENT = [{"guid": "unseen-photo", "mimeType": "image/png", "transferState": 5}]


class _Request:
    query = {"password": "test-password"}
    headers = {}

    def __init__(self, event_type, **record):
        self.payload = {"type": event_type, "data": record}

    async def read(self):
        return json.dumps(self.payload).encode()


@pytest.mark.asyncio
@pytest.mark.parametrize("pending", ["none", "text", "media"])
async def test_metadata_completion_admits_only_retained_content(pending):
    adapter = BlueBubblesAdapter(PlatformConfig(enabled=True, extra={
        "server_url": "http://bluebubbles.invalid", "password": "test-password",
        "send_read_receipts": False, "require_mention": False,
    }))
    admitted, requests = [], []
    download_ready = False

    async def accept(event):
        event._gateway_accepted = True
        admitted.append(event)

    adapter.handle_message = accept

    def transport(request):
        requests.append(request)
        assert request.method == "GET"
        assert request.url.params["password"] == "test-password"
        if request.url.path == "/api/v1/attachment/unseen-photo/download":
            return (httpx.Response(200, content=_PNG, headers={"content-type": "image/png"})
                    if download_ready else httpx.Response(503))
        assert request.url.path == "/api/v1/message/retained-message"
        assert request.url.params["with"] == "chats,attachments"
        if pending == "media":
            return httpx.Response(200, json={"data": {"chats": _GROUP, "attachments": _ATTACHMENT}})
        return httpx.Response(500)

    async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as client:
        adapter.client = client
        if pending != "none":
            fields = {"chats": _GROUP} if pending == "media" else {}
            first = await adapter._handle_webhook(_Request(
                "new-message", guid="retained-message", text="hello group",
                handle={"address": "peer@example.com"}, **fields,
            ))
            assert first.status == 200
            assert len(admitted) == (1 if pending == "media" else 0)
        if pending == "media":
            unavailable = await adapter._handle_webhook(_Request(
                "updated-message", guid="retained-message", attachments=_ATTACHMENT))
            assert unavailable.status == 503
            assert len(admitted) == 1
            download_ready = True

        update = _Request("updated-message", guid="retained-message", chats=_GROUP, dateRead=123)
        completed = await adapter._handle_webhook(update)
        assert completed.status == 200
        expected = ([] if pending == "none" else ["hello group"] if pending == "text"
                    else ["hello group", "(attachment)"])
        assert [event.text for event in admitted] == expected
        assert all((event.source.chat_id, event.source.chat_type) ==
                   ("iMessage;+;actual-group", "group") for event in admitted)
        if pending == "media":
            assert admitted[0].media_urls == []
            assert admitted[1].media_types == ["image/png"]
            assert len(admitted[1].media_urls) == 1
            assert Path(admitted[1].media_urls[0]).read_bytes() == _PNG
        requests_before_replay = len(requests)
        repeated = await adapter._handle_webhook(update)
        assert repeated.status == 200
        assert [event.text for event in admitted] == expected
        assert len(requests) == requests_before_replay
        if pending == "none":
            assert requests == []
