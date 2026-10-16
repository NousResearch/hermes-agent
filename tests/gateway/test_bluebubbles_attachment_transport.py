"""Real HTTP/cache coverage complementing PR #127424's admission unit tests."""
import asyncio
import base64
from collections import Counter
from contextlib import asynccontextmanager
import json
from pathlib import Path

import httpx
import pytest

from gateway.config import PlatformConfig
from gateway.platforms.bluebubbles import BlueBubblesAdapter

PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8"
    "AAwMCAO+aN4sAAAAASUVORK5CYII="
)
CHAT = "any;-;+15555550100"
PHOTO = {"guid": "photo-a", "mimeType": "image/png", "transferName": "a.png"}


class Request:
    query = {"password": "test-only"}
    headers = {}

    def __init__(self, text, attachments):
        self.body = json.dumps({"type": "new-message", "data": {
            "guid": "message-a", "text": text, "isFromMe": False,
            "handle": {"address": "+15555550100"},
            "chats": [{"guid": CHAT, "style": 45}], "attachments": attachments,
        }}).encode()

    async def read(self):
        return self.body


async def settle(adapter):
    while adapter._background_tasks:
        await asyncio.wait_for(
            asyncio.gather(*list(adapter._background_tasks), return_exceptions=True), 8
        )
        await asyncio.sleep(0)  # Allow tracked-task removal callbacks to run.


@asynccontextmanager
async def open_adapter(monkeypatch, before_download=None):
    adapter = BlueBubblesAdapter(PlatformConfig(enabled=True, extra={
        "server_url": "http://example.invalid", "password": "test-only",
        "send_read_receipts": False, "require_mention": False,
    }))
    paths = {}
    original_fetch = adapter._fetch_attachment_once

    async def fetch(guid, metadata):
        path = await original_fetch(guid, metadata)
        if path:
            paths[path] = guid
        return path

    async def transport(request):
        if before_download:
            await before_download(request)
        return httpx.Response(200, content=PNG)

    monkeypatch.setattr(adapter, "_fetch_attachment_once", fetch)
    async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as client:
        adapter.client = client
        try:
            yield adapter, paths
        finally:
            tasks = list(adapter._background_tasks)
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)


def attachment_counts(events, paths):
    return Counter(paths[path] for event in events for path in event.media_urls)


@pytest.mark.asyncio
@pytest.mark.parametrize("caption", ["two photos", ""])
async def test_concurrent_collectors_deliver_each_downloaded_file_once(monkeypatch, caption):
    blocked, release = asyncio.Event(), asyncio.Event()
    admitted = asyncio.Event()
    accepted = []

    async def before_download(request):
        if request.url.path.endswith("/photo-b/download") and asyncio.current_task().get_name() == "collector-one":
            blocked.set()
            await release.wait()

    async with open_adapter(monkeypatch, before_download) as (adapter, paths):
        async def admit(event):
            event._gateway_accepted = True
            accepted.append(event)
            if len(event.media_urls) == 2:
                admitted.set()

        monkeypatch.setattr(adapter, "handle_message", admit)
        request = Request(caption, [PHOTO, {**PHOTO, "guid": "photo-b"}])
        first = asyncio.create_task(adapter._handle_webhook(request), name="collector-one")
        try:
            await asyncio.wait_for(blocked.wait(), 5)
            second = await asyncio.wait_for(adapter._handle_webhook(request), 8)
            await asyncio.wait_for(admitted.wait(), 5)
            release.set()
            loser = await asyncio.wait_for(first, 8)
            await settle(adapter)
            assert second.status == loser.status == 200
            assert attachment_counts(accepted, paths) == {"photo-a": 1, "photo-b": 1}
            assert all(Path(path).read_bytes() == PNG for event in accepted for path in event.media_urls)
        finally:
            release.set()
            await asyncio.wait_for(asyncio.gather(first, return_exceptions=True), 8)


@pytest.mark.asyncio
async def test_recovery_retries_actual_missing_handler_until_base_admits(monkeypatch):
    async with open_adapter(monkeypatch) as (adapter, paths):
        accepted, rounds = [], []
        handled = asyncio.Event()
        real_sleep = asyncio.sleep

        async def gateway_handler(event):
            accepted.append(event)
            handled.set()

        async def next_round(delay):
            rounds.append(delay)
            if len(rounds) == 2:
                adapter.set_message_handler(gateway_handler)
            await real_sleep(0)

        monkeypatch.setattr("gateway.platforms.bluebubbles.asyncio.sleep", next_round)
        source = adapter.build_source(chat_id=CHAT, chat_type="dm", user_id="+15555550100")
        adapter._late_recovery_pending.add(PHOTO["guid"])
        # Do not replace handle_message: first refusal and second acceptance use the real base.
        await asyncio.wait_for(adapter._recover_late_attachments([PHOTO], source=source), 8)
        await settle(adapter)
        if PHOTO["guid"] in adapter._late_recovery_done:
            await asyncio.wait_for(handled.wait(), 5)
        assert attachment_counts(accepted, paths) == {"photo-a": 1}
        assert all(Path(path).read_bytes() == PNG for event in accepted for path in event.media_urls)
        assert PHOTO["guid"] in adapter._late_recovery_done
