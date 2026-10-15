"""Failed split sends must not resend or report success for an incomplete reply."""

import aiohttp
from aiohttp import web
from aiohttp.test_utils import TestServer
import pytest

from gateway.config import PlatformConfig
from plugins.platforms.webex.adapter import WebexAdapter


@pytest.mark.asyncio
async def test_failed_second_chunk_preserves_receipt_without_plain_fallback(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    requests = []

    async def post_message(request):
        requests.append(await request.json())
        if len(requests) == 2:
            return web.json_response({"message": "temporary failure"}, status=503)
        return web.json_response({"id": f"message-{len(requests)}"})

    app = web.Application()
    app.router.add_post("/messages", post_message)
    async with TestServer(app) as server, aiohttp.ClientSession() as session:
        monkeypatch.setattr(
            "plugins.platforms.webex.adapter.API_BASE", str(server.make_url("/")).rstrip("/")
        )
        adapter = WebexAdapter(PlatformConfig(token="token"))
        adapter._session = session
        result = await adapter._send_with_retry(
            "room", "x" * 8000, metadata={"thread_id": "thread"}, base_delay=0,
        )

    assert result.success is False
    assert len(requests) == 2
    assert result.message_id == "message-1"
    assert result.continuation_message_ids == ()
    assert result.raw_response["partial_overflow"] is True
    assert result.raw_response["delivered_chunks"] == 1
    assert result.raw_response["total_chunks"] == 2
    assert all(payload["parentId"] == "thread" for payload in requests)
