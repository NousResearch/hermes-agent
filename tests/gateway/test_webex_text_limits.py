"""Webex text delivery observes the API's UTF-8 byte limit without losing text."""

import asyncio
import re

import aiohttp
from aiohttp import web
from aiohttp.test_utils import TestServer
import pytest
import pytest_asyncio

from gateway.config import PlatformConfig
from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig
from plugins.platforms.webex.adapter import WebexAdapter


@pytest_asyncio.fixture
async def webex_http(monkeypatch):
    requests = []
    messages = {}
    first_sent = asyncio.Event()

    async def get_message(request):
        return web.json_response(messages[request.match_info["message_id"]])

    async def write_message(request):
        if request.content_type == "multipart/form-data":
            payload = {}
            async for part in await request.multipart():
                body = bytes(await part.read())
                payload[part.name] = body if part.filename else body.decode()
        else:
            payload = await request.json()
        requests.append((request.method, payload))
        if len(payload.get("markdown", "").encode("utf-8")) > 7439:
            return web.json_response({"message": "message exceeds 7439 bytes"}, status=400)
        message_id = request.match_info.get("message_id") or f"message-{len(messages)}"
        messages[message_id] = {**messages.get(message_id, {}), **payload, "id": message_id}
        first_sent.set()
        return web.json_response({"id": message_id})

    app = web.Application()
    app.router.add_get("/messages/{message_id}", get_message)
    app.router.add_post("/messages", write_message)
    app.router.add_put("/messages/{message_id}", write_message)
    async with TestServer(app) as server, aiohttp.ClientSession() as session:
        monkeypatch.setattr(
            "plugins.platforms.webex.adapter.API_BASE", str(server.make_url("/")).rstrip("/")
        )
        adapter = WebexAdapter(PlatformConfig(token="token"))
        adapter._session = session
        yield adapter, requests, messages, first_sent


@pytest.mark.asyncio
@pytest.mark.parametrize("text", ["x" * 8000, "漢" * 2500, "😀" * 2000], ids=["ascii", "han", "emoji"])
@pytest.mark.parametrize("operation", ["send", "stream"])
async def test_live_delivery_preserves_text_within_utf8_limit(webex_http, text, operation):
    adapter, requests, messages, first_sent = webex_http
    if operation == "send":
        result = await adapter.send("room", text, metadata={"thread_id": "thread"})
        assert result.success
    else:
        consumer = GatewayStreamConsumer(
            adapter, "room", StreamConsumerConfig(edit_interval=0, buffer_threshold=1, cursor=""),
            metadata={"thread_id": "thread"},
        )
        consumer.on_delta(text[:100])
        task = asyncio.create_task(consumer.run())
        await first_sent.wait()
        consumer.on_delta(text[100:])
        consumer.finish()
        await task
        assert consumer._final_response_sent
        assert any(method == "PUT" for method, _payload in requests)

    delivered = [message["markdown"] for message in messages.values()]
    assert len(delivered) > 1
    assert all(len(chunk.encode("utf-8")) <= 7439 for chunk in delivered)
    assert "".join(re.sub(r" \(\d+/\d+\)$", "", chunk) for chunk in delivered) == text
    assert all(message["roomId"] == "room" for message in messages.values())
    assert all(message["parentId"] == "thread" for message in messages.values())


@pytest.mark.asyncio
async def test_oversized_direct_edit_reports_limit_without_mutating_message(webex_http):
    adapter, requests, messages, _first_sent = webex_http
    messages["seed"] = {"id": "seed", "roomId": "room", "parentId": "thread", "markdown": "original"}

    result = await adapter.edit_message("room", "seed", "漢" * 2500, finalize=True)

    assert result.success is False and result.error_kind == "too_long"
    assert requests == []
    assert messages["seed"]["markdown"] == "original"


@pytest.mark.asyncio
async def test_large_markdown_code_blocks_remain_balanced(webex_http):
    adapter, _requests, messages, _first_sent = webex_http
    line = "print('漢😀')\n"
    result = await adapter.send("room", "```python\n" + line * 1000 + "```", metadata={"thread_id": "thread"})

    assert result.success
    assert len(messages) > 1
    assert sum(message["markdown"].count("print('漢😀')") for message in messages.values()) == 1000
    for message in messages.values():
        text = re.sub(r" \(\d+/\d+\)$", "", message["markdown"])
        assert text.startswith("```python\n") and text.endswith("```")
        assert len(message["markdown"].encode("utf-8")) <= 7439


@pytest.mark.asyncio
@pytest.mark.parametrize("caption", ["漢" * 2500, "😀" * 2000], ids=["han", "emoji"])
@pytest.mark.parametrize("media_kind", ["local_file", "image_url"])
async def test_live_media_preserves_oversized_captions(webex_http, tmp_path, caption, media_kind):
    adapter, requests, messages, _first_sent = webex_http
    metadata = {"thread_id": "thread"}
    if media_kind == "local_file":
        path = tmp_path / "report.txt"
        path.write_bytes(b"report contents")
        result = await adapter.send_document("room", str(path), caption=caption, metadata=metadata)
        assert result.success
        assert messages[result.message_id]["files"] == b"report contents"
    else:
        result = await adapter.send_image("room", "https://example.com/image.png", caption=caption, metadata=metadata)
        assert result.success
        assert messages[result.message_id]["files"] == ["https://example.com/image.png"]

    delivered_text = [message.get("markdown", "") for message in messages.values()]
    assert "".join(re.sub(r" \(\d+/\d+\)$", "", chunk) for chunk in delivered_text) == caption
    assert all(len(chunk.encode("utf-8")) <= 7439 for chunk in delivered_text)
    assert all(message["roomId"] == "room" and message["parentId"] == "thread" for message in messages.values())
    assert set(result.continuation_message_ids) | {result.message_id} == set(messages)
    assert sum("files" in payload for _method, payload in requests) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("media_kind", ["local_file", "image_url"])
async def test_live_media_stops_after_failed_caption_delivery(monkeypatch, tmp_path, media_kind):
    requests = []

    async def refuse(request):
        requests.append(request.content_type)
        await request.read()
        return web.json_response({"message": "temporarily unavailable"}, status=503)

    app = web.Application()
    app.router.add_post("/messages", refuse)
    async with TestServer(app) as server, aiohttp.ClientSession() as session:
        monkeypatch.setattr("plugins.platforms.webex.adapter.API_BASE", str(server.make_url("/")).rstrip("/"))
        adapter = WebexAdapter(PlatformConfig(token="token"))
        adapter._session = session
        caption = "漢" * 2500
        if media_kind == "local_file":
            path = tmp_path / "report.txt"
            path.write_bytes(b"report contents")
            result = await adapter.send_document("room", str(path), caption=caption, metadata={"thread_id": "thread"})
        else:
            result = await adapter.send_image("room", "https://example.com/image.png", caption=caption, metadata={"thread_id": "thread"})

    assert result.success is False
    assert requests == ["application/json"]
