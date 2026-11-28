"""A proactive thread send (thread root, no reply target) continues from the newest thread event.

``m.thread.event_id`` stays on the root while the ``m.in_reply_to`` fallback points at the latest
thread event (resolved through the relations API), later chunks chain to the preceding chunk, a
failed lookup falls back to the root, and an explicit reply target skips the lookup.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest
from aiohttp import web

from gateway.config import PlatformConfig

ROOM = "!room:example.org"


def _make_adapter():
    from plugins.platforms.matrix.adapter import MatrixAdapter

    return MatrixAdapter(PlatformConfig(enabled=True, token="syt_test_token", extra={
        "homeserver": "https://matrix.example.org", "user_id": "@bot:example.org"}))


def _client(lookup):
    client = MagicMock()
    client.api.request = lookup
    client.send_message_event = AsyncMock(return_value="$evt")
    client.upload_media = AsyncMock(return_value="mxc://example.org/plain")
    return client


def _relations(client):
    return [call.args[2]["m.relates_to"] for call in client.send_message_event.await_args_list]


@pytest.mark.asyncio
async def test_proactive_send_replies_to_latest_thread_event():
    adapter = _make_adapter()
    adapter._client = _client(AsyncMock(return_value={"chunk": [{"event_id": "$latest"}]}))

    assert (await adapter.send(ROOM, "threaded", metadata={"thread_id": "$root"})).success
    assert _relations(adapter._client) == [{
        "rel_type": "m.thread", "event_id": "$root", "is_falling_back": True,
        "m.in_reply_to": {"event_id": "$latest"}}]
    adapter._client.api.request.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("reply_to,lookup,expected", [
    (None, AsyncMock(side_effect=RuntimeError("offline")), "$root"),
    (None, AsyncMock(return_value={"chunk": []}), "$root"),
    ("$explicit", AsyncMock(return_value={"chunk": [{"event_id": "$latest"}]}), "$explicit"),
])
async def test_fallback_respects_explicit_target_and_lookup_failure(reply_to, lookup, expected):
    adapter = _make_adapter()
    adapter._client = _client(lookup)

    result = await adapter.send(ROOM, "threaded", reply_to=reply_to, metadata={"thread_id": "$root"})
    assert result.success
    relation = _relations(adapter._client)[0]
    assert relation["event_id"] == "$root"
    assert relation["m.in_reply_to"] == {"event_id": expected}
    if reply_to:
        lookup.assert_not_awaited()


@pytest.mark.asyncio
async def test_media_in_thread_replies_to_latest_thread_event():
    adapter = _make_adapter()
    adapter._client = _client(AsyncMock(return_value={"chunk": [{"event_id": "$latest"}]}))

    result = await adapter._upload_and_send(
        ROOM, b"png", "chart.png", "image/png", "m.image", caption="Chart",
        metadata={"thread_id": "$root"})
    assert result.success
    relation = _relations(adapter._client)[0]
    assert relation["event_id"] == "$root"
    assert relation["m.in_reply_to"] == {"event_id": "$latest"}


@pytest.mark.asyncio
async def test_lookup_and_chunk_chaining_over_real_http_transport():
    """The real mautrix client hits the relations endpoint; every chunk keeps the root and
    replies to the latest thread event (first) or the preceding chunk (later)."""
    client_mod = pytest.importorskip("mautrix.client")  # the matrix extra is opt-in
    from mautrix.types import UserID

    received = []

    async def handle(request):
        assert request.headers["Authorization"] == "Bearer fixture-token"
        if request.method == "GET":
            assert request.path == f"/_matrix/client/v1/rooms/{ROOM}/relations/$root/m.thread/m.room.message"
            assert request.query["dir"] == "b"
            return web.json_response({"chunk": [{"event_id": "$latest"}]})
        received.append(await request.json())
        return web.json_response({"event_id": f"$sent{len(received)}"})

    app = web.Application()
    app.router.add_route("*", "/{tail:.*}", handle)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    client = client_mod.Client(mxid=UserID("@fixture:example.org"), base_url=f"http://127.0.0.1:{port}", token="fixture-token")
    try:
        adapter = _make_adapter()
        adapter._client = client
        adapter.max_message_length = 100
        assert (await adapter.send(ROOM, "word " * 60, metadata={"thread_id": "$root"})).success
        assert len(received) > 1
        for index, content in enumerate(received):
            relation = content["m.relates_to"]
            assert relation["event_id"] == "$root"
            assert relation["m.in_reply_to"]["event_id"] == ("$latest" if index == 0 else f"$sent{index}")
    finally:
        await client.api.session.close()
        await runner.cleanup()
