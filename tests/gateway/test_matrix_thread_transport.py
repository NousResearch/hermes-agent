"""Exercise proactive thread delivery through the real mautrix HTTP client."""
import pytest
from aiohttp import web
from mautrix.client import Client
from mautrix.types import UserID
from gateway.config import PlatformConfig
from plugins.platforms.matrix.adapter import MatrixAdapter


@pytest.mark.asyncio
async def test_proactive_thread_lookup_and_chunk_chaining_use_real_http_transport():
    received = []

    async def handle(request):
        assert request.headers['Authorization'] == 'Bearer fixture-token'
        if request.method == 'GET':
            assert request.path == '/_matrix/client/v1/rooms/!room:example.org/relations/$root/m.thread/m.room.message'
            assert request.query['dir'] == 'b'
            return web.json_response({'chunk': [{'event_id': '$latest'}]})
        content = await request.json()
        received.append(content)
        return web.json_response({'event_id': f'$sent{len(received)}'})

    app = web.Application()
    app.router.add_route('*', '/{tail:.*}', handle)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '127.0.0.1', 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    client = Client(mxid=UserID('@fixture:example.org'), base_url=f'http://127.0.0.1:{port}', token='fixture-token')
    try:
        adapter = MatrixAdapter(PlatformConfig(enabled=True))
        adapter._client = client
        adapter.max_message_length = 100
        result = await adapter.send('!room:example.org', 'word ' * 60, metadata={'thread_id': '$root'})
        assert result.success
        assert len(received) > 1
        for index, content in enumerate(received):
            relation = content['m.relates_to']
            assert relation['event_id'] == '$root'
            assert relation['m.in_reply_to']['event_id'] == ('$latest' if index == 0 else f'$sent{index}')
    finally:
        await client.api.session.close()
        await runner.cleanup()
