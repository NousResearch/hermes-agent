"""Inbound Mattermost attachments: fetched only for authorized senders, read under the media cap.

Drives the real adapter against a fake Mattermost REST API on 127.0.0.1 with the real
GatewayRunner authorization callback wired as startup wires it.
"""
import json
import os
from pathlib import Path

import aiohttp
import pytest
import pytest_asyncio
from aiohttp import web

from gateway.config import GatewayConfig, Platform, PlatformConfig

OWNER, STRANGER = "owner_user", "stranger_user"


@pytest_asyncio.fixture
async def mattermost(tmp_path, monkeypatch):
    from gateway.pairing import PairingStore
    from gateway.run import GatewayRunner
    from plugins.platforms.mattermost.adapter import MattermostAdapter

    for key in ("GATEWAY_ALLOWED_USERS", "GATEWAY_ALLOW_ALL_USERS", "MATTERMOST_ALLOW_ALL_USERS"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("MATTERMOST_ALLOWED_USERS", OWNER)
    monkeypatch.setattr("gateway.pairing.PAIRING_DIR", tmp_path / "pairing")

    sizes, hits = {}, []

    async def info(request):
        hits.append(request.path)
        return web.json_response({"name": f"{request.match_info['fid']}.pdf", "mime_type": "application/pdf"})

    async def body(request):
        hits.append(request.path)
        resp = web.StreamResponse()  # chunked, so no Content-Length can be rejected up front
        await resp.prepare(request)
        remaining = sizes[request.match_info["fid"]]
        try:
            while remaining > 0:
                await resp.write(b"x" * min(65536, remaining))
                remaining -= 65536
        except (ConnectionResetError, aiohttp.ClientConnectionResetError):
            pass
        return resp

    app = web.Application()
    app.router.add_get("/api/v4/files/{fid}/info", info)
    app.router.add_get("/api/v4/files/{fid}", body)
    server = web.AppRunner(app)
    await server.setup()
    site = web.TCPSite(server, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]

    adapter = MattermostAdapter(PlatformConfig(enabled=True, token="t", extra={"url": f"http://127.0.0.1:{port}"}))
    adapter._bot_user_id = "bot"
    adapter._session = aiohttp.ClientSession()
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(platforms={Platform.MATTERMOST: adapter.config})
    runner.adapters = {Platform.MATTERMOST: adapter}
    runner.pairing_store = PairingStore()
    adapter.set_authorization_check(runner._make_adapter_auth_check(Platform.MATTERMOST))
    events = []

    async def capture(event):
        events.append(event)
    adapter.handle_message = capture

    async def dm(post_id, user_id, file_sizes):
        sizes.update(file_sizes)
        post = {"id": post_id, "user_id": user_id, "channel_id": "dm_chan", "message": "see attached",
                "file_ids": list(file_sizes)}
        await adapter._handle_ws_event({"event": "posted", "data": {
            "channel_type": "D", "sender_name": "@someone", "post": json.dumps(post)}})
        return events[-1]

    yield dm, hits
    await adapter._session.close()
    await server.cleanup()


@pytest.mark.asyncio
async def test_attachments_are_fetched_only_for_authorized_senders(mattermost):
    dm, hits = mattermost

    denied = await dm("p1", STRANGER, {"stranger_doc": 1024})
    assert hits == []
    assert denied.media_urls is None
    assert denied.source.user_id == STRANGER  # still dispatched so the runner can deny/pair

    allowed = await dm("p2", OWNER, {"owner_doc": 1024})
    assert len(allowed.media_urls) == 1
    assert Path(allowed.media_urls[0]).stat().st_size == 1024


@pytest.mark.asyncio
async def test_attachment_body_is_read_under_inbound_media_cap(mattermost):
    from gateway.platforms.base import get_document_cache_dir, get_inbound_media_max_bytes

    Path(os.environ["HERMES_HOME"], "config.yaml").write_text("gateway:\n  max_inbound_media_bytes: 1048576\n")
    cap = get_inbound_media_max_bytes()
    assert cap == 1048576
    dm, _hits = mattermost

    event = await dm("p3", OWNER, {"huge_doc": 4 * cap})

    assert event.media_urls is None
    assert all(p.stat().st_size <= cap for p in get_document_cache_dir().iterdir())
