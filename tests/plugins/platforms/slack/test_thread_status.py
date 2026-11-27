from types import SimpleNamespace

from aiohttp import ClientSession, web
from aiohttp.test_utils import TestServer
import pytest
from slack_sdk.web.async_client import AsyncWebClient

from gateway.config import PlatformConfig
from plugins.platforms.slack.adapter import SlackAdapter


@pytest.mark.asyncio
@pytest.mark.parametrize("configured", [None, "is pouncing… 🐾"])
async def test_thread_status_preserves_display_text_and_clears(configured):
    statuses = {}

    async def set_status(request):
        payload = await request.json()
        if request.path == "/agents.sessions.setStatus":
            if payload["status"] not in {"active", "processing", "suspended", "closed"}:
                return web.json_response({"ok": False, "error": "invalid_arguments"})
        target = (payload["channel_id"], payload["thread_ts"])
        statuses[target] = payload["status"]
        return web.json_response({"ok": True})

    app = web.Application()
    app.router.add_post("/assistant.threads.setStatus", set_status)
    app.router.add_post("/agents.sessions.setStatus", set_status)
    async with TestServer(app) as server, ClientSession() as session:
        client = AsyncWebClient(
            token="xoxb-test", base_url=str(server.make_url("/")), session=session,
        )
        adapter = SlackAdapter(PlatformConfig(
            enabled=True, token="xoxb-test", typing_status_text=configured,
        ))
        adapter._app = SimpleNamespace(client=client)
        metadata = {"thread_id": "171234.0001"}
        target = ("C123", "171234.0001")

        await adapter.send_typing("C123", metadata=metadata)
        assert statuses == {target: configured or "is thinking..."}

        adapter.set_status_text("C123", "is running pytest…")
        await adapter.send_typing("C123", metadata=metadata)
        assert statuses == {target: "is running pytest…"}

        await adapter.stop_typing("C123", metadata=metadata)
        assert statuses == {target: ""}
