"""Webex cooldowns survive final-response persistence and runtime redelivery."""

import asyncio
import sqlite3

import aiohttp
from aiohttp import web
from aiohttp.test_utils import TestServer
import pytest

from gateway import delivery_ledger as dl
from gateway.config import Platform, PlatformConfig
from gateway.platforms.event import MessageEvent
from gateway.run import GatewayRunner
from gateway.session import SessionSource
from plugins.platforms.webex.adapter import WebexAdapter


def _obligation():
    with dl._connect() as connection:
        connection.row_factory = sqlite3.Row
        return dict(connection.execute("SELECT * FROM delivery_obligations").fetchone())


@pytest.mark.asyncio
async def test_long_webex_cooldown_survives_final_and_runtime_redelivery(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(dl, "_db_path", lambda: tmp_path / "state.db")
    requests = []

    async def post_message(request):
        requests.append(await request.json())
        if len(requests) <= 2:
            return web.json_response(
                {"message": "rate limited"}, status=429, headers={"Retry-After": "600"}
            )
        return web.json_response({"id": "delivered"})

    app = web.Application()
    app.router.add_post("/messages", post_message)
    platform = Platform("webex")
    event = MessageEvent(
        text="question",
        message_id="incoming-message",
        source=SessionSource(
            platform=platform, chat_id="room", chat_type="group", thread_id="thread"
        ),
    )
    async with TestServer(app) as server, aiohttp.ClientSession() as session:
        monkeypatch.setattr(
            "plugins.platforms.webex.adapter.API_BASE",
            str(server.make_url("/")).rstrip("/"),
        )
        adapter = WebexAdapter(PlatformConfig(token="webex-test-token"))
        adapter._session = session
        result, _ = await adapter.send_final_ledgered(
            event,
            "agent:main:webex:group:room:thread:thread",
            "final answer",
            {"thread_id": "thread"},
            reply_to=None,
        )
        assert result.success is False and result.retry_after == 600
        row = await asyncio.to_thread(_obligation)
        assert row["state"] == "failed" and row["attempts"] == 0
        assert await asyncio.to_thread(
            dl.sweep_failed_for_runtime, "webex", now=row["updated_at"] + 599
        ) == []
        assert (await asyncio.to_thread(_obligation))["attempts"] == 0

        runner = object.__new__(GatewayRunner)
        runner.adapters = {platform: adapter}
        runner._profile_adapters = {}
        runner._active_profile_name = lambda: "default"
        runner._running = False  # Drive deadlines explicitly; no real-time timer waits.
        claimed = await asyncio.to_thread(
            dl.sweep_failed_for_runtime, "webex", now=row["updated_at"] + 601
        )
        assert len(claimed) == 1
        assert await runner._redeliver_claimed_obligations(claimed) == 0
        row = await asyncio.to_thread(_obligation)
        assert row["state"] == "failed" and row["attempts"] == 1
        assert await asyncio.to_thread(
            dl.sweep_failed_for_runtime, "webex", now=row["updated_at"] + 599
        ) == []
        assert (await asyncio.to_thread(_obligation))["attempts"] == 1

        claimed = await asyncio.to_thread(
            dl.sweep_failed_for_runtime, "webex", now=row["updated_at"] + 601
        )
        assert len(claimed) == 1
        assert await runner._redeliver_claimed_obligations(claimed) == 1

    row = await asyncio.to_thread(_obligation)
    assert row["state"] == "delivered" and row["attempts"] == 2
    assert len(requests) == 3
    assert all(payload["parentId"] == "thread" for payload in requests)
    assert requests[1]["markdown"] == dl.FLOOD_MARKER + "final answer"
    assert requests[2]["markdown"] == dl.FLOOD_MARKER + "final answer"
