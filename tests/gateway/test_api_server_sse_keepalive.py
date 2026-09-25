"""API-server SSE keepalive cadence while the agent is silent (preflight compression, long
tool calls). Remote OpenAI-compatible clients commonly drop an idle stream after ~20 s and the
disconnect interrupts the turn, so every SSE writer must emit a comment frame before that
window closes (issue #111512)."""

import asyncio
import types

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms import api_server
from gateway.platforms import api_server_openai_routes as routes
from gateway.platforms.api_server import APIServerAdapter
from gateway.platforms.api_server_openai_routes import _iter_stream_items

REMOTE_CLIENT_IDLE_DEADLINE_S = 20.0


class _RecordingResponse:
    def __init__(self) -> None:
        self.writes: list[bytes] = []

    async def write(self, data: bytes) -> None:
        self.writes.append(data)


@pytest.fixture
def adapter():
    return APIServerAdapter(PlatformConfig(enabled=True, extra={}))


@pytest.mark.asyncio
async def test_idle_openai_stream_writes_keepalive_before_remote_client_deadline(monkeypatch):
    """Chat/Responses streams (``_iter_stream_items``) keep the wire alive inside the client window."""
    assert api_server.CHAT_COMPLETIONS_SSE_KEEPALIVE_SECONDS <= REMOTE_CLIENT_IDLE_DEADLINE_S

    # Module-local fake clock: each read is one keepalive interval later, so the poll loop
    # observes an idle stream without waiting real seconds (asyncio keeps the real clock).
    reads = [0.0]
    step = api_server.CHAT_COMPLETIONS_SSE_KEEPALIVE_SECONDS

    def monotonic():
        reads[0] += step
        return reads[0]

    monkeypatch.setattr(routes, "time", types.SimpleNamespace(monotonic=monotonic))
    stream_q: asyncio.Queue = asyncio.Queue()
    agent_task = asyncio.get_running_loop().create_future()
    response = _RecordingResponse()

    async def _finish_after_first_keepalive():
        while not response.writes:
            await asyncio.sleep(0.05)
        agent_task.set_result(None)

    finisher = asyncio.create_task(_finish_after_first_keepalive())
    items = [item async for item in _iter_stream_items(stream_q, agent_task, response)]
    await finisher

    assert items == []
    assert response.writes[0] == b": keepalive\n\n"


@pytest.mark.asyncio
async def test_idle_run_events_stream_uses_shared_keepalive_cadence(monkeypatch, adapter):
    """``GET /v1/runs/{id}/events`` follows the same keepalive constant as the OpenAI routes."""
    monkeypatch.setattr(api_server, "CHAT_COMPLETIONS_SSE_KEEPALIVE_SECONDS", 0.2)
    from gateway.platforms.api_server_runs import _RunStream

    stream = _RunStream()
    adapter._run_streams["run_idle"] = stream
    adapter._set_run_status("run_idle", "running")
    monkeypatch.setattr(adapter, "_request_owns_run", lambda request, run_id: True)

    app = web.Application()
    app["api_server_adapter"] = adapter
    app.router.add_get("/v1/runs/{run_id}/events", adapter._handle_run_events)

    async def _close_after_idle():
        await asyncio.sleep(0.5)
        stream.put_nowait(None)

    async with TestClient(TestServer(app)) as cli:
        closer = asyncio.create_task(_close_after_idle())
        resp = await cli.get("/v1/runs/run_idle/events")
        body = await resp.text()
        await closer

    assert resp.status == 200
    assert body.count(": keepalive\n\n") >= 1
