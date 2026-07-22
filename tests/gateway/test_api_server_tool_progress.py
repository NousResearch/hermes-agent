"""Tests for the ``X-Hermes-Tool-Progress: off`` opt-out on /v1/chat/completions.

The API server emits custom ``event: hermes.tool.progress`` SSE frames so chat
frontends can show live tool activity. Strict OpenAI-compatible clients (the
``openai`` SDK, LiveKit's voice plugin, etc.) parse every frame as a
``chat.completion.chunk`` and crash on these custom frames (no ``choices`` field
-> ``chunk.choices`` is None -> "NoneType object is not iterable"). Such clients
send ``X-Hermes-Tool-Progress: off`` to suppress the UI-only frames; the tools
still run server-side and the full answer still streams.
"""

import asyncio
import queue

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter


def _make_adapter() -> APIServerAdapter:
    return APIServerAdapter(PlatformConfig(enabled=True))


async def _collect_sse(suppress: bool) -> str:
    """Drive ``_write_sse_chat_completion`` through a real aiohttp request and
    return the raw SSE body. The stream yields one tool-progress frame, one
    content chunk, then the end sentinel."""
    adapter = _make_adapter()

    async def handler(request: web.Request) -> web.StreamResponse:
        stream_q: "queue.Queue" = queue.Queue()
        stream_q.put(("__tool_progress__", {"tool_name": "terminal", "delta": "running"}))
        stream_q.put("hello world")
        stream_q.put(None)  # end-of-stream sentinel

        async def _agent():
            return (None, {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2})

        agent_task = asyncio.create_task(_agent())
        return await adapter._write_sse_chat_completion(
            request,
            completion_id="cmpl-test",
            model="test-model",
            created=0,
            stream_q=stream_q,
            agent_task=agent_task,
        )

    app = web.Application()
    app.router.add_get("/sse", handler)
    async with TestClient(TestServer(app)) as client:
        headers = {"X-Hermes-Tool-Progress": "off"} if suppress else {}
        resp = await client.get("/sse", headers=headers)
        assert resp.status == 200
        return await resp.text()


@pytest.mark.asyncio
async def test_tool_progress_emitted_by_default():
    """Without the header, the custom progress frame is present (chat frontends)."""
    body = await _collect_sse(suppress=False)
    assert "event: hermes.tool.progress" in body
    assert "hello world" in body  # content still streams
    assert "data: [DONE]" in body


@pytest.mark.asyncio
async def test_tool_progress_suppressed_with_header():
    """With ``X-Hermes-Tool-Progress: off`` the custom frame is gone, but the
    answer (and the standard chunks a strict OpenAI client expects) still stream."""
    body = await _collect_sse(suppress=True)
    assert "event: hermes.tool.progress" not in body
    assert "hello world" in body  # answer still streams; only the UI frame is skipped
    assert "data: [DONE]" in body
