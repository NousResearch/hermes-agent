"""``platforms.api_server.status_events: false`` drops the bare ``hermes.status``
SSE frames from streaming Chat Completions while keeping content chunks intact.

The frames carry ``{"kind": "lifecycle", "text": ...}`` — the deterministic
memory-recall indicator, warnings and compaction notices. They have no
``choices`` key, so a strict OpenAI client (opencode, the Vercel AI SDK) that
validates every ``data:`` line as a ``chat.completion.chunk`` aborts the whole
request with ``Invalid input: expected array, received undefined``.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

_STATUS_FRAME = {"kind": "lifecycle", "text": "🧠 Holographic — recalled 4 memories"}


def _stream_body(platform_cfg):
    from aiohttp import web
    from gateway.config import PlatformConfig
    from gateway.platforms.api_server import APIServerAdapter, ThreadSafeAsyncQueue

    # from_dict is the production loader: a top-level platform key lands in ``extra``.
    adapter = APIServerAdapter(PlatformConfig.from_dict(platform_cfg))
    written = []

    async def fake_agent():
        return {"final_response": "done", "completed": True}, {
            "input_tokens": 1, "output_tokens": 1, "total_tokens": 2}

    async def run():
        stream_q = ThreadSafeAsyncQueue()
        stream_q.put_nowait(("__status__", _STATUS_FRAME))
        stream_q.put_nowait("done")
        stream_q.put_nowait(None)
        agent_task = asyncio.ensure_future(fake_agent())
        resp = AsyncMock(spec=web.StreamResponse)
        resp.write = AsyncMock(side_effect=lambda data: written.append(data))
        resp.prepare = AsyncMock()
        req = MagicMock()
        req.headers = {}
        with patch("gateway.platforms.api_server.web.StreamResponse", return_value=resp):
            await adapter._write_sse_chat_completion(req, "cmpl-1", "m", 1, stream_q, agent_task)

    asyncio.run(run())
    return b"".join(written).decode()


def test_status_frames_emitted_by_default():
    body = _stream_body({"enabled": True, "token": "k"})
    assert "event: hermes.status" in body
    assert "recalled 4 memories" in body
    assert '"content": "done"' in body


def test_status_events_false_suppresses_frames_but_keeps_content():
    body = _stream_body({"enabled": True, "token": "k", "status_events": False})
    assert "hermes.status" not in body
    assert "recalled 4 memories" not in body
    assert '"content": "done"' in body
