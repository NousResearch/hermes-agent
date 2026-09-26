"""The /v1/responses SSE stream wraps the assistant's ``output_text`` in its content part events.

The openai SDK's ``responses.stream()`` accumulator creates ``message.content[i]`` only from
``response.content_part.added`` and indexes it on every ``output_text.delta``, so a stream
without that event raised ``IndexError`` on the first delta of every turn.
"""

import asyncio
import time
import uuid
from unittest.mock import patch

import httpx
import openai
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter, ThreadSafeAsyncQueue
from tests.gateway.test_api_server_reasoning_stream import (
    _fake_writer_env, _frames, adapter,  # noqa: F401
)


@pytest.mark.asyncio
async def test_responses_stream_wraps_message_text_in_its_content_part(adapter):
    import gateway.platforms.api_server as api_mod
    request, written, fake_response = _fake_writer_env()
    stream_q = ThreadSafeAsyncQueue()

    async def _agent():
        stream_q.put_nowait("Hello ")
        stream_q.put_nowait("world.")
        return {"final_response": "Hello world.", "completed": True}, None

    agent_task = asyncio.ensure_future(_agent())
    agent_task.add_done_callback(lambda _f: stream_q.put_nowait(None))
    with patch.object(api_mod.web, "StreamResponse", return_value=fake_response):
        await adapter._write_sse_responses(
            request=request, response_id=f"resp_{uuid.uuid4().hex[:28]}", model="hermes-agent",
            created_at=int(time.time()), stream_q=stream_q, agent_task=agent_task, agent_ref=[None],
            conversation_history=[], user_message="q", instructions=None, conversation=None,
            store=False, session_id=None)
    frames = _frames(written)
    assert [d["sequence_number"] for _e, d in frames] == list(range(len(frames)))
    message = next(d for e, d in frames
                   if e == "response.output_item.added" and d["item"]["type"] == "message")
    ids = {"item_id": message["item"]["id"], "output_index": message["output_index"],
           "content_index": 0}
    events = [e for e, _d in frames]
    # The writer may coalesce deltas; one run of them stands for the whole text.
    runs = [e for i, e in enumerate(events) if i == 0 or e != events[i - 1]]
    assert runs == [
        "response.created", "response.output_item.added", "response.content_part.added",
        "response.output_text.delta", "response.output_text.done", "response.content_part.done",
        "response.output_item.done", "response.completed"]
    parts = {e: d for e, d in frames if e.startswith("response.content_part.")}
    for event, text in (("response.content_part.added", ""),
                        ("response.content_part.done", "Hello world.")):
        assert {k: parts[event][k] for k in ids} == ids
        assert parts[event]["part"] == {"type": "output_text", "text": text, "annotations": []}


_KEY = "sk-content-part-" + "x" * 32


@pytest.mark.asyncio
@pytest.mark.parametrize("with_reasoning", [False, True], ids=["text", "after_reasoning"])
async def test_openai_sdk_responses_stream_accumulates_the_answer(with_reasoning, monkeypatch):
    """The official SDK's streaming helper, over HTTP, against the real /v1/responses handler.
    A reasoning item first moves the message to output_index 1."""
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": _KEY}))

    async def _run_agent(**kw):
        if with_reasoning:
            kw["reasoning_callback"]("thinking...")
        kw["stream_delta_callback"]("Hello ")
        kw["stream_delta_callback"]("world.")
        return ({"final_response": "Hello world.", "messages": [], "api_calls": 1},
                {"input_tokens": 3, "output_tokens": 4, "total_tokens": 7})

    monkeypatch.setattr(adapter, "_run_agent", _run_agent)
    app = web.Application()
    app.router.add_post("/v1/responses", adapter._handle_responses)
    server = TestServer(app)
    await server.start_server()
    try:
        async with httpx.AsyncClient(trust_env=False) as http_client:
            client = openai.AsyncOpenAI(
                base_url=str(server.make_url("/v1")), api_key=_KEY, max_retries=0,
                http_client=http_client)
            snapshots = []
            async with client.responses.stream(model="hermes-agent", input="hi") as stream:
                async for event in stream:
                    if event.type == "response.output_text.delta":
                        snapshots.append(event.snapshot)
                final = await stream.get_final_response()
    finally:
        await server.close()
    assert snapshots and snapshots[-1] == "Hello world."
    assert final.output_text == "Hello world."
