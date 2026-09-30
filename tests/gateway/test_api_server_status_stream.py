"""Agent status lines reach the OpenAI-compatible SSE writers as ``hermes.status`` events (#85426):
the auto-recovery countdown must not leave an API client staring at a silent socket."""

import asyncio
import time
import uuid
from unittest.mock import patch

import pytest

from gateway.platforms.api_server import ThreadSafeAsyncQueue
from tests.gateway.test_api_server_reasoning_stream import (
    _fake_writer_env, _frames, _stub_create_agent_runtime, adapter,  # noqa: F401
)

_LADDER = "⏳ Provider temporarily unavailable — retrying automatically in 15s (cycle 1/5); cancel the request to stop"


@pytest.mark.asyncio
async def test_chat_completions_stream_forwards_agent_status_callback(adapter, monkeypatch):
    """``_spawn_stream_agent`` wires ``status_callback`` into ``AIAgent(...)``; a status line arrives
    as an ``event: hermes.status`` frame and never as answer ``content``."""
    import gateway.platforms.api_server as api_mod

    class FakeAgent:
        def __init__(self, **kwargs):
            self._status_callback = kwargs.get("status_callback")
            self._stream_delta_callback = kwargs.get("stream_delta_callback")
            self.session_id = kwargs.get("session_id")

        def run_conversation(self, **kwargs):
            self._status_callback("lifecycle", _LADDER)
            self._stream_delta_callback("answer")
            return {"final_response": "answer", "completed": True}

    _stub_create_agent_runtime(monkeypatch, FakeAgent)
    monkeypatch.setattr(adapter, "_ensure_session_db", lambda: None)
    request, written, fake_response = _fake_writer_env()
    stream_q = ThreadSafeAsyncQueue()
    agent_task, agent_ref = adapter._spawn_stream_agent(
        stream_q, user_message="q", conversation_history=[], session_id="api-session")
    with patch.object(api_mod.web, "StreamResponse", return_value=fake_response):
        await adapter._write_sse_chat_completion(
            request, "chatcmpl-x", "hermes-agent", int(time.time()), stream_q, agent_task, agent_ref)
    frames = _frames(written)
    assert [d for e, d in frames if e == "hermes.status"] == [{"kind": "lifecycle", "text": _LADDER}]
    content = "".join(d["choices"][0]["delta"].get("content") or "" for e, d in frames if e is None)
    assert content == "answer"


@pytest.mark.asyncio
async def test_responses_stream_emits_status_event_outside_output_items(adapter):
    import gateway.platforms.api_server as api_mod
    request, written, fake_response = _fake_writer_env()
    stream_q = ThreadSafeAsyncQueue()

    async def _agent():
        stream_q.put_nowait(("__status__", {"kind": "lifecycle", "text": _LADDER}))
        stream_q.put_nowait("final text")
        return {"final_response": "final text", "completed": True}, None

    agent_task = asyncio.ensure_future(_agent())
    agent_task.add_done_callback(lambda _f: stream_q.put_nowait(None))
    with patch.object(api_mod.web, "StreamResponse", return_value=fake_response):
        await adapter._write_sse_responses(
            request=request, response_id=f"resp_{uuid.uuid4().hex[:28]}", model="hermes-agent",
            created_at=int(time.time()), stream_q=stream_q, agent_task=agent_task, agent_ref=[None],
            conversation_history=[], user_message="q", instructions=None, conversation=None,
            store=False, session_id=None)
    frames = _frames(written)
    assert [d for e, d in frames if e == "hermes.status"] == [{"kind": "lifecycle", "text": _LADDER}]
    completed = next(d for e, d in frames if e == "response.completed")
    assert [o["type"] for o in completed["response"]["output"]] == ["message"]
