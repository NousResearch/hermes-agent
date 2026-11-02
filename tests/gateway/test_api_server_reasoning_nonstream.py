"""Reasoning on the non-streaming OpenAI-compatible routes and the Responses input parser (#99552).

Non-streaming ``/v1/chat/completions`` carries ``message.reasoning_content`` and non-streaming
``/v1/responses`` a ``reasoning`` output item, read from the assistant messages the agent
persisted; a client replaying a prior response's output list (its ``reasoning`` item included)
as the next ``input`` / ``conversation_history`` must not get a 400 or an empty user turn.
"""

from unittest.mock import patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter

REASONING = "Let me think about this carefully."


def _result(user_text: str) -> dict:
    """Transcript-shaped agent result whose assistant message carries structured reasoning."""
    return {"final_response": "42", "completed": True,
            "messages": [{"role": "user", "content": user_text},
                         {"role": "assistant", "content": "42", "reasoning": REASONING}]}


def _app() -> tuple:
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={}))
    app = web.Application()
    app["api_server_adapter"] = adapter
    app.router.add_post("/v1/chat/completions", adapter._handle_chat_completions)
    app.router.add_post("/v1/responses", adapter._handle_responses)
    app.router.add_get("/v1/responses/{response_id}", adapter._handle_get_response)
    return TestClient(TestServer(app)), adapter


@pytest.mark.asyncio
async def test_non_streaming_routes_carry_reasoning_once():
    """Non-stream chat: ``message.reasoning_content`` equals the persisted reasoning exactly
    (no delta+fallback doubling); non-stream responses: one completed ``reasoning`` item
    before the message, also present on ``GET /v1/responses/{id}`` replay."""
    client, adapter = _app()
    async with client:
        async def _fake_run_agent(**kw):
            return _result(kw["user_message"]), {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}

        with patch.object(adapter, "_run_agent", side_effect=_fake_run_agent):
            r = await client.post("/v1/chat/completions", json={
                "model": "hermes-agent", "messages": [{"role": "user", "content": "q"}]})
            assert r.status == 200
            message = (await r.json())["choices"][0]["message"]
            assert message["reasoning_content"] == REASONING
            assert REASONING not in message["content"]

            r = await client.post("/v1/responses", json={"model": "hermes-agent", "input": "q", "store": True})
            assert r.status == 200
            data = await r.json()
            assert [o["type"] for o in data["output"]] == ["reasoning", "message"]
            assert data["output"][0]["status"] == "completed"
            assert data["output"][0]["summary"] == [{"type": "summary_text", "text": REASONING}]
            replay = await (await client.get(f"/v1/responses/{data['id']}")).json()
            assert [o["type"] for o in replay["output"]] == ["reasoning", "message"]


@pytest.mark.asyncio
async def test_responses_input_ignores_echoed_reasoning_items():
    """A ``{type: reasoning}`` item replayed in ``input`` or ``conversation_history`` is skipped:
    no 400, no empty ``user`` message in the history the agent receives."""
    client, adapter = _app()
    async with client:
        captured = {}

        async def _fake_run_agent(**kw):
            captured["history"] = kw["conversation_history"]
            captured["user"] = kw["user_message"]
            return _result(kw["user_message"]), {}

        reasoning_item = {"type": "reasoning", "id": "rs_1",
                          "summary": [{"type": "summary_text", "text": "thought"}]}
        with patch.object(adapter, "_run_agent", side_effect=_fake_run_agent):
            r = await client.post("/v1/responses", json={
                "model": "hermes-agent", "store": False,
                "conversation_history": [{"role": "user", "content": "h0"}, reasoning_item,
                                         {"role": "assistant", "content": "a0"}],
                "input": [{"role": "user", "content": "first"}, reasoning_item,
                          {"type": "message", "role": "assistant",
                           "content": [{"type": "output_text", "text": "hi"}]},
                          {"role": "user", "content": "second"}]})
        assert r.status == 200, await r.text()
        assert captured["user"] == "second"
        assert [(m["role"], m["content"]) for m in captured["history"]] == [
            ("user", "h0"), ("assistant", "a0"), ("user", "first"), ("assistant", "hi")]
