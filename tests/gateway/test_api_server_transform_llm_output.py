"""``transform_llm_output`` on the api_server streaming routes (#119323).

The deltas carry the raw model reply; the hook rewrites the final afterwards. Streaming
``/v1/chat/completions`` and ``/v1/responses`` must still deliver the rewrite, like their
non-streaming twins do.
"""

import json
from contextlib import ExitStack
from unittest.mock import patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from agent.turn_finalizer import apply_llm_output_transform
from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter

RAW = "Original model reply."
TRANSFORMS = {
    "append": lambda text: f"{text}\n\n[PLUGIN footer]",
    "replace": lambda text: f"[PLUGIN warning] {text}",
}


def _app_with_fake_turn(adapter, transform, stack):
    class FakeAgent:
        def __init__(self, cb):
            self.cb, self.session_id, self.model, self.platform = cb, "s", "fake", "api_server"

        def run_conversation(self, user_message, conversation_history, task_id=None, **kw):
            for part in ("Original ", "model ", "reply."):
                self.cb(part)
            final, transformed, pre = apply_llm_output_transform(self, RAW, turn_id="t1")
            return {"final_response": final, "response_transformed": transformed,
                    "pre_transform_response": pre, "messages": [], "api_calls": 1, "completed": True}

        def __getattr__(self, name):
            return None

    def invoke_hook(name, **kw):
        return [transform(kw["response_text"])] if name == "transform_llm_output" else []

    app = web.Application()
    app.router.add_post("/v1/chat/completions", adapter._handle_chat_completions)
    app.router.add_post("/v1/responses", adapter._handle_responses)
    finish = lambda agent, result, session_id, **kw: (result, {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2})
    stack.enter_context(patch("hermes_cli.lifecycle.invoke_hook", side_effect=invoke_hook))
    stack.enter_context(patch.object(
        adapter, "_create_agent", side_effect=lambda **kw: FakeAgent(kw.get("stream_delta_callback"))))
    stack.enter_context(patch.object(adapter, "_finish_turn_result", side_effect=finish))
    return app


def _sse_payloads(body):
    return [json.loads(line[6:]) for line in body.splitlines()
            if line.startswith("data: ") and line != "data: [DONE]"]


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", sorted(TRANSFORMS))
async def test_streaming_routes_deliver_transformed_final(kind):
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={}))
    expected = TRANSFORMS[kind](RAW)
    with ExitStack() as stack:
        app = _app_with_fake_turn(adapter, TRANSFORMS[kind], stack)
        async with TestClient(TestServer(app)) as cli:
            r = await cli.post("/v1/chat/completions", json={
                "model": "m", "messages": [{"role": "user", "content": "hi"}], "stream": True})
            chunks = _sse_payloads(await r.text())
            streamed = "".join(c["choices"][0]["delta"].get("content") or "" for c in chunks if c.get("choices"))
            if kind == "append":
                assert streamed == expected
            else:
                assert streamed.startswith(RAW) and streamed.endswith(expected)

            r = await cli.post("/v1/responses", json={"model": "m", "input": "hi", "stream": True, "store": False})
            events = _sse_payloads(await r.text())
            done = [e["text"] for e in events if e.get("type") == "response.output_text.done"]
            completed = [e["response"] for e in events if e.get("type") == "response.completed"]
            assert done == [expected]
            assert completed[0]["output"][-1]["content"][0]["text"] == expected
            if kind == "append":
                deltas = "".join(e["delta"] for e in events if e.get("type") == "response.output_text.delta")
                assert deltas == expected
