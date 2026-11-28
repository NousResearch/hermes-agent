"""API server: a ``hold_stream`` redactor's session streams no raw text to OpenAI-compatible clients.

Real ``APIServerAdapter`` routes (aiohttp test server) building a real ``AIAgent`` against a
loopback model. With the redactor declared ``hold_stream=True``, a streaming
``/v1/chat/completions`` and ``/v1/responses`` request receive the rewritten final as their only
text: the SSE stream stays open (role chunk / ``response.created``) while the reply is held,
then carries the rewrite. (A plain transform streams the raw deltas first and appends the
rewrite: ``test_api_server_transform_llm_output.py``.)
"""

import json
import os
from pathlib import Path

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from tests.fakes.fake_llm_provider import FakeLLMServer
from tests.fakes.redactor_plugin import REWRITE, raw_leaks, redactor_home, redactor_turn


def _sse(body):
    return [json.loads(line[6:]) for line in body.splitlines() if line.startswith("data: ") and line != "data: [DONE]"]


@pytest.mark.asyncio
async def test_held_api_streams_carry_only_the_rewrite():
    from hermes_cli import plugins

    home = Path(os.environ["HERMES_HOME"])
    with FakeLLMServer(redactor_turn() + redactor_turn()) as srv:
        redactor_home(home, srv.base_url, hold=True)
        plugins._reset_plugin_managers_for_tests()
        adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={}))
        app = web.Application()
        app.router.add_post("/v1/chat/completions", adapter._handle_chat_completions)
        app.router.add_post("/v1/responses", adapter._handle_responses)
        try:
            async with TestClient(TestServer(app)) as client:
                chat = await client.post("/v1/chat/completions", json={
                    "model": "fake-model", "stream": True, "messages": [{"role": "user", "content": "config?"}]})
                chunks = [c["choices"][0]["delta"].get("content") for c in _sse(await chat.text()) if c.get("choices")]
                responses = await client.post("/v1/responses", json={
                    "model": "fake-model", "stream": True, "store": False, "input": "config?"})
                events = _sse(await responses.text())
        finally:
            plugins._reset_plugin_managers_for_tests()

    assert len(srv.main_requests()) == 4  # each request ran its commentary + tool round, then answered
    assert raw_leaks(chunks) == [] and "".join(c for c in chunks if c) == REWRITE
    deltas = [e["delta"] for e in events if e.get("type") == "response.output_text.delta"]
    assert raw_leaks(deltas) == [] and "".join(deltas) == REWRITE
    assert [e["text"] for e in events if e.get("type") == "response.output_text.done"] == [REWRITE]
