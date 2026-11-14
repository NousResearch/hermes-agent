"""/v1/chat/completions ignores trailing empty assistant placeholders (#154758).

Some clients (Open WebUI) end ``messages`` with an empty assistant turn: the slot the reply
will fill. The endpoint took the last message as the user's turn, so those requests failed
with HTTP 400 "No user message found in messages".
"""

from unittest.mock import MagicMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter


def _app(adapter: APIServerAdapter) -> web.Application:
    app = web.Application()
    app["api_server_adapter"] = adapter
    app.router.add_post("/v1/chat/completions", adapter._handle_chat_completions)
    return app


async def _post(messages):
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    captured = {}

    async def _stub(**kwargs):
        captured.update(kwargs)
        return ({"final_response": "Hi.", "messages": [], "api_calls": 1},
                {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})

    async with TestClient(TestServer(_app(adapter))) as cli:
        with patch.object(adapter, "_run_agent", new=MagicMock(side_effect=_stub)):
            resp = await cli.post("/v1/chat/completions",
                                  json={"model": "hermes-agent", "messages": messages})
            return resp.status, await resp.text(), captured


@pytest.mark.asyncio
@pytest.mark.parametrize("placeholders", [
    [{"role": "assistant", "content": ""}],
    [{"role": "assistant", "content": "   "}],
    [{"role": "assistant", "content": []}],
    [{"role": "assistant"}],
    [{"role": "assistant", "content": ""}, {"role": "assistant", "content": ""}],
])
async def test_trailing_empty_assistant_is_not_taken_as_the_user_turn(placeholders):
    status, body, captured = await _post(
        [{"role": "user", "content": "Earlier"}, {"role": "assistant", "content": "Reply"},
         {"role": "user", "content": "Hello"}, *placeholders])
    assert status == 200, body
    assert captured["user_message"] == "Hello"
    assert captured["conversation_history"] == [
        {"role": "user", "content": "Earlier"}, {"role": "assistant", "content": "Reply"}]


@pytest.mark.asyncio
async def test_a_conversation_with_no_user_text_is_still_refused():
    status, body, _ = await _post([{"role": "user", "content": " "}, {"role": "assistant", "content": ""}])
    assert status == 400
    assert "No user message found" in body
