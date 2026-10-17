"""A failed agent run's error text is redacted on streaming ``/v1/chat/completions`` too.

The non-streaming reply, the ``X-Hermes-Error`` header and both ``/v1/responses`` paths pass the
error through ``_redact_api_error_text`` (#37733); the final SSE chunk of a streamed chat completion
put the raw text in ``error.message`` and ``hermes.error``, so a provider error carrying a key was
masked with ``stream: false`` and sent verbatim with ``stream: true``.
"""

import json
from unittest.mock import MagicMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter

SECRET = "sk-proj-chatstreamleak1234567890abcdef"
ERROR = f"provider auth failed OPENAI_API_KEY={SECRET}"


async def _post(tmp_path, monkeypatch, stream, *, create_agent):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": "test-local-key"}))
    app = web.Application()
    app.router.add_post("/v1/chat/completions", adapter._handle_chat_completions)
    with patch.object(adapter, "_create_agent", side_effect=create_agent):
        async with TestClient(TestServer(app)) as client:
            response = await client.post(
                "/v1/chat/completions", headers={"Authorization": "Bearer test-local-key"},
                json={"messages": [{"role": "user", "content": "hi"}], "stream": stream})
            return response.status, await response.text()


def _terminal_frame(wire):
    frames = [json.loads(line[6:]) for line in wire.splitlines()
              if line.startswith("data: ") and line != "data: [DONE]"]
    return frames[-1]


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_failed_result_error_is_redacted(tmp_path, monkeypatch, stream):
    agent = MagicMock(session_id="session", _last_compaction_in_place=False, session_prompt_tokens=1,
                      session_completion_tokens=0, session_total_tokens=1)
    agent.run_conversation.return_value = {
        "final_response": "", "completed": False, "failed": True, "error": ERROR,
        "messages": [], "api_calls": 1}

    status, wire = await _post(tmp_path, monkeypatch, stream, create_agent=lambda **_: agent)

    assert SECRET not in wire
    assert "OPENAI_API_KEY=" in wire
    if stream:
        terminal = _terminal_frame(wire)
        assert terminal["choices"][0]["finish_reason"] == "error"
        assert "OPENAI_API_KEY=" in terminal["error"]["message"]
        assert terminal["hermes"]["failed"] is True
    else:
        assert status == 502


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_raised_agent_error_is_redacted(tmp_path, monkeypatch, stream):
    def create_agent(**_):
        raise RuntimeError(ERROR)

    status, wire = await _post(tmp_path, monkeypatch, stream, create_agent=create_agent)

    assert SECRET not in wire
    assert "OPENAI_API_KEY=" in wire
    if stream:
        terminal = _terminal_frame(wire)
        assert terminal["choices"][0]["finish_reason"] == "error"
        assert terminal["error"]["type"] == "RuntimeError"
    else:
        assert status == 500
