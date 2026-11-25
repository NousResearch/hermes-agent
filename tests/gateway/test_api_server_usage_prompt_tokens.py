"""``platforms.api_server.usage_prompt_tokens: context`` reports the last prompt's size as the
OpenAI-compatible ``prompt_tokens``/``input_tokens`` instead of the turn's summed input, for
clients that read it as context use (gauges, auto-compaction). Default: unchanged."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent.turn_usage import last_prompt_usage
from gateway.platforms.api_server_openai_routes import (
    _chat_usage_payload, _responses_usage_payload)

# One turn, three model calls on a ~9k-token context: the input counters sum 27k.
TURN = {"input_tokens": 27000, "output_tokens": 300, "total_tokens": 27300,
        "last_prompt_tokens": 9000}


def test_default_reports_the_turn_sum():
    assert _chat_usage_payload(TURN) == {
        "prompt_tokens": 27000, "completion_tokens": 300, "total_tokens": 27300}
    assert _responses_usage_payload(TURN) == {
        "input_tokens": 27000, "output_tokens": 300, "total_tokens": 27300}


def test_context_reports_the_last_prompt():
    assert _chat_usage_payload(TURN, True) == {
        "prompt_tokens": 9000, "completion_tokens": 300, "total_tokens": 9300}
    assert _responses_usage_payload(TURN, True) == {
        "input_tokens": 9000, "output_tokens": 300, "total_tokens": 9300}


def test_context_falls_back_to_the_sum_when_no_prompt_was_recorded():
    usage = {key: value for key, value in TURN.items() if key != "last_prompt_tokens"}
    assert _chat_usage_payload(usage, True)["prompt_tokens"] == 27000


@pytest.mark.parametrize("value, expected", [
    (9000, {"last_prompt_tokens": 9000}), (0, {}), (None, {}), (True, {}), ("9000", {})])
def test_last_prompt_usage_only_reports_a_recorded_size(value, expected):
    agent = MagicMock()
    agent.context_compressor.last_prompt_tokens = value
    assert last_prompt_usage(agent) == expected


def test_last_prompt_usage_without_a_compressor():
    assert last_prompt_usage(object()) == {}


def _streamed_usage(platform_cfg):
    """The ``usage`` block of the final Chat Completions chunk for a turn reporting ``TURN``."""
    from aiohttp import web
    from gateway.config import PlatformConfig
    from gateway.platforms.api_server import APIServerAdapter, ThreadSafeAsyncQueue

    # from_dict is the production loader: a top-level platform key lands in ``extra``.
    adapter = APIServerAdapter(PlatformConfig.from_dict(platform_cfg))
    written = []

    async def fake_agent():
        return {"final_response": "done", "completed": True}, dict(TURN)

    async def run():
        stream_q = ThreadSafeAsyncQueue()
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
    frames = [line[6:] for line in b"".join(written).decode().splitlines()
              if line.startswith("data: {")]
    usages = [json.loads(frame)["usage"] for frame in frames if '"usage"' in frame]
    assert usages, frames
    return usages[-1]


def test_streamed_chat_completion_keeps_the_turn_sum_by_default():
    assert _streamed_usage({"enabled": True, "token": "k"})["prompt_tokens"] == 27000


def test_streamed_chat_completion_reports_the_context_when_configured():
    usage = _streamed_usage({"enabled": True, "token": "k", "usage_prompt_tokens": "context"})
    assert usage == {"prompt_tokens": 9000, "completion_tokens": 300, "total_tokens": 9300}


@pytest.mark.asyncio
async def test_run_agent_records_the_last_prompt_size():
    from tests.gateway.test_api_server import _make_adapter

    adapter = _make_adapter()
    agent = MagicMock()
    agent.run_conversation.return_value = {"final_response": "ok"}
    agent.session_prompt_tokens, agent.session_completion_tokens = 27000, 300
    agent.session_total_tokens = 27300
    agent.context_compressor.last_prompt_tokens = 9000
    with patch.object(adapter, "_create_agent", return_value=agent):
        _, usage = await adapter._run_agent(
            user_message="hello", conversation_history=[], session_id="s-1")
    assert usage["input_tokens"] == 27000
    assert usage["last_prompt_tokens"] == 9000
