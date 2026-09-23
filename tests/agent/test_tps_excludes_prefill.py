"""The status-bar tokens/s reads decode speed, not whole-call time.

A local model serving a 100k-token prompt spends most of the call in prefill, and under
speculative decoding (MTP) the stream still carries every accepted token in ``usage``, so
the only wrong ingredient in ``output_tokens / seconds`` was the seconds. The streaming call
stamps ``first_token_at`` on the first delta that carries text, reasoning or a tool call
(``first_chunk_at`` fires on vLLM's empty role delta, sent before prefill starts) and hands
the first-token-to-end span to the usage recorder as ``_last_api_decode_seconds``; a call
that streamed nothing falls back to the whole duration.
"""

from __future__ import annotations

import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from run_agent import AIAgent
from tests.agent.test_first_chunk_at_hook import _make_stream_chunk
from tests.agent.test_run_agent import _make_tool_defs


@pytest.fixture()
def agent():
    with (
        patch("model_tools.get_tool_definitions", return_value=_make_tool_defs("web_search")),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        a = AIAgent(api_key="test-key-1234567890", base_url="http://127.0.0.1:18300/v1",
                    quiet_mode=True, skip_context_files=True, skip_memory=True)
        a.client = MagicMock()
        a._cached_system_prompt = "You are helpful."
        a._use_prompt_caching = False
        a.compression_enabled = False
        a.save_trajectories = False
        a.api_mode = "chat_completions"
        a._interrupt_requested = False
        a._last_api_first_chunk_at = None
        a._last_api_decode_seconds = None
        return a


PREFILL_SECONDS = 0.3


def _vllm_like_stream():
    """Role delta at once, tokens only after prefill, as vLLM streams."""
    yield _make_stream_chunk(content=None)
    time.sleep(PREFILL_SECONDS)
    yield _make_stream_chunk(content="Hello")
    yield _make_stream_chunk(content="!", finish_reason="stop", model="qwen3.8-flash-next")


@patch("run_agent.AIAgent._create_request_openai_client")
@patch("run_agent.AIAgent._close_request_openai_client")
def test_decode_span_starts_at_the_first_token_not_the_role_delta(_close, create, agent):
    client = MagicMock()
    client.chat.completions.create.return_value = _vllm_like_stream()
    create.return_value = client

    started = time.time()
    response = agent._interruptible_streaming_api_call({})
    whole_call = time.time() - started

    assert response.choices[0].message.content == "Hello!"
    assert agent._last_api_first_chunk_at - started < PREFILL_SECONDS / 2  # the empty role delta
    assert whole_call >= PREFILL_SECONDS
    assert 0.0 <= agent._last_api_decode_seconds < PREFILL_SECONDS / 2


def _usage(output_tokens):
    return SimpleNamespace(prompt_tokens=100_000, completion_tokens=output_tokens,
                           total_tokens=100_000 + output_tokens, prompt_tokens_details=None,
                           completion_tokens_details=None)


def _record(agent, api_duration, decode_seconds):
    from agent import turn_usage
    agent._last_api_decode_seconds = decode_seconds
    turn_usage.record_response_usage(
        agent, SimpleNamespace(usage=_usage(300), id=None, model="qwen3.8-flash-next"),
        messages=[{"role": "user", "content": "hi"}], api_call_count=1,
        api_duration=api_duration, compression_attempts=0, max_compression_attempts=3)


def test_usage_recorder_keeps_whole_call_and_decode_spans_side_by_side(agent):
    _record(agent, api_duration=45.0, decode_seconds=6.0)
    _record(agent, api_duration=5.0, decode_seconds=None)  # nothing streamed: whole call

    assert list(agent._api_latency_history) == [45.0, 5.0]
    assert list(agent._api_decode_history) == [6.0, 5.0]
    assert list(agent._api_output_history) == [300, 300]


def test_status_bar_tps_is_decode_speed_after_a_long_prefill(agent):
    from tui_gateway.server import _get_usage

    _record(agent, api_duration=45.0, decode_seconds=6.0)

    usage = _get_usage(agent)
    assert usage["avg_latency_s"] == 45.0
    assert usage["avg_tps"] == 50.0  # 300 tokens in 6 s of generation, not 300 / 45
