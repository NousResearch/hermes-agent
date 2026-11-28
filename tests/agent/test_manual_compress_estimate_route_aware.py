"""Manual /compress must price the request with the same wire-truth predicate as
the automatic trigger (``turn_context._preflight_request_tokens``): stale reasoning
is charged only when the route actually replays it."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from agent.conversation_compression_manual import compress_now, parse_compress_args
from agent.model_metadata import estimate_request_tokens_rough

STRIPS = ("chat_completions", "groq", "qwen/qwen3-32b", "https://api.groq.com/openai/v1")
ECHOES = ("", "deepseek", "deepseek-reasoner", "https://api.deepseek.com")


def _history():
    msgs = []
    for i in range(6):
        msgs.append({"role": "user", "content": f"ask {i}"})
        msgs.append({
            "role": "assistant",
            "content": f"done {i}",
            "reasoning_content": "r" * 40_000,
        })
    return msgs


def _agent(route):
    agent = MagicMock()
    agent._cached_system_prompt, agent.tools, agent.context_compressor = (
        "sys",
        None,
        None,
    )
    agent._compression_skipped_due_to_lock = None
    agent.api_mode, agent.provider, agent.model, agent.base_url = route
    agent._compress_context.return_value = (
        [{"role": "assistant", "content": "summary"}],
        "",
    )
    return agent


@pytest.mark.parametrize("route,wire", [(STRIPS, False), (ECHOES, True)])
def test_compress_now_prices_stale_thinking_like_the_route(route, wire):
    agent, history = _agent(route), _history()
    expected = estimate_request_tokens_rough(
        history, system_prompt="sys", charge_stale_thinking=wire
    )
    result = compress_now(agent, history, parse_compress_args(""))
    assert result.before_tokens == expected
    # The compressor receives the same pressure figure the automatic trigger would.
    assert agent._compress_context.call_args.kwargs["approx_tokens"] == expected
