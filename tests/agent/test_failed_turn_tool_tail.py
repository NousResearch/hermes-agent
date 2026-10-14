"""A failed turn that ends after a tool round never leaves a raw tool row as the durable tail.

Thinking-exhausted / repetition truncation, content-policy refusals and non-retryable API errors
return without ``finalize_turn``. When they follow a tool round, the transcript ended at the tool
result, so the next prompt landed ``tool → user`` and strict providers resumed the stale tool
work. ``run_conversation`` closes it at the same seam that closes an open user tail.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import httpx
import pytest
from openai import BadRequestError

from hermes_state import SessionDB
from run_agent import AIAgent
from tests.agent.test_run_agent import _mock_response
from tools.registry import registry

_TOOL = "failed_turn_tool_tail_probe"
_TOOL_SCHEMA = {
    "name": _TOOL,
    "description": "probe tool for the failed-turn tool-tail test",
    "parameters": {"type": "object", "properties": {}, "required": []},
}
registry.register(
    name=_TOOL, toolset="utility", schema=_TOOL_SCHEMA,
    handler=lambda args, **_kw: "probe ok", override=True,
)


def _thinking_exhausted():
    return _mock_response(content="<think>" + "reasoning " * 40 + "</think>", finish_reason="length")


def _content_filter_refusal():
    return _mock_response(content="I can't help with that.", finish_reason="content_filter")


def _non_retryable_400():
    request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
    raise BadRequestError(
        "invalid request: bad field", response=httpx.Response(400, request=request),
        body={"error": {"message": "invalid request: bad field"}},
    )


@pytest.mark.parametrize("terminal", [_thinking_exhausted, _content_filter_refusal, _non_retryable_400])
def test_failed_turn_after_a_tool_round_closes_the_tool_tail(tmp_path, terminal):
    db = SessionDB(tmp_path / "state.db")
    try:
        with (
            patch("model_tools.get_tool_definitions", return_value=[{"type": "function", "function": _TOOL_SCHEMA}]),
            patch("model_tools.check_toolset_requirements", return_value={}),
            patch("agent.process_bootstrap.OpenAI"),
            patch("agent.model_metadata.fetch_model_metadata", return_value={}),
        ):
            agent = AIAgent(
                api_key="test-key-1234567890", base_url="https://openrouter.ai/api/v1", model="test/model",
                quiet_mode=True, skip_context_files=True, skip_memory=True, session_db=db, session_id="tool-tail",
            )
        agent.client = MagicMock()
        agent._disable_streaming = True
        agent._cached_system_prompt = "You are helpful."
        agent._use_prompt_caching = False
        agent.tool_delay = 0
        agent.compression_enabled = False
        agent.save_trajectories = False

        def call(kw):
            if kw["messages"][-1].get("role") == "tool":
                return terminal()
            tool_call = SimpleNamespace(id="call_1", type="function", function=SimpleNamespace(name=_TOOL, arguments="{}"))
            return _mock_response(content=None, finish_reason="tool_calls", tool_calls=[tool_call])

        agent._interruptible_api_call = call
        result = agent.run_conversation("run the probe")

        assert result["completed"] is False
        assert any(m.get("role") == "tool" for m in result["messages"])  # the tool round did run
        assert result["messages"][-1]["role"] == "assistant"
        assert db.latest_conversation_role("tool-tail") == "assistant"
    finally:
        db.close()
