"""Gemma 4 stripped-token handling in agent.turn_response_intake.normalize_model_response."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

# Intake late-imports the loop; load it at collection, before the real-home tripwire arms.
import agent.conversation_loop
from agent.turn_response_intake import normalize_model_response

GEMMA = "coder3101/gemma-4-31B-it-heretic:featherless-ai"


def _intake(content: str, finish_reason: str = "stop", model: str = GEMMA):
    message = SimpleNamespace(content=content, tool_calls=None, finish_reason=finish_reason, reasoning=None)
    agent = MagicMock(
        model=model, api_mode="chat_completions", tools=None, quiet_mode=True,
        tool_progress_callback=None, _incomplete_scratchpad_retries=0,
    )
    agent._get_transport.return_value.normalize_response.return_value = message
    verdict = normalize_model_response(
        agent, response=MagicMock(), messages=[], api_messages=[], conversation_history=[],
        api_call_count=1, api_duration=0.1, api_start_time=0.0, api_request_id="req",
        effective_task_id="task", turn_id="turn",
    )
    calls = [(tc.name, json.loads(tc.arguments)) for tc in verdict.assistant_message.tool_calls or []]
    return verdict, calls


def test_glued_parallel_calls_become_tool_calls():
    verdict, calls = _intake("call:get_weather{city:Paris}call:get_weather{city:London}")
    assert calls == [("get_weather", {"city": "Paris"}), ("get_weather", {"city": "London"})]
    assert verdict.finish_reason == "tool_calls"
    assert verdict.assistant_message.content is None


def test_leaked_thought_prefix_is_stripped_and_call_extracted():
    verdict, calls = _intake("thought\nI need the weather in Oslo.`city`call:get_weather{city:Oslo}")
    assert calls == [("get_weather", {"city": "Oslo"})]
    assert verdict.assistant_message.content == "I need the weather in Oslo.`city`"


def test_length_truncated_reply_is_never_parsed():
    content = "call:write_file{path:a.txt,content:x}call:terminal{command:echo hi}"
    verdict, calls = _intake(content, finish_reason="length")
    assert calls == []
    assert verdict.assistant_message.content == content
    assert verdict.finish_reason == "length"


def test_non_gemma_model_is_untouched():
    content = "call:get_weather{city:Paris}"
    verdict, calls = _intake(content, model="gpt-4o")
    assert calls == []
    assert verdict.assistant_message.content == content
