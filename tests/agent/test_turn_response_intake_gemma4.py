"""Gemma 4 stripped-token recovery: apply_gemma4_text_protocol and its intake wiring."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

from agent.gemma4_text_tool_parser import apply_gemma4_text_protocol
from agent.turn_response_intake import normalize_model_response

GEMMA = "coder3101/gemma-4-31B-it-heretic:featherless-ai"
TOOLS = [
    {"type": "function", "function": {"name": "get_weather", "parameters": {
        "type": "object", "properties": {"city": {"type": "string"}}}}},
]


def _message(content: str, finish_reason: str = "stop"):
    return SimpleNamespace(content=content, tool_calls=None, finish_reason=finish_reason, reasoning=None)


def _apply(content: str, *, finish_reason: str = "stop", model: str = GEMMA, provider: str = "hf",
           streamed: str = ""):
    agent = SimpleNamespace(model=model, provider=provider, tools=TOOLS, _current_streamed_assistant_text=streamed)
    message = _message(content, finish_reason)
    apply_gemma4_text_protocol(agent, message)
    calls = [(tc.name, json.loads(tc.arguments)) for tc in message.tool_calls or []]
    return message, calls, agent


def test_glued_parallel_calls_become_tool_calls():
    message, calls, _ = _apply("call:get_weather{city:Paris}call:get_weather{city:London}")
    assert calls == [("get_weather", {"city": "Paris"}), ("get_weather", {"city": "London"})]
    assert message.finish_reason == "tool_calls"
    assert message.content is None


def test_thought_preamble_on_a_tool_turn_becomes_reasoning():
    raw = "thought\nI need the weather in Oslo.`city`call:get_weather{city:Oslo}"
    message, calls, agent = _apply(raw, streamed=raw)
    assert calls == [("get_weather", {"city": "Oslo"})]
    assert message.reasoning == "I need the weather in Oslo.`city`"
    assert message.content is None
    assert agent._current_streamed_assistant_text == ""


def test_thought_prefixed_final_answer_stays_visible():
    message, calls, _ = _apply("thought\nThe user wants 2+2.\n\n2 + 2 = 4")
    assert calls == []
    assert message.reasoning is None
    assert message.content == "The user wants 2+2.\n\n2 + 2 = 4"


def test_preamble_without_thought_prefix_stays_visible():
    message, calls, _ = _apply("Let me check.call:get_weather{city:Paris}")
    assert calls == [("get_weather", {"city": "Paris"})]
    assert message.reasoning is None
    assert message.content == "Let me check."


def test_native_reasoning_is_not_overwritten():
    agent = SimpleNamespace(model=GEMMA, provider="hf", tools=TOOLS, _current_streamed_assistant_text="")
    message = _message("thought\nPlan.call:get_weather{city:Rome}")
    message.reasoning = "native reasoning"
    apply_gemma4_text_protocol(agent, message)
    assert message.reasoning == "native reasoning"
    assert message.content == "Plan."


def test_length_truncated_reply_is_never_parsed_for_calls():
    message, calls, _ = _apply("thought\nx\ncall:get_weather{city:Oslo}", finish_reason="length")
    assert calls == []
    assert message.content == "x\ncall:get_weather{city:Oslo}"
    assert message.finish_reason == "length"


def test_full_channel_tokens_become_reasoning():
    message, _, _ = _apply("<|channel>thought\nCheck the city.<channel|>It is sunny.")
    assert message.reasoning == "Check the city."
    assert message.content == "It is sunny."


def test_provider_aliases_and_custom_routes_are_active():
    for provider in ("hf", "huggingface", "custom", "custom:local-vllm"):
        _, calls, _ = _apply("call:get_weather{city:Paris}", provider=provider)
        assert calls == [("get_weather", {"city": "Paris"})], provider


def test_providers_with_native_tool_parsing_are_untouched():
    message, calls, _ = _apply("call:get_weather{city:Paris}", provider="openrouter")
    assert calls == []
    assert message.content == "call:get_weather{city:Paris}"


def test_non_gemma_model_is_untouched():
    message, calls, _ = _apply("thought\ncall:get_weather{city:Paris}", model="gpt-4o")
    assert calls == []
    assert message.content == "thought\ncall:get_weather{city:Paris}"


def test_streamed_record_follows_the_rewrite():
    raw = "Let me check.call:get_weather{city:Paris}"
    _, _, agent = _apply(raw, streamed=raw)
    assert agent._current_streamed_assistant_text == "Let me check."


def test_intake_runs_the_protocol():
    message = _message("call:get_weather{city:Paris}call:get_weather{city:London}")
    agent = MagicMock(model=GEMMA, provider="hf", api_mode="chat_completions", tools=TOOLS, quiet_mode=True,
                      tool_progress_callback=None, _incomplete_scratchpad_retries=0)
    agent._get_transport.return_value.normalize_response.return_value = message
    verdict = normalize_model_response(
        agent, response=MagicMock(), messages=[], api_messages=[], conversation_history=[],
        api_call_count=1, api_duration=0.1, api_start_time=0.0, api_request_id="req",
        effective_task_id="task", turn_id="turn",
    )
    assert [tc.name for tc in verdict.assistant_message.tool_calls] == ["get_weather", "get_weather"]
    assert verdict.finish_reason == "tool_calls"
