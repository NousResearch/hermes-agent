"""``literal_think_tags`` regressions from an independent review of the first revision.

Pinned: a named provider never inherits a sibling entry's opt-in (explicit false wins); a queued
gateway delta keeps the decision of the response that produced it; the iteration-limit summary,
the inline-reasoning extraction and the incomplete-scratchpad check all honour the opt-in.
All responses are fixtures; no model or platform requests are made.
"""
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agent.agent_runtime_helpers import strip_think_blocks
from agent.route_capabilities import literal_think_tags_capability
from gateway.stream_consumer import GatewayStreamConsumer

ROUTE = "https://router.example.invalid/v1"
MODEL = "fixture-model"
FLAG = "literal_think_tags"


def route(entries, requested):
    return SimpleNamespace(model=MODEL, base_url=ROUTE, provider="custom",
                           requested_provider=requested, _custom_providers=entries)


@pytest.mark.parametrize("requested", ["custom:b", "b"])
def test_named_provider_explicit_false_is_not_overridden_by_sibling(requested):
    agent = route([
        {"name": "a", "base_url": ROUTE, "models": {MODEL: {FLAG: True}}},
        {"name": "b", "base_url": ROUTE, "models": {MODEL: {FLAG: False}}},
    ], requested)
    assert {
        "capability": literal_think_tags_capability(agent),
        "visible": strip_think_blocks(agent, "<think>PRIVATE_FIXTURE</think>Answer"),
    } == {"capability": False, "visible": "Answer"}


@pytest.mark.parametrize("requested", ["custom:b", "b"])
def test_named_provider_without_opt_in_does_not_inherit_sibling_provider_flag(requested):
    agent = route([
        {"name": "a", "base_url": ROUTE, "capabilities": {FLAG: True}},
        {"name": "b", "base_url": ROUTE},
    ], requested)
    assert literal_think_tags_capability(agent) is False


def build_agent(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / ".env").write_text("", encoding="utf-8")
    (tmp_path / "config.yaml").write_text(
        f"providers:\n  local-llm:\n    api: {ROUTE}\n    models:\n      {MODEL}:\n        {FLAG}: true\n",
        encoding="utf-8",
    )
    from run_agent import AIAgent
    agent = AIAgent(model=MODEL, api_key="sk-dummy", base_url=ROUTE, quiet_mode=True,
                    skip_context_files=True, skip_memory=True, platform="discord")
    agent._disable_streaming = True
    return agent


def test_queued_completed_segment_keeps_its_producing_route_policy(tmp_path, monkeypatch):
    agent = build_agent(tmp_path, monkeypatch)
    consumer = GatewayStreamConsumer(MagicMock(), "review_chat")
    consumer.keeps_think_tags = agent._stream_keeps_think_tags
    agent.stream_delta_callback = consumer.on_delta
    agent._reset_stream_delivery_tracking()
    text = "The literal delimiter is </think>."
    agent._fire_stream_delta(text)
    consumer.on_delta(None)  # End the successful response/tool segment.

    # Worker moves route before the asynchronous consumer has drained that segment.
    agent.provider = "openrouter"
    agent.base_url = "https://openrouter.ai/api/v1"
    agent.model = "other-model"
    agent._reset_stream_delivery_tracking()
    assert agent._stream_keeps_think_tags() is False
    tick = consumer._drain_queue()
    assert tick.got_segment_break is True
    consumer._flush_think_buffer()
    assert consumer._accumulated == text


def reply(text, reasoning="PRIVATE_FIXTURE"):
    return SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content=text, reasoning=None, reasoning_content=reasoning,
                                    reasoning_details=None, tool_calls=None),
            finish_reason="stop",
        )], usage=None, model=MODEL,
    )


def test_max_iteration_summary_keeps_literal_pair(tmp_path, monkeypatch):
    from agent.chat_completion_helpers import handle_max_iterations
    agent = build_agent(tmp_path, monkeypatch)
    text = "The syntax is <think>example</think>."
    monkeypatch.setattr(agent, "_interruptible_api_call", lambda _: reply(text))
    messages = [{"role": "user", "content": "Summarize the example."}]
    result = handle_max_iterations(agent, messages, 1)
    assert result == text
    assert messages[-1]["content"] == text


@pytest.mark.parametrize("reasoning", [None, ""])
def test_literal_content_is_not_copied_to_empty_native_reasoning(tmp_path, monkeypatch, reasoning):
    agent = build_agent(tmp_path, monkeypatch)
    text = "Example: <think>literal sample</think>."
    message = reply(text, reasoning=reasoning).choices[0].message
    row = agent._build_assistant_message(message, "stop")
    assert row["content"] == text
    assert not row.get("reasoning_content")
    assert not row.get("reasoning")


def test_lone_advertised_scratchpad_tag_is_not_an_incomplete_response(tmp_path, monkeypatch):
    agent = build_agent(tmp_path, monkeypatch)
    from agent.think_scrubber import THINK_TAG_NAMES
    name = "REASONING_SCRATCHPAD"
    assert name in THINK_TAG_NAMES
    text = "<" + name + ">"
    monkeypatch.setattr(agent, "_interruptible_api_call", lambda _: reply(text))
    result = agent.run_conversation("Reply with the literal opening tag as text.")
    assert result["final_response"] == text
    assert result["api_calls"] == 1
