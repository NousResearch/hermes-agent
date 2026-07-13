from __future__ import annotations

from run_agent import AIAgent


def _agent_without_init():
    agent = AIAgent.__new__(AIAgent)
    agent._stream_needs_break = False
    agent._stream_think_scrubber = None
    agent._stream_context_scrubber = None
    agent.stream_delta_callback = lambda text: False
    agent._stream_callback = None
    agent._current_streamed_assistant_text = ""
    agent._current_visible_streamed_assistant_text = ""
    agent._strip_think_blocks = lambda content: content
    return agent


def test_stream_delta_records_received_text_without_marking_visible() -> None:
    agent = _agent_without_init()

    AIAgent._fire_stream_delta(agent, "draft body")

    assert agent._current_streamed_assistant_text == "draft body"
    assert agent._current_visible_streamed_assistant_text == ""


def test_stream_delta_none_return_remains_visible_for_legacy_callbacks() -> None:
    agent = _agent_without_init()
    agent.stream_delta_callback = lambda text: None

    AIAgent._fire_stream_delta(agent, "visible body")

    assert agent._current_streamed_assistant_text == "visible body"
    assert agent._current_visible_streamed_assistant_text == "visible body"


def test_interim_content_was_streamed_uses_visible_text_only() -> None:
    agent = _agent_without_init()
    agent._current_streamed_assistant_text = "hidden draft"
    agent._current_visible_streamed_assistant_text = ""

    assert AIAgent._interim_content_was_streamed(agent, "hidden draft") is False
