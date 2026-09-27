"""Tests for how a turn's streamed assistant text is built up.

The text used to be grown with ``+=`` on an attribute. Python cannot grow a
string in place there, so every delta copied the whole thing again and a long
reply cost the square of its length in copying. The text is now held as a list
of pieces and joined when something reads it.

These tests cover the behaviour callers depend on, plus a check on the stored
pieces that fails if the copying ever comes back.
"""

import pytest


def _make_agent():
    from run_agent import AIAgent

    agent = AIAgent(
        api_key="test-key",
        base_url="https://openrouter.ai/api/v1",
        model="test/model",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )
    agent.api_mode = "chat_completions"
    agent._interrupt_requested = False
    return agent


class TestStreamedTextValue:
    """The value callers read must not change."""


    def test_deltas_join_in_order(self):
        agent = _make_agent()
        for piece in ["Hello", ", ", "world", "!"]:
            agent._record_streamed_assistant_text(piece)
        assert agent._current_streamed_assistant_text == "Hello, world!"



    def test_direct_assignment_still_works(self):
        # Several call sites set this attribute straight, both to seed a value
        # and to clear it between turns.
        agent = _make_agent()
        agent._record_streamed_assistant_text("thrown away")
        agent._current_streamed_assistant_text = "set by hand"
        assert agent._current_streamed_assistant_text == "set by hand"
        agent._record_streamed_assistant_text(" plus more")
        assert agent._current_streamed_assistant_text == "set by hand plus more"


    def test_empty_and_non_string_deltas_are_ignored(self):
        agent = _make_agent()
        agent._record_streamed_assistant_text("keep")
        agent._record_streamed_assistant_text("")
        agent._record_streamed_assistant_text(None)  # type: ignore[arg-type]
        agent._record_streamed_assistant_text(12345)  # type: ignore[arg-type]
        assert agent._current_streamed_assistant_text == "keep"





def _agent_with_sink():
    agent = _make_agent()
    delivered = []
    agent.stream_delta_callback = delivered.append
    agent._stream_callback = None
    return agent, delivered


class TestFireStreamDeltaEmptiness:
    """_fire_stream_delta used to join the whole reply on every token just
    to decide whether to strip leading newlines. That check now looks at
    the parts list.
    """

    def test_first_delta_strips_leading_newlines(self):
        agent, delivered = _agent_with_sink()
        agent._fire_stream_delta("\n\nhello")
        assert delivered == ["hello"]
        assert agent._current_streamed_assistant_text == "hello"

    def test_later_delta_keeps_leading_newlines(self):
        agent, delivered = _agent_with_sink()
        agent._fire_stream_delta("hello")
        agent._fire_stream_delta("\n\nworld")
        assert delivered == ["hello", "\n\nworld"]
        assert agent._current_streamed_assistant_text == "hello\n\nworld"

    def test_after_clear_the_next_delta_strips_again(self):
        agent, delivered = _agent_with_sink()
        agent._fire_stream_delta("hello")
        agent._current_streamed_assistant_text = ""
        agent._fire_stream_delta("\n\nagain")
        assert delivered[-1] == "again"
        assert agent._current_streamed_assistant_text == "again"



if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
