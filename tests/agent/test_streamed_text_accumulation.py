"""Tests for how a turn's streamed assistant text is built up.

The text used to be grown with ``+=`` on an attribute. Python cannot grow a
string in place there, so every delta copied the whole thing again and a long
reply cost the square of its length in copying. The text is now held as a list
of pieces and joined when something reads it.

These tests cover the behaviour callers depend on, plus a check on the stored
pieces that fails if the copying ever comes back.
"""
from unittest.mock import patch

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

    def test_starts_empty(self):
        agent = _make_agent()
        assert agent._current_streamed_assistant_text == ""

    def test_deltas_join_in_order(self):
        agent = _make_agent()
        for piece in ["Hello", ", ", "world", "!"]:
            agent._record_streamed_assistant_text(piece)
        assert agent._current_streamed_assistant_text == "Hello, world!"

    def test_reading_twice_gives_the_same_answer(self):
        agent = _make_agent()
        agent._record_streamed_assistant_text("one ")
        agent._record_streamed_assistant_text("two")
        first = agent._current_streamed_assistant_text
        second = agent._current_streamed_assistant_text
        assert first == second == "one two"

    def test_reading_does_not_stop_later_deltas(self):
        agent = _make_agent()
        agent._record_streamed_assistant_text("before ")
        assert agent._current_streamed_assistant_text == "before "
        agent._record_streamed_assistant_text("after")
        assert agent._current_streamed_assistant_text == "before after"

    def test_direct_assignment_still_works(self):
        # Several call sites set this attribute straight, both to seed a value
        # and to clear it between turns.
        agent = _make_agent()
        agent._record_streamed_assistant_text("thrown away")
        agent._current_streamed_assistant_text = "set by hand"
        assert agent._current_streamed_assistant_text == "set by hand"
        agent._record_streamed_assistant_text(" plus more")
        assert agent._current_streamed_assistant_text == "set by hand plus more"

    def test_clearing_resets_to_empty(self):
        agent = _make_agent()
        agent._record_streamed_assistant_text("left over")
        agent._current_streamed_assistant_text = ""
        assert agent._current_streamed_assistant_text == ""
        agent._record_streamed_assistant_text("new turn")
        assert agent._current_streamed_assistant_text == "new turn"

    def test_empty_and_non_string_deltas_are_ignored(self):
        agent = _make_agent()
        agent._record_streamed_assistant_text("keep")
        agent._record_streamed_assistant_text("")
        agent._record_streamed_assistant_text(None)  # type: ignore[arg-type]
        agent._record_streamed_assistant_text(12345)  # type: ignore[arg-type]
        assert agent._current_streamed_assistant_text == "keep"

    def test_superseded_writer_is_still_fenced_out(self):
        # The single-writer guard (#65991) must keep working now that the
        # text is stored as pieces.
        agent = _make_agent()
        agent._record_streamed_assistant_text("allowed")
        with patch.object(agent, "_stream_writer_superseded", return_value=True):
            agent._record_streamed_assistant_text("blocked")
        assert agent._current_streamed_assistant_text == "allowed"


class TestStreamedTextCost:
    """Adding a delta must not touch the text already collected.

    Checked by looking at the stored pieces rather than by timing, so the
    test gives the same answer on a busy CI box as it does on a quiet one.
    """

    def test_each_delta_is_stored_as_its_own_piece(self):
        agent = _make_agent()
        for i in range(500):
            agent._record_streamed_assistant_text(f"delta-{i} ")
        # One piece per delta means nothing joined or copied the text that was
        # already there. If a delta ever rebuilds the whole string again, this
        # collapses to a single piece and the test fails.
        assert len(agent._streamed_assistant_text_parts) == 500

    def test_reading_the_text_does_not_collapse_the_pieces(self):
        # Collapsing on read would drop any delta that lands between the join
        # and the write back, so reading has to leave the pieces alone.
        agent = _make_agent()
        for i in range(10):
            agent._record_streamed_assistant_text(str(i))
        assert agent._current_streamed_assistant_text == "0123456789"
        assert len(agent._streamed_assistant_text_parts) == 10

    def test_a_long_reply_is_assembled_correctly(self):
        agent = _make_agent()
        delta = "x" * 8
        for _ in range(20000):
            agent._record_streamed_assistant_text(delta)
        assert agent._current_streamed_assistant_text == delta * 20000
        assert len(agent._streamed_assistant_text_parts) == 20000


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

    def test_fire_path_stores_one_piece_per_delta(self):
        agent, _delivered = _agent_with_sink()
        for i in range(200):
            agent._fire_stream_delta(f"d{i} ")
        assert len(agent._streamed_assistant_text_parts) == 200


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
