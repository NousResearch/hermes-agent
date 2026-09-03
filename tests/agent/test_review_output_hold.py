"""Advisor-enabled sessions must not stream candidates before review."""

from unittest.mock import MagicMock

from agent.review_checkpoints import (
    discard_review_output,
    release_review_output,
)
from run_agent import AIAgent


def _agent():
    agent = AIAgent.__new__(AIAgent)
    agent._review_hold_output = True
    agent._review_held_stream_chunks = []
    agent._review_held_stream_chars = 0
    agent._review_held_stream_overflow = False
    agent._review_release_interim_once = False
    agent._stream_needs_break = False
    agent._stream_think_scrubber = None
    agent._stream_context_scrubber = None
    agent._current_streamed_assistant_text = ""
    agent._stream_writer_tls = None
    agent._stream_writer_token = 0
    agent._stream_callback = None
    agent.stream_delta_callback = MagicMock()
    agent._strip_think_blocks = lambda text: text
    agent._record_streamed_assistant_text = MagicMock()
    agent._stream_hook_base_payload = lambda: {}
    return agent


def test_candidate_delta_is_held_until_release():
    agent = _agent()

    agent._fire_stream_delta("candidate answer")

    agent.stream_delta_callback.assert_not_called()
    assert agent._review_held_stream_chunks == ["candidate answer"]

    assert release_review_output(agent) is True
    agent.stream_delta_callback.assert_called_once_with("candidate answer")
    agent._record_streamed_assistant_text.assert_called_once_with("candidate answer")
    assert agent._review_held_stream_chunks == []


def test_rejected_candidate_is_discarded_without_surface_delivery():
    agent = _agent()
    agent._fire_stream_delta("candidate answer")

    discard_review_output(agent)

    agent.stream_delta_callback.assert_not_called()
    assert agent._review_held_stream_chunks == []


def test_overflow_fails_closed_for_candidate_release():
    agent = _agent()
    agent._review_held_stream_chunks = ["partial"]
    agent._review_held_stream_overflow = True

    assert release_review_output(agent) is False
    agent.stream_delta_callback.assert_not_called()
    assert agent._review_held_stream_chunks == []
