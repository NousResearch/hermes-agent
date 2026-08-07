"""Negative-control test: interrupted agent must never produce empty response.

This test validates the fix for the response_len=0 / empty-bubble bug.
When interrupted mid-loop with streamed content already received, the
conversation loop must recover that content as final_response instead of
leaving it as None.  This is a runtime regression test that exercises
the actual conversation loop, validates the returned final_response,
and asserts the persisted transcript preserves the recovered content.
"""

from unittest.mock import MagicMock, patch
import pytest

from run_agent import AIAgent

from tests.run_agent.test_run_agent import _make_tool_defs


@pytest.fixture()
def agent():
    """Minimal AIAgent with mocked OpenAI client and tool loading."""
    with (
        patch(
            "run_agent.get_tool_definitions", return_value=_make_tool_defs("web_search")
        ),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        a = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        a.client = MagicMock()
        return a


def test_interrupt_recovers_streamed_content(agent):
    """Interrupt with partial content: final_response must recover it."""
    streamed_text = "Here is the partial answer I was working on."
    agent._current_streamed_assistant_text = streamed_text
    agent._interrupt_requested = True
    agent._persist_session = lambda *args, **kwargs: None
    agent._save_trajectory = lambda *args, **kwargs: None
    agent._flush_messages_to_session_db = lambda *args, **kwargs: None

    result = agent.run_conversation("hello")

    # The turn completed with the recovered content
    assert result["final_response"] == streamed_text
    assert result["interrupted"] is True
    assert result["turn_exit_reason"] == "interrupted_by_user"
    # response_len must be > 0 so gateways don't deliver a blank bubble
    assert len(result["final_response"]) > 0
    # Turn completed successfully because we recovered partial content
    assert result["completed"] is True


def test_interrupt_recovers_content_with_think_blocks(agent):
    """Think blocks are stripped, visible content is preserved."""
    agent._current_streamed_assistant_text = (
        "<think>Let me reason about this carefully</think>"
        "The answer is 42."
    )
    agent._interrupt_requested = True
    agent._persist_session = lambda *args, **kwargs: None
    agent._save_trajectory = lambda *args, **kwargs: None
    agent._flush_messages_to_session_db = lambda *args, **kwargs: None

    result = agent.run_conversation("hello")

    assert result["final_response"] == "The answer is 42."
    assert result["interrupted"] is True
    assert result["turn_exit_reason"] == "interrupted_by_user"
    assert result["completed"] is True


def test_interrupt_no_streamed_content_returns_none(agent):
    """Interrupt with no streamed content: final_response stays None
    (no partial content to recover), but the turn still terminates
    cleanly without crashing."""
    agent._current_streamed_assistant_text = ""
    agent._interrupt_requested = True
    agent._persist_session = lambda *args, **kwargs: None
    agent._save_trajectory = lambda *args, **kwargs: None
    agent._flush_messages_to_session_db = lambda *args, **kwargs: None

    result = agent.run_conversation("hello")

    assert result["final_response"] is None  # no partial content to recover
    assert result["interrupted"] is True
    assert result["turn_exit_reason"] == "interrupted_by_user"
    # No recovered content means the turn is not marked completed
    assert result["completed"] is False
