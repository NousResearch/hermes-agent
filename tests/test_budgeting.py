"""Tests for summarize_turns() in agent.modules.budgeting."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from agent.modules.budgeting import summarize_turns


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_turns(n: int) -> list[dict]:
    return [{"role": "user" if i % 2 == 0 else "assistant", "content": f"msg {i}"} for i in range(n)]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_empty_turns_returns_empty_summary():
    summary, recent = summarize_turns([], 10)
    assert summary == ""
    assert recent == []


def test_under_budget_returns_input_unchanged():
    turns = _make_turns(5)
    summary, recent = summarize_turns(turns, keep_recent=10)
    assert summary == ""
    assert recent == turns
    # Must not mutate the input list
    assert len(recent) == 5


def test_under_budget_no_llm_call():
    turns = _make_turns(5)
    with patch("agent.modules.budgeting._invoke_summarizer") as mock_inv:
        summarize_turns(turns, keep_recent=10)
        mock_inv.assert_not_called()


def test_over_budget_calls_llm_and_returns_summary():
    turns = _make_turns(15)
    mock_summary = "Mocked conversation summary"

    with patch("agent.modules.budgeting._invoke_summarizer", return_value=mock_summary) as mock_inv:
        summary, recent = summarize_turns(turns, keep_recent=5)

    mock_inv.assert_called_once()
    assert summary == mock_summary
    assert len(recent) == 5


def test_recent_turns_are_last_chronologically():
    turns = _make_turns(15)

    with patch("agent.modules.budgeting._invoke_summarizer", return_value="summary"):
        _, recent = summarize_turns(turns, keep_recent=5)

    # The last 5 turns (indices 10..14) should be kept
    assert recent == turns[10:]
    assert recent[0]["content"] == "msg 10"
    assert recent[-1]["content"] == "msg 14"


def test_custom_model_passed_through():
    turns = _make_turns(12)

    with patch("agent.modules.budgeting._invoke_summarizer", return_value="s") as mock_inv:
        summarize_turns(turns, keep_recent=5, summarizer_model="claude-opus-4-5")

    _, kwargs = mock_inv.call_args
    # _invoke_summarizer(conversation_text, model) — positional args
    args, _ = mock_inv.call_args
    assert args[1] == "claude-opus-4-5"


def test_exactly_at_budget_no_compression():
    turns = _make_turns(10)
    with patch("agent.modules.budgeting._invoke_summarizer") as mock_inv:
        summary, recent = summarize_turns(turns, keep_recent=10)
    mock_inv.assert_not_called()
    assert summary == ""
    assert len(recent) == 10
