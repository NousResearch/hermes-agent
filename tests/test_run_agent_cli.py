"""Tests for run_agent CLI entry point (main function)."""

import pytest

from run_agent import main as _main_impl


def _call_main(query, monkeypatch):
    """Call main() with a mocked AIAgent so no real API calls are made."""
    from unittest.mock import MagicMock

    mock_agent = MagicMock()
    mock_agent.run_conversation.return_value = {
        "completed": True,
        "api_calls": 1,
        "messages": [],
        "final_response": "test response",
    }
    mock_agent._convert_to_trajectory_format.return_value = []

    monkeypatch.setattr("run_agent.AIAgent", lambda **kw: mock_agent)

    _main_impl(query=query, model="test-model", max_turns=1)


def test_query_string_passed_unchanged(monkeypatch, capsys):
    """A plain string query should pass through unchanged."""
    _call_main("hello world", monkeypatch)
    captured = capsys.readouterr()
    assert "User Query: hello world" in captured.out


def test_query_tuple_joined_with_comma(monkeypatch, capsys):
    """fire.Fire tuple query should be joined back into a string."""
    _call_main(("你好", "你是什么模型"), monkeypatch)
    captured = capsys.readouterr()
    assert "User Query: 你好, 你是什么模型" in captured.out


def test_query_list_joined_with_comma(monkeypatch, capsys):
    """List query should be joined back into a string."""
    _call_main(["hello", "what model"], monkeypatch)
    captured = capsys.readouterr()
    assert "User Query: hello, what model" in captured.out


def test_query_none_uses_default(monkeypatch, capsys):
    """None query should fall back to the default Python 3.13 prompt."""
    _call_main(None, monkeypatch)
    captured = capsys.readouterr()
    assert "Python 3.13" in captured.out
