"""Tests for the /failures CLI command handler dispatch."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest


def _make_handler():
    """Extract handler from CLI class (same pattern as test_eval_command.py)."""
    import importlib
    cli_mod = importlib.import_module("cli")
    cls = None
    for name in dir(cli_mod):
        obj = getattr(cli_mod, name)
        if isinstance(obj, type) and hasattr(obj, "_handle_failures_command"):
            cls = obj
            break
    assert cls is not None, "Could not find CLI class with _handle_failures_command"
    stub = object.__new__(cls)
    return stub


class TestFailuresCommandDispatch:
    def test_help_on_no_args(self, capsys):
        handler = _make_handler()
        handler._handle_failures_command("/failures")
        captured = capsys.readouterr()
        assert "/failures recent" in captured.out
        assert "/failures top" in captured.out
        assert "/failures show" in captured.out

    def test_help_on_unknown_subcommand(self, capsys):
        handler = _make_handler()
        handler._handle_failures_command("/failures foobar")
        captured = capsys.readouterr()
        assert "/failures recent" in captured.out

    @patch("agent.failure_analysis.storage.FailureStore")
    def test_recent(self, mock_store_cls, capsys):
        mock_store = MagicMock()
        mock_store.list_recent.return_value = [
            {"id": "f-1", "severity": "medium", "created_at": time.time(),
             "failure_type": "eval", "failure_subtype": "failed_check",
             "summary": "check failed", "fingerprint": "fp12345678901234"},
        ]
        mock_store_cls.return_value = mock_store

        handler = _make_handler()
        handler._handle_failures_command("/failures recent")
        captured = capsys.readouterr()
        assert "eval.failed_check" in captured.out

    @patch("agent.failure_analysis.storage.FailureStore")
    def test_top(self, mock_store_cls, capsys):
        mock_store = MagicMock()
        mock_store.top_fingerprints.return_value = [
            {"count": 3, "failure_type": "tool", "failure_subtype": "timeout",
             "fingerprint": "fp-aaa", "summary": "timed out",
             "latest_at": time.time(), "tool_name": None},
        ]
        mock_store_cls.return_value = mock_store

        handler = _make_handler()
        handler._handle_failures_command("/failures top")
        captured = capsys.readouterr()
        assert "tool.timeout" in captured.out
        assert "3x" in captured.out

    @patch("agent.failure_analysis.storage.FailureStore")
    def test_show(self, mock_store_cls, capsys):
        mock_store = MagicMock()
        mock_store.get_by_fingerprint.return_value = [
            {"failure_type": "eval", "failure_subtype": "regression",
             "severity": "high", "created_at": time.time(),
             "summary": "score dropped", "tool_name": None, "model": None,
             "source_surface": "eval", "eval_run_id": "run-001",
             "session_id": None},
        ]
        mock_store_cls.return_value = mock_store

        handler = _make_handler()
        handler._handle_failures_command("/failures show fp-test")
        captured = capsys.readouterr()
        assert "fp-test" in captured.out
        assert "Occurrences: 1" in captured.out

    def test_show_no_arg(self, capsys):
        handler = _make_handler()
        handler._handle_failures_command("/failures show")
        captured = capsys.readouterr()
        assert "Usage" in captured.out
