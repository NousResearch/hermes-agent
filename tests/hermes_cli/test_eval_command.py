"""Tests for the /eval CLI command handler dispatch.

These tests exercise the handler logic directly without standing up the full
CLI TUI.  They mock the underlying runner/storage to avoid real model calls.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from agent.evals.executor import AgentResult
from agent.evals.types import CaseResult, CaseStatus, RunSummary


# ---------------------------------------------------------------------------
# Minimal stub of the CLI class to test _handle_eval_command in isolation
# ---------------------------------------------------------------------------

def _make_handler():
    """Import and return a bound _handle_eval_command from the CLI class.

    We instantiate a lightweight object that has just enough of the CLI's
    interface for the handler to work (it only uses print, logger, and
    self._handle_eval_command).
    """
    # We need the actual method from cli.py.  Importing the full CLI class
    # pulls in heavy deps, so we extract the method and bind it to a stub.
    import importlib
    import types

    # Import the module (not the class) to get the unbound method
    cli_mod = importlib.import_module("cli")
    cls = None
    for name in dir(cli_mod):
        obj = getattr(cli_mod, name)
        if isinstance(obj, type) and hasattr(obj, "_handle_eval_command"):
            cls = obj
            break

    assert cls is not None, "Could not find CLI class with _handle_eval_command"

    # Create a lightweight stub instead of fully initializing the class
    stub = object.__new__(cls)
    return stub


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestEvalCommandDispatch:
    """Test that /eval subcommands dispatch correctly."""

    def test_eval_help_on_no_args(self, capsys):
        handler = _make_handler()
        handler._handle_eval_command("/eval")
        captured = capsys.readouterr()
        assert "/eval run" in captured.out
        assert "/eval recent" in captured.out
        assert "/eval show" in captured.out

    def test_eval_help_on_unknown_subcommand(self, capsys):
        handler = _make_handler()
        handler._handle_eval_command("/eval foobar")
        captured = capsys.readouterr()
        assert "/eval run" in captured.out

    def test_eval_list(self, capsys):
        handler = _make_handler()
        handler._handle_eval_command("/eval list")
        captured = capsys.readouterr()
        assert "Suites:" in captured.out
        assert "smoke" in captured.out
        assert "file-create-and-read" in captured.out

    @patch("agent.evals.runner.run_suite")
    @patch("agent.evals.storage.EvalStore")
    def test_eval_run_smoke(self, mock_store_cls, mock_run_suite, capsys):
        summary = RunSummary(
            run_id="test-run",
            suite_name="smoke",
            case_count=6,
            passed_count=4,
            failed_count=2,
            avg_score=0.67,
            case_results=[],
        )
        mock_run_suite.return_value = summary
        mock_store = MagicMock()
        mock_store_cls.return_value = mock_store

        handler = _make_handler()
        handler._handle_eval_command("/eval run smoke")
        captured = capsys.readouterr()

        mock_run_suite.assert_called_once_with("smoke")
        assert "test-run" in captured.out
        assert "smoke" in captured.out

    @patch("agent.evals.runner.run_single")
    @patch("agent.evals.storage.EvalStore")
    def test_eval_run_single_case(self, mock_store_cls, mock_run_single, capsys):
        summary = RunSummary(
            run_id="single-run",
            suite_name="single:file-create-and-read",
            case_count=1,
            passed_count=1,
            failed_count=0,
            avg_score=1.0,
            case_results=[],
        )
        mock_run_single.return_value = summary
        mock_store = MagicMock()
        mock_store_cls.return_value = mock_store

        handler = _make_handler()
        handler._handle_eval_command("/eval run file-create-and-read")
        captured = capsys.readouterr()

        mock_run_single.assert_called_once_with("file-create-and-read")
        assert "single-run" in captured.out

    @patch("agent.evals.storage.EvalStore")
    def test_eval_recent(self, mock_store_cls, capsys):
        mock_store = MagicMock()
        mock_store.list_runs.return_value = [
            {"id": "r1", "created_at": time.time(), "suite_name": "smoke",
             "passed_count": 6, "case_count": 6, "avg_score": 1.0, "label": ""},
        ]
        mock_store_cls.return_value = mock_store

        handler = _make_handler()
        handler._handle_eval_command("/eval recent")
        captured = capsys.readouterr()

        assert "r1" in captured.out
        assert "6/6" in captured.out

    @patch("agent.evals.storage.EvalStore")
    def test_eval_show(self, mock_store_cls, capsys):
        mock_store = MagicMock()
        mock_store.get_run_with_results.return_value = {
            "id": "r1",
            "created_at": time.time(),
            "suite_name": "smoke",
            "case_count": 1,
            "passed_count": 1,
            "failed_count": 0,
            "avg_score": 1.0,
            "label": "",
            "case_results": [
                {"case_id": "c1", "status": "passed", "total_score": 1.0,
                 "duration_ms": 50, "failure_summary": ""},
            ],
        }
        mock_store_cls.return_value = mock_store

        handler = _make_handler()
        handler._handle_eval_command("/eval show r1")
        captured = capsys.readouterr()

        assert "r1" in captured.out
        assert "✓ c1" in captured.out

    def test_eval_show_no_arg(self, capsys):
        handler = _make_handler()
        handler._handle_eval_command("/eval show")
        captured = capsys.readouterr()
        assert "Usage" in captured.out

    @patch("agent.evals.runner.run_suite")
    @patch("agent.evals.storage.EvalStore")
    def test_eval_run_unknown_target_tries_single(self, mock_store_cls, mock_run_suite, capsys):
        """When target is not a suite name, it tries run_single."""
        from agent.evals.cases import list_suites
        mock_run_suite.side_effect = KeyError("not a suite")
        mock_store = MagicMock()
        mock_store_cls.return_value = mock_store

        handler = _make_handler()
        # "nonexistent" is not a suite, handler should call run_single via the fallback
        handler._handle_eval_command("/eval run nonexistent")
        captured = capsys.readouterr()
        # Should show error since "nonexistent" is not a valid case either
        assert "Unknown suite or case" in captured.out or "error" in captured.out.lower()
