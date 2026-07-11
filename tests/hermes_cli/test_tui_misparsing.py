"""Tests for the '-tui' (single-dash) misparsing warning.

``hermes -tui`` silently sets ``--toolsets=ui`` instead of launching the TUI,
because ``-t`` is the short flag for ``--toolsets``.  The helper
``_warn_probable_tui_misparsing`` detects this and offers to relaunch with
``--tui``.
"""

import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from hermes_cli.main import _warn_probable_tui_misparsing


class TestWarnProbableTuiMisparsing:
    """Unit tests for ``_warn_probable_tui_misparsing``."""

    def test_returns_false_when_toolsets_not_ui(self):
        """No warning when toolsets is something other than 'ui'."""
        args = SimpleNamespace(toolsets="terminal,file", tui=False)
        assert _warn_probable_tui_misparsing(args) is False

    def test_returns_false_when_toolsets_none(self):
        """No warning when --toolsets was not passed at all."""
        args = SimpleNamespace(toolsets=None, tui=False)
        assert _warn_probable_tui_misparsing(args) is False

    def test_returns_false_when_tui_flag_also_set(self):
        """No warning when both --toolsets=ui and --tui are present.

        The user explicitly passed both; they know what they're doing.
        """
        args = SimpleNamespace(toolsets="ui", tui=True)
        assert _warn_probable_tui_misparsing(args) is False

    def test_returns_false_when_no_toolsets_attr(self):
        """Graceful when args has no 'toolsets' attribute at all."""
        args = SimpleNamespace(tui=False)
        assert _warn_probable_tui_misparsing(args) is False

    def test_warns_on_stderr_in_noninteractive(self, capsys):
        """In a non-TTY context, prints a warning to stderr and returns False."""
        args = SimpleNamespace(toolsets="ui", tui=False)
        with patch("sys.stdin") as mock_stdin, patch("sys.stdout") as mock_stdout:
            mock_stdin.isatty.return_value = False
            mock_stdout.isatty.return_value = False
            result = _warn_probable_tui_misparsing(args)
        assert result is False
        captured = capsys.readouterr()
        assert "Warning" in captured.err
        assert "hermes --tui" in captured.err

    def test_user_accepts_original_toolsets_ui(self, monkeypatch, capsys):
        """User types 'y' → continues with --toolsets=ui, returns False."""
        args = SimpleNamespace(toolsets="ui", tui=False)
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)
        monkeypatch.setattr("sys.stdout.isatty", lambda: True)
        monkeypatch.setattr("builtins.input", lambda _="": "y")
        result = _warn_probable_tui_misparsing(args)
        assert result is False
        captured = capsys.readouterr()
        assert "--toolsets=ui" in captured.out

    def test_user_declines_relaunches_tui(self, monkeypatch, capsys):
        """User types 'n' → returns True (caller should set args.tui=True)."""
        args = SimpleNamespace(toolsets="ui", tui=False)
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)
        monkeypatch.setattr("sys.stdout.isatty", lambda: True)
        monkeypatch.setattr("builtins.input", lambda _="": "n")
        result = _warn_probable_tui_misparsing(args)
        assert result is True
        captured = capsys.readouterr()
        assert "Launching --tui" in captured.out

    def test_user_enter_defaults_to_no(self, monkeypatch):
        """Empty input (just pressing Enter) defaults to 'No' → relaunch."""
        args = SimpleNamespace(toolsets="ui", tui=False)
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)
        monkeypatch.setattr("sys.stdout.isatty", lambda: True)
        monkeypatch.setattr("builtins.input", lambda _="": "")
        result = _warn_probable_tui_misparsing(args)
        assert result is True

    def test_ctrl_c_exits_cleanly(self, monkeypatch):
        """KeyboardInterrupt during input exits with code 0."""
        args = SimpleNamespace(toolsets="ui", tui=False)
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)
        monkeypatch.setattr("sys.stdout.isatty", lambda: True)
        monkeypatch.setattr("builtins.input", lambda _="": (_ for _ in ()).throw(KeyboardInterrupt))
        with pytest.raises(SystemExit) as exc_info:
            _warn_probable_tui_misparsing(args)
        assert exc_info.value.code == 0

    def test_eof_exits_cleanly(self, monkeypatch):
        """EOFError during input exits with code 0."""
        args = SimpleNamespace(toolsets="ui", tui=False)
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)
        monkeypatch.setattr("sys.stdout.isatty", lambda: True)
        monkeypatch.setattr("builtins.input", lambda _="": (_ for _ in ()).throw(EOFError))
        with pytest.raises(SystemExit) as exc_info:
            _warn_probable_tui_misparsing(args)
        assert exc_info.value.code == 0
