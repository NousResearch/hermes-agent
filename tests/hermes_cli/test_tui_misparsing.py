"""Tests for the '-tui' (single-dash) misparsing warning.

``hermes -tui`` silently sets ``--toolsets=ui`` instead of launching the TUI,
because ``-t`` is the short flag for ``--toolsets``. The helper detects the
literal raw argv token and offers to relaunch with ``--tui``.
"""

from unittest.mock import patch

import pytest

from hermes_cli.main import _warn_probable_tui_misparsing


class TestWarnProbableTuiMisparsing:
    """Unit tests for ``_warn_probable_tui_misparsing``."""

    def test_returns_false_without_literal_typo(self):
        """A valid explicit --toolsets ui invocation must not be changed."""
        assert _warn_probable_tui_misparsing(["--toolsets", "ui"]) is False

    def test_parser_distinguishes_typo_from_explicit_toolset(self):
        """Argparse produces the same value; raw argv preserves the distinction."""
        from hermes_cli._parser import build_top_level_parser

        parser, _subparsers, _chat_parser = build_top_level_parser()
        typo = parser.parse_args(["-tui"])
        explicit = parser.parse_args(["--toolsets", "ui"])

        assert typo.toolsets == explicit.toolsets == "ui"
        assert _warn_probable_tui_misparsing(["-tui"]) is False  # non-TTY
        assert _warn_probable_tui_misparsing(["--toolsets", "ui"]) is False

    def test_returns_false_when_explicit_tui_flag_is_also_present(self):
        """An explicit --tui already provides the intended interface."""
        assert _warn_probable_tui_misparsing(["--tui", "-tui"]) is False

    def test_warns_on_stderr_in_noninteractive(self, capsys):
        """In a non-TTY context, prints a warning to stderr and returns False."""
        with patch("sys.stdin") as mock_stdin, patch("sys.stdout") as mock_stdout:
            mock_stdin.isatty.return_value = False
            mock_stdout.isatty.return_value = False
            result = _warn_probable_tui_misparsing(["-tui"])
        assert result is False
        captured = capsys.readouterr()
        assert "Warning" in captured.err
        assert "hermes --tui" in captured.err

    def test_user_accepts_original_toolsets_ui(self, monkeypatch, capsys):
        """User types 'y' → continues with --toolsets=ui, returns False."""
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)
        monkeypatch.setattr("sys.stdout.isatty", lambda: True)
        monkeypatch.setattr("builtins.input", lambda _="": "y")
        result = _warn_probable_tui_misparsing(["-tui"])
        assert result is False
        captured = capsys.readouterr()
        assert "--toolsets=ui" in captured.out

    def test_user_declines_relaunches_tui(self, monkeypatch, capsys):
        """User types 'n' → returns True (caller should set args.tui=True)."""
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)
        monkeypatch.setattr("sys.stdout.isatty", lambda: True)
        monkeypatch.setattr("builtins.input", lambda _="": "n")
        result = _warn_probable_tui_misparsing(["-tui"])
        assert result is True
        captured = capsys.readouterr()
        assert "Launching --tui" in captured.out

    def test_user_enter_defaults_to_no(self, monkeypatch):
        """Empty input (just pressing Enter) defaults to 'No' → relaunch."""
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)
        monkeypatch.setattr("sys.stdout.isatty", lambda: True)
        monkeypatch.setattr("builtins.input", lambda _="": "")
        result = _warn_probable_tui_misparsing(["-tui"])
        assert result is True

    def test_ctrl_c_exits_cleanly(self, monkeypatch):
        """KeyboardInterrupt during input exits with code 0."""
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)
        monkeypatch.setattr("sys.stdout.isatty", lambda: True)
        monkeypatch.setattr("builtins.input", lambda _="": (_ for _ in ()).throw(KeyboardInterrupt))
        with pytest.raises(SystemExit) as exc_info:
            _warn_probable_tui_misparsing(["-tui"])
        assert exc_info.value.code == 0

    def test_eof_exits_cleanly(self, monkeypatch):
        """EOFError during input exits with code 0."""
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)
        monkeypatch.setattr("sys.stdout.isatty", lambda: True)
        monkeypatch.setattr("builtins.input", lambda _="": (_ for _ in ()).throw(EOFError))
        with pytest.raises(SystemExit) as exc_info:
            _warn_probable_tui_misparsing(["-tui"])
        assert exc_info.value.code == 0
