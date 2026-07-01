"""Tests for `hermes terminal-setup`."""

from __future__ import annotations

import subprocess
import sys
from io import StringIO
from unittest.mock import patch, MagicMock

import pytest



class TestDetectTerminal:
    def test_detects_iterm2_via_term_program(self, monkeypatch):
        from hermes_cli.terminal_setup import _detect_terminal
        monkeypatch.setenv("TERM_PROGRAM", "iTerm.app")
        monkeypatch.delenv("LC_TERMINAL", raising=False)
        monkeypatch.delenv("TERM", raising=False)
        monkeypatch.delenv("KITTY_WINDOW_ID", raising=False)
        monkeypatch.delenv("GHOSTTY_RESOURCES_DIR", raising=False)
        monkeypatch.delenv("VSCODE_PID", raising=False)
        assert _detect_terminal() == "iterm2"

    def test_detects_iterm2_via_lc_terminal(self, monkeypatch):
        from hermes_cli.terminal_setup import _detect_terminal
        monkeypatch.setenv("TERM_PROGRAM", "something-else")
        monkeypatch.setenv("LC_TERMINAL", "iTerm2")
        monkeypatch.delenv("TERM", raising=False)
        monkeypatch.delenv("KITTY_WINDOW_ID", raising=False)
        monkeypatch.delenv("GHOSTTY_RESOURCES_DIR", raising=False)
        monkeypatch.delenv("VSCODE_PID", raising=False)
        assert _detect_terminal() == "iterm2"

    def test_detects_terminal_app(self, monkeypatch):
        from hermes_cli.terminal_setup import _detect_terminal
        monkeypatch.setenv("TERM_PROGRAM", "Apple_Terminal")
        monkeypatch.delenv("LC_TERMINAL", raising=False)
        monkeypatch.delenv("KITTY_WINDOW_ID", raising=False)
        monkeypatch.delenv("GHOSTTY_RESOURCES_DIR", raising=False)
        monkeypatch.delenv("VSCODE_PID", raising=False)
        assert _detect_terminal() == "terminal_app"

    def test_detects_kitty_via_term(self, monkeypatch):
        from hermes_cli.terminal_setup import _detect_terminal
        monkeypatch.setenv("TERM_PROGRAM", "")
        monkeypatch.setenv("TERM", "xterm-kitty")
        monkeypatch.delenv("LC_TERMINAL", raising=False)
        monkeypatch.delenv("KITTY_WINDOW_ID", raising=False)
        monkeypatch.delenv("GHOSTTY_RESOURCES_DIR", raising=False)
        monkeypatch.delenv("VSCODE_PID", raising=False)
        assert _detect_terminal() == "kitty"

    def test_detects_kitty_via_window_id(self, monkeypatch):
        from hermes_cli.terminal_setup import _detect_terminal
        monkeypatch.setenv("TERM_PROGRAM", "")
        monkeypatch.setenv("TERM", "xterm-256color")
        monkeypatch.setenv("KITTY_WINDOW_ID", "1")
        monkeypatch.delenv("LC_TERMINAL", raising=False)
        monkeypatch.delenv("GHOSTTY_RESOURCES_DIR", raising=False)
        monkeypatch.delenv("VSCODE_PID", raising=False)
        assert _detect_terminal() == "kitty"

    def test_detects_wezterm(self, monkeypatch):
        from hermes_cli.terminal_setup import _detect_terminal
        monkeypatch.setenv("TERM_PROGRAM", "WezTerm")
        monkeypatch.setenv("TERM", "xterm-256color")
        monkeypatch.delenv("LC_TERMINAL", raising=False)
        monkeypatch.delenv("KITTY_WINDOW_ID", raising=False)
        monkeypatch.delenv("GHOSTTY_RESOURCES_DIR", raising=False)
        monkeypatch.delenv("VSCODE_PID", raising=False)
        assert _detect_terminal() == "wezterm"

    def test_detects_ghostty_via_resources_dir(self, monkeypatch):
        from hermes_cli.terminal_setup import _detect_terminal
        monkeypatch.setenv("TERM_PROGRAM", "")
        monkeypatch.setenv("TERM", "xterm-256color")
        monkeypatch.setenv("GHOSTTY_RESOURCES_DIR", "/Applications/Ghostty.app/Contents/Resources")
        monkeypatch.delenv("LC_TERMINAL", raising=False)
        monkeypatch.delenv("KITTY_WINDOW_ID", raising=False)
        monkeypatch.delenv("VSCODE_PID", raising=False)
        assert _detect_terminal() == "ghostty"

    def test_detects_vscode_via_pid(self, monkeypatch):
        from hermes_cli.terminal_setup import _detect_terminal
        monkeypatch.setenv("TERM_PROGRAM", "")
        monkeypatch.setenv("TERM", "xterm-256color")
        monkeypatch.setenv("VSCODE_PID", "12345")
        monkeypatch.delenv("LC_TERMINAL", raising=False)
        monkeypatch.delenv("KITTY_WINDOW_ID", raising=False)
        monkeypatch.delenv("GHOSTTY_RESOURCES_DIR", raising=False)
        assert _detect_terminal() == "vscode"

    def test_unknown_terminal(self, monkeypatch):
        from hermes_cli.terminal_setup import _detect_terminal
        for var in ("TERM_PROGRAM", "LC_TERMINAL", "TERM", "KITTY_WINDOW_ID",
                    "GHOSTTY_RESOURCES_DIR", "VSCODE_PID", "TERM_EMULATOR"):
            monkeypatch.delenv(var, raising=False)
        assert _detect_terminal() == "unknown"



class TestITerm2GlobalKeyMap:
    def test_no_conflict_when_defaults_returns_nonzero(self, monkeypatch):
        from hermes_cli.terminal_setup import _iterm2_has_shift_return_keybinding
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        with patch("subprocess.run", return_value=mock_result):
            assert _iterm2_has_shift_return_keybinding() is False

    def test_detects_shift_return_keybinding(self, monkeypatch):
        from hermes_cli.terminal_setup import _iterm2_has_shift_return_keybinding
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = (
            '{\n'
            '    "0xd-0x20000" =     {\n'
            '        Action = 11;\n'
            '        Text = "";\n'
            '    };\n'
            '}'
        )
        with patch("subprocess.run", return_value=mock_result):
            assert _iterm2_has_shift_return_keybinding() is True

    def test_no_conflict_when_output_lacks_shift_return(self, monkeypatch):
        from hermes_cli.terminal_setup import _iterm2_has_shift_return_keybinding
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = (
            '{\n'
            '    "0xf700-0x280000" = {\n'
            '        Action = 10;\n'
            '    };\n'
            '}'
        )
        with patch("subprocess.run", return_value=mock_result):
            assert _iterm2_has_shift_return_keybinding() is False

    def test_remove_returns_true_on_success(self, monkeypatch):
        from hermes_cli.terminal_setup import _remove_iterm2_global_keymap
        mock_result = MagicMock()
        mock_result.returncode = 0
        with patch("subprocess.run", return_value=mock_result):
            assert _remove_iterm2_global_keymap() is True

    def test_remove_returns_false_on_failure(self, monkeypatch):
        from hermes_cli.terminal_setup import _remove_iterm2_global_keymap
        mock_result = MagicMock()
        mock_result.returncode = 1
        with patch("subprocess.run", return_value=mock_result):
            assert _remove_iterm2_global_keymap() is False

    def test_remove_returns_false_on_exception(self, monkeypatch):
        from hermes_cli.terminal_setup import _remove_iterm2_global_keymap
        with patch("subprocess.run", side_effect=OSError("not found")):
            assert _remove_iterm2_global_keymap() is False



class TestWizardRouting:

    def _run(self, terminal_token: str, monkeypatch, user_input: str = "n"):
        from hermes_cli import terminal_setup as ts_mod

        monkeypatch.setattr(ts_mod, "_detect_terminal", lambda: terminal_token)

        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = ""
        monkeypatch.setattr(subprocess, "run", lambda *a, **kw: mock_proc)

        monkeypatch.setattr("builtins.input", lambda _: user_input)

        captured = StringIO()
        monkeypatch.setattr(sys, "stdout", captured)

        ts_mod.run_terminal_setup(args=None)
        return captured.getvalue()

    def test_iterm2_no_conflict(self, monkeypatch):
        output = self._run("iterm2", monkeypatch)
        assert output  # must produce some output

    def test_iterm2_with_conflict_user_accepts(self, monkeypatch):
        from hermes_cli import terminal_setup as ts_mod
        monkeypatch.setattr(ts_mod, "_detect_terminal", lambda: "iterm2")
        monkeypatch.setattr(ts_mod, "_iterm2_has_shift_return_keybinding", lambda: True)
        monkeypatch.setattr(ts_mod, "_remove_iterm2_global_keymap", lambda: True)
        monkeypatch.setattr("builtins.input", lambda _: "y")
        captured = StringIO()
        monkeypatch.setattr(sys, "stdout", captured)
        ts_mod.run_terminal_setup(args=None)
        assert "GlobalKeyMap" in captured.getvalue()

    def test_iterm2_with_conflict_user_declines(self, monkeypatch):
        from hermes_cli import terminal_setup as ts_mod
        monkeypatch.setattr(ts_mod, "_detect_terminal", lambda: "iterm2")
        monkeypatch.setattr(ts_mod, "_iterm2_has_shift_return_keybinding", lambda: True)
        monkeypatch.setattr(ts_mod, "_remove_iterm2_global_keymap", lambda: True)
        monkeypatch.setattr("builtins.input", lambda _: "n")
        captured = StringIO()
        monkeypatch.setattr(sys, "stdout", captured)
        ts_mod.run_terminal_setup(args=None)
        assert "Skipped" in captured.getvalue()

    def test_iterm2_remove_fails_shows_manual_hint(self, monkeypatch):
        from hermes_cli import terminal_setup as ts_mod
        monkeypatch.setattr(ts_mod, "_detect_terminal", lambda: "iterm2")
        monkeypatch.setattr(ts_mod, "_iterm2_has_shift_return_keybinding", lambda: True)
        monkeypatch.setattr(ts_mod, "_remove_iterm2_global_keymap", lambda: False)
        monkeypatch.setattr("builtins.input", lambda _: "y")
        captured = StringIO()
        monkeypatch.setattr(sys, "stdout", captured)
        ts_mod.run_terminal_setup(args=None)
        out = captured.getvalue()
        assert "defaults delete" in out

    def test_terminal_app(self, monkeypatch):
        output = self._run("terminal_app", monkeypatch)
        assert "Alt+Enter" in output

    def test_kitty(self, monkeypatch):
        output = self._run("kitty", monkeypatch)
        assert "kitty" in output.lower()

    def test_wezterm(self, monkeypatch):
        output = self._run("wezterm", monkeypatch)
        assert "WezTerm" in output or "wezterm" in output.lower()

    def test_ghostty(self, monkeypatch):
        output = self._run("ghostty", monkeypatch)
        assert "Ghostty" in output or "ghostty" in output.lower()

    def test_vscode(self, monkeypatch):
        output = self._run("vscode", monkeypatch)
        assert "/terminal-setup" in output

    def test_unknown(self, monkeypatch):
        output = self._run("unknown", monkeypatch)
        assert output  # must produce some output



class TestCLIParserWiring:
    def test_subparser_accepts_terminal_setup(self):
        import argparse
        from hermes_cli.subcommands.terminal_setup import build_terminal_setup_parser

        sentinel = object()
        root = argparse.ArgumentParser()
        subs = root.add_subparsers(dest="command")
        build_terminal_setup_parser(subs, cmd_terminal_setup=lambda a: sentinel)
        args = root.parse_args(["terminal-setup"])
        assert args.command == "terminal-setup"
        assert callable(args.func)

    def test_terminal_setup_in_builtin_subcommands(self):
        from hermes_cli.main import _BUILTIN_SUBCOMMANDS
        assert "terminal-setup" in _BUILTIN_SUBCOMMANDS
