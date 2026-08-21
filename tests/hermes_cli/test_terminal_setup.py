"""Tests for the safe classic-CLI ``hermes terminal-setup`` helper."""

from __future__ import annotations

import argparse


def test_detect_terminal_prefers_windows_terminal(monkeypatch):
    from hermes_cli.terminal_setup import detect_terminal

    monkeypatch.setenv("WT_SESSION", "session-id")
    monkeypatch.setenv("TERM_PROGRAM", "vscode")

    assert detect_terminal() == "windows-terminal"


def test_detect_terminal_identifies_vscode(monkeypatch):
    from hermes_cli.terminal_setup import detect_terminal

    monkeypatch.delenv("WT_SESSION", raising=False)
    monkeypatch.setenv("TERM_PROGRAM", "vscode")

    assert detect_terminal() == "vscode"


def test_terminal_setup_reports_only_classic_cli_guidance(monkeypatch, capsys):
    from hermes_cli.terminal_setup import run_terminal_setup

    monkeypatch.setenv("WT_SESSION", "session-id")
    run_terminal_setup()

    output = capsys.readouterr().out
    assert "classic `hermes` CLI" in output
    assert "does not change terminal, shell, or system configuration" in output
    assert "Ctrl+Enter" in output
    assert "TUI" not in output


def test_terminal_setup_does_not_write_or_launch_processes(monkeypatch):
    from hermes_cli.terminal_setup import run_terminal_setup

    def fail(*_args, **_kwargs):
        raise AssertionError("terminal-setup must not invoke external processes")

    monkeypatch.setattr("subprocess.run", fail)
    monkeypatch.setattr("os.system", fail)

    run_terminal_setup()


def test_terminal_setup_parser_registers_a_handler():
    from hermes_cli.subcommands.terminal_setup import build_terminal_setup_parser

    root = argparse.ArgumentParser()
    subparsers = root.add_subparsers(dest="command")
    handler = lambda _args: None
    build_terminal_setup_parser(subparsers, cmd_terminal_setup=handler)

    args = root.parse_args(["terminal-setup"])
    assert args.command == "terminal-setup"
    assert args.func is handler


def test_terminal_setup_is_fully_registered_in_the_top_level_cli_source():
    from pathlib import Path
    import hermes_cli.main as main

    source = Path(main.__file__).read_text(encoding="utf-8")
    assert "from hermes_cli.subcommands.terminal_setup import build_terminal_setup_parser" in source
    assert "def cmd_terminal_setup(args):" in source
    assert "build_terminal_setup_parser(subparsers, cmd_terminal_setup=cmd_terminal_setup)" in source
    assert "terminal-setup" in main._BUILTIN_SUBCOMMANDS


def test_terminal_setup_gives_iterm2_specific_safe_instructions(monkeypatch, capsys):
    from hermes_cli.terminal_setup import run_terminal_setup

    monkeypatch.delenv("WT_SESSION", raising=False)
    monkeypatch.setenv("TERM_PROGRAM", "iTerm.app")
    run_terminal_setup()

    output = capsys.readouterr().out
    assert "Profiles → Keys" in output
    assert "Report modifiers using CSI u" in output


def test_terminal_setup_gives_vscode_settings_snippet(monkeypatch, capsys):
    from hermes_cli.terminal_setup import run_terminal_setup

    monkeypatch.delenv("WT_SESSION", raising=False)
    monkeypatch.setenv("VSCODE_PID", "123")
    run_terminal_setup()

    assert '"terminal.integrated.enableKittyKeyboardProtocol": true' in capsys.readouterr().out


def test_terminal_setup_scopes_windows_terminal_to_preview_kitty_support(monkeypatch, capsys):
    from hermes_cli.terminal_setup import run_terminal_setup

    monkeypatch.setenv("WT_SESSION", "session-id")
    run_terminal_setup()

    output = capsys.readouterr().out
    assert "Windows Terminal Preview 1.25+" in output
    assert "Kitty keyboard protocol" in output
