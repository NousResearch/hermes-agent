"""The TUI exit epilogue must stay quiet when a late KeyboardInterrupt lands in it.

``_print_tui_exit_summary`` runs after the TUI process has already exited; a
second Ctrl+C arriving during its read-only DB/config load used to escape
``except Exception`` (KeyboardInterrupt is a BaseException) and dump a raw
traceback into the shell — same class as #83341 / #80256.
"""

import hermes_state
import hermes_cli.main as hermes_main
from hermes_cli.main_tui_launch import _print_tui_exit_summary


def _raise_interrupt(*args, **kwargs):
    raise KeyboardInterrupt


def test_exit_summary_swallows_keyboard_interrupt_from_db_read(monkeypatch):
    monkeypatch.setattr(hermes_state, "SessionDB", _raise_interrupt)

    # Must return quietly — the TUI already exited; no traceback may reach the shell.
    _print_tui_exit_summary("20260927_134833_9d0de4ee")


def test_exit_summary_swallows_keyboard_interrupt_from_last_session_lookup(monkeypatch):
    monkeypatch.setattr(hermes_main, "_resolve_last_session", _raise_interrupt)

    _print_tui_exit_summary(None)
