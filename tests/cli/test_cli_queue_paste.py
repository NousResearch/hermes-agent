"""Regression tests for collapsed paste references passed to /queue."""

from queue import Queue
from unittest.mock import patch
from types import SimpleNamespace

from cli import HermesCLI


def test_queue_expands_collapsed_paste_reference(tmp_path):
    pasted = "first\nmiddle\nlast"
    paste_file = tmp_path / "paste.txt"
    paste_file.write_text(pasted, encoding="utf-8")
    placeholder = f"[Pasted text #1: 3 lines → {paste_file}]"
    cli_obj = HermesCLI.__new__(HermesCLI)
    cli_obj._agent_running = False
    cli_obj._pending_input = Queue()
    cli_obj._pending_resume_sessions = None

    with patch("cli._cprint"):
        assert cli_obj.process_command(f"/queue {placeholder}") is True

    assert cli_obj._pending_input.get_nowait() == pasted


def test_chat_preview_keeps_collapsed_paste_marker(tmp_path):
    pasted = "first\nmiddle\nlast"
    paste_file = tmp_path / "paste.txt"
    paste_file.write_text(pasted, encoding="utf-8")
    placeholder = f"[Pasted text #1: 3 lines → {paste_file}]"
    previewed, sent = [], []
    cli_obj = HermesCLI.__new__(HermesCLI)
    cli_obj.final_response_markdown = "strip"
    cli_obj._pending_resume_sessions = None
    cli_obj._pending_input = Queue()
    cli_obj._print_user_message_preview = previewed.append
    cli_obj._tui_unwrap_input = lambda value: (value, False, False)
    cli_obj._typed_voice_stop = lambda value: False
    cli_obj.handle_bang_shell = lambda value: False
    cli_obj._turn_summary_begin = lambda: None
    cli_obj._tui_after_turn = lambda: None
    cli_obj._app = SimpleNamespace(invalidate=lambda: None)
    cli_obj.chat = lambda value, **kwargs: sent.append(value)

    with patch("cli.print"):
        cli_obj._tui_process_one_input(placeholder)

    assert previewed == [placeholder]
    assert sent == [pasted]
