"""Regression tests for collapsed paste references passed to /queue."""

from queue import Queue
from unittest.mock import patch

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
