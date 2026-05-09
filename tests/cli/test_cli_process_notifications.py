"""Tests for CLI background process notifications."""

import queue
from unittest.mock import MagicMock, patch

import cli as cli_mod
from cli import HermesCLI


def _bare_cli():
    instance = object.__new__(HermesCLI)
    instance._pending_input = queue.Queue()
    instance._app = MagicMock()
    return instance


def test_completion_notification_renders_terminal_notice_and_is_not_pending_input():
    instance = _bare_cli()
    evt = {
        "type": "completion",
        "session_id": "proc_1",
        "command": "python job.py",
        "exit_code": 0,
        "output": "done",
    }

    with patch.object(cli_mod, "_cprint") as cprint:
        assert instance._print_process_notification(evt) is True

    printed = "\n".join(str(call.args[0]) for call in cprint.call_args_list)
    assert "Background process notification" in printed
    assert "Background process proc_1 completed" in printed
    assert "done" in printed
    assert instance._pending_input.empty()
    instance._app.invalidate.assert_called()


def test_watch_disabled_notification_renders_terminal_notice():
    instance = _bare_cli()
    evt = {
        "type": "watch_disabled",
        "session_id": "proc_2",
        "message": "watch patterns disabled after repeated notifications",
    }

    with patch.object(cli_mod, "_cprint") as cprint:
        assert instance._print_process_notification(evt) is True

    printed = "\n".join(str(call.args[0]) for call in cprint.call_args_list)
    assert "Background process notification" in printed
    assert "watch patterns disabled" in printed
    assert instance._pending_input.empty()
