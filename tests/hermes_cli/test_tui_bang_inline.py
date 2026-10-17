"""`!<command>` typed in the prompt_toolkit TUI while the agent is busy must run immediately.

A bang command typed while the agent runs was pushed into `_pending_input`, which the process
loop only drains after the current turn ends — so `!hermes-account 2` looked dead for an entire
long turn. Slash commands already have an inline path for the same reason (/model, /steer, /bg);
bang commands cost zero tokens and run locally, so they get the same immediate dispatch.
"""
from __future__ import annotations

import queue
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


def _shell(*, busy: bool):
    from cli import HermesCLI
    shell = object.__new__(HermesCLI)
    shell._tui_enter_overlay = lambda event: False
    shell._tui_multiline_shortcuts = False
    shell._attached_images = []
    shell._agent_running = busy
    shell.agent = None
    shell.config = {}
    shell._pending_input = queue.Queue()
    shell._interrupt_queue = queue.Queue()
    shell._inline_pastes = lambda buf: None
    shell._tui_last_text_change = 0.0
    shell._should_handle_model_command_inline = lambda text, has_images=False: False
    shell._should_handle_steer_command_inline = lambda text, has_images=False: False
    shell._should_handle_background_command_inline = lambda text, has_images=False: False
    shell._tui_enter_while_busy = MagicMock()
    return shell


def _enter(shell, text):
    buf = SimpleNamespace(text=text, cursor_position=len(text), reset=MagicMock())
    app = SimpleNamespace(current_buffer=buf, invalidate=MagicMock(), is_running=False, exit=MagicMock())
    shell._tui_handle_enter(SimpleNamespace(app=app))
    return buf


def test_busy_bang_runs_inline_instead_of_waiting_for_the_turn():
    shell = _shell(busy=True)
    with patch.object(type(shell), "handle_bang_shell", return_value=True) as bang:
        buf = _enter(shell, "!hermes-account 2")
    bang.assert_called_once_with("!hermes-account 2")
    assert shell._pending_input.empty()
    shell._tui_enter_while_busy.assert_not_called()
    buf.reset.assert_called_once()


def test_idle_bang_runs_inline_and_never_becomes_a_turn():
    shell = _shell(busy=False)
    with patch.object(type(shell), "handle_bang_shell", return_value=True) as bang:
        _enter(shell, "!echo hi")
    bang.assert_called_once()
    assert shell._pending_input.empty()


def test_plain_prompt_while_busy_is_untouched_by_the_bang_path():
    shell = _shell(busy=True)
    with patch.object(type(shell), "handle_bang_shell") as bang:
        _enter(shell, "fix the bug!")
    bang.assert_not_called()
    shell._tui_enter_while_busy.assert_called_once()


def test_bang_declined_by_handler_falls_back_to_the_local_queue():
    """Gateway/cron contexts (handle_bang_shell -> False) keep the old queued-local behaviour."""
    shell = _shell(busy=True)
    with patch.object(type(shell), "handle_bang_shell", return_value=False):
        _enter(shell, "!git status")
    assert shell._pending_input.get_nowait() == "!git status"
    shell._tui_enter_while_busy.assert_not_called()
