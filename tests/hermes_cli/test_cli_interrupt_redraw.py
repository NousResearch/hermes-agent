"""Post-interrupt redraw in the classic CLI (regression for #157627).

After Ctrl+C / Ctrl+Q stops a turn, ``_recover_terminal_after_interrupt`` erases the viewport and
replays the output history. The replay reproduces the screen only when every transcript row on it
was recorded, and its writes must not interleave with prompt_toolkit's own paints, so they belong
on the application's event loop rather than on the REPL worker thread that ran the turn.
"""

from __future__ import annotations

import re
from types import SimpleNamespace

import pytest
from prompt_toolkit.buffer import Buffer

import cli as cli_mod
import hermes_cli.cli_tui_mixin as mixin_mod

_SGR = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


@pytest.fixture(autouse=True)
def _fresh_output_history():
    cli_mod._configure_output_history(True, 200)
    yield
    cli_mod._configure_output_history(True, 200)


class _RunningTurnCli(mixin_mod.CLITuiMixin):
    """A CLI whose agent is mid-turn with an empty prompt: Ctrl+C / Ctrl+Q interrupt it."""

    def __init__(self):
        self._agent_running = True
        self.agent = object()
        self._attached_images = []
        self._last_ctrl_c_time = 0.0
        self._should_exit = False

    _close_model_picker = _close_command_palette = _submit_slash_confirm_response = None

    def _tui_cancel_voice_recording(self, event):
        return False

    def _tui_cancel_foreground_ui(self, event, *, closers):
        return False

    def _tui_clear_blocking_overlays(self, event):
        return False


def _rows(text: str) -> list[str]:
    return [line.strip() for line in _SGR.sub("", text).splitlines() if line.strip()]


@pytest.mark.parametrize("handler,presses", [
    ("_tui_handle_ctrl_c", 1),  # "Interrupting agent... (press Ctrl+C again to force exit)"
    ("_tui_handle_ctrl_c", 2),  # ... then "Force exiting..."
    ("_tui_handle_ctrl_q", 1),  # "Interrupting agent..."
])
def test_replay_after_interrupt_repaints_the_rows_the_keypress_printed(monkeypatch, capsys, handler, presses):
    monkeypatch.setattr(mixin_mod, "request_hard_interrupt", lambda agent: None)
    # Every paint path ends on the terminal (stdout here), recorded or not.
    monkeypatch.setattr(cli_mod, "_pt_print", print)
    monkeypatch.setattr(cli_mod, "_pt_print_ansi", lambda text, raw="": print(text))
    monkeypatch.setattr(cli_mod, "_PT_ANSI", lambda text: text)
    cli = _RunningTurnCli()
    event = SimpleNamespace(app=SimpleNamespace(current_buffer=Buffer(), invalidate=lambda: None,
                                                exit=lambda: setattr(cli, "_should_exit", True)))

    for _ in range(presses):
        getattr(cli, handler)(event)
    on_screen = _rows(capsys.readouterr().out)
    assert on_screen, "the keypress printed nothing to compare against"

    cli_mod._replay_output_history()

    assert _rows(capsys.readouterr().out) == on_screen
