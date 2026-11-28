"""Post-interrupt redraw in the classic CLI (regression for #157627).

After Ctrl+C / Ctrl+Q stops a turn, ``_recover_terminal_after_interrupt`` erases the viewport and
replays the output history. The replay reproduces the screen only when every transcript row on it
was recorded, and its writes must not interleave with prompt_toolkit's own paints, so they belong
on the application's event loop rather than on the REPL worker thread that ran the turn.
"""

from __future__ import annotations

import asyncio
import re
import threading
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from prompt_toolkit.buffer import Buffer

import cli as cli_mod
import hermes_cli.cli_tui_mixin as mixin_mod
from cli import HermesCLI
from tests.hermes_cli.test_cli_force_redraw import _fake_app
from tests.waiting import wait_until

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


@pytest.mark.parametrize("entry", ["post_interrupt", "slash_redraw"])
def test_redraw_requested_off_the_loop_paints_on_the_app_loop(entry):
    """The REPL worker runs both the post-interrupt recovery and ``/redraw``: their erase and replay
    must run on the loop thread, where they cannot interleave with the loop's own paints."""
    loop = asyncio.new_event_loop()
    loop_thread = threading.Thread(target=loop.run_forever, name="app-loop", daemon=True)
    loop_thread.start()
    try:
        app = _fake_app(rows=24, columns=80, chrome=3, painted=[80, 80, 80])
        app.loop = loop
        writers, finished = [], threading.Event()
        output = app.renderer.output
        for name in ("cursor_goto", "erase_end_of_line", "write", "write_raw", "flush"):
            getattr(output, name).side_effect = lambda *a, **k: writers.append(threading.current_thread().name)
        app.invalidate.side_effect = lambda: (writers.append(threading.current_thread().name), finished.set())
        cli = object.__new__(HermesCLI)
        cli._app = app
        cli_mod._record_output_history("transcript row before the interrupt")

        with patch("hermes_cli.curses_ui.flush_stdin"), patch.object(cli_mod, "_cprint"):
            if entry == "post_interrupt":
                cli._recover_terminal_after_interrupt()
            else:
                cli._cmd_redraw("/redraw")
            wait_until(finished.is_set, what="the redraw to finish")
    finally:
        loop.call_soon_threadsafe(loop.stop)
        loop_thread.join(timeout=5)
        loop.close()

    assert writers and set(writers) == {"app-loop"}, writers
