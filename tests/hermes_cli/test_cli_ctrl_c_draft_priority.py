"""Ctrl+C / Ctrl+Q in the classic CLI: a typed draft wins over interrupting a running turn.

Mirrors the Ink TUI's ``resolveCtrlCComposerAction``: pressing Ctrl+C to wipe a half-typed
follow-up must not kill the in-flight turn; the next press on an empty prompt interrupts.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from prompt_toolkit.buffer import Buffer

import hermes_cli.cli_tui_mixin as mixin_mod


class _Cli(mixin_mod.CLITuiMixin):
    def __init__(self):
        self._agent_running = True
        self.agent = object()
        self._attached_images = []
        self._last_ctrl_c_time = 0.0
        self._should_exit = False

    # Voice / modal / overlay layers sit above the draft check and are covered elsewhere.
    _close_model_picker = _close_command_palette = _submit_slash_confirm_response = None

    def _tui_cancel_voice_recording(self, event):
        return False

    def _tui_cancel_foreground_ui(self, event, closers):
        return False

    def _tui_clear_blocking_overlays(self, event):
        return False


@pytest.mark.parametrize("handler", ["_tui_handle_ctrl_c", "_tui_handle_ctrl_q"])
def test_draft_is_cleared_before_a_running_turn_is_interrupted(monkeypatch, handler):
    interrupts = []
    monkeypatch.setattr(mixin_mod, "request_hard_interrupt", interrupts.append)
    cli = _Cli()
    buffer = Buffer()
    buffer.text = "half typed follow-up"
    app = SimpleNamespace(current_buffer=buffer, invalidate=lambda: None,
                          exit=lambda: setattr(cli, "_should_exit", True))
    event = SimpleNamespace(app=app)

    getattr(cli, handler)(event)
    assert buffer.text == "" and interrupts == [] and not cli._should_exit

    getattr(cli, handler)(event)
    assert interrupts == [cli.agent] and not cli._should_exit
