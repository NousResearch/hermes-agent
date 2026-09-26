"""#75461: interrupt_debug.log is a persistent on-disk file, so a credential in a
steered/queued user message must never reach it verbatim.

Behavior contract, not a source-text check: drive the real write path with a
Telegram-token-shaped payload and assert the bytes that land in the file.
"""

import queue
import types
from pathlib import Path

from hermes_cli.cli_chat_turn_mixin import CLIChatTurnMixin
from hermes_cli.cli_tui_mixin import CLITuiMixin

# Shape that triggered the original leak.
TOKEN = "1234567890:" + "A" * 35


def _assert_token_masked(log, contents):
    assert log.exists()
    assert "A" * 35 not in contents
    assert "1234567890:***" in contents


def _run_busy_submit(tmp_path, monkeypatch, text):
    """Drive the legacy interrupt-queue branch of the busy-submit path."""
    import cli

    monkeypatch.setattr(cli, "_hermes_home", tmp_path)
    mixin = CLITuiMixin.__new__(CLITuiMixin)
    mixin.busy_input_mode = "interrupt"
    mixin.agent = types.SimpleNamespace()  # no redirect() -> legacy queue branch
    mixin._agent_running = True
    mixin._interrupt_queue = types.SimpleNamespace(
        put=lambda payload: None,
    )
    CLITuiMixin._tui_enter_while_busy(mixin, text, [], {"text": text})
    return Path(tmp_path) / "interrupt_debug.log"


def test_queued_interrupt_message_is_redacted_on_disk(tmp_path, monkeypatch):
    log = _run_busy_submit(tmp_path, monkeypatch, f"my key is {TOKEN}")

    _assert_token_masked(log, log.read_text(encoding="utf-8"))


def _fire_running_agent_interrupt(tmp_path, monkeypatch, text):
    """Drive the agent-thread side: the queued message arriving at a live turn.

    ``_chat_monitor_agent_thread`` polls ``self._interrupt_queue`` while the
    agent thread is alive, so the stub thread reports itself alive exactly once
    — long enough for the queued message to be dequeued, written, and for the
    loop to exit on the next poll.
    """
    import cli

    monkeypatch.setattr(cli, "_hermes_home", tmp_path)

    pending = [text]

    class _InterruptQueue:
        def get(self, timeout=None):
            if pending:
                return pending.pop(0)
            raise queue.Empty()

    mixin = CLIChatTurnMixin.__new__(CLIChatTurnMixin)
    mixin._interrupt_queue = _InterruptQueue()
    mixin._pending_input = types.SimpleNamespace(put=lambda payload: pending.append(payload))
    mixin._clarify_state = None
    mixin._clarify_freetext = None
    mixin._voice_mode = False
    mixin.agent = types.SimpleNamespace(
        interrupt=lambda msg: None,
        _active_children=[],
        _interrupt_requested=True,
    )
    mixin._clear_active_overlays_for_interrupt = lambda: None

    turn = types.SimpleNamespace(stop_event=types.SimpleNamespace(set=lambda: None))
    entered = []

    def _is_alive():
        # True on the first poll (enters the loop), False afterwards (exits).
        if not entered:
            entered.append(True)
            return True
        return False

    CLIChatTurnMixin._chat_monitor_agent_thread(
        mixin, turn, types.SimpleNamespace(is_alive=_is_alive, join=lambda timeout=None: None)
    )
    return Path(tmp_path) / "interrupt_debug.log"


def test_running_agent_interrupt_is_redacted_on_disk(tmp_path, monkeypatch):
    log = _fire_running_agent_interrupt(tmp_path, monkeypatch, f"my key is {TOKEN}")

    _assert_token_masked(log, log.read_text(encoding="utf-8"))
