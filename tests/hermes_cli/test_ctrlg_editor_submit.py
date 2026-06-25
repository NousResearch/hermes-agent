"""Tests for Ctrl+G external-editor submit in the classic CLI.

Ctrl+G opens the current draft in ``$EDITOR``; on a clean save the draft is
submitted (TUI parity) rather than left in the input area. Submission in the
CLI is driven by the custom Enter keybinding, not the buffer accept_handler,
so ``_open_external_editor`` chains a done-callback that calls
``_submit_editor_buffer``. These exercise that submit helper directly.
"""

import queue

from cli import HermesCLI


class _FakeBuf:
    def __init__(self, text: str):
        self.text = text
        self.reset_called = False

    def reset(self, append_to_history: bool = False):
        self.reset_called = True
        self.text = ""


def _make(agent_running: bool = False, busy: str = "queue") -> HermesCLI:
    c = HermesCLI.__new__(HermesCLI)
    c._pending_input = queue.Queue()
    c._interrupt_queue = queue.Queue()
    c._agent_running = agent_running
    c.busy_input_mode = busy
    c._app = None
    c._should_exit = False
    return c


def test_idle_prompt_routed_to_pending_input():
    c = _make()
    buf = _FakeBuf("Explain vector databases.\nKeep it short.")

    c._submit_editor_buffer(buf)

    assert c._pending_input.get_nowait() == "Explain vector databases.\nKeep it short."
    assert buf.reset_called


def test_empty_save_does_not_submit():
    c = _make()
    buf = _FakeBuf("   \n  \n")

    c._submit_editor_buffer(buf)

    assert c._pending_input.empty()
    # An empty save must not clear-and-submit a blank turn.
    assert not buf.reset_called


def test_running_queue_mode_queues_for_next_turn():
    c = _make(agent_running=True, busy="queue")
    buf = _FakeBuf("next turn please")

    c._submit_editor_buffer(buf)

    assert c._pending_input.get_nowait() == "next turn please"
    assert c._interrupt_queue.empty()


def test_running_interrupt_mode_uses_interrupt_queue():
    c = _make(agent_running=True, busy="interrupt")
    buf = _FakeBuf("interrupt this")

    c._submit_editor_buffer(buf)

    assert c._interrupt_queue.get_nowait() == "interrupt this"
    assert c._pending_input.empty()


def test_slash_command_dispatched_not_queued():
    c = _make()
    seen = {}
    c.process_command = lambda command: seen.setdefault("cmd", command) or True
    buf = _FakeBuf("/status")

    c._submit_editor_buffer(buf)

    assert seen.get("cmd") == "/status"
    assert c._pending_input.empty()
