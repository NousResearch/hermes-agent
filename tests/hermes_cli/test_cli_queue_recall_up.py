"""Up on an empty composer takes the newest queued prompt back for editing (Copilot CLI gesture).

Invariants: the recalled item leaves the queue (newest first, images re-attached, foreign payloads
untouched); the recalled draft re-queues on Enter even under ``busy_input_mode: interrupt``; the
arm drops as soon as the draft is gone so an unrelated later Enter routes normally again.
"""

from queue import Queue
from unittest.mock import MagicMock, patch

from prompt_toolkit.buffer import Buffer

from cli import HermesCLI, _VoiceInputMessage


def _cli(*queued):
    cli_obj = HermesCLI.__new__(HermesCLI)
    cli_obj._agent_running = True
    cli_obj._pending_input = Queue()
    cli_obj._interrupt_queue = Queue()
    cli_obj._attached_images = []
    cli_obj._queue_recall_armed = False
    cli_obj.busy_input_mode = "interrupt"
    cli_obj.agent = None
    for item in queued:
        cli_obj._pending_input.put(item)
    return cli_obj


def test_up_on_empty_composer_pops_newest_queued_prompt_and_reattaches_images():
    cli_obj = _cli("first", _VoiceInputMessage("spoken"), ("newest", ["img.png"]))
    buf = Buffer()

    with patch("cli._cprint"):
        assert cli_obj._tui_recall_queued_prompt(buf) is True
    assert buf.text == "newest" and buf.cursor_position == len("newest")
    assert cli_obj._attached_images == ["img.png"]
    assert cli_obj._queue_recall_armed is True

    with patch("cli._cprint"):
        assert cli_obj._tui_recall_queued_prompt(buf) is True
    assert buf.text == "spoken"  # voice sentinel recalls as plain text
    assert list(cli_obj._pending_input.queue) == ["first"]

    # A non-recallable payload at the tail is left alone and nothing is popped behind it.
    cli_obj._pending_input.put({"opaque": True})
    with patch("cli._cprint"):
        assert cli_obj._tui_recall_queued_prompt(buf) is False
    assert list(cli_obj._pending_input.queue) == ["first", {"opaque": True}]


def test_recalled_prompt_requeues_on_enter_under_interrupt_mode_then_arm_clears():
    # Five lines: over the fallback paste-collapse threshold, so a recall that looked like a
    # paste to _tui_on_text_changed would be folded into a [Pasted text #N] placeholder.
    queued = "queued while busy\nline 2\nline 3\nline 4\nline 5"
    cli_obj = _cli(queued)
    cli_obj.config = {}
    cli_obj._tui_last_text_change = 0.0
    cli_obj._tui_prev_text_len = 0
    cli_obj._tui_prev_newline_count = 0
    cli_obj._tui_paste_just_collapsed = False
    cli_obj._skip_paste_collapse = False
    cli_obj._recover_terminal_input_modes = MagicMock()
    buf = Buffer(on_text_changed=cli_obj._tui_on_text_changed)
    event = MagicMock()
    event.app.current_buffer = buf

    with patch("cli._cprint"):
        cli_obj._tui_history_up(event)  # the real Up handler, paste-collapse guard included
        assert buf.text == queued
        buf.insert_text(" (edited)")
        # Enter while busy: an armed recall re-queues instead of honouring `interrupt`.
        with patch("agent.onboarding.is_seen", return_value=True):
            cli_obj._tui_enter_while_busy(buf.text, [], buf.text)
    assert list(cli_obj._pending_input.queue) == [queued + " (edited)"]
    assert cli_obj._interrupt_queue.empty()
    assert cli_obj._queue_recall_armed is False  # consumed by that submit
    with patch("cli._cprint"), patch("agent.onboarding.is_seen", return_value=True):
        cli_obj._tui_enter_while_busy("fresh text", [], "fresh text")
    assert cli_obj._interrupt_queue.get_nowait() == "fresh text"  # normal interrupt routing again

    # Recall, then discard the draft with Esc Esc: the arm must drop with the draft.
    buf.reset()
    with patch("cli._cprint"):
        cli_obj._tui_history_up(event)
    assert buf.text == queued + " (edited)" and cli_obj._queue_recall_armed is True
    cli_obj._tui_handle_double_escape(event)
    assert buf.text == "" and cli_obj._queue_recall_armed is False
