"""#110737: an interrupting message that carries image attachments is a ``(text, images)``
tuple on ``_interrupt_queue``; the re-queue block in ``_chat_render_turn`` must hand it to
``_pending_input`` intact instead of string-joining it (TypeError, swallowed by ``chat()``)."""

import queue
from pathlib import Path
from unittest.mock import MagicMock

from hermes_cli.cli_chat_turn_mixin import CLIChatTurnMixin


class _Stub(CLIChatTurnMixin):
    def __init__(self):
        self._interrupt_queue = queue.Queue()
        self._pending_input = queue.Queue()
        self._voice_tts = None
        self._voice_continuous = False
        self.agent = MagicMock(max_iterations=500)
        self._chat_print_reasoning_box = lambda turn: None
        self._chat_print_response_panel = lambda turn, response: None
        self._emit_focus_recovery_line = lambda: None
        self._ring_bell = lambda **kwargs: None


def _interrupted_turn(payload):
    turn = MagicMock()
    turn.mute_notification_reply = False
    turn.result = {"interrupted": True, "interrupt_message": payload, "final_response": ""}
    turn.use_streaming_tts = False
    return turn


def test_tuple_payload_is_requeued_with_its_images():
    cli = _Stub()
    images = [Path("/tmp/clip.png")]

    cli._chat_render_turn(_interrupted_turn(("look at this", images)), MagicMock(), None)

    assert cli._pending_input.get_nowait() == ("look at this", images)


def test_queued_parts_join_text_and_merge_images():
    cli = _Stub()
    cli._interrupt_queue.put("second")
    cli._interrupt_queue.put(("third", [Path("/tmp/b.png")]))

    cli._chat_render_turn(_interrupted_turn(("first", [Path("/tmp/a.png")])), MagicMock(), None)

    assert cli._pending_input.get_nowait() == (
        "first\nsecond\nthird", [Path("/tmp/a.png"), Path("/tmp/b.png")],
    )
