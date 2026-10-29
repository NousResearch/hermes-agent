"""The CLI must not render the local wait-interrupt sentinel as an assistant reply.

When a turn is cancelled before any assistant bytes stream, the agent returns
``"Operation interrupted: waiting for model response (N.Ns elapsed)."`` as
``final_response``. The gateway (#7921), ACP and the TUI suppress it as
cancellation metadata; ``_chat_render_turn`` printed, spoke and returned it.
"""

import queue
from unittest.mock import MagicMock

from agent.conversation_loop import INTERRUPT_WAITING_FOR_MODEL_PREFIX
from hermes_cli.cli_chat_turn_mixin import CLIChatTurnMixin, is_wait_interrupt_metadata

SENTINEL = f"{INTERRUPT_WAITING_FOR_MODEL_PREFIX}20.0s elapsed)."


class _Stub(CLIChatTurnMixin):
    def __init__(self):
        self._interrupt_queue = queue.Queue()
        self._pending_input = queue.Queue()
        self._voice_tts = None
        self._voice_continuous = False
        self.agent = MagicMock(max_iterations=500, provider="p", model="m")
        self.provider = "p"
        self.model = "m"
        self.rendered = []
        self._chat_print_reasoning_box = lambda turn: None
        self._chat_print_response_panel = lambda turn, response: self.rendered.append(response)
        self._emit_focus_recovery_line = lambda: None
        self._ring_bell = lambda **kwargs: None


def _turn(**result):
    turn = MagicMock()
    turn.mute_notification_reply = False
    turn.use_streaming_tts = False
    turn.result = result
    return turn


def test_wait_sentinel_is_not_rendered_and_redirect_is_requeued():
    cli = _Stub()
    turn = _turn(final_response=SENTINEL, interrupted=True, partial=True, completed=False,
                 interrupt_message="use the other file")

    response = cli._chat_render_turn(turn, MagicMock(), None)

    assert response == ""
    assert cli.rendered == [""]
    assert cli._pending_input.get_nowait() == "use the other file"


def test_wait_sentinel_partial_does_not_become_an_error_panel():
    cli = _Stub()
    turn = _turn(final_response=SENTINEL, interrupted=True, partial=True, completed=False)

    assert cli._chat_render_turn(turn, MagicMock(), None) == ""
    assert cli.rendered == [""]


def test_genuine_failure_on_an_interrupted_turn_still_surfaces():
    cli = _Stub()
    turn = _turn(final_response=SENTINEL, interrupted=True, failed=True, completed=False,
                 error="401 unauthorized")

    response = cli._chat_render_turn(turn, MagicMock(), None)

    assert response and response != SENTINEL
    assert "401" in response


def test_real_partial_text_is_kept():
    cli = _Stub()
    partial = "Here is the first half of the answer"
    turn = _turn(final_response=partial, interrupted=True, completed=False)

    assert cli._chat_render_turn(turn, MagicMock(), None) == partial


def test_sentinel_text_is_kept_when_the_turn_was_not_interrupted():
    cli = _Stub()
    turn = _turn(final_response=SENTINEL, completed=True)

    assert cli._chat_render_turn(turn, MagicMock(), None) == SENTINEL


def test_recognizer_is_prefix_exact():
    assert is_wait_interrupt_metadata(SENTINEL)
    assert is_wait_interrupt_metadata(f"  {SENTINEL}\n")
    assert not is_wait_interrupt_metadata("The log said Operation interrupted: waiting for model response (")
    assert not is_wait_interrupt_metadata(None)
