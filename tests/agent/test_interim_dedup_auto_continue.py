"""An auto-continue turn must not re-deliver the interrupted turn's final message.

Observed 2026-10-01 (Brandon's keg/keg-price session ``20261001_160012_21b96d``): a
large-context Codex turn was interrupted mid-run, Hermes auto-continued it, and the
same final answer reached the desktop as several NEW bubbles.

Why the existing guard missed it
--------------------------------
``agent/conversation_loop.py`` resets ``agent._delivered_interim_texts`` at the start of
every user turn (the gateway caches one agent across turns, so per-turn state must not
leak). An auto-continue is submitted as a *new user turn* carrying the crash-recovery
note, so the reset wiped the record of what the interrupted turn had already delivered
before the re-run could compare against it. The dedup itself
(``stream_delivery._interim_text_was_delivered``) was working exactly as designed; it
simply had nothing left to compare.

The fix carries the delivered-text set across an auto-continue boundary only. An ordinary
next user message still starts clean, so a user who genuinely repeats themselves still
gets a fresh answer.
"""

from types import SimpleNamespace

import pytest

from agent.stream_delivery import StreamDeliveryMixin


def _reset(agent, persist_user_display_kind):
    """Import inside the test: conftest isolates HERMES_HOME per test, and importing the
    conversation loop at collection time reads the real Hermes home."""
    from agent.conversation_loop import _reset_per_turn_state

    _reset_per_turn_state(agent, persist_user_display_kind=persist_user_display_kind)


class _Agent(StreamDeliveryMixin):
    """Just enough surface for the interim-delivery dedup path.

    ``_visible_commentary`` and ``_strip_think_blocks`` are supplied as plain methods
    because the mixin calls them as bound attributes; the rest mirror ``agent_init``'s
    field defaults that the dedup path reads.
    """

    def __init__(self):
        self._delivered_interim_texts = set()
        self._current_streamed_assistant_text = ""
        self.show_commentary = True
        self.delivered = []
        # Real signature: _deliver_interim passes already_streamed= as a keyword.
        self.interim_assistant_callback = (
            lambda text, *, already_streamed=False: self.delivered.append(text)
        )

    def _strip_think_blocks(self, text):
        return text

    def _visible_commentary(self, text):
        return text


ANSWER = "Burma Brewing invoice 5-13, total $1,744.08 across 52 kegs."


def _deliver(agent, text):
    agent._emit_interim_assistant_message({"role": "assistant", "content": text})


def test_same_turn_repeat_is_suppressed():
    """Baseline: within one turn the dedup already works (the guard we must not break)."""
    agent = _Agent()
    _deliver(agent, ANSWER)
    _deliver(agent, ANSWER)
    assert agent.delivered == [ANSWER]


def test_auto_continue_turn_carries_delivered_texts():
    """The per-turn reset helper preserves the record across an auto-continue turn."""
    agent = _Agent()
    _deliver(agent, ANSWER)
    assert ANSWER in agent._delivered_interim_texts

    # Auto-continue turn: same agent, recovery note as the user turn.
    _reset(agent, "auto_continue")

    # The re-run produces the same final answer again -> must NOT reach the UI twice.
    _deliver(agent, ANSWER)
    assert agent.delivered == [ANSWER], "auto-continue re-delivered an already-delivered message"


def test_ordinary_next_user_turn_still_resets():
    """A real new user message must start clean (no cross-turn suppression)."""
    agent = _Agent()
    _deliver(agent, ANSWER)

    _reset(agent, None)

    assert agent._delivered_interim_texts == set()
    _deliver(agent, ANSWER)
    assert agent.delivered == [ANSWER, ANSWER], "a genuine repeat by the user must still be answered"


def test_model_switch_turn_still_resets():
    """Only the auto-continue boundary carries over; other synthetic rows do not."""
    agent = _Agent()
    _deliver(agent, ANSWER)

    _reset(agent, "model_switch")

    assert agent._delivered_interim_texts == set()


def test_auto_continue_reset_is_idempotent_and_bounded():
    """Repeated auto-continue turns keep one set, not an unbounded cross-session leak."""
    agent = _Agent()
    _deliver(agent, ANSWER)
    first = set(agent._delivered_interim_texts)

    for _ in range(3):
        _reset(agent, "auto_continue")
        _deliver(agent, "A different follow-up line.")

    assert first <= agent._delivered_interim_texts
    assert len(agent.delivered) == 2, "each distinct message is still delivered once"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
