from gateway.calls.native.streaming.interruption import InterruptionPolicy
from gateway.calls.native.streaming.types import (
    InterruptionSignal, InterruptionParams, InterruptionAction, TurnEvent,
    TurnEventKind, TranscriptEvent, TranscriptKind,
)

P = InterruptionParams()


def _sig(**kw):
    base = dict(call_id="c", at_ms=0, assistant_speaking=True, turn_event=None,
               latest_partial=None, playhead=None, params=P,
               ms_since_speech_start=0, ms_since_assistant_silent_partial=0)
    base.update(kw)
    return InterruptionSignal(**base)


def test_not_speaking_waits():
    d = InterruptionPolicy().decide(_sig(assistant_speaking=False))
    assert d.action is InterruptionAction.WAIT


def test_backchannel_ignored():
    te = TurnEvent(call_id="c", kind=TurnEventKind.POSSIBLE_BACKCHANNEL, at_ms=0)
    d = InterruptionPolicy().decide(_sig(turn_event=te))
    assert d.action is InterruptionAction.IGNORE


def test_real_interrupt():
    partial = TranscriptEvent(call_id="c", kind=TranscriptKind.PARTIAL, text="stop talking now")
    d = InterruptionPolicy().decide(_sig(latest_partial=partial, ms_since_speech_start=400))
    assert d.action is InterruptionAction.INTERRUPT


def test_one_word_below_min_words_waits():
    partial = TranscriptEvent(call_id="c", kind=TranscriptKind.PARTIAL, text="uh")
    d = InterruptionPolicy().decide(_sig(latest_partial=partial, ms_since_speech_start=400))
    assert d.action is InterruptionAction.WAIT


def test_false_positive_resume_after_timeout():
    d = InterruptionPolicy().decide(_sig(ms_since_speech_start=P.false_interruption_timeout_ms + 1))
    assert d.action is InterruptionAction.RESUME


def test_backchannel_takes_priority_over_long_speech():
    """Backchannel returns IGNORE even if speech duration and word-count would qualify for INTERRUPT."""
    te = TurnEvent(call_id="c", kind=TurnEventKind.POSSIBLE_BACKCHANNEL, at_ms=0)
    partial = TranscriptEvent(call_id="c", kind=TranscriptKind.PARTIAL, text="stop talking now please")
    d = InterruptionPolicy().decide(_sig(
        turn_event=te,
        latest_partial=partial,
        ms_since_speech_start=P.min_speech_ms + 500,
    ))
    assert d.action is InterruptionAction.IGNORE


def test_interrupt_checked_before_resume_timeout():
    """A signal past false_interruption_timeout_ms that ALSO has enough words+duration → INTERRUPT, not RESUME."""
    partial = TranscriptEvent(call_id="c", kind=TranscriptKind.PARTIAL, text="stop talking now")
    d = InterruptionPolicy().decide(_sig(
        latest_partial=partial,
        ms_since_speech_start=P.false_interruption_timeout_ms + 500,
    ))
    assert d.action is InterruptionAction.INTERRUPT
