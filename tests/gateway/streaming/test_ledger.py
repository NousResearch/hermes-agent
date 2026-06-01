from gateway.calls.native.streaming.ledger import HeardSpanLedger
from gateway.calls.native.streaming.types import FlushResult, PlaybackMark, TurnEndReason


def test_full_playback_heard_all():
    led = HeardSpanLedger("call")
    full = "hello there friend"
    led.note_mark(PlaybackMark("call", char_offset=len(full), text_so_far=full, at_ms=100))
    rec = led.record(user_transcript="hi", full_text=full, reason=TurnEndReason.COMPLETED)
    assert rec.assistant_heard_text == full
    assert rec.assistant_abandoned_text == ""
    assert rec.interrupted is False


def test_partial_heard_truncation():
    led = HeardSpanLedger("call")
    full = "the quick brown fox jumps"
    led.note_mark(PlaybackMark("call", char_offset=9, text_so_far="the quick", at_ms=80))
    led.note_flush(FlushResult(dropped_frames=4, dropped_ms=80, last_sent_mark=None), full)
    rec = led.record(user_transcript="u", full_text=full, reason=TurnEndReason.BARGED_IN)
    assert rec.assistant_heard_text == "the quick"
    assert rec.assistant_abandoned_text == full[9:]
    assert rec.interrupted is True
    assert rec.ended_reason is TurnEndReason.BARGED_IN


def test_barge_in_during_thinking_nothing_heard():
    led = HeardSpanLedger("call")
    full = "discarded answer"
    rec = led.record(user_transcript="u", full_text=full, reason=TurnEndReason.BARGED_IN)
    assert rec.assistant_heard_text == ""
    assert rec.assistant_abandoned_text == full
    assert rec.interrupted is True


def test_false_interruption_marks_interrupted():
    led = HeardSpanLedger("call")
    full = "actually i was going to say"
    # Heard some prefix via a mark
    led.note_mark(PlaybackMark("call", char_offset=10, text_so_far="actually i", at_ms=50))
    rec = led.record(user_transcript="oh", full_text=full, reason=TurnEndReason.FALSE_INTERRUPTION)
    assert rec.interrupted is True
    assert rec.assistant_heard_text == "actually i"
    assert rec.assistant_abandoned_text == full[10:]
    assert rec.ended_reason is TurnEndReason.FALSE_INTERRUPTION


def test_note_mark_monotonic():
    led = HeardSpanLedger("call")
    full = "one two three four five"
    # Record a later mark first (char_offset=12)
    led.note_mark(PlaybackMark("call", char_offset=12, text_so_far="one two thre", at_ms=60))
    # Then record an earlier (out-of-order) mark with smaller offset — must be ignored
    led.note_mark(PlaybackMark("call", char_offset=7, text_so_far="one two", at_ms=30))
    rec = led.record(user_transcript="u", full_text=full, reason=TurnEndReason.BARGED_IN)
    # Heard span should still be offset=12, not regressed to 7
    assert rec.assistant_heard_text == "one two thre"
    assert rec.assistant_abandoned_text == full[12:]


def test_turn_index_passthrough():
    led = HeardSpanLedger("call", turn_index=3)
    full = "anything"
    rec = led.record(user_transcript="u", full_text=full, reason=TurnEndReason.COMPLETED)
    assert rec.turn_index == 3


def test_note_flush_with_mark_advances_heard_span():
    # The mark-carrying flush path (real transports return a non-None last_sent_mark).
    led = HeardSpanLedger("call")
    full = "alpha beta gamma delta"
    flush = FlushResult(
        dropped_frames=2,
        dropped_ms=40,
        last_sent_mark=PlaybackMark("call", char_offset=10, text_so_far="alpha beta", at_ms=70),
    )
    led.note_flush(flush, full)
    rec = led.record(user_transcript="u", full_text=full, reason=TurnEndReason.BARGED_IN)
    assert rec.assistant_heard_text == "alpha beta"
    assert rec.assistant_abandoned_text == full[10:]
    assert rec.interrupted is True


def test_note_flush_mark_does_not_regress_existing_span():
    # A flush mark behind an already-recorded mark must not shrink the heard span.
    led = HeardSpanLedger("call")
    full = "alpha beta gamma delta"
    led.note_mark(PlaybackMark("call", char_offset=16, text_so_far="alpha beta gamma", at_ms=80))
    flush = FlushResult(
        dropped_frames=1,
        dropped_ms=20,
        last_sent_mark=PlaybackMark("call", char_offset=5, text_so_far="alpha", at_ms=90),
    )
    led.note_flush(flush, full)
    rec = led.record(user_transcript="u", full_text=full, reason=TurnEndReason.BARGED_IN)
    assert rec.assistant_heard_text == "alpha beta gamma"  # not regressed to "alpha"
