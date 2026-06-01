import dataclasses
import pytest
from gateway.calls.native.streaming.types import (
    AudioFrame, MediaFormat, BrainEvent, BrainEventKind, TurnEndReason, CallTurnRecord,
)


def test_audioframe_duration_ms_derived():
    media = MediaFormat(sample_rate=16000, channels=1, frame_ms=20)
    frame = AudioFrame(pcm16=b"\x00" * 640, media=media, timestamp_ms=0, seq=0)
    assert frame.duration_ms == 20


def test_brainevent_is_final():
    e = BrainEvent(call_id="c", kind=BrainEventKind.FINAL_TEXT, text="hi")
    assert e.is_final is True
    assert BrainEvent(call_id="c", kind=BrainEventKind.PARTIAL_TEXT).is_final is False


def test_callturnrecord_is_frozen():
    r = CallTurnRecord(call_id="c", turn_index=0, user_transcript="u", assistant_heard_text="h")
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.user_transcript = "x"  # type: ignore[misc]
    assert r.ended_reason is TurnEndReason.COMPLETED
