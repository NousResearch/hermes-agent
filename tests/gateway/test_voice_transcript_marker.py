"""A transcribed voice message must reach the model marked, and reach the transcript unmarked.

The gateway used to hand speech-to-text output to the model as ordinary text: the agent could
not tell a spoken message from a typed one, so it "corrected" phonetic proper nouns and answered
in tables a voice interface cannot read. The prefix (and the instruction it carries) is the same
one the CLI already applies API-call-locally via ``turn.voice_prefix``.
"""
import asyncio
from types import SimpleNamespace

from gateway.run import GatewayRunner
from hermes_constants import VOICE_INPUT_PREFIX


def _event(transcribed: bool) -> SimpleNamespace:
    return SimpleNamespace(_gateway_stt_transcribed=transcribed)


def test_transcribed_voice_is_marked_for_the_model_only():
    transcript = '"il faut deux kilos de tomates et une baguette"'

    model_text, persisted = GatewayRunner._hmwa_apply_voice_transcript_marker(
        _event(True), transcript, None,
    )

    assert model_text == f"{VOICE_INPUT_PREFIX}{transcript}"
    assert persisted == transcript


def test_typed_text_and_failed_transcription_stay_untouched():
    typed = "il faut deux kilos de tomates"
    assert GatewayRunner._hmwa_apply_voice_transcript_marker(_event(False), typed, None) == (typed, None)

    # Transcription failure leaves a neutral note; labelling it as a transcript would be a lie.
    note = "[voice message could not be transcribed automatically; the audio is available at: /x.ogg]"
    assert GatewayRunner._hmwa_apply_voice_transcript_marker(_event(False), note, note) == (note, note)


def test_cached_transcript_still_marks_the_turn():
    """A rebuilt/copied pending event carries the cached transcript; the flag must follow it,
    or the marker disappears silently on interrupt and drain paths."""
    event = SimpleNamespace(
        _gateway_pending_stt_text="salut, deux kilos de tomates",
        _gateway_pending_stt_transcripts=["salut, deux kilos de tomates"],
    )

    text, transcripts = asyncio.run(
        GatewayRunner._transcribe_pending_audio_event_once(object.__new__(GatewayRunner), event)
    )

    assert text == "salut, deux kilos de tomates"
    assert transcripts == ["salut, deux kilos de tomates"]
    assert event._gateway_stt_transcribed is True
