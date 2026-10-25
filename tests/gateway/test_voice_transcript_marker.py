"""A transcribed voice message must reach the model marked, and reach the transcript unmarked.

The gateway used to hand speech-to-text output to the model as ordinary text: the agent could
not tell a spoken message from a typed one, so it corrected phonetic proper nouns and answered
in tables a voice interface cannot read. The CLI already carries a voice prefix API-call-locally
(``turn.voice_prefix``); this is the gateway equivalent.
"""
from types import SimpleNamespace

from gateway.run import GatewayRunner


def _event(transcribed: bool) -> SimpleNamespace:
    return SimpleNamespace(_gateway_stt_transcribed=transcribed)


def test_transcribed_voice_is_marked_for_the_model_only():
    transcript = '"il faut deux kilos de tomates et une baguette"'

    model_text, persisted = GatewayRunner._hmwa_apply_voice_transcript_marker(
        _event(True), transcript, None,
    )

    assert model_text == f"{GatewayRunner.VOICE_TRANSCRIPT_MARKER}{transcript}"
    assert persisted == transcript


def test_typed_text_and_failed_transcription_stay_untouched():
    typed = "il faut deux kilos de tomates"
    assert GatewayRunner._hmwa_apply_voice_transcript_marker(_event(False), typed, None) == (typed, None)

    # Transcription failure leaves a neutral note; labelling it as a transcript would be a lie.
    note = "[voice message could not be transcribed automatically; the audio is available at: /x.ogg]"
    assert GatewayRunner._hmwa_apply_voice_transcript_marker(_event(False), note, note) == (note, note)
