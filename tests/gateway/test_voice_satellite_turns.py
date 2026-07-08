"""Unit tests for the voice_satellite turn state machine."""

from tests.gateway._plugin_adapter_loader import load_plugin_adapter

_adapter = load_plugin_adapter("voice_satellite")
audio = _adapter._import_sibling("audio")
tm = _adapter._import_sibling("turn_machine")

SPEECH = b"\x00\x30" * 1600  # 0.1s @16k of loud samples (0x3000 = 12288 rms-ish)
SILENCE = b"\x00\x00" * 1600


def make_machine(**kwargs):
    factory = lambda: audio.EndpointDetector(
        silence_threshold=200, silence_duration=0.2, min_speech_seconds=0.1
    )
    return tm.TurnMachine(factory, **kwargs)


def test_full_turn_lifecycle():
    m = make_machine()
    assert m.phase is tm.TurnPhase.IDLE
    assert m.on_pipeline_start(now=0.0) is True
    assert m.phase is tm.TurnPhase.LISTENING

    assert m.on_audio(SPEECH, 0.1, 16000, now=0.1) is None
    assert m.on_audio(SILENCE, 0.1, 16000, now=0.2) is None
    action = m.on_audio(SILENCE, 0.1, 16000, now=0.3)
    assert action is not None and action[0] == "transcribe"
    assert action[2] == 16000
    assert m.phase is tm.TurnPhase.TRANSCRIBING

    assert m.on_transcript_ready("hello") == ("dispatch", "hello")
    assert m.phase is tm.TurnPhase.THINKING
    m.on_reply_started()
    assert m.phase is tm.TurnPhase.SPEAKING
    m.on_playback_done()
    assert m.phase is tm.TurnPhase.IDLE


def test_empty_transcript_aborts_to_idle():
    m = make_machine()
    m.on_pipeline_start(now=0.0)
    m.on_audio(SPEECH, 0.1, 16000, now=0.1)
    m.on_audio(SILENCE, 0.3, 16000, now=0.4)
    assert m.on_transcript_ready("") == ("abort",)
    assert m.phase is tm.TurnPhase.IDLE


def test_listen_timeout_aborts():
    m = make_machine(listen_timeout_seconds=1.0)
    m.on_pipeline_start(now=100.0)
    assert m.on_audio(SILENCE, 0.1, 16000, now=100.1) is None
    assert m.on_audio(SILENCE, 0.1, 16000, now=101.5) == ("abort",)
    assert m.phase is tm.TurnPhase.IDLE


def test_audio_ignored_outside_listening_and_reentry_ignored():
    m = make_machine()
    assert m.on_audio(SPEECH, 0.1, 16000, now=0.0) is None  # IDLE: ignored
    assert m.phase is tm.TurnPhase.IDLE
    m.on_pipeline_start(now=0.0)
    assert m.on_pipeline_start(now=0.1) is False  # already in a turn
    m.to_idle()
    assert m.phase is tm.TurnPhase.IDLE
