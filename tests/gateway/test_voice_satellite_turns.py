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
    assert action[3] == m.turn_id
    assert m.phase is tm.TurnPhase.TRANSCRIBING

    assert m.on_transcript_ready("hello", action[3]) == ("dispatch", "hello")
    assert m.phase is tm.TurnPhase.THINKING
    m.on_reply_started()
    assert m.phase is tm.TurnPhase.SPEAKING
    m.on_playback_done()
    assert m.phase is tm.TurnPhase.IDLE


def test_empty_transcript_aborts_to_idle():
    m = make_machine()
    m.on_pipeline_start(now=0.0)
    m.on_audio(SPEECH, 0.1, 16000, now=0.1)
    action = m.on_audio(SILENCE, 0.3, 16000, now=0.4)
    assert m.on_transcript_ready("", action[3]) == ("abort",)
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


def test_transcript_in_wrong_phase_is_ignored_without_state_change():
    m = make_machine()
    assert m.on_transcript_ready("late result", m.turn_id) == ("stale",)  # IDLE
    assert m.phase is tm.TurnPhase.IDLE
    m.on_pipeline_start(now=0.0)
    assert m.on_transcript_ready("late result", m.turn_id) == ("stale",)  # LISTENING
    # A wrong-phase callback must not tear down the live turn.
    assert m.phase is tm.TurnPhase.LISTENING


def test_stale_transcript_from_previous_turn_cannot_hijack_new_turn():
    m = make_machine()
    m.on_pipeline_start(now=0.0)
    m.on_audio(SPEECH, 0.1, 16000, now=0.1)
    old_action = m.on_audio(SILENCE, 0.3, 16000, now=0.4)
    old_turn = old_action[3]
    m.to_idle()  # link dropped / turn aborted while STT was in flight

    m.on_pipeline_start(now=1.0)  # new wake before the old STT returns
    m.on_audio(SPEECH, 0.1, 16000, now=1.1)
    new_action = m.on_audio(SILENCE, 0.3, 16000, now=1.4)
    assert m.phase is tm.TurnPhase.TRANSCRIBING

    # The old turn's STT result lands now: rejected, new turn untouched.
    assert m.on_transcript_ready("stale words", old_turn) == ("stale",)
    assert m.phase is tm.TurnPhase.TRANSCRIBING
    # The new turn's own transcript still dispatches normally.
    assert m.on_transcript_ready("fresh words", new_action[3]) == (
        "dispatch", "fresh words",
    )


def test_on_reply_started_allows_thinking_and_idle_dedups_speaking():
    m = make_machine()
    m.on_reply_started()  # IDLE: announce path
    assert m.phase is tm.TurnPhase.SPEAKING
    m.on_reply_started()  # duplicate: no-op
    assert m.phase is tm.TurnPhase.SPEAKING
    m.on_playback_done()
    assert m.phase is tm.TurnPhase.IDLE
