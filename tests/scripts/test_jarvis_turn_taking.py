import importlib.util
import os
import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
JARVIS_DIR = ROOT / "scripts" / "jarvis-voice"


def load_local_module(name: str):
    path = JARVIS_DIR / f"{name}.py"
    assert path.exists(), f"{path} missing"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def install_jarvis_import_stubs(monkeypatch):
    google = types.ModuleType("google")
    genai = types.ModuleType("google.genai")
    genai.types = types.SimpleNamespace()
    numpy = types.ModuleType("numpy")
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.genai", genai)
    monkeypatch.setitem(sys.modules, "google.genai.types", genai.types)
    monkeypatch.setitem(sys.modules, "numpy", numpy)
    monkeypatch.setitem(sys.modules, "sounddevice", types.SimpleNamespace())


class _FakeType:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def install_genai_type_stubs(monkeypatch):
    google = types.ModuleType("google")
    genai = types.ModuleType("google.genai")
    genai.types = types.SimpleNamespace(
        LiveConnectConfig=_FakeType,
        ThinkingConfig=_FakeType,
        SessionResumptionConfig=_FakeType,
        RealtimeInputConfig=_FakeType,
        AutomaticActivityDetection=_FakeType,
        EndSensitivity=types.SimpleNamespace(END_SENSITIVITY_HIGH="END_SENSITIVITY_HIGH"),
        ActivityHandling=types.SimpleNamespace(NO_INTERRUPTION="NO_INTERRUPTION"),
        TurnCoverage=types.SimpleNamespace(TURN_INCLUDES_ONLY_ACTIVITY="TURN_INCLUDES_ONLY_ACTIVITY"),
    )
    numpy = types.ModuleType("numpy")
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.genai", genai)
    monkeypatch.setitem(sys.modules, "google.genai.types", genai.types)
    monkeypatch.setitem(sys.modules, "numpy", numpy)
    monkeypatch.setitem(sys.modules, "sounddevice", types.SimpleNamespace())


def test_default_vad_waits_for_natural_pause(monkeypatch):
    install_jarvis_import_stubs(monkeypatch)
    monkeypatch.delenv("JARVIS_VAD_SILENCE_MS", raising=False)

    jarvis_live = load_local_module("jarvis_live")

    assert jarvis_live._read_vad_silence_ms() == 900


def test_invalid_vad_value_falls_back_to_natural_pause(monkeypatch):
    install_jarvis_import_stubs(monkeypatch)
    monkeypatch.setenv("JARVIS_VAD_SILENCE_MS", "not-a-number")

    jarvis_live = load_local_module("jarvis_live")

    assert jarvis_live._read_vad_silence_ms() == 900


def test_manual_activity_config_disables_server_vad(monkeypatch):
    install_genai_type_stubs(monkeypatch)
    monkeypatch.delenv("JARVIS_VAD_SILENCE_MS", raising=False)

    jarvis_live = load_local_module("jarvis_live")
    config = jarvis_live.build_config(handle=None, manual_activity=True)

    realtime = config.realtime_input_config
    assert realtime.automatic_activity_detection.disabled is True
    assert realtime.activity_handling == "NO_INTERRUPTION"
    assert realtime.turn_coverage == "TURN_INCLUDES_ONLY_ACTIVITY"


def test_auto_activity_config_keeps_no_interruption(monkeypatch):
    install_genai_type_stubs(monkeypatch)
    monkeypatch.delenv("JARVIS_VAD_SILENCE_MS", raising=False)

    jarvis_live = load_local_module("jarvis_live")
    config = jarvis_live.build_config(handle=None, manual_activity=False)

    realtime = config.realtime_input_config
    assert realtime.automatic_activity_detection.silence_duration_ms == 900
    assert realtime.activity_handling == "NO_INTERRUPTION"


def test_read_bool_env_accepts_on_values(monkeypatch):
    turn_taking = load_local_module("turn_taking")
    monkeypatch.setenv("JARVIS_START_ACTIVE", "1")

    assert turn_taking.read_bool_env("JARVIS_START_ACTIVE") is True


def test_read_bool_env_rejects_off_values(monkeypatch):
    turn_taking = load_local_module("turn_taking")
    monkeypatch.setenv("JARVIS_START_ACTIVE", "0")

    assert turn_taking.read_bool_env("JARVIS_START_ACTIVE") is False


def test_turn_gate_does_not_interrupt_playback_by_default():
    turn_taking = load_local_module("turn_taking")
    gate = turn_taking.TurnTakingGate(interrupt_rms=2400.0, interrupt_frames=3)

    assert gate.should_interrupt_playback(2500.0) is False
    assert gate.should_interrupt_playback(2600.0) is False
    assert gate.should_interrupt_playback(2700.0) is False


def test_turn_gate_can_interrupt_when_explicitly_enabled():
    turn_taking = load_local_module("turn_taking")
    gate = turn_taking.TurnTakingGate(
        interrupt_rms=2400.0,
        interrupt_frames=3,
        voice_interrupt_enabled=True,
    )

    assert gate.should_interrupt_playback(2500.0) is False
    assert gate.should_interrupt_playback(2600.0) is False
    assert gate.should_interrupt_playback(2700.0) is True


def test_turn_gate_keeps_short_tail_after_normal_playback():
    turn_taking = load_local_module("turn_taking")
    gate = turn_taking.TurnTakingGate(playback_tail_sec=0.7)

    gate.note_playback(10.0)

    assert gate.in_playback_tail(10.6) is True
    assert gate.in_playback_tail(10.8) is False


def test_server_interrupt_does_not_clear_active_playback_by_default():
    turn_taking = load_local_module("turn_taking")
    gate = turn_taking.TurnTakingGate()
    gate.note_playback(10.0)

    assert gate.should_clear_for_server_interrupt(now=10.2, playback_active=True) is False
    assert gate.should_clear_for_server_interrupt(now=11.0, playback_active=False) is True


def test_local_speech_detector_marks_start_and_end():
    turn_taking = load_local_module("turn_taking")
    detector = turn_taking.LocalSpeechTurnDetector(speech_rms=500.0, silence_ms=300)

    assert detector.observe(rms=700.0, frame_ms=100.0) == "start"
    assert detector.observe(rms=650.0, frame_ms=100.0) == "speech"
    assert detector.observe(rms=0.0, frame_ms=100.0) is None
    assert detector.observe(rms=0.0, frame_ms=100.0) is None
    assert detector.observe(rms=0.0, frame_ms=100.0) == "end"


def test_local_speech_detector_can_ignore_short_spikes():
    turn_taking = load_local_module("turn_taking")
    detector = turn_taking.LocalSpeechTurnDetector(
        speech_rms=500.0,
        silence_ms=300,
        min_speech_frames=2,
    )

    assert detector.observe(rms=700.0, frame_ms=100.0) is None
    assert detector.observe(rms=0.0, frame_ms=100.0) is None
    assert detector.observe(rms=700.0, frame_ms=100.0) is None
    assert detector.observe(rms=700.0, frame_ms=100.0) == "start"
