"""Tests for ``hermes_cli.voice`` — the TUI gateway's voice wrapper.

The module is imported *lazily* by ``tui_gateway/server.py`` so that a
box with missing audio deps fails at call time (returning a clean RPC
error) rather than at gateway startup. These tests therefore only
assert the public contract the gateway depends on: the three symbols
exist, ``stop_and_transcribe`` is a no-op when nothing is recording,
and ``speak_text`` tolerates empty input without touching the provider
stack.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestPublicAPI:
    def test_gateway_symbols_importable(self):
        """Match the exact import shape tui_gateway/server.py uses."""
        from hermes_cli.voice import (
            speak_text,
            start_recording,
            stop_and_transcribe,
        )

        assert callable(start_recording)
        assert callable(stop_and_transcribe)
        assert callable(speak_text)


class TestStopWithoutStart:
    def test_returns_none_when_no_recording_active(self, monkeypatch):
        """Idempotent no-op: stop before start must not raise or touch state."""
        import hermes_cli.voice as voice

        monkeypatch.setattr(voice, "_recorder", None)

        assert voice.stop_and_transcribe() is None


class TestSpeakTextGuards:
    @pytest.mark.parametrize("text", ["", "   ", "\n\t  "])
    def test_empty_text_is_noop(self, text):
        """Empty / whitespace-only text must return without importing tts_tool
        (the gateway spawns a thread per call, so a no-op on empty input
        keeps the thread pool from churning on trivial inputs)."""
        from hermes_cli.voice import speak_text

        # Should simply return None without raising.
        assert speak_text(text) is None


class TestContinuousAPI:
    """Continuous (VAD) mode API — CLI-parity loop entry points."""

    def test_continuous_exports(self):
        from hermes_cli.voice import (
            is_continuous_active,
            start_continuous,
            stop_continuous,
        )

        assert callable(start_continuous)
        assert callable(stop_continuous)
        assert callable(is_continuous_active)

    def test_not_active_by_default(self, monkeypatch):
        import hermes_cli.voice as voice

        # Isolate from any state left behind by other tests in the session.
        monkeypatch.setattr(voice, "_continuous_active", False)
        monkeypatch.setattr(voice, "_continuous_recorder", None)

        assert voice.is_continuous_active() is False

    def test_stop_continuous_idempotent_when_inactive(self, monkeypatch):
        """stop_continuous must not raise when no loop is active — the
        gateway's voice.toggle off path calls it unconditionally."""
        import hermes_cli.voice as voice

        monkeypatch.setattr(voice, "_continuous_active", False)
        monkeypatch.setattr(voice, "_continuous_recorder", None)

        # Should return cleanly without exceptions
        assert voice.stop_continuous() is None
        assert voice.is_continuous_active() is False

    def test_double_start_is_idempotent(self, monkeypatch):
        """A second start_continuous while already active is a no-op — prevents
        two overlapping capture threads fighting over the microphone when the
        UI double-fires (e.g. both /voice on and Ctrl+B within the same tick)."""
        import hermes_cli.voice as voice

        monkeypatch.setattr(voice, "_continuous_active", True)
        called = {"n": 0}

        class FakeRecorder:
            def start(self, on_silence_stop=None):
                called["n"] += 1

            def cancel(self):
                pass

        monkeypatch.setattr(voice, "_continuous_recorder", FakeRecorder())

        voice.start_continuous(on_transcript=lambda _t: None)

        # The guard inside start_continuous short-circuits before rec.start()
        assert called["n"] == 0


class TestContinuousLoopSimulation:
    """End-to-end simulation of the VAD loop with a fake recorder.

    Proves auto-restart works: the silence callback must trigger transcribe →
    on_transcript → re-call rec.start(on_silence_stop=same_cb). Also covers
    the 3-strikes no-speech halt.
    """

    @pytest.fixture
    def fake_recorder(self, monkeypatch):
        import hermes_cli.voice as voice

        # Reset module state between tests.
        monkeypatch.setattr(voice, "_continuous_active", False)
        monkeypatch.setattr(voice, "_continuous_recorder", None)
        monkeypatch.setattr(voice, "_continuous_no_speech_count", 0)
        monkeypatch.setattr(voice, "_continuous_on_transcript", None)
        monkeypatch.setattr(voice, "_continuous_on_status", None)
        monkeypatch.setattr(voice, "_continuous_on_silent_limit", None)

        class FakeRecorder:
            _silence_threshold = 200
            _silence_duration = 3.0
            is_recording = False

            def __init__(self):
                self.start_calls = 0
                self.last_callback = None
                self.stopped = 0
                self.cancelled = 0
                # Preset WAV path returned by stop()
                self.next_stop_wav = "/tmp/fake.wav"

            def start(self, on_silence_stop=None):
                self.start_calls += 1
                self.last_callback = on_silence_stop
                self.is_recording = True

            def stop(self):
                self.stopped += 1
                self.is_recording = False
                return self.next_stop_wav

            def cancel(self):
                self.cancelled += 1
                self.is_recording = False

        rec = FakeRecorder()
        monkeypatch.setattr(voice, "create_audio_recorder", lambda: rec)
        # Skip real file ops in the silence callback.
        monkeypatch.setattr(voice.os.path, "isfile", lambda _p: False)
        return rec

    def test_loop_auto_restarts_after_transcript(self, fake_recorder, monkeypatch):
        import hermes_cli.voice as voice

        monkeypatch.setattr(
            voice,
            "transcribe_recording",
            lambda _p: {"success": True, "transcript": "hello world"},
        )
        monkeypatch.setattr(voice, "is_whisper_hallucination", lambda _t: False)

        transcripts = []
        statuses = []

        voice.start_continuous(
            on_transcript=lambda t: transcripts.append(t),
            on_status=lambda s: statuses.append(s),
        )

        assert fake_recorder.start_calls == 1
        assert statuses == ["listening"]

        # Simulate AudioRecorder's silence detector firing.
        fake_recorder.last_callback()

        assert transcripts == ["hello world"]
        assert fake_recorder.start_calls == 2  # auto-restarted
        assert statuses == ["listening", "transcribing", "listening"]
        assert voice.is_continuous_active() is True

        voice.stop_continuous()

    def test_silent_limit_halts_loop_after_three_strikes(self, fake_recorder, monkeypatch):
        import hermes_cli.voice as voice

        # Transcription returns no speech — fake_recorder.stop() returns the
        # path, but transcribe returns empty text, counting as silence.
        monkeypatch.setattr(
            voice,
            "transcribe_recording",
            lambda _p: {"success": True, "transcript": ""},
        )
        monkeypatch.setattr(voice, "is_whisper_hallucination", lambda _t: False)

        transcripts = []
        silent_limit_fired = []

        voice.start_continuous(
            on_transcript=lambda t: transcripts.append(t),
            on_silent_limit=lambda: silent_limit_fired.append(True),
        )

        # Fire silence callback 3 times
        for _ in range(3):
            fake_recorder.last_callback()

        assert transcripts == []
        assert silent_limit_fired == [True]
        assert voice.is_continuous_active() is False
        assert fake_recorder.cancelled >= 1

    def test_stop_during_transcription_discards_restart(self, fake_recorder, monkeypatch):
        """User hits Ctrl+B mid-transcription: the in-flight transcript must
        still fire (it's a real utterance), but the loop must NOT restart."""
        import hermes_cli.voice as voice

        stop_triggered = {"flag": False}

        def late_transcribe(_p):
            # Simulate stop_continuous arriving while we're inside transcribe
            voice.stop_continuous()
            stop_triggered["flag"] = True
            return {"success": True, "transcript": "final word"}

        monkeypatch.setattr(voice, "transcribe_recording", late_transcribe)
        monkeypatch.setattr(voice, "is_whisper_hallucination", lambda _t: False)

        transcripts = []
        voice.start_continuous(on_transcript=lambda t: transcripts.append(t))

        initial_starts = fake_recorder.start_calls  # 1
        fake_recorder.last_callback()

        assert stop_triggered["flag"] is True
        # Loop is stopped — no auto-restart
        assert fake_recorder.start_calls == initial_starts
        # The in-flight transcript was suppressed because we stopped mid-flight
        assert transcripts == []
        assert voice.is_continuous_active() is False
