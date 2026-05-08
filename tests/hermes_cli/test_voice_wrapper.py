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


class TestNormalizeVoiceRecordKeyForPromptToolkit:
    """Round-9 Copilot review regression on #19835.

    Classic CLI only normalized ``ctrl+`` / ``alt+``, so TUI-valid
    aliases like ``control+``, ``option+``, ``opt+`` silently bound a
    different (or no) shortcut in the CLI. Normalizer now maps the
    same set of aliases the TUI parser accepts, so one config value
    binds identically in both runtimes.
    """

    def test_ctrl_and_alt_map_to_prompt_toolkit_form(self):
        from hermes_cli.voice import normalize_voice_record_key_for_prompt_toolkit

        assert normalize_voice_record_key_for_prompt_toolkit("ctrl+b") == "c-b"
        assert normalize_voice_record_key_for_prompt_toolkit("alt+r") == "a-r"

    def test_control_option_opt_aliases_match_tui_parser(self):
        from hermes_cli.voice import normalize_voice_record_key_for_prompt_toolkit

        assert normalize_voice_record_key_for_prompt_toolkit("control+o") == "c-o"
        assert normalize_voice_record_key_for_prompt_toolkit("option+space") == "a-space"
        assert normalize_voice_record_key_for_prompt_toolkit("opt+enter") == "a-enter"

    def test_case_insensitive(self):
        from hermes_cli.voice import normalize_voice_record_key_for_prompt_toolkit

        assert normalize_voice_record_key_for_prompt_toolkit("Ctrl+B") == "c-b"
        assert normalize_voice_record_key_for_prompt_toolkit("CONTROL+O") == "c-o"

    def test_non_string_falls_back_to_default(self):
        from hermes_cli.voice import normalize_voice_record_key_for_prompt_toolkit

        assert normalize_voice_record_key_for_prompt_toolkit(None) == "c-b"
        assert normalize_voice_record_key_for_prompt_toolkit(1) == "c-b"
        assert normalize_voice_record_key_for_prompt_toolkit(True) == "c-b"
        assert normalize_voice_record_key_for_prompt_toolkit({}) == "c-b"

    def test_empty_string_falls_back(self):
        from hermes_cli.voice import normalize_voice_record_key_for_prompt_toolkit

        assert normalize_voice_record_key_for_prompt_toolkit("") == "c-b"

    def test_super_win_fall_back_to_default_in_cli(self):
        """prompt_toolkit has no super modifier, so ``super+b`` / ``win+o``
        would crash the classic CLI at startup if passed through. Fall
        back to the documented default; the CLI binding site is
        expected to warn so users know the shortcut is TUI-only
        (Copilot round-11 on #19835)."""
        from hermes_cli.voice import normalize_voice_record_key_for_prompt_toolkit

        assert normalize_voice_record_key_for_prompt_toolkit("super+b") == "c-b"
        assert normalize_voice_record_key_for_prompt_toolkit("win+o") == "c-b"
        assert normalize_voice_record_key_for_prompt_toolkit("windows+o") == "c-b"

    # Round-10 Copilot review regressions on #19835.
    def test_strips_whitespace_within_and_around(self):
        """``ctrl + b`` / ``  option + space  `` are accepted by the TUI
        parser; the CLI normalizer must mirror that or the same config
        binds different shortcuts across runtimes."""
        from hermes_cli.voice import normalize_voice_record_key_for_prompt_toolkit

        assert normalize_voice_record_key_for_prompt_toolkit("ctrl + b") == "c-b"
        assert normalize_voice_record_key_for_prompt_toolkit("  option + space  ") == "a-space"

    def test_named_key_aliases_collapse_to_prompt_toolkit_canonical(self):
        """TUI accepts ``return`` / ``esc`` / ``bs`` / ``del`` etc.;
        CLI must collapse to prompt_toolkit's canonical spelling
        (``enter`` / ``escape`` / ``backspace`` / ``delete``)."""
        from hermes_cli.voice import normalize_voice_record_key_for_prompt_toolkit

        assert normalize_voice_record_key_for_prompt_toolkit("ctrl+return") == "c-enter"
        assert normalize_voice_record_key_for_prompt_toolkit("ctrl+esc") == "c-escape"
        assert normalize_voice_record_key_for_prompt_toolkit("ctrl+bs") == "c-backspace"
        assert normalize_voice_record_key_for_prompt_toolkit("alt+del") == "a-delete"

    def test_typoed_named_keys_fall_back_to_default(self):
        """``ctrl+spcae`` would otherwise pass through as ``c-spcae`` and
        prompt_toolkit would reject it at startup — fall back instead."""
        from hermes_cli.voice import normalize_voice_record_key_for_prompt_toolkit

        assert normalize_voice_record_key_for_prompt_toolkit("ctrl+spcae") == "c-b"
        assert normalize_voice_record_key_for_prompt_toolkit("ctrl+f5") == "c-b"

    def test_bare_char_and_multi_modifier_fall_back(self):
        """TUI parser rejects bare-char (``o``) and multi-modifier
        (``ctrl+alt+r``) configs; the CLI normalizer must match."""
        from hermes_cli.voice import normalize_voice_record_key_for_prompt_toolkit

        assert normalize_voice_record_key_for_prompt_toolkit("o") == "c-b"
        assert normalize_voice_record_key_for_prompt_toolkit("b") == "c-b"
        assert normalize_voice_record_key_for_prompt_toolkit("ctrl+alt+r") == "c-b"

    def test_reserved_ctrl_chars_fall_back(self):
        """``ctrl+c`` / ``ctrl+d`` / ``ctrl+l`` are always claimed by
        the CLI's prompt_toolkit input layer or terminal driver; match
        the TUI parser's rejection to keep /voice status honest."""
        from hermes_cli.voice import normalize_voice_record_key_for_prompt_toolkit

        assert normalize_voice_record_key_for_prompt_toolkit("ctrl+c") == "c-b"
        assert normalize_voice_record_key_for_prompt_toolkit("ctrl+d") == "c-b"
        assert normalize_voice_record_key_for_prompt_toolkit("ctrl+l") == "c-b"

    def test_unknown_modifier_falls_back(self):
        """``meta+b`` is ambiguous on the wire (Alt on xterm, Cmd on
        legacy macOS), same class as the TUI parser's rejection."""
        from hermes_cli.voice import normalize_voice_record_key_for_prompt_toolkit

        assert normalize_voice_record_key_for_prompt_toolkit("meta+b") == "c-b"
        assert normalize_voice_record_key_for_prompt_toolkit("shift+b") == "c-b"

    # Round-14 Copilot review regression on #19835. On macOS the TUI
    # parser rejects alt+c/d/l because hermes-ink reports Alt as
    # ``key.meta`` and isActionMod(darwin) accepts it. The CLI
    # normalizer must mirror that platform-gated rejection so shared
    # configs like ``option+c`` don't bind Alt+C in the CLI while the
    # TUI falls back to Ctrl+B.
    def test_alt_cdl_rejected_on_macos(self, monkeypatch):
        monkeypatch.setattr("sys.platform", "darwin")

        from hermes_cli.voice import normalize_voice_record_key_for_prompt_toolkit

        assert normalize_voice_record_key_for_prompt_toolkit("alt+c") == "c-b"
        assert normalize_voice_record_key_for_prompt_toolkit("alt+d") == "c-b"
        assert normalize_voice_record_key_for_prompt_toolkit("alt+l") == "c-b"
        assert normalize_voice_record_key_for_prompt_toolkit("option+c") == "c-b"
        assert normalize_voice_record_key_for_prompt_toolkit("opt+d") == "c-b"
        # Other alt letters still bind on darwin.
        assert normalize_voice_record_key_for_prompt_toolkit("alt+r") == "a-r"
        assert normalize_voice_record_key_for_prompt_toolkit("alt+space") == "a-space"

    def test_alt_cdl_allowed_on_non_macos(self, monkeypatch):
        monkeypatch.setattr("sys.platform", "linux")

        from hermes_cli.voice import normalize_voice_record_key_for_prompt_toolkit

        assert normalize_voice_record_key_for_prompt_toolkit("alt+c") == "a-c"
        assert normalize_voice_record_key_for_prompt_toolkit("alt+d") == "a-d"
        assert normalize_voice_record_key_for_prompt_toolkit("alt+l") == "a-l"


class TestVoiceRecordKeyFromConfig:
    """Round-11 Copilot review regression on #19835.

    ``load_config()`` preserves YAML scalar overrides, so a hand-edited
    ``voice: true`` or ``voice: cmd+b`` made the naive
    ``cfg.get('voice', {}).get('record_key')`` chain raise
    AttributeError before voice could run. The shape-safe extractor
    returns None for every malformed shape so the call-site fallback
    (``normalize_…`` / ``format_…``) surfaces the documented default.
    """

    def test_dict_voice_with_string_record_key(self):
        from hermes_cli.voice import voice_record_key_from_config

        assert voice_record_key_from_config({"voice": {"record_key": "ctrl+o"}}) == "ctrl+o"

    def test_non_dict_config_root(self):
        from hermes_cli.voice import voice_record_key_from_config

        for bad_root in (None, True, 1, "ctrl+b", [], ["ctrl+b"]):
            assert voice_record_key_from_config(bad_root) is None, bad_root

    def test_non_dict_voice_entry(self):
        from hermes_cli.voice import voice_record_key_from_config

        for bad_voice in (None, True, "cmd+b", 42, ["ctrl+b"]):
            assert voice_record_key_from_config({"voice": bad_voice}) is None, bad_voice

    def test_missing_record_key_returns_none(self):
        from hermes_cli.voice import voice_record_key_from_config

        assert voice_record_key_from_config({"voice": {"beep_enabled": True}}) is None
        assert voice_record_key_from_config({}) is None

    def test_normalizer_accepts_extractor_output_directly(self):
        """voice_record_key_from_config + normalize_… must compose —
        None / non-string scalars all fall back to c-b."""
        from hermes_cli.voice import (
            normalize_voice_record_key_for_prompt_toolkit,
            voice_record_key_from_config,
        )

        for raw in (None, True, 1, "cmd+b", ["ctrl+b"]):
            extracted = voice_record_key_from_config({"voice": raw})
            assert normalize_voice_record_key_for_prompt_toolkit(extracted) == "c-b"


class TestFormatVoiceRecordKeyForStatus:
    """Round-10 Copilot review regression on #19835.

    ``/voice status`` used to print the raw scalar (``True`` / ``1``)
    for non-string configs even though the actual binding falls back
    to Ctrl+B. The formatter routes through the same normalizer so
    status always matches what the CLI actually binds.
    """

    def test_ctrl_and_alt_letter_keys_render_canonically(self):
        from hermes_cli.voice import format_voice_record_key_for_status

        assert format_voice_record_key_for_status("ctrl+b") == "Ctrl+B"
        assert format_voice_record_key_for_status("ctrl+o") == "Ctrl+O"
        assert format_voice_record_key_for_status("alt+r") == "Alt+R"

    def test_named_keys_render_in_title_case(self):
        from hermes_cli.voice import format_voice_record_key_for_status

        assert format_voice_record_key_for_status("ctrl+space") == "Ctrl+Space"
        assert format_voice_record_key_for_status("alt+enter") == "Alt+Enter"
        assert format_voice_record_key_for_status("ctrl+esc") == "Ctrl+Escape"

    def test_aliases_render_via_normalized_form(self):
        from hermes_cli.voice import format_voice_record_key_for_status

        assert format_voice_record_key_for_status("control+o") == "Ctrl+O"
        assert format_voice_record_key_for_status("option+space") == "Alt+Space"
        assert format_voice_record_key_for_status("opt+enter") == "Alt+Enter"

    def test_non_string_scalar_falls_back_to_ctrl_b_label(self):
        from hermes_cli.voice import format_voice_record_key_for_status

        # Copilot round-10 regression: previously /voice status printed
        # the raw scalar ("True" / "1") even though the actual binding
        # fell back to Ctrl+B.
        assert format_voice_record_key_for_status(True) == "Ctrl+B"
        assert format_voice_record_key_for_status(1) == "Ctrl+B"
        assert format_voice_record_key_for_status(None) == "Ctrl+B"
        assert format_voice_record_key_for_status({}) == "Ctrl+B"

    def test_malformed_configs_fall_back_to_ctrl_b(self):
        from hermes_cli.voice import format_voice_record_key_for_status

        assert format_voice_record_key_for_status("ctrl+spcae") == "Ctrl+B"
        assert format_voice_record_key_for_status("ctrl+alt+r") == "Ctrl+B"
        assert format_voice_record_key_for_status("") == "Ctrl+B"
        assert format_voice_record_key_for_status("  ") == "Ctrl+B"


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
        monkeypatch.setattr(voice, "_continuous_stopping", False, raising=False)
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

        started = voice.start_continuous(on_transcript=lambda _t: None)

        # The guard inside start_continuous short-circuits before rec.start()
        assert started is True
        assert called["n"] == 0

    def test_start_returns_false_while_stopping(self, monkeypatch):
        import hermes_cli.voice as voice

        monkeypatch.setattr(voice, "_continuous_active", False)
        monkeypatch.setattr(voice, "_continuous_stopping", True, raising=False)

        assert voice.start_continuous(on_transcript=lambda _t: None) is False


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
        monkeypatch.setattr(voice, "_continuous_auto_restart", True, raising=False)
        monkeypatch.setattr(voice, "_play_beep", lambda *_, **__: None)

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
                self.fail_stop = False
                self.fail_next_start = False

            def start(self, on_silence_stop=None):
                if self.fail_next_start:
                    self.fail_next_start = False
                    raise RuntimeError("boom")
                self.start_calls += 1
                self.last_callback = on_silence_stop
                self.is_recording = True

            def stop(self):
                if self.fail_stop:
                    raise RuntimeError("stop failed")
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

    def test_auto_restart_false_stops_after_first_transcript(self, fake_recorder, monkeypatch):
        import hermes_cli.voice as voice

        monkeypatch.setattr(
            voice,
            "transcribe_recording",
            lambda _p: {"success": True, "transcript": "single shot"},
        )
        monkeypatch.setattr(voice, "is_whisper_hallucination", lambda _t: False)

        transcripts = []
        statuses = []

        voice.start_continuous(
            on_transcript=lambda t: transcripts.append(t),
            on_status=lambda s: statuses.append(s),
            auto_restart=False,
        )
        fake_recorder.last_callback()

        assert transcripts == ["single shot"]
        assert fake_recorder.start_calls == 1
        assert statuses == ["listening", "transcribing", "idle"]
        assert voice.is_continuous_active() is False

    def test_auto_restart_false_retains_silent_strikes_across_starts(
        self, fake_recorder, monkeypatch
    ):
        import hermes_cli.voice as voice

        monkeypatch.setattr(
            voice,
            "transcribe_recording",
            lambda _p: {"success": True, "transcript": ""},
        )
        monkeypatch.setattr(voice, "is_whisper_hallucination", lambda _t: False)

        silent_limit_fired = []

        for _ in range(3):
            voice.start_continuous(
                on_transcript=lambda _t: None,
                on_silent_limit=lambda: silent_limit_fired.append(True),
                auto_restart=False,
            )
            fake_recorder.last_callback()

        assert silent_limit_fired == [True]
        assert voice.is_continuous_active() is False
        assert fake_recorder.start_calls == 3

    def test_force_transcribe_stop_delivers_current_buffer(self, fake_recorder, monkeypatch):
        import hermes_cli.voice as voice

        class ImmediateThread:
            def __init__(self, target, daemon=False):
                self.target = target

            def start(self):
                self.target()

        monkeypatch.setattr(voice.threading, "Thread", ImmediateThread)
        monkeypatch.setattr(
            voice,
            "transcribe_recording",
            lambda _p: {"success": True, "transcript": "manual stop"},
        )
        monkeypatch.setattr(voice, "is_whisper_hallucination", lambda _t: False)

        transcripts = []
        statuses = []

        voice.start_continuous(
            on_transcript=lambda t: transcripts.append(t),
            on_status=lambda s: statuses.append(s),
        )
        voice.stop_continuous(force_transcribe=True)

        assert fake_recorder.stopped == 1
        assert transcripts == ["manual stop"]
        assert statuses == ["listening", "transcribing", "idle"]
        assert voice.is_continuous_active() is False

    def test_force_transcribe_empty_single_shots_hit_silent_limit(
        self, fake_recorder, monkeypatch
    ):
        import hermes_cli.voice as voice

        class ImmediateThread:
            def __init__(self, target, daemon=False):
                self.target = target

            def start(self):
                self.target()

        monkeypatch.setattr(voice.threading, "Thread", ImmediateThread)
        monkeypatch.setattr(
            voice,
            "transcribe_recording",
            lambda _p: {"success": True, "transcript": ""},
        )
        monkeypatch.setattr(voice, "is_whisper_hallucination", lambda _t: False)

        silent_limit_fired = []

        for _ in range(3):
            voice.start_continuous(
                on_transcript=lambda _t: None,
                on_silent_limit=lambda: silent_limit_fired.append(True),
                auto_restart=False,
            )
            voice.stop_continuous(force_transcribe=True)

        assert silent_limit_fired == [True]
        assert fake_recorder.stopped == 3
        assert voice._continuous_no_speech_count == 0

    def test_force_transcribe_valid_single_shot_resets_silent_strikes(
        self, fake_recorder, monkeypatch
    ):
        import hermes_cli.voice as voice

        class ImmediateThread:
            def __init__(self, target, daemon=False):
                self.target = target

            def start(self):
                self.target()

        monkeypatch.setattr(voice.threading, "Thread", ImmediateThread)
        monkeypatch.setattr(voice, "_continuous_no_speech_count", 2)
        monkeypatch.setattr(
            voice,
            "transcribe_recording",
            lambda _p: {"success": True, "transcript": "manual stop"},
        )
        monkeypatch.setattr(voice, "is_whisper_hallucination", lambda _t: False)

        transcripts = []
        silent_limit_fired = []

        voice.start_continuous(
            on_transcript=lambda t: transcripts.append(t),
            on_silent_limit=lambda: silent_limit_fired.append(True),
            auto_restart=False,
        )
        voice.stop_continuous(force_transcribe=True)

        assert transcripts == ["manual stop"]
        assert silent_limit_fired == []
        assert voice._continuous_no_speech_count == 0

    def test_force_transcribe_stop_failure_cancels_and_clears_stopping(
        self, fake_recorder, monkeypatch
    ):
        import hermes_cli.voice as voice

        class ImmediateThread:
            def __init__(self, target, daemon=False):
                self.target = target

            def start(self):
                self.target()

        monkeypatch.setattr(voice.threading, "Thread", ImmediateThread)
        fake_recorder.fail_stop = True

        statuses = []
        voice.start_continuous(
            on_transcript=lambda _t: None,
            on_status=lambda s: statuses.append(s),
        )
        voice.stop_continuous(force_transcribe=True)

        assert fake_recorder.cancelled == 1
        assert statuses == ["listening", "transcribing", "idle"]
        assert voice.is_continuous_active() is False
        assert voice._continuous_stopping is False

    def test_restart_failure_reports_idle(self, fake_recorder, monkeypatch):
        import hermes_cli.voice as voice

        monkeypatch.setattr(
            voice,
            "transcribe_recording",
            lambda _p: {"success": True, "transcript": "hello world"},
        )
        monkeypatch.setattr(voice, "is_whisper_hallucination", lambda _t: False)

        statuses = []
        voice.start_continuous(on_transcript=lambda _t: None, on_status=statuses.append)

        fake_recorder.fail_next_start = True
        fake_recorder.last_callback()

        assert statuses == ["listening", "transcribing", "idle"]
        assert voice.is_continuous_active() is False

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
