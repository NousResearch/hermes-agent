"""Regression tests for the test-suite audio-playback guard.

The incident: a plain test run spoke the words "partial answer complete" out
of a developer's speakers. That string is a fake ``final_response`` from
``tests/test_tui_gateway_server.py``. The route was entirely in-process — no
leaked shell variable, so ``scripts/run_tests.sh``'s ``env -i`` was no help:

  1. A test drives the ``voice.toggle`` RPC with ``action="tts"``. The
     handler flips the flag by writing the real process environment:
     ``os.environ["HERMES_VOICE_TTS"] = "1"``.
  2. The flag outlives that test (``monkeypatch.delenv`` on an *absent* key
     records no undo entry), so every later test in the process sees it.
  3. Any later test that drives a turn to completion hits the TTS dispatch in
     ``prompt.submit``, which calls ``hermes_cli.voice.speak_text`` on a
     daemon thread with the turn's final response text.
  4. ``speak_text`` needs no API key to be audible — ``tools/tts_tool.py``
     defaults to the keyless ``edge`` provider.

Two independent defences in ``tests/conftest.py``, one test each below:

  * ``_HERMES_BEHAVIORAL_VARS`` now blanks ``HERMES_VOICE`` /
    ``HERMES_VOICE_TTS`` at every test setup, so step 2 cannot cross a test
    boundary.
  * ``_audio_playback_guard`` stubs ``speak_text`` outright, so step 3 stays
    silent even *within* the test that set the flag itself.

The second is the load-bearing one: env blanking alone cannot stop code under
test from re-setting the variable mid-test.
"""

import os
import subprocess
import sys
import types
import wave

import pytest

from tui_gateway import server


def test_voice_toggle_still_leaks_the_env_var_but_speech_is_stubbed(monkeypatch):
    """The dangerous primitive is neutralised even when the flag IS set.

    This reproduces the incident's first step for real — it drives the same
    ``voice.toggle`` RPC that the original culprit test does, and asserts the
    handler really does write ``os.environ`` (so the guard is being tested
    against live behaviour, not a straw man). It then walks the second step:
    ``speak_text`` is called exactly as ``prompt.submit`` calls it, with the
    exact string the user heard — and must not reach the TTS backend.
    """
    monkeypatch.setattr(server, "_load_cfg", lambda: {"voice": {}})
    monkeypatch.setitem(
        sys.modules,
        "tools.voice_mode",
        types.SimpleNamespace(
            check_voice_requirements=lambda: {"available": True, "details": ""}
        ),
    )
    monkeypatch.setenv("HERMES_VOICE", "1")

    resp = server.handle_request(
        {"id": "tts", "method": "voice.toggle", "params": {"action": "tts"}}
    )

    # The handler mutates the real process environment. This is the leak;
    # it is upstream behaviour we are guarding against, not asserting away.
    assert resp["result"]["tts"] is True
    assert os.environ.get("HERMES_VOICE_TTS") == "1"
    assert server._voice_tts_enabled() is True

    # Any call into the TTS backend from here on would be real synthesis and
    # real playback. Stand a recorder in front of it — a *recorder*, not a
    # raising stub, because ``speak_text`` wraps its whole body in
    # ``except Exception`` and would swallow an AssertionError, quietly
    # turning this test green whether or not the guard is doing anything.
    import tools.tts_tool as tts_tool

    calls = []
    monkeypatch.setattr(
        tts_tool,
        "text_to_speech_tool",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    # Called exactly as tui_gateway/server.py's prompt.submit completion path
    # calls it (a late `from hermes_cli.voice import speak_text`), with the
    # exact fixture string that came out of the speakers.
    from hermes_cli.voice import speak_text

    assert speak_text("partial answer complete") is None
    assert calls == [], (
        "audio guard breached: speak_text reached the TTS backend — on an "
        "unguarded run this is real synthesis through the keyless 'edge' "
        "provider, played through the speakers"
    )


def test_voice_env_does_not_leak_into_the_next_test():
    """Second defence: the flag the previous test set must not have survived.

    Ordering matters — this test only means anything because it runs after the
    one above, in the same process, which left ``HERMES_VOICE_TTS=1`` set.
    Before ``HERMES_VOICE``/``HERMES_VOICE_TTS`` were added to
    ``_HERMES_BEHAVIORAL_VARS``, this assertion failed.
    """
    assert "HERMES_VOICE_TTS" not in os.environ
    assert "HERMES_VOICE" not in os.environ
    assert server._voice_tts_enabled() is False
    assert server._voice_mode_enabled() is False


def test_guard_can_be_opted_out_of_explicitly():
    """The stub is a guard, not a lobotomy — the real function is reachable."""
    import hermes_cli.voice as voice

    assert voice.speak_text.__name__ == "_blocked_speak_text"


@pytest.mark.real_audio_playback
def test_bypass_marker_restores_the_real_speak_text():
    """``@pytest.mark.real_audio_playback`` hands back the real primitive.

    Asserts identity only — it does not call it, which would speak aloud.
    """
    import hermes_cli.voice as voice

    assert voice.speak_text.__name__ == "speak_text"


def _write_silent_wav(path):
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\x00\x00" * 160)


def _force_system_player_path(monkeypatch, vm, which_result):
    """Steer ``play_audio_file`` to the afplay spawn regardless of host."""
    monkeypatch.setattr(vm, "_sounddevice_output_allowed", lambda: False)
    monkeypatch.setattr(vm.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(vm.shutil, "which", lambda name: which_result)


def test_system_player_spawn_is_neutralised(tmp_path, monkeypatch):
    """Third route (2026-09-04 incident): beeps and "Hello world." on macOS.

    ``play_beep`` and ``tools.tts_tool.stream_tts_to_speaker`` never touch
    ``hermes_cli.voice`` — both end in ``tools.voice_mode.play_audio_file``,
    which on macOS spawns ``afplay``. Drive that exact path with
    ``subprocess.Popen`` replaced by a *subclass* of the real one (so the
    guard still classes it as a real spawn, exactly like the live-system
    guard's wrapper) whose constructor raises. Unguarded, that raise is
    swallowed by the player loop and ``play_audio_file`` reports ``False``;
    guarded, the spawn never happens and it reports ``True``.
    """
    import tools.voice_mode as vm

    assert vm._spawn_system_player.__name__ == "_silent_spawn_system_player"

    class _ExplodingPopen(subprocess.Popen):
        def __init__(self, *args, **kwargs):
            raise AssertionError("real system-player spawn reached subprocess.Popen")

    monkeypatch.setattr(subprocess, "Popen", _ExplodingPopen)

    wav = tmp_path / "silent.wav"
    _write_silent_wav(wav)
    _force_system_player_path(monkeypatch, vm, "/usr/bin/afplay")

    assert vm.play_audio_file(str(wav)) is True


def test_system_player_guard_hands_through_to_a_test_stubbed_popen(tmp_path, monkeypatch):
    """Tests that stub ``subprocess.Popen`` to assert on argv/env must keep
    seeing the spawn — the guard defers to them instead of swallowing it."""
    import tools.voice_mode as vm

    seen = []

    class _Done:
        returncode = 0

        def wait(self, timeout=None):
            return 0

    def _recording_popen(cmd, **kwargs):
        seen.append((list(cmd), kwargs))
        return _Done()

    monkeypatch.setattr(subprocess, "Popen", _recording_popen)

    wav = tmp_path / "silent.wav"
    _write_silent_wav(wav)
    _force_system_player_path(monkeypatch, vm, "/usr/bin/afplay")

    assert vm.play_audio_file(str(wav)) is True
    assert seen and seen[0][0][0] == "afplay"
    assert "env" in seen[0][1]


@pytest.mark.real_audio_playback
def test_bypass_marker_restores_the_real_system_player_spawn():
    """Identity only — calling it would start a real player."""
    import tools.voice_mode as vm

    assert vm._spawn_system_player.__name__ == "_spawn_system_player"
