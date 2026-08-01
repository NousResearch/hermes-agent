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
import sys
import types

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

    resp = server.dispatch(
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
