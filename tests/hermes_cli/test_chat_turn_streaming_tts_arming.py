"""Streaming-TTS arm gate regression (#84046): ``_chat_setup_turn_audio`` used to gate
streaming TTS on a ``sounddevice`` import, which raises on macOS (PortAudio/CoreAudio
init triggers a kTCCServiceMediaLibrary prompt) even though the speaker path
deliberately avoids sounddevice on Darwin (tempfile -> afplay; see
tts_tool_speaker._device_usable). The swallow-``except`` then disabled streaming TTS
for every working provider on macOS.

The arm gate must not probe sounddevice at all: enablement is
``check_tts_requirements()`` alone, and where sounddevice is unusable a working
provider still arms streaming TTS.

These contracts are host-independent — the arm decision is data (the requirements
probe), not host state — so the tests run unmarked on every lane; the macOS arm of
the *speaker output policy* itself (sounddevice never imported on Darwin) is
covered for real by ``tests/tools/test_tts_macos_output.py`` via
``@pytest.mark.platforms``.

Behavior contracts, mocked per test; no real audio, no config on disk.
"""

import os
import sys
import threading
import types

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from cli import HermesCLI, _ChatTurn


class _StreamingTTSArming:
    """Drive ``HermesCLI._chat_setup_turn_audio`` without a real CLI instance."""

    def make_cli(self):
        cli = HermesCLI.__new__(HermesCLI)
        cli._voice_mode = False
        cli._voice_continuous = False
        cli._voice_tts = True
        cli._voice_tts_done = threading.Event()
        cli._voice_last_tts_text = ""
        cli.show_timestamps = False
        cli.streaming_enabled = False
        cli.timestamp_format = "%H:%M"
        return cli

    def make_turn(self):
        turn = _ChatTurn()
        turn.mute_notification_reply = False
        return turn


def _arm_turn(monkeypatch, requirements, sounddevice_factory=None):
    """Arm a voice-tts turn and return ``(turn, sounddevice_probe)``.

    ``sounddevice_factory`` replaces ``tts_tool._import_sounddevice``; the default
    spy raises the same OSError the import raises on macOS (CoreAudio TCC prompt)
    and records that the arm gate called it at all — the gate must never call it,
    so the raise doubles as proof that arming does not depend on the import.
    """
    from tools import tts_tool

    probe = []

    def _spy():
        probe.append(1)
        raise OSError("PaAlsa [ CoreAudio TCC media-library prompt ]")

    monkeypatch.setattr(tts_tool, "_import_sounddevice", sounddevice_factory or _spy)
    monkeypatch.setattr(tts_tool, "check_tts_requirements", lambda: requirements)

    speaker_calls = []
    started = threading.Event()

    def _fake_stream(text_queue, stop_event, done_event, *a, **k):
        speaker_calls.append(True)
        done_event.set()
        started.set()

    import tools.tts_tool_speaker as speaker_mod
    monkeypatch.setattr(speaker_mod, "stream_tts_to_speaker", _fake_stream)

    cli = _StreamingTTSArming().make_cli()
    turn = _StreamingTTSArming().make_turn()
    cli._chat_setup_turn_audio(turn, "hello", False)

    if turn.use_streaming_tts:
        assert turn.text_queue is not None
        assert turn.stop_event is not None
        turn.text_queue.put(None)  # drain sentinel so the fake speaker exits
        started.wait(timeout=2)
    return turn, probe, bool(speaker_calls)


def test_streaming_tts_arms_without_importing_sounddevice(monkeypatch):
    """#84046: the arm gate must not import sounddevice at all — on macOS the
    import raises (CoreAudio TCC prompt) even though the speaker path never uses
    it there, so the import disabled streaming TTS for working providers. A
    raising import (or none at all) must not stand between requirements and
    arming."""
    turn, probe, spoke = _arm_turn(monkeypatch, requirements=True)
    assert turn.use_streaming_tts is True, "streaming TTS must arm without a sounddevice probe"
    assert probe == [], "the arm gate must not probe sounddevice"
    assert spoke is True, "the TTS speaker thread must have been started"


def test_streaming_tts_arms_with_a_plain_working_import_too(monkeypatch):
    """The gate is import-agnostic in the other direction: where sounddevice
    imports fine it still wires solely from check_tts_requirements()."""
    turn, probe, _ = _arm_turn(
        monkeypatch,
        requirements=True,
        sounddevice_factory=lambda: probe.append(1) or types.ModuleType("sounddevice"),
    )
    assert turn.use_streaming_tts is True
    assert probe == [], "arming must come from check_tts_requirements() alone, never the probe"


def test_streaming_tts_stays_off_when_provider_unavailable(monkeypatch):
    """Arming remains honest: with the provider unavailable the turn stays unarmed
    (fallback to whole-response TTS), and no sounddevice probe happens there either."""
    turn, probe, _ = _arm_turn(monkeypatch, requirements=False)
    assert turn.use_streaming_tts is False
    assert turn.text_queue is None
    assert probe == [], "no sounddevice probe on the unarmed path"


def test_streaming_tts_not_armed_without_voice_tts(monkeypatch):
    """Non-voice-tts turns are untouched: no probe, no queue, no thread."""
    from tools import tts_tool

    probe = []
    monkeypatch.setattr(tts_tool, "_import_sounddevice",
                        lambda: probe.append(1) or types.ModuleType("sounddevice"))

    cli = _StreamingTTSArming().make_cli()
    cli._voice_tts = False
    turn = _StreamingTTSArming().make_turn()

    cli._chat_setup_turn_audio(turn, "hello", False)
    assert turn.use_streaming_tts is False
    assert turn.text_queue is None
    assert probe == []
