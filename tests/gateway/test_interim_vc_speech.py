"""Interim VC speech: gating and routing contracts.

Regression context: mid-turn commentary between tool calls was silent in bound voice
channels. ``play_interim_in_voice`` speaks it additively (queued behind replies, no
rewrite, fire-and-forget) gated on ``discord.voice_fx.stream_replies.speak_interims``.
"""

import asyncio

import pytest  # noqa: F401  (kept for parity with sibling gateway tests)


class _FakeMixer:
    def __init__(self):
        self.stream_calls = []

    def play_speech_streaming(self, *, gain=None, fade_in_ms=40):
        self.stream_calls.append(gain)

        class _Child:
            def push(self, pcm):
                pass

            def end(self):
                pass

        return _Child()


def _make_adapter(monkeypatch, *, speak_interims, max_chars=300):
    """A bare object bound to the REAL adapter methods under test, with stubbed edges."""
    from plugins.platforms.discord.adapter import DiscordAdapter

    cfg = {
        "enabled": True,
        "speak_interims": speak_interims,
        "interim_max_chars": max_chars,
        "piece_chars": 250,
        "speech_gain": 1.0,
    }
    mixer = _FakeMixer()
    adapter = type("AdapterStub", (), {})()
    adapter._voice_fx_cfg = {"stream_replies": cfg}
    adapter._voice_mixers = {123: mixer}
    adapter._stream_tts_tasks = {}
    adapter.prepare_tts_text = lambda t: t
    adapter._reset_voice_timeout = lambda gid: None
    adapter.pump_calls = []

    async def fake_pump(guild_id, pieces, *, label="reply"):
        adapter.pump_calls.append((guild_id, list(pieces), label))
        return True

    # Bind the real methods under test onto the stub; stub the pump edge.
    adapter._stream_tts_cfg = DiscordAdapter._stream_tts_cfg.__get__(adapter)
    adapter._stream_tts_enabled = DiscordAdapter._stream_tts_enabled.__get__(adapter)
    adapter._stream_tts_piece_chars = DiscordAdapter._stream_tts_piece_chars.__get__(adapter)
    adapter._STREAM_PIECE_CHARS_DEFAULT = 250
    adapter.play_interim_in_voice = DiscordAdapter.play_interim_in_voice.__get__(adapter)
    adapter._pump_pieces_into_mixer = fake_pump
    return adapter, mixer


def _split(monkeypatch):
    import tools.tts_tool_delivery as delivery

    def fake_split(t, piece_chars):
        return [t]

    monkeypatch.setattr(delivery, "_split_text_for_tts", fake_split)


def test_interim_speech_disabled_returns_false(monkeypatch):
    _split(monkeypatch)
    adapter, mixer = _make_adapter(monkeypatch, speak_interims=False)
    ok = asyncio.run(adapter.play_interim_in_voice(123, "Working on it now."))
    assert ok is False
    assert adapter.pump_calls == []
    assert mixer.stream_calls == []  # never opened a speech child


def test_interim_speech_enabled_pumps_without_drain_or_rewrite(monkeypatch):
    _split(monkeypatch)
    adapter, _mixer = _make_adapter(monkeypatch, speak_interims=True)
    ok = asyncio.run(adapter.play_interim_in_voice(123, "Working on it now."))
    assert ok is True
    # Routed into the shared pump as an interim, fire-and-forget (no drain wait).
    # (The pump itself opens the mixer speech child - covered by the reply-path mixer tests.)
    assert adapter.pump_calls == [(123, ["Working on it now."], "interim")]
    # Fire-and-forget: never registered for voice-leave teardown.
    assert 123 not in adapter._stream_tts_tasks


def test_interim_speech_skips_over_max_chars(monkeypatch):
    _split(monkeypatch)
    adapter, mixer = _make_adapter(monkeypatch, speak_interims=True, max_chars=10)
    ok = asyncio.run(adapter.play_interim_in_voice(123, "This is way too long for an interim."))
    assert ok is False
    assert adapter.pump_calls == []
    assert mixer.stream_calls == []


def test_interim_speech_zero_max_chars_is_uncapped(monkeypatch):
    _split(monkeypatch)
    adapter, mixer = _make_adapter(monkeypatch, speak_interims=True, max_chars=0)
    long_text = "x" * 5000
    ok = asyncio.run(adapter.play_interim_in_voice(123, long_text))
    assert ok is True
    assert adapter.pump_calls and adapter.pump_calls[0][1] == [long_text]


def test_interim_speech_streaming_disabled_noop(monkeypatch):
    """stream_replies.enabled is the master switch: speak_interims alone must not speak."""
    _split(monkeypatch)
    adapter, mixer = _make_adapter(monkeypatch, speak_interims=True)
    adapter._voice_fx_cfg["stream_replies"]["enabled"] = False
    ok = asyncio.run(adapter.play_interim_in_voice(123, "Working on it now."))
    assert ok is False
    assert mixer.stream_calls == []


def _stub_interim_rewrite(monkeypatch, calls):
    import tools.tts_rewrite as tr

    def fake_rewrite(text, *, interim=False):
        calls.append((text, interim))
        return text.upper()

    monkeypatch.setattr(tr, "rewrite_text_for_speech", fake_rewrite)


def test_interim_speech_rewrite_applied_interim_variant(monkeypatch):
    """interim_rewrite (default on) runs the speed-first prompt: interim=True kwarg."""
    _split(monkeypatch)
    adapter, _mixer = _make_adapter(monkeypatch, speak_interims=True)
    calls = []
    _stub_interim_rewrite(monkeypatch, calls)
    asyncio.run(adapter.play_interim_in_voice(123, "Working on it now."))
    assert calls == [("Working on it now.", True)]  # the INTERIM prompt variant, not the reply one
    # The rewritten text feeds the pump.
    assert adapter.pump_calls == [(123, ["WORKING ON IT NOW."], "interim")]


def test_interim_speech_rewrite_disabled_keeps_raw(monkeypatch):
    _split(monkeypatch)
    adapter, _mixer = _make_adapter(monkeypatch, speak_interims=True)
    adapter._voice_fx_cfg["stream_replies"]["interim_rewrite"] = False
    calls = []
    _stub_interim_rewrite(monkeypatch, calls)
    asyncio.run(adapter.play_interim_in_voice(123, "Working on it now."))
    assert calls == []  # knob off: no rewrite call at all
    assert adapter.pump_calls == [(123, ["Working on it now."], "interim")]


# --- Prebaked ack cache (play_ack_in_voice) -------------------------------

class _SpeechMixer:
    def __init__(self):
        self.speech_calls = []

    def play_speech(self, pcm, *, gain=None):
        self.speech_calls.append((pcm, gain))


def _make_ack_adapter(monkeypatch, tmp_path):
    """Bare stub bound to the real ack methods, with the cache pointed at tmp_path."""
    from plugins.platforms.discord.adapter import DiscordAdapter

    adapter = type("AdapterStub", (), {})()
    adapter._voice_fx_cfg = {"ack_enabled": True, "ack_phrases": ["Test phrase one."], "speech_gain": 1.0}
    mixer = _SpeechMixer()
    adapter._voice_mixers = {111: mixer}
    adapter._reset_voice_timeout = lambda gid: None
    adapter._lead_silence_bytes = lambda: b""
    adapter._ACK_CACHE_DIR = str(tmp_path / "ack")
    adapter._ack_cache_path = DiscordAdapter._ack_cache_path.__get__(adapter)
    adapter._ack_cached_pcm = DiscordAdapter._ack_cached_pcm.__get__(adapter)
    adapter.play_ack_in_voice = DiscordAdapter.play_ack_in_voice.__get__(adapter)
    return adapter, mixer


def test_play_ack_prebakes_once_then_replays_from_disk(monkeypatch, tmp_path):
    """First play renders into the cache; second play must NOT re-synthesize."""
    adapter, mixer = _make_ack_adapter(monkeypatch, tmp_path)
    bakes = []
    import json as _json
    import struct as _struct
    import wave as _wave

    def fake_tts(*, text, output_path):
        bakes.append(text)
        # Provider ignores the requested extension; writes .ogg (like the real one).
        # Content is a real tiny WAV so ffmpeg (content-based detection) decodes it.
        actual = output_path + ".ogg"
        with _wave.open(actual, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(24000)
            w.writeframes(_struct.pack("<h", 0) * 2400)  # 100 ms silence
        return _json.dumps({"success": True, "file_path": actual})

    monkeypatch.setattr("tools.tts_tool.text_to_speech_tool", fake_tts)
    ok1 = asyncio.run(adapter.play_ack_in_voice(111, "Test phrase one."))
    assert ok1 is True and bakes == ["Test phrase one."]
    assert len(mixer.speech_calls) == 1
    ok2 = asyncio.run(adapter.play_ack_in_voice(111, "Test phrase one."))
    assert ok2 is True and bakes == ["Test phrase one."]  # no second bake
    assert len(mixer.speech_calls) == 2  # but it still played
    # Cache holds exactly one wav keyed by the phrase hash.
    files = [f for f in __import__("os").listdir(adapter._ACK_CACHE_DIR) if f.endswith(".wav")]
    assert files == ["ack_" + __import__("hashlib").sha256(b"Test phrase one.").hexdigest()[:16] + ".wav"]


def test_play_ack_bake_failure_is_noop(monkeypatch, tmp_path):
    """TTS failure -> play_ack_in_voice False, no speech pushed, no cache file."""
    adapter, mixer = _make_ack_adapter(monkeypatch, tmp_path)
    import json as _json

    def fake_tts(*, text, output_path):
        return _json.dumps({"success": False, "error": "down"})

    monkeypatch.setattr("tools.tts_tool.text_to_speech_tool", fake_tts)
    ok = asyncio.run(adapter.play_ack_in_voice(111, "Test phrase one."))
    assert ok is False
    assert mixer.speech_calls == []
    assert __import__("os").listdir(adapter._ACK_CACHE_DIR) == []
