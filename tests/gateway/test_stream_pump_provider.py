"""Streaming TTS pump: provider-agnostic synthesis and mixer pumping contracts.

Regression context: the pump used to POST pieces to a hardcoded speech-server URL.
It now renders each piece through the configured ``tts.provider`` via
``text_to_speech_tool`` and decodes with ``decode_to_pcm`` - these tests pin the
provider dispatch, the per-piece pause/lead-silence wiring, the stop-on-failure
behavior, and the reply-vs-interim teardown-registration split.
"""

import asyncio

from plugins.platforms.discord import voice_mixer as _vm


def _make_pump_adapter(cfg=None):
    """A bare stub bound to the REAL pump + synthesize methods, with stubbed edges.

    ``synthesize_piece`` is a controllable async stub recording calls; tests either
    script its results or bind the real one to exercise the provider dispatch.
    """
    from plugins.platforms.discord.adapter import DiscordAdapter

    cfg = {"enabled": True, "piece_chars": 250, "piece_pause_ms": 0,
           "speech_gain": 1.0, **(cfg or {})}
    mixer = _RecordingMixer()
    adapter = type("AdapterStub", (), {})()
    adapter._voice_fx_cfg = {"stream_replies": cfg}
    adapter._voice_mixers = {12345: mixer}
    adapter._stream_tts_tasks = {}
    adapter._reset_voice_timeout = lambda gid: None
    adapter._playback_timeout_seconds = 1
    adapter._lead_silence_bytes = lambda: b"\x00" * (_vm.BYTES_PER_MS * 100)
    adapter.piece_calls = []

    async def fake_synthesize(piece, timeout_s):
        adapter.piece_calls.append((piece, timeout_s))
        results = adapter.synth_results
        if adapter.synth_raises is not None:
            raise adapter.synth_raises
        result = results.pop(0) if results else b"PCM"
        return result

    adapter.synth_results = []
    adapter.synth_raises = None
    adapter._synthesize_piece = fake_synthesize
    adapter._stream_tts_cfg = DiscordAdapter._stream_tts_cfg.__get__(adapter)
    adapter._stream_tts_piece_pause_s = DiscordAdapter._stream_tts_piece_pause_s.__get__(adapter)
    adapter._playback_timeout_limit = DiscordAdapter._playback_timeout_limit.__get__(adapter)
    adapter._pump_pieces_into_mixer = DiscordAdapter._pump_pieces_into_mixer.__get__(adapter)
    return adapter, mixer


class _RecordingMixer:
    def __init__(self, speech_active=False):
        self.pushed = []
        self.end_count = 0
        self.stop_count = 0
        self.stream_gains = []
        self.speech_active = speech_active

    def play_speech_streaming(self, *, gain=None, fade_in_ms=40):
        self.stream_gains.append(gain)
        mixer = self

        class _Child:
            def push(self, pcm):
                mixer.pushed.append(pcm)

            def end(self):
                mixer.end_count += 1

        return _Child()

    def stop_speech(self):
        self.stop_count += 1
        self.speech_active = False


# --- _synthesize_piece: provider dispatch via text_to_speech_tool ----------

def _patch_provider(monkeypatch, impl):
    import tools.tts_tool as tts_tool
    monkeypatch.setattr(tts_tool, "text_to_speech_tool", impl)


def _bind_real_synthesize(adapter):
    from plugins.platforms.discord.adapter import DiscordAdapter
    adapter._synthesize_piece = DiscordAdapter._synthesize_piece.__get__(adapter)


def test_synthesize_piece_decodes_provider_file_path(monkeypatch, tmp_path):
    """Provider success: decode runs on the file the provider SAYS it wrote, not the
    requested temp path - providers may return a different extension/format."""
    audio = tmp_path / "piece.ogg"
    audio.write_bytes(b"fake-audio")
    seen = {}

    def fake_tool(text, output_path, skip_rewrite=False):
        seen["text"] = text
        return '{"success": true, "file_path": "%s"}' % audio

    _patch_provider(monkeypatch, fake_tool)
    decoded = {}

    def fake_decode(path):
        decoded["path"] = path
        return b"PCM-DATA"

    monkeypatch.setattr(_vm, "decode_to_pcm", fake_decode)
    adapter, _mixer = _make_pump_adapter()
    _bind_real_synthesize(adapter)

    pcm = asyncio.run(adapter._synthesize_piece("hello piece", 30.0))

    assert seen["text"] == "hello piece"
    assert decoded["path"] == str(audio)  # actual provider path, not piece.wav
    assert pcm == b"PCM-DATA"


def test_synthesize_piece_provider_exception_returns_none(monkeypatch):
    def boom(text, output_path):
        raise RuntimeError("provider down")

    _patch_provider(monkeypatch, boom)
    adapter, _mixer = _make_pump_adapter()
    _bind_real_synthesize(adapter)

    assert asyncio.run(adapter._synthesize_piece("hello", 30.0)) is None


def test_synthesize_piece_timeout_returns_none(monkeypatch):
    async def slow(text, output_path):
        await asyncio.sleep(5)

    import tools.tts_tool as tts_tool
    monkeypatch.setattr(tts_tool, "text_to_speech_tool", lambda *a, **k: slow(*a, **k))
    adapter, _mixer = _make_pump_adapter()
    _bind_real_synthesize(adapter)

    assert asyncio.run(adapter._synthesize_piece("hello", 0.05)) is None


def test_synthesize_piece_failed_result_returns_none(monkeypatch, tmp_path):
    """success=false, or success=true with a nonexistent file path: no decode."""
    _patch_provider(monkeypatch, lambda text, output_path: '{"success": false}')
    adapter, _mixer = _make_pump_adapter()
    _bind_real_synthesize(adapter)
    assert asyncio.run(adapter._synthesize_piece("hello", 30.0)) is None

    _patch_provider(
        monkeypatch,
        lambda text, output_path: '{"success": true, "file_path": "%s"}' % (tmp_path / "missing.wav"))
    assert asyncio.run(adapter._synthesize_piece("hello", 30.0)) is None


# --- _pump_pieces_into_mixer: mixer wiring ---------------------------------

def test_pump_no_mixer_returns_false():
    adapter, _mixer = _make_pump_adapter()
    adapter._voice_mixers = {}
    ok = asyncio.run(adapter._pump_pieces_into_mixer(12345, ["a"], label="reply"))
    assert ok is False


def test_pump_reply_pushes_lead_silence_then_pieces_and_ends():
    adapter, mixer = _make_pump_adapter()
    ok = asyncio.run(adapter._pump_pieces_into_mixer(
        12345, ["piece one", "piece two"], label="reply"))
    assert ok is True
    # First piece: lead silence prepended; second: no pause (piece_pause_ms=0).
    lead = adapter._lead_silence_bytes()
    assert mixer.pushed == [lead + b"PCM", b"PCM"]
    assert mixer.end_count == 1  # child closed exactly once
    # Reply pumps register for voice-leave teardown; the slot pops itself once the
    # pump task ends (done-callback), without the caller ever awaiting it.
    async def _scenario():
        ok = await adapter._pump_pieces_into_mixer(
            12345, ["piece one", "piece two"], label="reply")
        assert ok is True
        pump_task = adapter._stream_tts_tasks.get(12345)
        assert pump_task is not None  # registered immediately, before any audio
        while not pump_task.done():  # the pump runs on its own; yield until it ends
            await asyncio.sleep(0)
        await asyncio.sleep(0)  # one more tick: the done-callback pops the slot
        assert 12345 not in adapter._stream_tts_tasks
    asyncio.run(_scenario())


def test_pump_inter_piece_pause_inserted(monkeypatch):
    adapter, mixer = _make_pump_adapter(cfg={"piece_pause_ms": 100})
    asyncio.run(adapter._pump_pieces_into_mixer(12345, ["a", "b"]))
    pause = b"\x00" * (_vm.BYTES_PER_MS * 100)
    assert mixer.pushed == [adapter._lead_silence_bytes() + b"PCM", pause + b"PCM"]


def test_pump_stops_on_first_failed_piece():
    """Provider failure on piece 2: piece 3 is never requested, child still ends."""
    adapter, mixer = _make_pump_adapter()
    adapter.synth_results = [b"PCM", None, b"PCM"]
    ok = asyncio.run(adapter._pump_pieces_into_mixer(
        12345, ["one", "two", "three"], label="reply"))
    assert ok is True  # stream took ownership even though it stopped early
    assert [c[0] for c in adapter.piece_calls] == ["one", "two"]
    assert mixer.pushed == [adapter._lead_silence_bytes() + b"PCM"]
    assert mixer.end_count == 1


def test_pump_crash_still_ends_child():
    adapter, mixer = _make_pump_adapter()
    adapter.synth_raises = RuntimeError("boom")
    asyncio.run(adapter._pump_pieces_into_mixer(12345, ["a"], label="reply"))
    assert mixer.pushed == []
    assert mixer.end_count == 1  # crashed pump must not leave the child open


def test_pump_interim_never_registers_teardown_task():
    adapter, _mixer = _make_pump_adapter()
    asyncio.run(adapter._pump_pieces_into_mixer(12345, ["note"], label="interim"))
    assert 12345 not in adapter._stream_tts_tasks


def test_pump_passes_configured_speech_gain():
    adapter, mixer = _make_pump_adapter(cfg={"speech_gain": 1.5})
    asyncio.run(adapter._pump_pieces_into_mixer(12345, ["a"]))
    assert mixer.stream_gains == [1.5]


def test_pump_returns_while_synthesis_still_running():
    """The latency contract: the pump call returns BEFORE piece synthesis completes, so
    the caller's text send is never gated on how long the provider takes."""

    async def _slow_synthesize(piece, timeout_s):
        await asyncio.sleep(5.0)  # would blow any drain-wait if we awaited it
        return b"PCM"

    adapter, mixer = _make_pump_adapter()
    adapter._synthesize_piece = _slow_synthesize

    async def _scenario():
        import time
        start = time.monotonic()
        ok = await adapter._pump_pieces_into_mixer(12345, ["a", "b"], label="reply")
        elapsed = time.monotonic() - start
        assert ok is True
        assert elapsed < 1.0  # returned at start, not after the 5 s syntheses
        assert 12345 in adapter._stream_tts_tasks  # pump still registered/running

    asyncio.run(_scenario())


def test_pump_returns_before_playback_finishes():
    """The pump returns as soon as playback STARTS - even mid-speech - so the caller's
    text send is never gated on synthesis or playback drain."""
    adapter, mixer = _make_pump_adapter(cfg={"speech_gain": 1.0})
    mixer.speech_active = True  # playback "in progress"
    ok = asyncio.run(adapter._pump_pieces_into_mixer(
        12345, ["a"], label="reply"))
    assert ok is True
    assert mixer.stop_count == 0  # never waits, never force-stops


def test_pump_reply_passes_pieces_to_provider_in_order():
    adapter, _mixer = _make_pump_adapter()
    asyncio.run(adapter._pump_pieces_into_mixer(12345, ["x", "y", "z"]))
    assert [c[0] for c in adapter.piece_calls] == ["x", "y", "z"]

def _make_lane_adapter(cfg=None):
    """Pump adapter bound to the REAL lane entries (reply + interim) with stubbed edges."""
    from plugins.platforms.discord.adapter import DiscordAdapter

    adapter, mixer = _make_pump_adapter(cfg)
    adapter._stream_tts_enabled = DiscordAdapter._stream_tts_enabled.__get__(adapter)
    adapter._stream_tts_piece_chars = DiscordAdapter._stream_tts_piece_chars.__get__(adapter)
    adapter._STREAM_PIECE_CHARS_DEFAULT = DiscordAdapter._STREAM_PIECE_CHARS_DEFAULT
    adapter.prepare_tts_text = lambda text: text
    adapter.play_reply_streaming_in_voice = DiscordAdapter.play_reply_streaming_in_voice.__get__(adapter)
    adapter.play_interim_in_voice = DiscordAdapter.play_interim_in_voice.__get__(adapter)
    return adapter, mixer


def _patch_lane_rewrite(monkeypatch, rewrite_calls):
    """Replace the LLM rewrite with a marker-substituting fake; records (text, interim)."""
    import tools.tts_rewrite as tts_rewrite

    def fake_rewrite(text, interim=False):
        rewrite_calls.append((text, interim))
        return text.replace("reply", "REWRITTEN-REPLY")

    monkeypatch.setattr(tts_rewrite, "rewrite_text_for_speech", fake_rewrite)


def _patch_lane_provider(monkeypatch, provider_calls):
    """Recording provider fake that writes a decodable file at the requested path."""
    import json

    import tools.tts_tool as tts_tool

    def fake_tool(text, output_path, skip_rewrite=False):
        provider_calls.append({"text": text, "skip_rewrite": skip_rewrite})
        with open(output_path, "wb") as handle:
            handle.write(b"fake-audio")
        return json.dumps({"success": True, "file_path": output_path})

    monkeypatch.setattr(tts_tool, "text_to_speech_tool", fake_tool)


def test_reply_lane_rewrites_whole_text_once_before_split(monkeypatch):
    """Main, steered, queued, background and media-final lanes all converge on this
    entry: the FULL reply is rewritten exactly once, then split, and every piece
    reaches the provider rewritten and with skip_rewrite set."""
    adapter, _mixer = _make_lane_adapter()
    _bind_real_synthesize(adapter)
    rewrite_calls = []
    provider_calls = []
    _patch_lane_rewrite(monkeypatch, rewrite_calls)
    _patch_lane_provider(monkeypatch, provider_calls)

    text = " ".join("Sentence %d of the streaming reply." % n for n in range(1, 31))
    assert len(text) >= 400
    monkeypatch.setattr(_vm, "decode_to_pcm", lambda path: b"PCM")

    async def _scenario():
        assert await adapter.play_reply_streaming_in_voice(12345, text) is True
        await adapter._stream_tts_tasks.get(12345)  # drain for deterministic asserts

    asyncio.run(_scenario())

    assert [t for t, _interim in rewrite_calls] == [text]  # once, whole
    assert provider_calls
    assert all(call["skip_rewrite"] is True for call in provider_calls)
    assert all("REWRITTEN-REPLY" in call["text"] for call in provider_calls)
    assert len(provider_calls) > 1  # long reply was actually split


def test_reply_lane_skips_rewrite_under_min_chars(monkeypatch):
    adapter, _mixer = _make_lane_adapter()
    _bind_real_synthesize(adapter)
    rewrite_calls = []
    provider_calls = []
    _patch_lane_rewrite(monkeypatch, rewrite_calls)
    _patch_lane_provider(monkeypatch, provider_calls)
    monkeypatch.setattr(_vm, "decode_to_pcm", lambda path: b"PCM")

    async def _scenario():
        assert await adapter.play_reply_streaming_in_voice(12345, "Short spoken reply.") is True
        await adapter._stream_tts_tasks.get(12345)

    asyncio.run(_scenario())

    assert rewrite_calls == []
    assert provider_calls[0]["text"] == "Short spoken reply."


def test_reply_lane_supersedes_inflight_reply():
    """A second reply from any lane (steer supersede) cancels the in-flight pump so it
    stops pushing PCM; already-queued audio still finishes via mixer child reuse."""
    adapter, _mixer = _make_lane_adapter()

    async def slow_first(piece, timeout_s):
        if "First" in piece:
            await asyncio.sleep(5.0)
        return b"PCM"

    adapter._synthesize_piece = slow_first

    async def _scenario():
        assert await adapter.play_reply_streaming_in_voice(
            12345, "First reply sentence. " * 15) is True
        old_task = adapter._stream_tts_tasks.get(12345)
        assert not old_task.done()
        assert await adapter.play_reply_streaming_in_voice(
            12345, "Second reply sentence.") is True
        new_task = adapter._stream_tts_tasks.get(12345)
        assert new_task is not old_task
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert old_task.done()  # cancel delivered and processed
        await asyncio.wait([old_task])

    asyncio.run(_scenario())


def test_interim_lane_rewrites_interim_never_supersedes_reply(monkeypatch):
    """The interim lane uses the interim rewrite and must never cancel a playing reply."""
    adapter, mixer = _make_lane_adapter(cfg={"speak_interims": True, "interim_max_chars": 0})
    rewrite_calls = []
    _patch_lane_rewrite(monkeypatch, rewrite_calls)

    async def slow_reply(piece, timeout_s):
        if "First" in piece:
            await asyncio.sleep(5.0)
        return b"PCM"

    adapter._synthesize_piece = slow_reply

    async def _scenario():
        assert await adapter.play_reply_streaming_in_voice(
            12345, "First reply sentence. " * 15) is True
        reply_task = adapter._stream_tts_tasks.get(12345)
        assert not reply_task.done()
        assert await adapter.play_interim_in_voice(12345, "Quick interim note.") is True
        assert adapter._stream_tts_tasks.get(12345) is reply_task  # untouched
        await asyncio.sleep(0)
        assert not reply_task.done()
        reply_task.cancel()
        await asyncio.wait([reply_task])

    asyncio.run(_scenario())

    assert rewrite_calls == [("Quick interim note.", True)]  # interim prompt path
    assert mixer.pushed  # interim audio queued into the mixer
