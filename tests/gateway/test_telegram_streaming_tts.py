"""Telegram buffers native streaming PCM into early, complete voice messages."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.base import AudioFormat, StreamingTTSHandle
from gateway.streaming_tts_consumer import StreamingTTSConsumer
from plugins.platforms.telegram import telegram_streaming_tts as stream_mod
from plugins.platforms.telegram.adapter import TelegramAdapter


def _adapter() -> TelegramAdapter:
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="test"))
    adapter._bot = MagicMock()
    adapter._send_voice_bubble = AsyncMock(return_value=MagicMock(message_id=1))
    return adapter


def test_original_handle_positional_order_stays_compatible():
    handle = StreamingTTSHandle("123", AudioFormat(), False, True)
    assert handle.aborted is True
    assert handle.defer_audible_until_flush is False


def test_supports_only_native_mono_s16_pcm_with_ffmpeg(monkeypatch):
    adapter = _adapter()
    monkeypatch.setattr(stream_mod.shutil, "which", lambda name: "/usr/bin/ffmpeg")
    assert adapter.supports_streaming_tts("123", AudioFormat()) is True
    assert adapter.supports_streaming_tts("123", AudioFormat(channels=2)) is False
    assert adapter.supports_streaming_tts("123", AudioFormat(sample_width=1)) is False
    adapter._bot = None
    assert adapter.supports_streaming_tts("123", AudioFormat()) is False


def test_pcm_encoder_uses_stdin_stdout_without_temp_files(monkeypatch):
    monkeypatch.setattr(stream_mod.shutil, "which", lambda name: "/usr/bin/ffmpeg")
    run = MagicMock(return_value=SimpleNamespace(returncode=0, stdout=b"OggS-audio", stderr=b""))
    monkeypatch.setattr(stream_mod.subprocess, "run", run)
    pcm = b"\x01\x00" * 24000

    encoded = stream_mod.encode_pcm_ogg_opus(pcm, AudioFormat())

    assert encoded == b"OggS-audio"
    command = run.call_args.args[0]
    assert command[command.index("-f") + 1] == "s16le"
    assert "pipe:0" in command and "pipe:1" in command
    assert run.call_args.kwargs["input"] == pcm
    assert stream_mod.pcm_duration_seconds(pcm, AudioFormat()) == 1


@pytest.mark.asyncio
async def test_clause_flush_sends_voice_and_abort_is_idempotent(monkeypatch):
    monkeypatch.setattr(stream_mod.shutil, "which", lambda name: "/usr/bin/ffmpeg")
    encode = MagicMock(return_value=b"OggS-audio")
    monkeypatch.setattr(stream_mod, "encode_pcm_ogg_opus", encode)
    adapter = _adapter()
    metadata = {"thread_id": "7"}
    handle = await adapter.begin_streaming_tts("123", AudioFormat(), metadata=metadata)
    metadata["thread_id"] = "changed"

    await adapter.write_streaming_tts(handle, b"\x00\x00" * 24000)
    assert handle.audible is False
    adapter._send_voice_bubble.assert_not_awaited()

    await adapter.flush_streaming_tts(handle)

    assert handle.audible is True
    assert handle.defer_audible_until_flush is True
    assert handle.metadata == {"thread_id": "7"}
    assert encode.call_args.args[0] == b"\x00\x00" * 24000
    args = adapter._send_voice_bubble.await_args.args
    assert args[0].getvalue() == b"OggS-audio"
    assert args[1] == "123"
    assert args[3] == {"thread_id": "7"}
    assert args[5] == 1

    await adapter.flush_streaming_tts(handle)  # empty buffer: no duplicate
    assert adapter._send_voice_bubble.await_count == 1

    await adapter.write_streaming_tts(handle, b"\x00\x00" * 48000)
    await adapter.finish_streaming_tts(handle)
    assert adapter._send_voice_bubble.await_count == 2

    await adapter.write_streaming_tts(handle, b"discard")
    await adapter.abort_streaming_tts(handle)
    await adapter.abort_streaming_tts(handle)
    await adapter.write_streaming_tts(handle, b"late")
    await adapter.flush_streaming_tts(handle)
    assert adapter._send_voice_bubble.await_count == 2
    assert handle.aborted is True and not handle.pcm


@pytest.mark.asyncio
async def test_failed_first_upload_does_not_mark_audio_audible(monkeypatch):
    monkeypatch.setattr(stream_mod.shutil, "which", lambda name: "/usr/bin/ffmpeg")
    monkeypatch.setattr(stream_mod, "encode_pcm_ogg_opus", lambda *a: b"OggS-audio")
    adapter = _adapter()
    adapter._send_voice_bubble = AsyncMock(side_effect=RuntimeError("upload failed"))
    handle = await adapter.begin_streaming_tts("123", AudioFormat())
    await adapter.write_streaming_tts(handle, b"pcm")

    with pytest.raises(RuntimeError, match="upload failed"):
        await adapter.flush_streaming_tts(handle)

    assert handle.audible is False


@pytest.mark.asyncio
async def test_consumer_flushes_clause_and_enables_whole_file_suppression(monkeypatch):
    monkeypatch.setattr(stream_mod.shutil, "which", lambda name: "/usr/bin/ffmpeg")
    monkeypatch.setattr(stream_mod, "encode_pcm_ogg_opus", lambda *a: b"OggS-audio")
    adapter = _adapter()
    handle = await adapter.begin_streaming_tts("123", AudioFormat())
    consumer = StreamingTTSConsumer.__new__(StreamingTTSConsumer)
    consumer._adapter = adapter
    consumer._handle = handle
    consumer._aborted = False
    consumer._suppress_whole_file = False
    consumer._strip_markdown = lambda text: text
    consumer._streamer = SimpleNamespace(stream=lambda text: iter([b"first", b"second"]))

    await consumer._synthesise_and_write("Ein vollständiger Satz.")

    adapter._send_voice_bubble.assert_awaited_once()
    assert handle.audible is True
    assert consumer.suppress_whole_file is True


@pytest.mark.asyncio
async def test_abort_during_encoding_never_uploads_late_speech(monkeypatch):
    import asyncio
    import threading
    loop = asyncio.get_running_loop()
    started, release = asyncio.Event(), threading.Event()
    def encode(*args):
        loop.call_soon_threadsafe(started.set)
        if not release.wait(5):
            raise TimeoutError("test encoder was not released")
        return b"OggS-audio"
    monkeypatch.setattr(stream_mod.shutil, "which", lambda name: "/usr/bin/ffmpeg")
    monkeypatch.setattr(stream_mod, "encode_pcm_ogg_opus", encode)
    adapter = _adapter()
    handle = await adapter.begin_streaming_tts("123", AudioFormat())
    await adapter.write_streaming_tts(handle, b"pcm")
    task = asyncio.create_task(adapter.flush_streaming_tts(handle))
    try:
        await asyncio.wait_for(started.wait(), 2)
        await adapter.abort_streaming_tts(handle)
    finally:
        release.set()
        await asyncio.wait_for(task, 2)
    adapter._send_voice_bubble.assert_not_awaited()
    assert handle.aborted and not handle.audible


@pytest.mark.asyncio
@pytest.mark.parametrize("ambiguous", [True, False])
async def test_upload_acknowledgment_controls_whole_file_fallback(monkeypatch, ambiguous):
    import asyncio
    import httpx
    from telegram.error import BadRequest, NetworkError
    from plugins.platforms.telegram.telegram_media_retry import send_media_retrying
    error = NetworkError("upload acknowledgment lost") if ambiguous else BadRequest("invalid voice file")
    if ambiguous:
        error.__cause__ = httpx.ReadTimeout("response lost")
    async def send_once():
        raise error
    async def upload(*args):
        return await send_media_retrying(send_once, None, "test voice")
    monkeypatch.setattr(stream_mod.shutil, "which", lambda name: "/usr/bin/ffmpeg")
    monkeypatch.setattr(stream_mod, "encode_pcm_ogg_opus", lambda *args: b"OggS-audio")
    monkeypatch.setattr("tools.tts_streaming.resolve_streaming_provider", lambda cfg: SimpleNamespace(stream=lambda text: iter([b"pcm"])))
    adapter = _adapter()
    adapter._send_voice_bubble = AsyncMock(side_effect=upload)
    consumer = StreamingTTSConsumer(adapter, "123", {}, asyncio.get_running_loop())
    consumer._queue.put_nowait("One spoken clause.")
    consumer.finish()
    consumer.start()
    assert await consumer.wait_complete(timeout=3) is False
    assert consumer.done
    assert consumer.suppress_whole_file is ambiguous
    assert consumer.partial is ambiguous
    adapter._send_voice_bubble.assert_awaited_once()
