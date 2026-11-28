"""Gateway speech paths fire ``transform_spoken_text`` before synthesis (issue #152238).

Covers the runner's voice reply (``voice_reply``), the base adapter's auto-TTS (``auto_tts``) and
the streaming-TTS consumer (``streaming``). The plugin dispatch is faked at ``hermes_cli.plugins``
and the TTS tool / streamer are fakes; what the synthesizer receives is the assertion.
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource, build_session_key
from gateway.streaming_tts_consumer import StreamingTTSConsumer


def _fake_spoken_hook(monkeypatch, reply):
    calls = []

    def _invoke(hook_name, **kwargs):
        if hook_name != "transform_spoken_text":
            return []
        calls.append(kwargs)
        return [reply(kwargs)]

    monkeypatch.setattr("hermes_cli.plugins.has_hook", lambda name: name == "transform_spoken_text")
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", _invoke)
    return calls


def _broken_hook_dispatch(monkeypatch):
    monkeypatch.setattr("hermes_cli.plugins.has_hook", lambda name: True)
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", MagicMock(side_effect=RuntimeError("boom")))


def _fake_tts_tool(monkeypatch):
    """Record what the TTS tool is asked to speak; write a real file at the requested path."""
    spoken = []

    def _tts(*, text, output_path, **_kwargs):
        spoken.append(text)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "wb") as fh:
            fh.write(b"\x00" * 32)
        return json.dumps({"success": True, "file_path": output_path})

    monkeypatch.setattr("tools.tts_tool.text_to_speech_tool", _tts)
    return spoken


# --------------------------------------------------------------------------- voice_reply


def _voice_reply_runner():
    runner = object.__new__(GatewayRunner)
    send_voice = AsyncMock()
    runner.adapters = {Platform.TELEGRAM: SimpleNamespace(
        send_voice=send_voice, is_in_voice_channel=lambda *_a, **_k: False)}
    return runner, send_voice


def _text_event():
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="42", user_id="42", chat_type="dm")
    return MessageEvent(text="hi", message_type=MessageType.TEXT, source=source, message_id="m1")


@pytest.mark.asyncio
async def test_voice_reply_speaks_the_transformed_text(monkeypatch, tmp_path):
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))
    spoken = _fake_tts_tool(monkeypatch)
    calls = _fake_spoken_hook(monkeypatch, lambda kw: f"[sighs] {kw['text']}")
    runner, send_voice = _voice_reply_runner()

    await runner._send_voice_reply(_text_event(), "Fine, it passed.", session_id="sess-1")

    assert spoken == ["[sighs] Fine, it passed."]
    send_voice.assert_awaited_once()
    assert [(c["surface"], c["platform"], c["session_id"], c["index"]) for c in calls] == [
        ("voice_reply", "telegram", "sess-1", 0)]


@pytest.mark.asyncio
async def test_voice_reply_empty_replacement_sends_no_audio(monkeypatch, tmp_path):
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))
    spoken = _fake_tts_tool(monkeypatch)
    _fake_spoken_hook(monkeypatch, lambda kw: "")
    runner, send_voice = _voice_reply_runner()

    await runner._send_voice_reply(_text_event(), "Fine, it passed.", session_id="sess-1")

    assert spoken == []
    send_voice.assert_not_awaited()


@pytest.mark.asyncio
async def test_voice_reply_survives_a_broken_hook_dispatch(monkeypatch, tmp_path):
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))
    spoken = _fake_tts_tool(monkeypatch)
    _broken_hook_dispatch(monkeypatch)
    runner, send_voice = _voice_reply_runner()

    await runner._send_voice_reply(_text_event(), "Fine, it passed.")

    assert spoken == ["Fine, it passed."]
    send_voice.assert_awaited_once()


# --------------------------------------------------------------------------- auto_tts


class _VoiceAdapter(BasePlatformAdapter):
    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="fake-token"), Platform.DISCORD)
        self.sent = []
        self._should_auto_tts_for_chat = lambda _chat_id: True
        self.play_tts = AsyncMock(return_value=SendResult(success=True, message_id="tts-1"))
        self._session_store = SimpleNamespace(peek_session_id=lambda key: "sess-auto")

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        return True

    async def disconnect(self) -> None:
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        self.sent.append(content)
        return SendResult(success=True, message_id="1")

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        return None

    async def stop_typing(self, chat_id: str, metadata=None) -> None:
        return None

    async def get_chat_info(self, chat_id: str):
        return {"id": chat_id}


async def _run_voice_turn(reply_text):
    adapter = _VoiceAdapter()

    async def _hold_typing(*_args, **_kwargs):
        await asyncio.Event().wait()

    adapter._keep_typing = _hold_typing
    adapter.set_message_handler(lambda _event: asyncio.sleep(0, result=reply_text))
    event = MessageEvent(
        text="hello", message_type=MessageType.VOICE, message_id="voice-1",
        source=SessionSource(platform=Platform.DISCORD, chat_id="-1001", chat_type="group"))
    with patch("tools.tts_tool.check_tts_requirements", return_value=True):
        await adapter._process_message_background(event, build_session_key(event.source))
    return adapter


@pytest.mark.asyncio
async def test_auto_tts_speaks_the_transformed_text_and_sends_the_original(monkeypatch):
    spoken = _fake_tts_tool(monkeypatch)
    calls = _fake_spoken_hook(monkeypatch, lambda kw: f"[sighs] {kw['text']}")

    adapter = await _run_voice_turn("Fine, it passed.")

    assert spoken == ["[sighs] Fine, it passed."]
    adapter.play_tts.assert_awaited_once()
    assert adapter.sent == ["Fine, it passed."]
    assert [(c["surface"], c["platform"], c["session_id"]) for c in calls] == [
        ("auto_tts", "discord", "sess-auto")]


@pytest.mark.asyncio
async def test_auto_tts_empty_replacement_plays_nothing_but_keeps_the_text(monkeypatch):
    spoken = _fake_tts_tool(monkeypatch)
    _fake_spoken_hook(monkeypatch, lambda kw: "")

    adapter = await _run_voice_turn("Fine, it passed.")

    assert spoken == []
    adapter.play_tts.assert_not_awaited()
    assert adapter.sent == ["Fine, it passed."]


@pytest.mark.asyncio
async def test_auto_tts_survives_a_broken_hook_dispatch(monkeypatch):
    spoken = _fake_tts_tool(monkeypatch)
    _broken_hook_dispatch(monkeypatch)

    adapter = await _run_voice_turn("Fine, it passed.")

    assert spoken == ["Fine, it passed."]
    adapter.play_tts.assert_awaited_once()


# --------------------------------------------------------------------------- streaming consumer


class _Streamer:
    sample_rate, channels, sample_width = 24000, 1, 2

    def __init__(self):
        self.clauses = []

    def stream(self, text):
        self.clauses.append(text)
        yield b"\x01\x00" * 4


class _StreamingAdapter:
    name = "fake-voice"
    platform = Platform.DISCORD

    def __init__(self):
        self.written = []

    def supports_streaming_tts(self, chat_id, audio_format):
        return True

    async def begin_streaming_tts(self, chat_id, audio_format, metadata=None):
        from gateway.platforms.base import StreamingTTSHandle
        return StreamingTTSHandle(chat_id=chat_id, audio_format=audio_format)

    async def write_streaming_tts(self, handle, chunk):
        self.written.append(chunk)

    async def finish_streaming_tts(self, handle, *, interrupted=False):
        return None

    async def abort_streaming_tts(self, handle, error=None):
        return None


class _SilentStreamer(_Streamer):
    """A provider that answers every clause with no audio."""

    def stream(self, text):
        self.clauses.append(text)
        yield from ()


async def _stream_consumer(monkeypatch, deltas, streamer):
    monkeypatch.setattr("tools.tts_streaming.resolve_streaming_provider", lambda cfg: streamer)
    consumer = StreamingTTSConsumer(
        _StreamingAdapter(), "chat-1", {}, asyncio.get_running_loop(), session_id="sess-stream")
    consumer.start()
    for delta in deltas:
        consumer.on_delta(delta)
    consumer.finish()
    await consumer.wait_complete(timeout=5)
    return consumer


async def _stream_reply(monkeypatch, deltas):
    streamer = _Streamer()
    await _stream_consumer(monkeypatch, deltas, streamer)
    return streamer.clauses


@pytest.mark.asyncio
async def test_streaming_consumer_transforms_each_clause_with_its_index(monkeypatch):
    calls = _fake_spoken_hook(monkeypatch, lambda kw: f"[{kw['index']}] {kw['text']}")

    clauses = await _stream_reply(
        monkeypatch, ["The first sentence is here. ", "The second sentence is here. "])

    assert clauses == ["[0] The first sentence is here.", "[1] The second sentence is here."]
    assert {(c["surface"], c["platform"], c["session_id"]) for c in calls} == {
        ("streaming", "discord", "sess-stream")}


@pytest.mark.asyncio
async def test_streaming_consumer_skips_a_clause_replaced_with_empty(monkeypatch):
    _fake_spoken_hook(monkeypatch, lambda kw: "" if kw["index"] == 0 else None)

    clauses = await _stream_reply(
        monkeypatch, ["The first sentence is here. ", "The second sentence is here. "])

    assert clauses == ["The second sentence is here."]


_TWO_SENTENCES = ["The first sentence is here. ", "The second sentence is here. "]


@pytest.mark.asyncio
async def test_a_reply_the_plugin_silenced_entirely_is_not_spoken_again_as_a_whole(monkeypatch):
    """Silence the plugin chose is the reply's speech: no whole-reply fallback, no second call."""
    calls = _fake_spoken_hook(monkeypatch, lambda kw: "")
    streamer = _Streamer()

    consumer = await _stream_consumer(monkeypatch, _TWO_SENTENCES, streamer)

    assert streamer.clauses == []
    assert [c["index"] for c in calls] == [0, 1]
    assert consumer.suppress_whole_file is True
    assert consumer.completed is False


@pytest.mark.parametrize("reply", [lambda kw: None, lambda kw: "" if kw["index"] == 0 else None],
                         ids=["nothing-silenced", "one-sentence-silenced"])
@pytest.mark.asyncio
async def test_a_stream_that_produced_no_audio_still_falls_back_to_the_whole_reply(monkeypatch, reply):
    """Only silence the plugin asked for counts as handled; a provider that gave no audio does not."""
    _fake_spoken_hook(monkeypatch, reply)

    consumer = await _stream_consumer(monkeypatch, _TWO_SENTENCES, _SilentStreamer())

    assert consumer.suppress_whole_file is False
