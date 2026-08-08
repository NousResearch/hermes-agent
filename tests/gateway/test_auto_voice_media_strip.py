"""Delivery directives must never be spoken, on either TTS path.

A response that attaches a file carries a ``MEDIA:<path>`` line, and optionally
an ``[[audio_as_voice]]`` or ``[[as_document]]`` flag. Those are a delivery
contract with the platform adapter, not speech. The text delivery path already
strips them via ``_strip_media_directives`` before display, but both TTS paths
used to receive the raw text and synthesised the directive into the audio, so
the clip read an absolute local file path aloud after the real sentence.

Two independent paths are covered here:

* whole file, ``GatewayRunner._send_voice_reply``
* streaming, ``StreamingTTSConsumer``, which receives raw model deltas and is
  the path that runs when streaming TTS is enabled

All identifiers and paths below are synthetic fixtures.
"""

import asyncio
import json
import os
import queue
import tempfile
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform
from gateway.platforms.base import (
    AudioFormat,
    MessageEvent,
    MessageType,
    StreamingTTSHandle,
)
from gateway.run import GatewayRunner
from gateway.session import SessionSource
from gateway.streaming_tts_consumer import (
    StreamingTTSConsumer,
    _directive_holdback_index,
)

FIXTURE_CHAT_ID = "-1000000000001"
FIXTURE_USER_ID = "1000000002"
FIXTURE_MEDIA = "/tmp/hermes-fixture/audio/clip-0001.mp3"
FIXTURE_RESPONSE = "Here is the track you asked for.\n\nMEDIA:%s" % FIXTURE_MEDIA


def _assert_no_directive(spoken):
    assert "MEDIA:" not in spoken, "delivery directive leaked into speech: %r" % spoken
    assert FIXTURE_MEDIA not in spoken, "file path spoken aloud: %r" % spoken
    assert "clip-0001" not in spoken, "path fragment spoken aloud: %r" % spoken
    assert "audio_as_voice" not in spoken, "delivery flag spoken aloud: %r" % spoken


# ---------------------------------------------------------------------------
# Whole-file path: GatewayRunner._send_voice_reply
# ---------------------------------------------------------------------------


def _make_event():
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id=FIXTURE_CHAT_ID,
        user_id=FIXTURE_USER_ID,
        chat_type="group",
    )
    return MessageEvent(
        text="play it",
        message_type=MessageType.TEXT,
        source=source,
        message_id="m1",
    )


def _runner_with_adapter(send_voice_mock):
    runner = object.__new__(GatewayRunner)
    adapter = SimpleNamespace(
        send_voice=send_voice_mock,
        is_in_voice_channel=lambda *_a, **_k: False,
    )
    runner.adapters = {Platform.TELEGRAM: adapter}
    return runner


def _capture_tts(monkeypatch, seen):
    """Record the exact text handed to the whole-file TTS engine."""

    def _fake_text_to_speech_tool(*, text, output_path, **_kwargs):
        seen.append(text)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "wb") as fh:
            fh.write(b"\x00" * 32)
        return json.dumps({"success": True, "file_path": output_path})

    monkeypatch.setattr(
        "tools.tts_tool.text_to_speech_tool", _fake_text_to_speech_tool
    )
    # Identity markdown strip so these cases isolate the directive strip.
    monkeypatch.setattr("tools.tts_tool._strip_markdown_for_tts", lambda text: text)


@pytest.mark.asyncio
async def test_voice_reply_does_not_speak_media_directive(monkeypatch, tmp_path):
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))
    seen = []
    _capture_tts(monkeypatch, seen)

    runner = _runner_with_adapter(AsyncMock())
    await runner._send_voice_reply(_make_event(), FIXTURE_RESPONSE)

    assert seen, "TTS was never invoked"
    _assert_no_directive(seen[0])
    assert "Here is the track you asked for." in seen[0]


@pytest.mark.asyncio
async def test_voice_reply_strips_inline_delivery_flag(monkeypatch, tmp_path):
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))
    seen = []
    _capture_tts(monkeypatch, seen)

    runner = _runner_with_adapter(AsyncMock())
    await runner._send_voice_reply(
        _make_event(), "Round five, here we go. [[audio_as_voice]]"
    )

    assert seen, "TTS was never invoked"
    _assert_no_directive(seen[0])
    assert "Round five" in seen[0]


# ---------------------------------------------------------------------------
# Split-delta holdback helper
# ---------------------------------------------------------------------------


class TestDirectiveHoldback:
    """A directive half-arrived in a delta must never be forwarded."""

    def test_plain_text_is_never_held_back(self):
        text = "An ordinary sentence with no directives at all."
        assert _directive_holdback_index(text) == len(text)

    def test_empty_buffer(self):
        assert _directive_holdback_index("") == 0

    @pytest.mark.parametrize("frag", ["M", "ME", "MED", "MEDI", "MEDIA"])
    def test_partial_media_marker_is_held(self, frag):
        buf = "spoken words " + frag
        assert _directive_holdback_index(buf) == len("spoken words ")

    def test_open_media_path_is_held_until_newline(self):
        buf = "spoken words MEDIA:/tmp/hermes-fixture/audio/clip"
        assert _directive_holdback_index(buf) == len("spoken words ")

    def test_terminated_media_line_is_not_held(self):
        buf = "spoken words MEDIA:%s\nmore speech" % FIXTURE_MEDIA
        assert _directive_holdback_index(buf) == len(buf)

    def test_unclosed_double_bracket_is_held(self):
        buf = "spoken words [[audio_as"
        assert _directive_holdback_index(buf) == len("spoken words ")


# ---------------------------------------------------------------------------
# Streaming path: StreamingTTSConsumer
# ---------------------------------------------------------------------------


class RecordingStreamer:
    """Streaming provider that records every text handed to it."""

    sample_rate = 24000
    channels = 1
    sample_width = 2

    def __init__(self):
        self.spoken = []

    def stream(self, text: str):
        self.spoken.append(text)
        yield b"\x00\x00"


class RecordingAdapter:
    name = "fake-voice"

    def __init__(self):
        self.chunks = []

    async def write_streaming_tts(self, handle, chunk):
        self.chunks.append(chunk)


@pytest.fixture
def streaming_consumer(monkeypatch):
    """A consumer wired to a recording provider, with the drain loop bypassed."""
    streamer = RecordingStreamer()
    monkeypatch.setattr(
        "tools.tts_streaming.resolve_streaming_provider",
        lambda *_a, **_k: streamer,
    )
    loop = asyncio.new_event_loop()
    try:
        adapter = RecordingAdapter()
        consumer = StreamingTTSConsumer(adapter, FIXTURE_CHAT_ID, {}, loop)
        consumer._handle = StreamingTTSHandle(
            chat_id=FIXTURE_CHAT_ID, audio_format=AudioFormat()
        )
        assert consumer.active is True
        yield consumer, streamer
    finally:
        loop.close()


def _drain_to_provider(consumer):
    """Feed every queued clause through the real synthesis entry point."""
    while True:
        try:
            clause = consumer._queue.get_nowait()
        except queue.Empty:
            return
        if not isinstance(clause, str):
            continue
        asyncio.run(consumer._synthesise_and_write(clause))


def test_streaming_directive_in_one_delta_never_reaches_provider(streaming_consumer):
    consumer, streamer = streaming_consumer

    consumer.on_delta("Here is the track you asked for. ")
    consumer.on_delta("Enjoy the music tonight. \n\nMEDIA:%s\n" % FIXTURE_MEDIA)
    consumer.finish()
    _drain_to_provider(consumer)

    spoken = " ".join(streamer.spoken)
    _assert_no_directive(spoken)
    assert "Here is the track you asked for." in spoken


def test_streaming_directive_split_across_deltas_never_reaches_provider(
    streaming_consumer,
):
    """The model streams token by token, so a directive arrives in pieces."""
    consumer, streamer = streaming_consumer

    for delta in [
        "Here is the track you asked for. ",
        "Enjoy the music tonight. ",
        "\n\nMED",
        "IA:/tmp/hermes-",
        "fixture/audio/",
        "clip-0001",
        ".mp3\n",
    ]:
        consumer.on_delta(delta)
    consumer.finish()
    _drain_to_provider(consumer)

    spoken = " ".join(streamer.spoken)
    _assert_no_directive(spoken)
    assert "Here is the track you asked for." in spoken
    assert "Enjoy the music tonight." in spoken


def test_streaming_inline_flag_split_across_deltas_never_reaches_provider(
    streaming_consumer,
):
    consumer, streamer = streaming_consumer

    for delta in ["Round five, here we go. ", "[[audio", "_as_voice]]"]:
        consumer.on_delta(delta)
    consumer.finish()
    _drain_to_provider(consumer)

    spoken = " ".join(streamer.spoken)
    _assert_no_directive(spoken)
    assert "Round five" in spoken


def test_streaming_ordinary_text_still_reaches_provider(streaming_consumer):
    """The guard must not swallow normal speech."""
    consumer, streamer = streaming_consumer

    consumer.on_delta("The first sentence is here. ")
    consumer.on_delta("The second sentence follows it. ")
    consumer.finish()
    _drain_to_provider(consumer)

    spoken = " ".join(streamer.spoken)
    assert "The first sentence is here." in spoken
    assert "The second sentence follows it." in spoken
