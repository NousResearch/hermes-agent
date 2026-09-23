"""Queued-lane final responses must still be SPOKEN in a bound voice channel.

Steered turns deliver their final reply through ``_deliver_queued_first_response``
(the queued lane), which used to do a bare text send and never called the
streaming-TTS path - so the response after a voice steer was never spoken.
"""

import asyncio
from types import SimpleNamespace

from gateway.config import Platform


def _voice_adapter(spoken=True):
    """Minimal adapter stub exposing the voice-streaming surface the lane checks."""
    calls = []

    async def stream_fn(gid, text):
        calls.append((gid, text))
        return spoken

    adapter = SimpleNamespace(
        _voice_text_channels={12345: 67890},
        _stream_tts_enabled=lambda: True,
        play_reply_streaming_in_voice=stream_fn,
        is_in_voice_channel=lambda gid: True,
        stream_calls=calls,
    )

    @staticmethod
    def extract_media(text):
        return [], text

    adapter.extract_media = extract_media
    return adapter


def _plain_adapter():
    """Adapter with NO voice surface at all (non-Discord or voice-less platform)."""
    adapter = SimpleNamespace()

    @staticmethod
    def extract_media(text):
        return [], text

    adapter.extract_media = extract_media
    return adapter


def _run(adapter, response="Final spoken answer.", text_already_delivered=False,
         deliver_media=True):
    from gateway.run_notifications import GatewayNotificationsMixin

    mixin = object.__new__(GatewayNotificationsMixin)

    async def fake_send_queued(*a, **k):
        return None

    async def fake_deliver_media(*a, **k):
        return None

    mixin._send_queued_final_text = fake_send_queued
    mixin._deliver_media_from_response = fake_deliver_media
    source = SimpleNamespace(chat_id="67890", platform=Platform.DISCORD)
    asyncio.run(
        mixin._deliver_queued_first_response(
            response, source, adapter,
            text_already_delivered=text_already_delivered,
            deliver_media=deliver_media,
        )
    )


def test_queued_final_streamed_into_voice_channel():
    adapter = _voice_adapter()
    _run(adapter)
    assert adapter.stream_calls == [(12345, "Final spoken answer.")]


def test_queued_final_skips_voice_when_already_delivered():
    adapter = _voice_adapter()
    _run(adapter, text_already_delivered=True)
    assert adapter.stream_calls == []


def test_queued_final_skips_voice_when_streaming_disabled():
    adapter = _voice_adapter()
    adapter._stream_tts_enabled = lambda: False
    _run(adapter)
    assert adapter.stream_calls == []


def test_queued_final_skips_voice_when_chat_not_bound():
    adapter = _voice_adapter()
    adapter._voice_text_channels = {111: 999}  # different chat
    _run(adapter)
    assert adapter.stream_calls == []


def test_queued_final_survives_adapter_without_voice_surface():
    # Must not raise; nothing to call.
    _run(_plain_adapter())


def test_queued_final_stream_failure_is_fail_open():
    """A streaming exception must not block the queued text send."""

    class _BoomAdapter:
        _voice_text_channels = {111: 555}

        @staticmethod
        def _stream_tts_enabled():
            return True

        @staticmethod
        async def play_reply_streaming_in_voice(gid, text):
            raise RuntimeError("speech server down")

        @staticmethod
        def is_in_voice_channel(gid):
            return True

        @staticmethod
        def extract_media(text):
            return [], text

        async def send(self, *a, **k):
            raise AssertionError("voice failure must not fall through to a bare duplicate send in this lane guard")

    from gateway.run_notifications import GatewayNotificationsMixin

    mixin = object.__new__(GatewayNotificationsMixin)

    async def fake_send_queued(*a, **k):
        return None

    mixin._send_queued_final_text = fake_send_queued
    source = SimpleNamespace(chat_id="555", platform=Platform.DISCORD)
    # Must not raise.
    asyncio.run(
        mixin._deliver_queued_first_response("hello", source, _BoomAdapter(),
                                             deliver_media=False)
    )
