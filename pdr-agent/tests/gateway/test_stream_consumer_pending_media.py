"""A streamed reply must never show a MEDIA:/[[...]] directive while it is still being typed.

Regression: a voice-only reply ("MEDIA:/…/tts_….ogg") streamed token by token. Interim
edits fired before the path reached its extension, so the cleaner (which only strips
complete deliverable tags) let ``MEDIA:/home/…`` through. At finish the cleaned text was
empty, so nothing edited the bubble again and the partial path stayed above the voice note.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig
from tests.waiting import async_wait_until


def _adapter():
    adapter = MagicMock()
    adapter.send = AsyncMock(return_value=SimpleNamespace(success=True, message_id="m1"))
    adapter.edit_message = AsyncMock(return_value=SimpleNamespace(success=True))
    adapter.MAX_MESSAGE_LENGTH = 4096
    return adapter


def _shown(adapter):
    texts = [c.kwargs.get("content", "") for c in adapter.send.call_args_list]
    texts += [c.kwargs.get("content", "") for c in adapter.edit_message.call_args_list]
    return texts


async def _stream(deltas):
    adapter = _adapter()
    consumer = GatewayStreamConsumer(
        adapter, "chat_1", StreamConsumerConfig(edit_interval=0.0, buffer_threshold=1))
    task = asyncio.create_task(consumer.run())
    seen = ""
    for delta in deltas:
        consumer.on_delta(delta)
        seen += delta
        # Every chunk is eligible for an interim edit (interval 0): wait until the consumer
        # has absorbed it, then yield so any edit it decides on actually goes out.
        await async_wait_until(lambda: consumer._accumulated.endswith(seen[-8:])
                               and consumer._queue.empty())
        for _ in range(5):
            await asyncio.sleep(0)
    consumer.finish()
    await task
    return adapter


@pytest.mark.asyncio
async def test_voice_only_reply_never_shows_partial_media_path():
    adapter = await _stream([
        "MED", "IA:/home/u/.hermes/cache/audio/", "tts_20261014_143816", "_843235.ogg"])
    for text in _shown(adapter):
        assert "MEDIA" not in text and "/home/" not in text, text


@pytest.mark.asyncio
async def test_partial_voice_directive_is_never_shown():
    adapter = await _stream(["Te mando el audio.\n", "[[audio_", "as_voice]]\n",
                             "MEDIA:/home/u/.hermes/cache/audio/a", ".ogg"])
    for text in _shown(adapter):
        assert "[[" not in text and "MEDIA" not in text, text
    assert any("Te mando el audio." in t for t in _shown(adapter))


@pytest.mark.asyncio
async def test_prose_mentioning_media_still_streams():
    adapter = await _stream(["La palabra MEDIA aparece ", "en un texto normal ", "y sigue."])
    assert any("La palabra MEDIA aparece" in t for t in _shown(adapter))


@pytest.mark.asyncio
async def test_prose_with_media_colon_streams_before_the_line_ends():
    # "media:" followed by words, not a path, is never a directive: the open line keeps streaming.
    adapter = await _stream(["Platforms we cover, social media: ", "Twitter, Mastodon ", "and Bluesky",
                             " plus a few forums", " in one long line"])
    interim = [t for t in _shown(adapter) if "in one long line" not in t]
    assert any("Twitter, Mastodon" in t for t in interim), _shown(adapter)


@pytest.mark.asyncio
async def test_partial_tag_after_punctuation_is_never_shown():
    # A complete tag has no left boundary ("图片：MEDIA:/…" is stripped), so its prefix after
    # punctuation must be held too; showing "MED" and then removing it breaks prefix-stable frames.
    adapter = await _stream(["Report attached.\n", "图片：MED", "IA:/home/u/report.pdf"])
    for text in _shown(adapter):
        assert "MED" not in text, text
    assert any("Report attached." in t for t in _shown(adapter))
