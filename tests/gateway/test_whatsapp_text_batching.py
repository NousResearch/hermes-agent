"""Text-debounce batching for the WhatsApp adapter (issue #35301).

WhatsApp delivers rapid multi-message bursts (forwarded batches, paste-splits)
individually.  Without debounce each fragment triggers a separate agent
invocation, wasting tokens and flooding the user with reply fragments.  This
mirrors the Telegram/WeCom/Feishu pattern.

Batch delays are read from ``config.extra`` (config.yaml), not env vars.
"""

import asyncio
from unittest.mock import AsyncMock

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType
from gateway.session import SessionSource
from plugins.platforms.whatsapp.adapter import WhatsAppAdapter


def _make_adapter(**extra):
    base = {"session_name": "test"}
    base.update(extra)
    return WhatsAppAdapter(PlatformConfig(enabled=True, extra=base))


def _event(text):
    src = SessionSource(
        platform=Platform.WHATSAPP,
        chat_id="chat123",
        chat_type="dm",
        user_id="user1",
        user_name="tester",
    )
    return MessageEvent(text=text, message_type=MessageType.TEXT, source=src)


def test_batch_delays_overridden_via_config_extra():
    adapter = _make_adapter(
        text_batch_delay_seconds="2.5",
        text_batch_split_delay_seconds=7,
    )
    assert adapter._text_batch_delay_seconds == 2.5
    assert adapter._text_batch_split_delay_seconds == 7.0


def test_invalid_config_value_falls_back_to_default():
    adapter = _make_adapter(
        text_batch_delay_seconds="garbage",
        text_batch_split_delay_seconds=-3,
    )
    assert adapter._text_batch_delay_seconds == 5.0
    assert adapter._text_batch_split_delay_seconds == 10.0


async def _flush_with_recorded_delay(
    monkeypatch, adapter, event, *, last_chunk_len=None
):
    sleeps = []

    async def fake_sleep(delay):
        sleeps.append(delay)

    monkeypatch.setattr("plugins.platforms.whatsapp.adapter.asyncio.sleep", fake_sleep)
    adapter.handle_message = AsyncMock()
    key = adapter._text_batch_key(event)
    event._last_chunk_len = (  # type: ignore[attr-defined]
        len(event.text or "") if last_chunk_len is None else last_chunk_len
    )
    adapter._pending_text_batches[key] = event

    await adapter._flush_text_batch(key)

    adapter.handle_message.assert_awaited_once_with(event)
    return sleeps


def test_adaptive_batch_tiers_are_ordered():
    assert WhatsAppAdapter._TEXT_BATCH_FAST_LEN < WhatsAppAdapter._TEXT_BATCH_SHORT_LEN
    assert (
        WhatsAppAdapter._TEXT_BATCH_FAST_DELAY_S
        < WhatsAppAdapter._TEXT_BATCH_SHORT_DELAY_S
    )
    assert WhatsAppAdapter._TEXT_BATCH_FAST_DELAY_S > 0
    assert WhatsAppAdapter._TEXT_BATCH_SHORT_DELAY_S > 0


def test_fast_text_batch_uses_fast_delay(monkeypatch):
    adapter = _make_adapter(text_batch_delay_seconds=5.0)

    async def _drive():
        sleeps = await _flush_with_recorded_delay(
            monkeypatch, adapter, _event("f" * 320)
        )
        assert sleeps == [0.18]

    asyncio.run(_drive())


def test_medium_text_batch_uses_short_delay(monkeypatch):
    adapter = _make_adapter(text_batch_delay_seconds=5.0)

    async def _drive():
        sleeps = await _flush_with_recorded_delay(
            monkeypatch, adapter, _event("m" * 1024)
        )
        assert sleeps == [0.24]

    asyncio.run(_drive())


def test_tier_uses_total_batch_length_not_latest_chunk(monkeypatch):
    adapter = _make_adapter(text_batch_delay_seconds=5.0)

    async def _drive():
        sleeps = await _flush_with_recorded_delay(
            monkeypatch,
            adapter,
            _event("m" * 321),
            last_chunk_len=1,
        )
        assert sleeps == [0.24]

    asyncio.run(_drive())


def test_configured_delay_caps_adaptive_tiers(monkeypatch):
    adapter = _make_adapter(text_batch_delay_seconds=0.1)

    async def _drive():
        fast_sleeps = await _flush_with_recorded_delay(
            monkeypatch, adapter, _event("fast")
        )
        medium_sleeps = await _flush_with_recorded_delay(
            monkeypatch, adapter, _event("m" * 321)
        )
        assert fast_sleeps == [0.1]
        assert medium_sleeps == [0.1]

    asyncio.run(_drive())


def test_long_text_batch_uses_configured_delay(monkeypatch):
    adapter = _make_adapter(text_batch_delay_seconds=0.5)

    async def _drive():
        sleeps = await _flush_with_recorded_delay(
            monkeypatch, adapter, _event("l" * 1025)
        )
        assert sleeps == [0.5]

    asyncio.run(_drive())


def test_split_threshold_uses_configured_split_delay(monkeypatch):
    adapter = _make_adapter(
        text_batch_delay_seconds=0.5,
        text_batch_split_delay_seconds=1.5,
    )

    async def _drive():
        sleeps = await _flush_with_recorded_delay(
            monkeypatch,
            adapter,
            _event("s" * WhatsAppAdapter._SPLIT_THRESHOLD),
        )
        assert sleeps == [1.5]

    asyncio.run(_drive())
