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
from gateway.platforms.whatsapp import WhatsAppAdapter
from gateway.session import SessionSource


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


def test_batch_delays_default_from_config():
    adapter = _make_adapter()
    assert adapter._text_batch_delay_seconds == 5.0
    assert adapter._text_batch_split_delay_seconds == 10.0


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


def test_env_var_is_ignored(monkeypatch):
    # Config-only path: the legacy HERMES_* env var must NOT influence delays.
    monkeypatch.setenv("HERMES_WHATSAPP_TEXT_BATCH_DELAY_SECONDS", "99")
    adapter = _make_adapter()
    assert adapter._text_batch_delay_seconds == 5.0


def test_rapid_texts_collapse_into_single_dispatch():
    adapter = _make_adapter(
        text_batch_delay_seconds=0.05,
        text_batch_split_delay_seconds=0.05,
    )
    dispatched = []

    async def _capture(event):
        dispatched.append(event.text)

    adapter.handle_message = _capture

    async def _drive():
        adapter._enqueue_text_event(_event("one"))
        adapter._enqueue_text_event(_event("two"))
        adapter._enqueue_text_event(_event("three"))
        assert dispatched == []  # nothing flushed during the burst
        await asyncio.sleep(0.2)

    asyncio.run(_drive())
    assert dispatched == ["one\ntwo\nthree"]


def test_lone_message_dispatched_alone():
    adapter = _make_adapter(
        text_batch_delay_seconds=0.05,
        text_batch_split_delay_seconds=0.05,
    )
    dispatched = []

    async def _capture(event):
        dispatched.append(event.text)

    adapter.handle_message = _capture

    async def _drive():
        adapter._enqueue_text_event(_event("solo"))
        await asyncio.sleep(0.2)

    asyncio.run(_drive())
    assert dispatched == ["solo"]

async def _flush_with_recorded_delay(monkeypatch, adapter, event):
    sleeps = []

    async def fake_sleep(delay):
        sleeps.append(delay)

    monkeypatch.setattr("gateway.platforms.whatsapp.asyncio.sleep", fake_sleep)
    adapter.handle_message = AsyncMock()
    key = adapter._text_batch_key(event)
    event._last_chunk_len = len(event.text or "")  # type: ignore[attr-defined]
    adapter._pending_text_batches[key] = event

    await adapter._flush_text_batch(key)

    adapter.handle_message.assert_awaited_once_with(event)
    return sleeps


def test_adaptive_batch_tiers_mirror_telegram_ordering():
    assert WhatsAppAdapter._TEXT_BATCH_FAST_LEN < WhatsAppAdapter._TEXT_BATCH_SHORT_LEN
    assert WhatsAppAdapter._TEXT_BATCH_FAST_DELAY_S < WhatsAppAdapter._TEXT_BATCH_SHORT_DELAY_S
    assert WhatsAppAdapter._TEXT_BATCH_FAST_DELAY_S > 0
    assert WhatsAppAdapter._TEXT_BATCH_SHORT_DELAY_S > 0


def test_short_text_batch_uses_telegram_fast_delay(monkeypatch):
    adapter = _make_adapter(text_batch_delay_seconds=5.0)

    async def _drive():
        sleeps = await _flush_with_recorded_delay(monkeypatch, adapter, _event("short message"))
        assert sleeps == [WhatsAppAdapter._TEXT_BATCH_FAST_DELAY_S]

    asyncio.run(_drive())


def test_medium_text_batch_uses_telegram_short_delay(monkeypatch):
    adapter = _make_adapter(text_batch_delay_seconds=5.0)

    async def _drive():
        sleeps = await _flush_with_recorded_delay(
            monkeypatch,
            adapter,
            _event("m" * (WhatsAppAdapter._TEXT_BATCH_FAST_LEN + 1)),
        )
        assert sleeps == [WhatsAppAdapter._TEXT_BATCH_SHORT_DELAY_S]

    asyncio.run(_drive())


def test_configured_delay_can_cap_adaptive_tiers(monkeypatch):
    adapter = _make_adapter(text_batch_delay_seconds=0.1)

    async def _drive():
        short_sleeps = await _flush_with_recorded_delay(monkeypatch, adapter, _event("short"))
        medium_sleeps = await _flush_with_recorded_delay(
            monkeypatch,
            adapter,
            _event("m" * (WhatsAppAdapter._TEXT_BATCH_FAST_LEN + 1)),
        )
        assert short_sleeps == [0.1]
        assert medium_sleeps == [0.1]

    asyncio.run(_drive())


def test_long_text_batch_uses_configured_delay(monkeypatch):
    adapter = _make_adapter(text_batch_delay_seconds=0.5)

    async def _drive():
        sleeps = await _flush_with_recorded_delay(
            monkeypatch,
            adapter,
            _event("m" * (WhatsAppAdapter._TEXT_BATCH_SHORT_LEN + 1)),
        )
        assert sleeps == [0.5]

    asyncio.run(_drive())


def test_near_split_text_batch_uses_split_delay(monkeypatch):
    adapter = _make_adapter(text_batch_delay_seconds=0.5, text_batch_split_delay_seconds=1.5)

    async def _drive():
        sleeps = await _flush_with_recorded_delay(
            monkeypatch,
            adapter,
            _event("m" * WhatsAppAdapter._SPLIT_THRESHOLD),
        )
        assert sleeps == [1.5]

    asyncio.run(_drive())
