"""Text-debounce batching for the WhatsApp adapter (issue #35301).

WhatsApp delivers rapid multi-message bursts (forwarded batches, paste-splits)
individually.  Without debounce each fragment triggers a separate agent
invocation, wasting tokens and flooding the user with reply fragments.  This
mirrors the Telegram/WeCom/Feishu pattern.

Batch delays are read from ``config.extra`` (config.yaml), not env vars.
"""

import asyncio

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType
from plugins.platforms.whatsapp.adapter import WhatsAppAdapter
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


