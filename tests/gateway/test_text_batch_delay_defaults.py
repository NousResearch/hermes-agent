"""Text-batch debounce defaults must not exceed Telegram's cadence (#44883, #25056).

WhatsApp (5s/10s) and Weixin (3s/5s) used to hold every reply for seconds
before dispatching; Telegram waits 0.3s (1.0s near a split chunk).
"""

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.event import MessageEvent, MessageType
from gateway.platforms.weixin import WeixinAdapter
from gateway.session import SessionSource
from plugins.platforms.whatsapp.adapter import WhatsAppAdapter


@pytest.mark.parametrize(
    ("adapter_cls", "platform"),
    [(WhatsAppAdapter, Platform.WHATSAPP), (WeixinAdapter, Platform.WEIXIN)],
    ids=["whatsapp", "weixin"],
)
def test_default_text_batch_delays_match_telegram_cadence(adapter_cls, platform, tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = adapter_cls(PlatformConfig(enabled=True, extra={}))
    source = SessionSource(platform=platform, chat_id="c1", chat_type="dm", user_id="u1")
    short_event = MessageEvent(text="hello", message_type=MessageType.TEXT, source=source)
    short_event._last_chunk_len = len(short_event.text)

    assert adapter._text_batch_delay_for(short_event) <= 0.5
    assert adapter._text_batch_split_delay_seconds <= 1.5
