import asyncio
from unittest.mock import AsyncMock

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType, SessionSource


def _make_adapter():
    from gateway.platforms.telegram import TelegramAdapter

    config = PlatformConfig(enabled=True, token="test-token")
    adapter = object.__new__(TelegramAdapter)
    adapter._platform = Platform.TELEGRAM
    adapter.config = config
    adapter._pending_text_batches = {}
    adapter._pending_text_batch_tasks = {}
    adapter._text_batch_delay_seconds = 0.01
    adapter._text_batch_split_delay_seconds = 0.01
    adapter._TEXT_BATCH_FAST_LEN = 320
    adapter._TEXT_BATCH_SHORT_LEN = 1024
    adapter._TEXT_BATCH_FAST_DELAY_S = 0.01
    adapter._TEXT_BATCH_SHORT_DELAY_S = 0.01
    adapter._SPLIT_THRESHOLD = 3900
    adapter._active_sessions = {}
    adapter._pending_messages = {}
    adapter._message_handler = AsyncMock()
    adapter.handle_message = AsyncMock()
    adapter.send = AsyncMock()
    return adapter


def _make_event(text: str) -> MessageEvent:
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=SessionSource(platform=Platform.TELEGRAM, chat_id="12345", chat_type="dm"),
        message_id="99",
    )


def test_metatron_frontdoor_response_bypasses_agent_handler(monkeypatch):
    async def run_test():
        adapter = _make_adapter()

        async def fake_frontdoor(_text: str) -> str:
            return "THE hub is ready."

        monkeypatch.setattr(
            "gateway.platforms.telegram.handle_metatron_frontdoor_text",
            fake_frontdoor,
        )

        adapter._enqueue_text_event(_make_event("Metatron, initialize THE hub"))
        await asyncio.sleep(0.05)

        adapter.handle_message.assert_not_called()
        adapter.send.assert_called_once()
        assert adapter.send.call_args.args[1] == "THE hub is ready."

    asyncio.run(run_test())


def test_non_metatron_text_routes_to_agent_handler(monkeypatch):
    async def run_test():
        adapter = _make_adapter()

        async def fake_frontdoor(_text: str):
            return None

        monkeypatch.setattr(
            "gateway.platforms.telegram.handle_metatron_frontdoor_text",
            fake_frontdoor,
        )

        adapter._enqueue_text_event(_make_event("what is blocked today?"))
        await asyncio.sleep(0.05)

        adapter.handle_message.assert_called_once()
        adapter.send.assert_not_called()

    asyncio.run(run_test())
