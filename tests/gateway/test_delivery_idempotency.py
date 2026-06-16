import asyncio
from types import SimpleNamespace
from typing import Any, Dict, Optional

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.platforms.telegram import TelegramAdapter


class _RecordingAdapter(BasePlatformAdapter):
    def __init__(self):
        super().__init__(PlatformConfig(enabled=True), Platform.TELEGRAM)
        self.sent = []

    async def connect(self) -> bool:
        return True

    async def disconnect(self) -> None:
        return None

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        return {"name": chat_id, "type": "dm"}

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        message_id = f"m{len(self.sent) + 1}"
        self.sent.append(
            {
                "chat_id": chat_id,
                "content": content,
                "reply_to": reply_to,
                "metadata": dict(metadata or {}),
                "message_id": message_id,
            }
        )
        return SendResult(success=True, message_id=message_id)


class _RecordingTelegramBot:
    def __init__(self):
        self.sent_messages = []

    async def send_message(self, **kwargs):
        message_id = len(self.sent_messages) + 100
        self.sent_messages.append(dict(kwargs))
        return SimpleNamespace(message_id=message_id)

    async def send_chat_action(self, *args, **kwargs):
        return True


def test_final_delivery_same_run_same_content_is_sent_once():
    adapter = _RecordingAdapter()

    first = asyncio.run(
        adapter._send_final_once(
            chat_id="chat-1",
            content="Clean final",
            session_key="telegram:dm:chat-1:user-1",
            run_generation=7,
            metadata={"notify": True},
        )
    )
    second = asyncio.run(
        adapter._send_final_once(
            chat_id="chat-1",
            content="Clean final",
            session_key="telegram:dm:chat-1:user-1",
            run_generation=7,
            metadata={"notify": True},
        )
    )

    assert first.success is True
    assert second.success is True
    assert second.raw_response["duplicate_suppressed"] is True
    assert second.message_id == first.message_id
    assert [m["content"] for m in adapter.sent] == ["Clean final"]


def test_final_delivery_same_run_different_content_sends_followup():
    adapter = _RecordingAdapter()

    asyncio.run(
        adapter._send_final_once(
            chat_id="chat-1",
            content="Initial final",
            session_key="telegram:dm:chat-1:user-1",
            run_generation=7,
        )
    )
    changed = asyncio.run(
        adapter._send_final_once(
            chat_id="chat-1",
            content="Corrected final",
            session_key="telegram:dm:chat-1:user-1",
            run_generation=7,
        )
    )

    assert changed.success is True
    assert changed.message_id == "m2"
    assert [m["content"] for m in adapter.sent] == ["Initial final", "Corrected final"]


def test_final_delivery_new_generation_can_send_same_content_again():
    adapter = _RecordingAdapter()

    asyncio.run(
        adapter._send_final_once(
            chat_id="chat-1",
            content="Clean final",
            session_key="telegram:dm:chat-1:user-1",
            run_generation=7,
        )
    )
    asyncio.run(
        adapter._send_final_once(
            chat_id="chat-1",
            content="Clean final",
            session_key="telegram:dm:chat-1:user-1",
            run_generation=8,
        )
    )

    assert [m["message_id"] for m in adapter.sent] == ["m1", "m2"]


def test_telegram_final_delivery_same_run_same_content_hits_bot_api_once():
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="test-token"))
    bot = _RecordingTelegramBot()
    adapter._bot = bot
    adapter._rich_send_disabled = True

    first = asyncio.run(
        adapter._send_final_once(
            chat_id="12345",
            content="Clean final for Telegram",
            session_key="agent:main:telegram:dm:12345:999",
            run_generation=42,
            metadata={
                "notify": True,
                "direct_messages_topic_id": "999",
                "telegram_dm_topic_reply_fallback": True,
                "telegram_reply_to_message_id": "777",
            },
        )
    )
    duplicate = asyncio.run(
        adapter._send_final_once(
            chat_id="12345",
            content="Clean final for Telegram",
            session_key="agent:main:telegram:dm:12345:999",
            run_generation=42,
            metadata={
                "notify": True,
                "direct_messages_topic_id": "999",
                "telegram_dm_topic_reply_fallback": True,
                "telegram_reply_to_message_id": "777",
            },
        )
    )

    assert first.success is True
    assert duplicate.success is True
    assert duplicate.raw_response["duplicate_suppressed"] is True
    assert duplicate.message_id == first.message_id == "100"
    assert len(bot.sent_messages) == 1
    sent = bot.sent_messages[0]
    assert sent["chat_id"] == 12345
    assert sent["reply_to_message_id"] == 777
    assert "disable_notification" not in sent
