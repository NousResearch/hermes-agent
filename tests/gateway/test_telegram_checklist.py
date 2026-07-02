from unittest.mock import AsyncMock

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.telegram.adapter import TelegramAdapter


@pytest.mark.asyncio
async def test_telegram_send_checklist_uses_raw_bot_api_and_pins():
    adapter = TelegramAdapter(
        PlatformConfig(
            enabled=True,
            token="fake",
            extra={"secretary": {"allowed_business_connections": ["bc-1"]}},
        )
    )
    bot = AsyncMock()
    bot.do_api_request.return_value = {"message_id": 123}
    adapter._bot = bot

    result = await adapter.send_checklist(
        chat_id="548392727",
        title="Task: Native checklists",
        tasks=["Add sendChecklist", {"id": 7, "text": "Pin message"}],
        pin=True,
    )

    assert result.success is True
    assert result.message_id == "123"
    bot.do_api_request.assert_awaited_once_with(
        "sendChecklist",
        api_kwargs={
            "business_connection_id": "bc-1",
            "chat_id": 548392727,
            "checklist": {
                "title": "Task: Native checklists",
                "tasks": [
                    {"id": 1, "text": "Add sendChecklist"},
                    {"id": 7, "text": "Pin message"},
                ],
                "others_can_mark_tasks_as_done": True,
            },
            "disable_notification": True,
        },
    )
    bot.pin_chat_message.assert_awaited_once_with(
        chat_id=548392727,
        message_id=123,
        disable_notification=True,
    )


@pytest.mark.asyncio
async def test_telegram_send_checklist_requires_business_connection():
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="fake", extra={}))
    adapter._bot = AsyncMock()

    result = await adapter.send_checklist(
        chat_id="548392727",
        title="Task: Missing BC",
        tasks=["one"],
    )

    assert result.success is False
    assert result.retryable is False
    assert result.error == "business_connection_id_required_for_telegram_checklist"


def test_telegram_input_checklist_validation_limits():
    checklist = TelegramAdapter._input_checklist("T", ["a", "b"])
    assert checklist["tasks"] == [{"id": 1, "text": "a"}, {"id": 2, "text": "b"}]

    with pytest.raises(ValueError, match="at most 30"):
        TelegramAdapter._input_checklist("T", [str(i) for i in range(31)])

    with pytest.raises(ValueError, match="too long"):
        TelegramAdapter._input_checklist("T", ["x" * 101])
