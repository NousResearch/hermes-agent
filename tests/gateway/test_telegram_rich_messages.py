from types import SimpleNamespace

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.telegram import TelegramAdapter


class FakeTelegramBot:
    def __init__(self):
        self.rich_calls = []
        self.send_messages = []
        self.edit_messages = []
        self._next_id = 100

    def _message(self):
        self._next_id += 1
        return SimpleNamespace(message_id=self._next_id)

    async def do_api_request(self, method, data=None):
        self.rich_calls.append({"method": method, "data": data or {}})
        return {"message_id": f"rich-{len(self.rich_calls)}"}

    async def send_message(self, **kwargs):
        self.send_messages.append(kwargs)
        return self._message()

    async def edit_message_text(self, **kwargs):
        self.edit_messages.append(kwargs)
        return True


def _adapter(*, rich_messages=None):
    extra = {}
    if rich_messages is not None:
        extra["rich_messages"] = rich_messages
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="***", extra=extra))
    bot = FakeTelegramBot()
    adapter._bot = bot
    return adapter, bot


@pytest.mark.asyncio
async def test_short_ordinary_markdown_stays_legacy_when_rich_enabled():
    adapter, bot = _adapter(rich_messages=True)

    result = await adapter.send("123", "A **short** reply.")

    assert result.success is True
    assert bot.rich_calls == []
    assert len(bot.send_messages) == 1


@pytest.mark.asyncio
async def test_heading_and_bullet_list_use_rich_when_enabled():
    adapter, bot = _adapter(rich_messages=True)
    content = "# Summary\n- First item\n- Second item"

    result = await adapter.send("123", content)

    assert result.success is True
    assert bot.send_messages == []
    assert bot.rich_calls == [
        {
            "method": "sendRichMessage",
            "data": {"chat_id": 123, "text": content, "disable_notification": True},
        }
    ]


@pytest.mark.asyncio
async def test_heading_and_bullet_list_stay_legacy_when_rich_false():
    adapter, bot = _adapter(rich_messages=False)

    result = await adapter.send("123", "# Summary\n- First item\n- Second item")

    assert result.success is True
    assert bot.rich_calls == []
    assert len(bot.send_messages) == 1


@pytest.mark.asyncio
async def test_heading_and_bullet_list_stay_legacy_when_rich_missing():
    adapter, bot = _adapter()

    result = await adapter.send("123", "# Summary\n- First item\n- Second item")

    assert result.success is True
    assert bot.rich_calls == []
    assert len(bot.send_messages) == 1


@pytest.mark.asyncio
async def test_long_plain_content_uses_single_rich_send_when_enabled():
    adapter, bot = _adapter(rich_messages=True)
    content = "x" * (adapter.MAX_MESSAGE_LENGTH + 100)

    result = await adapter.send("123", content)

    assert result.success is True
    assert bot.send_messages == []
    assert [call["method"] for call in bot.rich_calls] == ["sendRichMessage"]
    assert bot.rich_calls[0]["data"]["text"] == content


@pytest.mark.asyncio
async def test_long_plain_content_uses_legacy_split_when_rich_missing():
    adapter, bot = _adapter()
    content = "x" * (adapter.MAX_MESSAGE_LENGTH + 100)

    result = await adapter.send("123", content)

    assert result.success is True
    assert bot.rich_calls == []
    assert len(bot.send_messages) > 1


@pytest.mark.asyncio
async def test_final_long_plain_edit_uses_rich_edit_when_enabled():
    adapter, bot = _adapter(rich_messages=True)
    content = "x" * (adapter.MAX_MESSAGE_LENGTH + 100)

    result = await adapter.edit_message("123", "10", content, finalize=True)

    assert result.success is True
    assert bot.edit_messages == []
    assert bot.rich_calls == [
        {
            "method": "editRichMessage",
            "data": {
                "chat_id": 123,
                "message_id": 10,
                "text": content,
                "disable_notification": True,
            },
        }
    ]


@pytest.mark.asyncio
async def test_final_long_plain_edit_uses_legacy_split_when_rich_false():
    adapter, bot = _adapter(rich_messages=False)
    content = "x" * (adapter.MAX_MESSAGE_LENGTH + 100)

    result = await adapter.edit_message("123", "10", content, finalize=True)

    assert result.success is True
    assert bot.rich_calls == []
    assert bot.edit_messages
    assert bot.send_messages


def test_needs_rich_rendering_empty_and_disabled_are_false():
    adapter, _ = _adapter(rich_messages=True)
    assert adapter._needs_rich_rendering("") is False

    adapter, _ = _adapter(rich_messages=False)
    assert adapter._needs_rich_rendering("# Heading\n- item") is False
