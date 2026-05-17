import pytest

from gateway.config import PlatformConfig
from gateway.platforms.telegram import TelegramAdapter


class _FakeMessage:
    message_id = 123


class _FakeBot:
    def __init__(self):
        self.calls = []

    async def send_message(self, **kwargs):
        self.calls.append(kwargs)
        return _FakeMessage()


@pytest.mark.asyncio
async def test_send_accepts_non_numeric_telegram_chat_id_without_local_int_cast():
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="123:fake"))
    bot = _FakeBot()
    adapter._bot = bot

    result = await adapter.send("lobobyte_bot", "hello")

    assert result.success is True
    assert bot.calls
    assert bot.calls[0]["chat_id"] == "lobobyte_bot"


def test_telegram_chat_id_arg_preserves_usernames_and_coerces_numeric_ids():
    assert TelegramAdapter._telegram_chat_id_arg("8558728265") == 8558728265
    assert TelegramAdapter._telegram_chat_id_arg("-1001234567890") == -1001234567890
    assert TelegramAdapter._telegram_chat_id_arg("lobobyte_bot") == "lobobyte_bot"
    assert TelegramAdapter._telegram_chat_id_arg("@lobobyte_bot") == "@lobobyte_bot"
