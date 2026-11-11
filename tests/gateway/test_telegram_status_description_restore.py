"""The opt-in status indicator borrows the bot's short description without destroying it (#78784).

The short description is the bot's profile text. The operator's own text is captured before the
first "Online" write, survives crashes and restarts (the live text is then the status string), and
comes back when the indicator is turned off.
"""

from types import SimpleNamespace

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.telegram.adapter import TelegramAdapter


class _Bot:
    """Bot API double holding one short description, like the real profile."""

    id = 4242

    def __init__(self, text):
        self.text = text

    async def get_my_short_description(self):
        return SimpleNamespace(short_description=self.text)

    async def set_my_short_description(self, short_description):
        self.text = short_description


def _adapter(tmp_path, bot, **extra):
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="***", extra=extra))
    adapter._update_receipt_dir = tmp_path  # the profile home the adapter was built in
    adapter._bot = bot
    return adapter


@pytest.mark.asyncio
async def test_operator_text_survives_restarts_and_returns_when_the_indicator_is_off(tmp_path):
    bot = _Bot("Plans trips and books trains")
    first = _adapter(tmp_path, bot, status_indicator=True)
    await first._set_status_indicator(online=True)
    assert bot.text == "Online"
    # A crash leaves "Online" up; the next boot must not take it for the operator's text.
    second = _adapter(tmp_path, bot, status_indicator=True)
    await second._set_status_indicator(online=True)
    await second._set_status_indicator(online=False)
    assert bot.text == "Offline"  # documented behaviour while the indicator is on

    off = _adapter(tmp_path, bot)  # operator turned status_indicator off
    await off._set_status_indicator(online=True)
    assert bot.text == "Plans trips and books trains"
    bot.text = "Edited later in BotFather"
    await _adapter(tmp_path, bot)._set_status_indicator(online=True)
    assert bot.text == "Edited later in BotFather"  # restored once, never again


@pytest.mark.asyncio
async def test_new_botfather_text_replaces_the_capture(tmp_path):
    bot = _Bot("v1 text")
    await _adapter(tmp_path, bot, status_indicator=True)._set_status_indicator(online=True)
    bot.text = "v2 text"  # operator edits the profile while the gateway is down
    await _adapter(tmp_path, bot, status_indicator=True)._set_status_indicator(online=True)

    await _adapter(tmp_path, bot)._set_status_indicator(online=True)

    assert bot.text == "v2 text"
