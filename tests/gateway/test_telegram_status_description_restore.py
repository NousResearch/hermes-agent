"""The opt-in status indicator borrows the bot's short description without destroying it (#78784).

The short description is the bot's profile text. The operator's own text is captured before the
first "Online" write, survives crashes and restarts (the live text is then the status string), and
comes back when the indicator is turned off.
"""

from types import SimpleNamespace

import pytest

from hermes_agent.gateway.config import PlatformConfig
from plugins.platforms.telegram.adapter import TelegramAdapter


class _Bot:
    """Bot API double holding one short description, like the real profile."""

    id = 4242

    def __init__(self, text):
        self.text = text
        self.read_fails = False

    async def get_my_short_description(self):
        if self.read_fails:
            raise TimeoutError("Bot API read timed out")
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


@pytest.mark.asyncio
async def test_status_text_is_never_taken_for_the_operators_and_never_written_without_a_capture(tmp_path):
    bot = _Bot("Plans trips and books trains")
    bot.read_fails = True  # the first capture cannot read the operator's text
    unsaved = _adapter(tmp_path, bot, status_indicator=True)
    await unsaved._set_status_indicator(online=True)
    await unsaved._set_status_indicator(online=False)
    assert bot.text == "Plans trips and books trains"  # no status write over the only copy

    bot.read_fails = False
    long_label = "Online " + "x" * 200  # longer than Telegram's 120-char cap
    await _adapter(tmp_path, bot, status_indicator=True, status_online=long_label)._set_status_indicator(online=True)
    # The operator renames both labels between runs: the old Hermes text is still not theirs.
    renamed = _adapter(tmp_path, bot, status_indicator=True, status_online="Up", status_offline="Down")
    await renamed._set_status_indicator(online=True)
    await _adapter(tmp_path, bot)._set_status_indicator(online=True)
    assert bot.text == "Plans trips and books trains"


@pytest.mark.asyncio
async def test_turning_the_indicator_off_keeps_a_newer_botfather_text(tmp_path):
    bot = _Bot("v1 text")
    run = _adapter(tmp_path, bot, status_indicator=True)
    await run._set_status_indicator(online=True)
    await run._set_status_indicator(online=False)
    bot.text = "v2 text"  # set in BotFather while the gateway is down, indicator then turned off

    await _adapter(tmp_path, bot)._set_status_indicator(online=True)

    assert bot.text == "v2 text"


@pytest.mark.asyncio
async def test_botfather_edit_made_while_connected_survives_offline_and_comes_back(tmp_path):
    bot = _Bot("Original profile")
    running = _adapter(tmp_path, bot, status_indicator=True)
    await running._set_status_indicator(online=True)
    bot.text = "New profile"  # operator edits BotFather while Hermes is connected
    await running._set_status_indicator(online=False)
    assert bot.text == "Offline"  # documented clean-disconnect indicator stays

    await _adapter(tmp_path, bot)._set_status_indicator(online=True)  # indicator turned off

    assert bot.text == "New profile"
