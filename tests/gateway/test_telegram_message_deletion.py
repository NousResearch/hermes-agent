"""Whether the Telegram bot can delete a user's message in a chat: a vault prompt is only posted
where its answer can be deleted on receipt (``gateway.vault_prompts.bound_owner``)."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from telegram.error import Forbidden

from gateway.config import Platform, PlatformConfig


def _adapter(member=None, error=None):
    from plugins.platforms.telegram.adapter import TelegramAdapter

    adapter = object.__new__(TelegramAdapter)
    adapter.platform = Platform.TELEGRAM
    adapter.config = PlatformConfig(enabled=True, token="***")
    adapter._bot = SimpleNamespace(id=999, get_chat_member=AsyncMock(return_value=member, side_effect=error))
    return adapter


@pytest.mark.asyncio
@pytest.mark.parametrize("member, expected", [
    (SimpleNamespace(status="creator"), True),
    (SimpleNamespace(status="administrator", can_delete_messages=True), True),
    (SimpleNamespace(status="administrator", can_delete_messages=False), False),
    (SimpleNamespace(status="member"), False),
])
async def test_group_rights(member, expected):
    adapter = _adapter(member)
    assert await adapter.can_delete_user_messages("-100777", "group") is expected
    assert adapter._bot.get_chat_member.await_args.kwargs["user_id"] == 999, "the bot's own rights"


@pytest.mark.asyncio
async def test_a_private_chat_needs_no_lookup():
    adapter = _adapter()
    assert await adapter.can_delete_user_messages("1001", "dm") is True
    adapter._bot.get_chat_member.assert_not_awaited()


@pytest.mark.asyncio
async def test_an_unreadable_right_is_no_right():
    adapter = _adapter(error=Forbidden("bot was kicked from the supergroup chat"))
    assert await adapter.can_delete_user_messages("-100777", "group") is False
