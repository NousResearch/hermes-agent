"""Inbound Telegram dice reach the agent through the same group gate as a location pin (#78719)."""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from tests.gateway.test_telegram_group_gating import _group_location_message, _make_adapter


def _dice_update(**msg_kw):
    msg = _group_location_message(**msg_kw)
    msg.location, msg.dice = None, SimpleNamespace(emoji="\U0001F3B2", value=4)
    return SimpleNamespace(update_id=3003, message=msg, effective_message=None)


@pytest.mark.parametrize(("require_mention", "delivered"), [(False, True), (True, False)])
def test_dice_follows_group_gate(require_mention, delivered):
    from plugins.platforms.telegram.telegram_dice import handle_dice_message

    adapter = _make_adapter(require_mention=require_mention, group_allowed_chats=["-100"])
    adapter.handle_message = AsyncMock()
    asyncio.run(handle_dice_message(adapter, _dice_update(), SimpleNamespace()))
    assert adapter.handle_message.await_count == int(delivered)
    if delivered:
        assert "\U0001F3B2 rolled 4" in adapter.handle_message.call_args[0][0].text
