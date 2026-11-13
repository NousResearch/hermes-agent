"""Inbound Telegram dice reach the agent through the same group gate as a location pin (#78719)."""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from tests.gateway.test_telegram_group_gating import _FakeSessionStore, _group_location_message, _make_adapter


def _dice_update(**msg_kw):
    msg = _group_location_message(**msg_kw)
    msg.location, msg.dice = None, SimpleNamespace(emoji="\U0001F3B2", value=4)
    return SimpleNamespace(update_id=3003, message=msg, effective_message=None)


@pytest.mark.parametrize(("require_mention", "delivered"), [(False, True), (True, False)])
def test_dice_follows_group_gate(require_mention, delivered):
    from hermes_agent.plugins.platforms.telegram.telegram_dice import handle_dice_message

    adapter = _make_adapter(require_mention=require_mention, group_allowed_chats=["-100"])
    adapter.handle_message = AsyncMock()
    asyncio.run(handle_dice_message(adapter, _dice_update(), SimpleNamespace()))
    assert adapter.handle_message.await_count == int(delivered)
    if delivered:
        assert "\U0001F3B2 rolled 4" in adapter.handle_message.call_args[0][0].text


def test_observed_dice_keeps_its_roll_without_a_turn():
    from hermes_agent.plugins.platforms.telegram.telegram_dice import handle_dice_message

    adapter = _make_adapter(require_mention=True, allowed_chats=["-100"], group_allowed_chats=["-100"],
                            observe_unmentioned_group_messages=True)
    adapter._session_store = store = _FakeSessionStore()
    adapter.handle_message = AsyncMock()
    asyncio.run(handle_dice_message(adapter, _dice_update(), SimpleNamespace()))
    adapter.handle_message.assert_not_awaited()
    (_sid, message, _skip_db), = store.messages
    assert message["observed"] is True and "\U0001F3B2 rolled 4" in message["content"]
