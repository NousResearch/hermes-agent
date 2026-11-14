"""A Telegram live location is one input: its GPS refreshes start a turn only after real movement.

Telegram delivers each refresh of a live location as an ``edited_message`` of the same message,
every few seconds to a minute for up to eight hours, and PTB dispatches those edits to the
location handler like new messages. Every tick used to become a full agent turn.
"""

import asyncio
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

from tests.gateway.test_telegram_group_gating import _FakeSessionStore, _group_location_message, _make_adapter

_T0 = datetime(2026, 11, 11, 12, 0, tzinfo=UTC)


def _pin(lat, *, live=True, edited_after=None, message_id=50, update_id=1):
    msg = _group_location_message(chat_id=111, lat=lat, lon=13.40)
    msg.chat = SimpleNamespace(id=111, type="private", title=None, is_forum=False)
    msg.message_id = message_id
    msg.date = _T0
    msg.edit_date = None if edited_after is None else _T0 + timedelta(seconds=edited_after)
    msg.location = SimpleNamespace(latitude=lat, longitude=13.40, live_period=3600 if live else None)
    return SimpleNamespace(update_id=update_id, message=msg, effective_message=msg)


def _deliver(adapter, updates):
    async def _run():
        for update in updates:
            await adapter._handle_location_message(update, SimpleNamespace())
    asyncio.run(_run())
    return [call.args[0].text.split("\n")[1] for call in adapter.handle_message.await_args_list]


def _adapter():
    adapter = _make_adapter(allow_from=["111"])
    adapter.handle_message = AsyncMock()
    return adapter


def test_a_live_share_is_one_turn_until_the_user_moves():
    adapter = _adapter()
    turns = _deliver(adapter, [
        _pin(52.52, update_id=1),
        # GPS jitter of about a metre, every 30 s: the same place
        *[_pin(52.52 + 0.00001 * i, edited_after=30 * i, update_id=1 + i) for i in range(1, 5)],
        _pin(52.53, edited_after=150, update_id=6),  # ~1.1 km away: new input
        _pin(52.53001, edited_after=180, update_id=7),  # jitter around the new spot
        _pin(52.53001, edited_after=180 + 901, update_id=8),  # parked past the quiet window
        _pin(52.53001, live=False, edited_after=1200, update_id=9),  # the share stopped
    ])
    assert turns == ["latitude: 52.52", "latitude: 52.53", "latitude: 52.53001"]


def test_static_pins_and_separate_live_shares_each_reach_the_agent():
    adapter = _adapter()
    turns = _deliver(adapter, [
        _pin(52.52, live=False, message_id=1),
        _pin(52.52, live=False, message_id=2),  # the same spot pinned again on purpose
        _pin(52.52, message_id=3),
        _pin(52.52, message_id=4),  # a second live share, its own message
    ])
    assert turns == ["latitude: 52.52"] * 4


def test_observed_group_records_a_live_share_once():
    """In a group that observes unmentioned messages, the refreshes of one live share must not
    each append a transcript row: the in-place check runs before the observe step."""
    adapter = _make_adapter(require_mention=True, allowed_chats=["-100"], group_allowed_chats=["-100"],
                            observe_unmentioned_group_messages=True)
    adapter.handle_message = AsyncMock()
    store = adapter._session_store = _FakeSessionStore()
    updates = [_pin(52.52, update_id=1),
               *[_pin(52.52 + 0.00001 * i, edited_after=30 * i, update_id=1 + i) for i in range(1, 5)]]
    for update in updates:
        update.message.chat = SimpleNamespace(id=-100, type="group", title="Test Group", is_forum=False)

    _deliver(adapter, updates)

    adapter.handle_message.assert_not_awaited()
    assert len(store.messages) == 1
