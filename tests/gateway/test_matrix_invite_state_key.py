"""Tests for Matrix invite state-key filtering (regression for #76292).

In bridged rooms the sync stream carries membership invite events addressed to
users OTHER than the bot (e.g. a Discord ghost inviting the Matrix owner).
``_on_invite`` only checked the event SENDER (``_is_authorized_user``) and never
the event TARGET (``event.state_key``), so invites for other users were joined
as if the bot had been invited, spamming "rejecting invite"/"joining" log noise
on every sync.

These tests drive ``_on_invite`` directly with a fake invite event whose
``state_key`` names a third party and assert the invite is skipped quietly
(no join scheduled, no warning logged), while an invite whose ``state_key``
is the bot's own user id still joins.
"""

import logging
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import PlatformConfig


def _make_adapter(user_id="@hermes:example.org"):
    """Create a MatrixAdapter with a known bot user id."""
    from plugins.platforms.matrix.adapter import MatrixAdapter

    config = PlatformConfig(
        enabled=True,
        token="syt_test_token",
        extra={
            "homeserver": "https://matrix.example.org",
            "user_id": user_id,
        },
    )
    adapter = MatrixAdapter(config)
    adapter._text_batch_delay_seconds = 0
    adapter.handle_message = AsyncMock()
    adapter._startup_ts = time.time() - 10
    # Authorize the inviter so the only gate under test is the state_key check.
    adapter._allowed_user_ids = {"@owner:example.org"}
    return adapter


def _make_invite_event(room_id="!bridged:example.org", sender="@owner:example.org", state_key=None):
    """Create a fake m.room.member invite event addressed via state_key."""
    content = SimpleNamespace(is_direct=False, membership="invite")
    return SimpleNamespace(
        room_id=room_id,
        sender=sender,
        state_key=state_key,
        content=content,
    )


async def _drain_invite_tasks(adapter):
    """Await any tasks _schedule_invite_join spawned off the sync path."""
    for task in list(adapter._invite_join_tasks.values()):
        await task


class TestOnInviteStateKey:
    @pytest.mark.asyncio
    async def test_invite_to_other_user_is_skipped_quietly(self, caplog):
        """An invite addressed to someone else schedules no join and warns no one.

        Known-benign noise must stay quiet: no WARNING+ records from the skip.
        """
        adapter = _make_adapter()
        adapter._join_room_by_id = AsyncMock(return_value=True)

        event = _make_invite_event(state_key="@discord-ghost:example.org")
        with caplog.at_level(logging.DEBUG):
            await adapter._on_invite(event)

        assert adapter._invite_join_tasks == {}
        adapter._join_room_by_id.assert_not_awaited()
        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert warnings == []

    @pytest.mark.asyncio
    async def test_invite_to_self_still_joins(self):
        """An invite whose state_key is the bot still schedules the join."""
        adapter = _make_adapter()
        adapter._join_room_by_id = AsyncMock(return_value=True)

        event = _make_invite_event(state_key="@hermes:example.org")
        await adapter._on_invite(event)
        await _drain_invite_tasks(adapter)

        adapter._join_room_by_id.assert_awaited_once_with("!bridged:example.org")

    @pytest.mark.asyncio
    async def test_invite_processed_when_own_user_id_unknown(self):
        """If the bot's own user id is not yet known, don't suppress invites.

        whoami resolution may not have run; the state_key check must not turn
        every invite into a silent drop in that window.
        """
        adapter = _make_adapter(user_id="")
        adapter._join_room_by_id = AsyncMock(return_value=True)

        event = _make_invite_event(state_key="@someone:example.org")
        await adapter._on_invite(event)
        await _drain_invite_tasks(adapter)

        adapter._join_room_by_id.assert_awaited_once_with("!bridged:example.org")

    @pytest.mark.asyncio
    async def test_invite_without_state_key_stays_fail_closed(self):
        """An invite with no resolvable target is processed, not skipped.

        state_key is required on real m.room.member events; if it is ever
        absent we keep the old fail-closed behavior rather than silently
        dropping a possibly-legitimate invite.
        """
        adapter = _make_adapter()
        adapter._join_room_by_id = AsyncMock(return_value=True)

        event = _make_invite_event()
        del event.state_key  # model an event with no target at all
        await adapter._on_invite(event)
        await _drain_invite_tasks(adapter)

        adapter._join_room_by_id.assert_awaited_once_with("!bridged:example.org")
