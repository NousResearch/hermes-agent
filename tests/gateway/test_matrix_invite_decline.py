"""Tests for declining invites from unauthorised inviters.

Rejecting an invite without leaving it leaves the room pending in
``rooms.invite``: the same warning repeats on every sync and restart, and the
invite never settles. Both gates — the live ``_on_invite`` handler and the
post-sync ``_schedule_pending_invite_joins`` reconcile pass — must leave the
room instead. A failed leave stays retryable; a successful one is remembered so
the room is not left repeatedly.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import PlatformConfig


def _make_adapter():
    """Create a MatrixAdapter with mocked config, one-user allowlist and a client."""
    from plugins.platforms.matrix.adapter import MatrixAdapter

    config = PlatformConfig(
        enabled=True,
        token="syt_test_token",
        extra={
            "homeserver": "https://matrix.example.org",
            "user_id": "@hermes:example.org",
        },
    )
    adapter = MatrixAdapter(config)
    adapter._allowed_user_ids = {"@alice:example.org"}
    adapter._join_room_by_id = AsyncMock(return_value=True)
    adapter._record_dm_room = AsyncMock()
    adapter._client = SimpleNamespace(leave_room=AsyncMock())
    return adapter


def _invite_event(room_id="!spam:evil.example", sender="@mallory:evil.example", is_direct=False):
    """A live invite event as mautrix delivers it to _on_invite."""
    return SimpleNamespace(
        room_id=room_id,
        sender=sender,
        content=SimpleNamespace(is_direct=is_direct),
    )


def _invite_sync_data(room_id="!spam:evil.example", sender="@mallory:evil.example"):
    """A sync payload with one pending invite, as seen by the reconcile pass."""
    return {
        "rooms": {
            "invite": {
                room_id: {
                    "invite_state": {
                        "events": [
                            {
                                "type": "m.room.member",
                                "state_key": "@hermes:example.org",
                                "sender": sender,
                                "content": {"membership": "invite", "is_direct": False},
                            }
                        ]
                    }
                }
            }
        },
        "next_batch": "s1",
    }


async def _drain(adapter, attr="_invite_decline_tasks"):
    for task in list(getattr(adapter, attr).values()):
        await task


class TestInviteDecline:
    @pytest.mark.asyncio
    async def test_live_unauthorized_invite_is_declined_not_joined(self):
        adapter = _make_adapter()

        await adapter._on_invite(_invite_event())
        await _drain(adapter)

        adapter._client.leave_room.assert_awaited_once_with("!spam:evil.example")
        adapter._join_room_by_id.assert_not_awaited()
        assert "!spam:evil.example" in adapter._declined_invites

    @pytest.mark.asyncio
    async def test_reconcile_unauthorized_invite_is_declined_not_joined(self):
        adapter = _make_adapter()

        adapter._schedule_pending_invite_joins(_invite_sync_data())
        await _drain(adapter)

        adapter._client.leave_room.assert_awaited_once_with("!spam:evil.example")
        adapter._join_room_by_id.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_authorized_inviter_is_joined_and_never_declined(self):
        adapter = _make_adapter()

        await adapter._on_invite(_invite_event(sender="@alice:example.org", is_direct=True))
        await _drain(adapter, "_invite_join_tasks")

        adapter._join_room_by_id.assert_awaited_once_with("!spam:evil.example")
        adapter._client.leave_room.assert_not_awaited()
        assert adapter._declined_invites == set()

    @pytest.mark.asyncio
    async def test_repeated_sync_does_not_decline_the_same_room_twice(self):
        adapter = _make_adapter()
        sync_data = _invite_sync_data()

        adapter._schedule_pending_invite_joins(sync_data)
        await _drain(adapter)
        adapter._schedule_pending_invite_joins(sync_data)
        await _drain(adapter)

        adapter._client.leave_room.assert_awaited_once_with("!spam:evil.example")

    @pytest.mark.asyncio
    async def test_failed_decline_is_retried_on_the_next_sync(self):
        adapter = _make_adapter()
        adapter._client.leave_room = AsyncMock(side_effect=[RuntimeError("m_unable"), None])
        sync_data = _invite_sync_data()

        adapter._schedule_pending_invite_joins(sync_data)
        await _drain(adapter)
        # A failure must not be remembered as a settled room.
        assert "!spam:evil.example" not in adapter._declined_invites

        adapter._schedule_pending_invite_joins(sync_data)
        await _drain(adapter)

        assert adapter._client.leave_room.await_count == 2
        assert "!spam:evil.example" in adapter._declined_invites

    @pytest.mark.asyncio
    async def test_decline_is_a_noop_without_a_client(self):
        adapter = _make_adapter()
        adapter._client = None

        adapter._schedule_pending_invite_joins(_invite_sync_data())
        await _drain(adapter)

        assert adapter._invite_decline_tasks == {}
        assert adapter._declined_invites == set()
