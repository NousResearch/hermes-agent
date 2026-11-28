"""An unauthorized Matrix invite must be declined, not only logged.

``_on_invite`` logged a warning and returned, which left the membership at
``invite``: the room reappears in every later sync and the same warning is
re-emitted after each gateway restart. Declining settles that state. The
allow-list gate itself must stay closed either way.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from mautrix.types import RoomID

from gateway.config import PlatformConfig

ROOM_ID = "!spam:evil.example"


def _make_adapter():
    """Create a MatrixAdapter with a one-user allowlist and a mocked client."""
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
    adapter._client = MagicMock()
    adapter._client.leave_room = AsyncMock()
    adapter._schedule_invite_join = MagicMock()
    return adapter


def _invite_event(sender="@mallory:evil.example", room_id=ROOM_ID, is_direct=False):
    return SimpleNamespace(
        room_id=room_id,
        sender=sender,
        content=SimpleNamespace(is_direct=is_direct),
    )


class TestDeclineUnauthorizedInvite:
    @pytest.mark.asyncio
    async def test_unauthorized_invite_is_declined_and_not_joined(self):
        adapter = _make_adapter()

        await adapter._on_invite(_invite_event())

        adapter._client.leave_room.assert_awaited_once_with(RoomID(ROOM_ID))
        adapter._schedule_invite_join.assert_not_called()

    @pytest.mark.asyncio
    async def test_failed_decline_does_not_raise_or_join(self):
        """A rejected leave must not escape into the sync dispatch path.

        The invite stays pending on the server, so the next sync sees it
        again and retries; the handler itself only has to stay contained.
        """
        adapter = _make_adapter()
        adapter._client.leave_room.side_effect = Exception("M_FORBIDDEN: not in room")

        await adapter._on_invite(_invite_event())

        adapter._client.leave_room.assert_awaited_once()
        adapter._schedule_invite_join.assert_not_called()

    @pytest.mark.asyncio
    async def test_authorized_invite_still_joins_and_is_not_declined(self):
        adapter = _make_adapter()

        await adapter._on_invite(_invite_event(sender="@alice:example.org"))

        adapter._client.leave_room.assert_not_awaited()
        adapter._schedule_invite_join.assert_called_once_with(
            ROOM_ID, is_direct=False, inviter="@alice:example.org"
        )
