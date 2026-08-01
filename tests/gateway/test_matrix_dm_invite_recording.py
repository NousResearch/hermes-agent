"""Tests for Matrix DM room recording on invite (issue #44679).

When the bot's Matrix account has no ``m.direct`` account data (common for
accounts created solely for Hermes), DM rooms are silently treated as groups.
This tests the fix that records DM rooms in ``m.direct`` when the invite
event carries ``is_direct: true``.
"""

import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import PlatformConfig


def _make_adapter(tmp_path=None):
    """Create a MatrixAdapter with mocked config."""
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
    adapter._text_batch_delay_seconds = 0
    adapter.handle_message = AsyncMock()
    adapter._startup_ts = time.time() - 10
    # Authorize the inviter used throughout this module so the invite-auth
    # gate in _on_invite (rejects auto-joins from non-allow-listed users)
    # lets the join through and the DM-recording side effects are exercised.
    adapter._allowed_user_ids = {"@alice:example.org"}
    return adapter


def _make_invite_event(
    room_id="!dm_room:example.org",
    sender="@alice:example.org",
    is_direct=True,
):
    """Create a fake invite event with is_direct in content."""
    content = SimpleNamespace(is_direct=is_direct)
    return SimpleNamespace(
        room_id=room_id,
        sender=sender,
        content=content,
    )


# ---------------------------------------------------------------------------
# _on_invite DM recording
# ---------------------------------------------------------------------------


class TestOnInviteRecordsDM:
    """_on_invite schedules a join that records the DM when is_direct is True.

    The join itself is non-blocking (``_schedule_invite_join`` spawns a task),
    so these tests drive ``_on_invite`` and then await the scheduled task to
    observe its side effects.
    """

    @staticmethod
    async def _drain_invite_tasks(adapter):
        """Await any tasks _schedule_invite_join spawned."""
        tasks = list(adapter._invite_join_tasks.values())
        for task in tasks:
            await task

    @pytest.mark.asyncio
    async def test_dm_invite_records_room(self):
        adapter = _make_adapter()
        adapter._join_room_by_id = AsyncMock(return_value=True)
        adapter._record_dm_room = AsyncMock()

        event = _make_invite_event(is_direct=True, sender="@alice:example.org")
        await adapter._on_invite(event)
        await self._drain_invite_tasks(adapter)

        adapter._join_room_by_id.assert_awaited_once_with("!dm_room:example.org")
        adapter._record_dm_room.assert_awaited_once_with(
            "!dm_room:example.org", "@alice:example.org"
        )

    @pytest.mark.asyncio
    async def test_non_dm_invite_does_not_record(self):
        adapter = _make_adapter()
        adapter._join_room_by_id = AsyncMock(return_value=True)
        adapter._record_dm_room = AsyncMock()

        event = _make_invite_event(is_direct=False)
        await adapter._on_invite(event)
        await self._drain_invite_tasks(adapter)

        adapter._join_room_by_id.assert_awaited_once()
        adapter._record_dm_room.assert_not_awaited()


# ---------------------------------------------------------------------------
# _record_dm_room
# ---------------------------------------------------------------------------


class TestRecordDMRoom:
    """_record_dm_room should update m.direct account data and local cache."""

    @pytest.mark.asyncio
    async def test_creates_m_direct_when_absent(self):
        """When m.direct doesn't exist (404), creates it from scratch."""
        adapter = _make_adapter()
        adapter._client = MagicMock()
        adapter._client.get_account_data = AsyncMock(side_effect=Exception("M_NOT_FOUND"))
        adapter._client.set_account_data = AsyncMock()

        await adapter._record_dm_room("!new:example.org", "@alice:example.org")

        adapter._client.set_account_data.assert_awaited_once_with(
            "m.direct", {"@alice:example.org": ["!new:example.org"]}
        )
        assert adapter._dm_rooms.get("!new:example.org") is True


    @pytest.mark.asyncio
    async def test_no_duplicate_room_in_m_direct(self):
        """If room is already in m.direct, does not append again."""
        adapter = _make_adapter()
        adapter._client = MagicMock()
        existing_data = {"@alice:example.org": ["!room:example.org"]}
        adapter._client.get_account_data = AsyncMock(return_value=existing_data)
        adapter._client.set_account_data = AsyncMock()

        await adapter._record_dm_room("!room:example.org", "@alice:example.org")

        adapter._client.set_account_data.assert_not_awaited()
        assert adapter._dm_rooms.get("!room:example.org") is True


