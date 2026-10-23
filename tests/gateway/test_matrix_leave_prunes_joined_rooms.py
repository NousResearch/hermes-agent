"""Matrix prunes rooms the bot left / was kicked from on each sync, so a later
re-invite isn't silently dropped by the stale joined-rooms short-circuits."""
import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig


def _make_adapter():
    from plugins.platforms.matrix.adapter import MatrixAdapter
    config = PlatformConfig(
        enabled=True,
        token="syt_test_token",
        extra={
            "homeserver": "https://matrix.example.org",
            "user_id": "@bot:example.org",
        },
    )
    return MatrixAdapter(config)


def _client():
    c = MagicMock()
    c.crypto = None
    c.sync_store = MagicMock()
    c.sync_store.put_next_batch = AsyncMock()
    c.handle_sync = MagicMock(return_value=[])
    return c


class TestMatrixLeavePrunesJoinedRooms:

    @pytest.mark.asyncio
    async def test_incremental_leave_prunes_only_the_left_room(self):
        """A room in rooms.leave (left or kicked) is dropped from _joined_rooms on an
        incremental sync; other joined rooms are untouched.

        Without this, _joined_rooms only ever grew between reconnects, so the
        short-circuits in _join_room_by_id / _schedule_invite_join /
        _schedule_pending_invite_joins (all keyed on `room_id in _joined_rooms`)
        treated a kicked/left room as still-joined and dropped a later re-invite.
        """
        adapter = _make_adapter()
        adapter._closing = False
        client = _client()
        adapter._client = client

        await adapter._absorb_sync(client, {"rooms": {"join": {
            "!stay:example.org": {}, "!gone:example.org": {}}}, "next_batch": "s1"})
        assert {"!stay:example.org", "!gone:example.org"} <= adapter._joined_rooms

        # Incremental sync: the bot is kicked from / leaves one room.
        await adapter._absorb_sync(client, {"rooms": {"leave": {
            "!gone:example.org": {}}}, "next_batch": "s2"})

        assert "!gone:example.org" not in adapter._joined_rooms  # pruned
        assert "!stay:example.org" in adapter._joined_rooms      # untouched

    @pytest.mark.asyncio
    async def test_room_in_both_join_and_leave_stays_joined(self):
        """When one sync reports a room in both rooms.join and rooms.leave
        (membership churned join->leave->join since `since`), the bot is currently
        joined, so the room must survive the prune.

        _joined_rooms is shared by reference with the crypto store and seeds the DM
        cache, so a spurious prune would drop a live room from both until the next
        reconnect.
        """
        adapter = _make_adapter()
        adapter._closing = False
        client = _client()
        adapter._client = client

        await adapter._absorb_sync(client, {"rooms": {
            "join": {"!r:example.org": {}},
            "leave": {"!r:example.org": {}},
        }, "next_batch": "s1"})

        assert "!r:example.org" in adapter._joined_rooms  # current join wins

    @pytest.mark.asyncio
    async def test_reinvite_after_leave_actually_rejoins(self):
        """End-to-end payoff: after the bot leaves a room, a later invite to it
        drives a real re-join — not just a set edit.

        _schedule_pending_invite_joins and _join_room_by_id both short-circuit on
        `room_id in _joined_rooms`, so without the leave-prune the re-invite is
        silently dropped and join_room is never called. This asserts join_room fires
        and the room is rejoined, without a gateway reconnect.
        """
        adapter = _make_adapter()
        adapter._closing = False
        adapter._user_id = "@bot:example.org"
        adapter._authorization_check = None
        adapter._allowed_user_ids = {"@boss:example.org"}
        adapter._refresh_dm_cache = AsyncMock()  # isolate: DM cache is unrelated here
        client = _client()
        client.join_room = AsyncMock()
        adapter._client = client

        # Bot joins, then is kicked / leaves the room.
        await adapter._absorb_sync(client, {"rooms": {
            "join": {"!r:example.org": {}}}, "next_batch": "s1"})
        await adapter._absorb_sync(client, {"rooms": {
            "leave": {"!r:example.org": {}}}, "next_batch": "s2"})
        assert "!r:example.org" not in adapter._joined_rooms
        client.join_room.assert_not_awaited()

        # A fresh invite to the same room from an authorized user.
        invite = {"invite_state": {"events": [{
            "type": "m.room.member",
            "state_key": "@bot:example.org",
            "sender": "@boss:example.org",
            "content": {"membership": "invite", "is_direct": False},
        }]}}
        await adapter._absorb_sync(client, {"rooms": {
            "invite": {"!r:example.org": invite}}, "next_batch": "s3"})

        # The join is scheduled off the sync path; drive the task to completion.
        await asyncio.gather(*list(adapter._invite_join_tasks.values()))

        client.join_room.assert_awaited_once()
        assert str(client.join_room.await_args.args[0]) == "!r:example.org"
        assert "!r:example.org" in adapter._joined_rooms  # rejoined, no reconnect
