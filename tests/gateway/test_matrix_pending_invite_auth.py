"""Tests for the inviter allowlist gate on pending-invite reconciliation.

``_on_invite`` only auto-joins when the inviter is allow-listed, but a
pending invite reconciled from sync's ``rooms.invite`` after a gateway
restart never fires ``_on_invite``. ``_schedule_pending_invite_joins``
must apply the same gate, reading the inviter from the stripped invite
state, or an invite from an arbitrary federated user that arrives while
the gateway is down gets auto-joined on restart.
"""

import logging
import time
from unittest.mock import AsyncMock

import pytest

from gateway.config import PlatformConfig


def _make_adapter():
    """Create a MatrixAdapter with mocked config and a one-user allowlist."""
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
    adapter._allowed_user_ids = {"@alice:example.org"}
    adapter._join_room_by_id = AsyncMock(return_value=True)
    adapter._record_dm_room = AsyncMock()
    return adapter


def _member_invite_event(
    state_key="@hermes:example.org",
    sender="@alice:example.org",
    is_direct=True,
    membership="invite",
):
    """Create a stripped m.room.member event as found in invite_state."""
    return {
        "type": "m.room.member",
        "state_key": state_key,
        "sender": sender,
        "content": {"membership": membership, "is_direct": is_direct},
    }


def _invite_sync_data(room_id="!pending_room:example.org", invite_state=None):
    """Create a sync payload with one pending invite room."""
    room = {} if invite_state is None else {"invite_state": invite_state}
    return {"rooms": {"invite": {room_id: room}}, "next_batch": "s1"}


async def _drain_invite_tasks(adapter):
    """Await any tasks _schedule_invite_join spawned."""
    for task in list(adapter._invite_join_tasks.values()):
        await task


class TestPendingInviteAuthorization:
    """_schedule_pending_invite_joins applies _on_invite's inviter gate.

    Rejection mirrors _on_invite exactly: no join is scheduled (so no
    entry lands in _invite_join_tasks), nothing is recorded in m.direct,
    and the pending invite is otherwise left untouched.
    """

    @pytest.mark.asyncio
    async def test_allowed_inviter_is_joined(self):
        adapter = _make_adapter()

        sync_data = _invite_sync_data(invite_state={"events": [_member_invite_event()]})
        adapter._schedule_pending_invite_joins(sync_data)
        await _drain_invite_tasks(adapter)

        adapter._join_room_by_id.assert_awaited_once_with("!pending_room:example.org")
        adapter._record_dm_room.assert_awaited_once_with(
            "!pending_room:example.org", "@alice:example.org"
        )

    @pytest.mark.asyncio
    async def test_unknown_bot_user_id_fails_closed(self):
        adapter = _make_adapter()
        adapter._user_id = ""

        sync_data = _invite_sync_data(invite_state={"events": [_member_invite_event()]})
        adapter._schedule_pending_invite_joins(sync_data)
        await _drain_invite_tasks(adapter)

        adapter._join_room_by_id.assert_not_awaited()
        adapter._record_dm_room.assert_not_awaited()
        assert adapter._invite_join_tasks == {}

    @pytest.mark.parametrize(
        "invite_state",
        [
            pytest.param(
                {"events": [_member_invite_event(sender="@mallory:evil.example")]},
                id="non-allowed-inviter",
            ),
            pytest.param(None, id="no-invite-state"),
            pytest.param({"events": []}, id="empty-events"),
            pytest.param(
                {"events": [_member_invite_event(state_key="@other:example.org")]},
                id="member-event-for-other-user",
            ),
            pytest.param(
                {"events": [_member_invite_event(sender="")]},
                id="missing-inviter",
            ),
        ],
    )
    @pytest.mark.asyncio
    async def test_unauthorized_or_unknown_inviter_is_not_joined(self, invite_state):
        """An inviter outside the allowlist, or one that cannot be read
        from the stripped invite state at all, fails closed like
        _on_invite: no join is scheduled and nothing is recorded."""
        adapter = _make_adapter()

        sync_data = _invite_sync_data(invite_state=invite_state)
        adapter._schedule_pending_invite_joins(sync_data)
        await _drain_invite_tasks(adapter)

        adapter._join_room_by_id.assert_not_awaited()
        adapter._record_dm_room.assert_not_awaited()
        assert adapter._invite_join_tasks == {}

    @pytest.mark.asyncio
    async def test_empty_allowlist_fails_closed(self):
        """With no allowlist configured, _on_invite rejects every invite;
        reconciliation must do the same."""
        adapter = _make_adapter()
        adapter._allowed_user_ids = set()

        sync_data = _invite_sync_data(invite_state={"events": [_member_invite_event()]})
        adapter._schedule_pending_invite_joins(sync_data)
        await _drain_invite_tasks(adapter)

        adapter._join_room_by_id.assert_not_awaited()
        adapter._record_dm_room.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_allow_all_env_bypasses_gate(self, monkeypatch):
        """GATEWAY_ALLOW_ALL_USERS disables the gate, exactly as it does
        for _on_invite, even when the inviter is unknown."""
        monkeypatch.setenv("GATEWAY_ALLOW_ALL_USERS", "true")
        adapter = _make_adapter()
        adapter._allowed_user_ids = set()

        sync_data = _invite_sync_data(invite_state=None)
        adapter._schedule_pending_invite_joins(sync_data)
        await _drain_invite_tasks(adapter)

        adapter._join_room_by_id.assert_awaited_once_with("!pending_room:example.org")
        adapter._record_dm_room.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_allow_all_logs_direct_invite_without_inviter(
        self, monkeypatch, caplog
    ):
        monkeypatch.setenv("GATEWAY_ALLOW_ALL_USERS", "true")
        adapter = _make_adapter()
        adapter._allowed_user_ids = set()

        sync_data = _invite_sync_data(
            invite_state={"events": [_member_invite_event(sender="")]}
        )
        with caplog.at_level(
            logging.WARNING,
            logger="plugins.platforms.matrix.adapter",
        ):
            adapter._schedule_pending_invite_joins(sync_data)
            await _drain_invite_tasks(adapter)

        adapter._join_room_by_id.assert_awaited_once_with("!pending_room:example.org")
        adapter._record_dm_room.assert_not_awaited()
        assert [record.getMessage() for record in caplog.records] == [
            "Matrix: joining direct invite to !pending_room:example.org "
            "without recording it in m.direct because the invite state "
            "has no inviter"
        ]
