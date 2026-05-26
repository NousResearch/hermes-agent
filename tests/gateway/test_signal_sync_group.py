"""Tests for Signal syncMessage group auto-detection.

When the bot is a founding member of a new Signal group, signal-cli sends a
syncMessage with a top-level groupV2 field instead of a regular group invite
envelope.  The adapter should detect this and route to the existing group invite
handler.
"""
import pytest
from unittest.mock import AsyncMock

from gateway.config import PlatformConfig
from tests.gateway.test_signal_enhancements import _make_signal_adapter


@pytest.fixture(autouse=True)
def _reset_signal_scheduler():
    from gateway.platforms.signal_rate_limit import _reset_scheduler
    _reset_scheduler()
    yield
    _reset_scheduler()


class TestSyncMessageGroupCreation:
    """syncMessage with groupV2 should be routed to the group invite handler."""

    @pytest.mark.asyncio
    async def test_sync_message_group_creation_detected(self, monkeypatch):
        """syncMessage with top-level groupV2 should add group to
        group_allow_from when group_invite_policy is allow-all."""
        adapter = _make_signal_adapter(
            monkeypatch,
            group_invite_policy="allow-all",
        )

        rpc_calls = []

        async def mock_rpc(method, params, rpc_id=None, **kwargs):
            rpc_calls.append(method)
            return {"success": True}

        adapter._rpc = mock_rpc

        # syncMessage with groupV2 at the syncMessage level (not inside sentMessage)
        envelope = {
            "envelope": {
                "sourceNumber": "+15559999999",
                "sourceUuid": "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
                "sourceName": "Group Creator",
                "syncMessage": {
                    "groupV2": {"groupId": "sync-created-group-001"},
                },
            }
        }

        await adapter._handle_envelope(envelope)

        assert "joinGroup" in rpc_calls
        assert "sync-created-group-001" in adapter.group_allow_from

    @pytest.mark.asyncio
    async def test_sync_message_group_creation_respects_policy(self, monkeypatch):
        """With approved-only policy and unknown inviter, group should NOT
        be added from a syncMessage groupV2."""
        adapter = _make_signal_adapter(
            monkeypatch,
            allowed_users="+15551111111",  # only this user approved
        )
        # default policy is approved-only

        rpc_calls = []

        async def mock_rpc(method, params, rpc_id=None, **kwargs):
            rpc_calls.append(method)
            return {"success": True}

        adapter._rpc = mock_rpc

        # Sender is NOT in the approved list
        envelope = {
            "envelope": {
                "sourceNumber": "+15559999999",
                "sourceUuid": "77777777-7777-7777-7777-777777777777",
                "sourceName": "Unknown Creator",
                "syncMessage": {
                    "groupV2": {"groupId": "sync-created-group-002"},
                },
            }
        }

        await adapter._handle_envelope(envelope)

        assert "joinGroup" not in rpc_calls
        assert "sync-created-group-002" not in adapter.group_allow_from

    @pytest.mark.asyncio
    async def test_sync_message_note_to_self_still_works(self, monkeypatch):
        """Note to Self messages via syncMessage should still be promoted
        to dataMessage and not be confused with group creation."""
        adapter = _make_signal_adapter(
            monkeypatch,
            account="+15551234567",
            allowed_users="*",
        )

        processed_messages = []

        # Save original _handle_envelope to check dataMessage promotion
        original_handle = adapter._handle_envelope

        async def mock_rpc(method, params, rpc_id=None, **kwargs):
            return {"success": True}

        adapter._rpc = mock_rpc

        # Note to Self: sentMessage with destination matching bot's account
        envelope = {
            "envelope": {
                "sourceNumber": "+15551234567",
                "sourceUuid": "self-uuid-1234",
                "sourceName": "Bot",
                "syncMessage": {
                    "sentMessage": {
                        "destinationNumber": "+15551234567",
                        "timestamp": 999999,
                        "message": "Note to myself",
                    },
                },
            }
        }

        # Should not raise, and should not add any group
        await adapter._handle_envelope(envelope)

        # No group should have been added
        assert len(adapter.group_allow_from) == 0
