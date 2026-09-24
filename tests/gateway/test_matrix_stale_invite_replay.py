"""Regression tests for stale Matrix invite replay from rooms.leave.

A full /sync includes historical timelines for rooms the bot has left.  Old
membership=invite events in those timelines must never trigger a fresh join;
current rooms.invite entries must continue through normal mautrix dispatch.
"""

from unittest.mock import MagicMock

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.matrix.adapter import MatrixAdapter


def _make_adapter() -> MatrixAdapter:
    config = PlatformConfig(
        enabled=True,
        token="syt_test_token",
        extra={
            "homeserver": "https://matrix.example.org",
            "user_id": "@hermes:example.org",
        },
    )
    return MatrixAdapter(config)


def test_filters_only_invite_membership_events_from_left_rooms():
    stale_invite = {
        "type": "m.room.member",
        "state_key": "@hermes:example.org",
        "content": {"membership": "invite", "is_direct": False},
    }
    later_leave = {
        "type": "m.room.member",
        "state_key": "@hermes:example.org",
        "content": {"membership": "leave"},
    }
    encrypted_message = {
        "type": "m.room.encrypted",
        "content": {"ciphertext": "preserve-me"},
    }
    current_invite = {
        "type": "m.room.member",
        "state_key": "@hermes:example.org",
        "content": {"membership": "invite", "is_direct": True},
    }
    sync_data = {
        "next_batch": "s123",
        "rooms": {
            "leave": {
                "!old:example.org": {
                    "state": {"events": [stale_invite]},
                    "timeline": {
                        "events": [stale_invite, encrypted_message, later_leave],
                        "limited": False,
                    },
                }
            },
            "invite": {
                "!current:example.org": {
                    "invite_state": {"events": [current_invite]}
                }
            },
        },
    }

    filtered = MatrixAdapter._without_stale_left_room_invites(sync_data)

    assert filtered is not sync_data
    assert filtered["rooms"]["leave"]["!old:example.org"]["state"]["events"] == []
    assert filtered["rooms"]["leave"]["!old:example.org"]["timeline"][
        "events"
    ] == [encrypted_message, later_leave]
    assert filtered["rooms"]["invite"] is sync_data["rooms"]["invite"]
    assert sync_data["rooms"]["leave"]["!old:example.org"]["timeline"][
        "events"
    ] == [stale_invite, encrypted_message, later_leave]


def test_no_match_returns_original_sync_object():
    sync_data = {
        "rooms": {
            "leave": {
                "!old:example.org": {
                    "timeline": {
                        "events": [
                            {
                                "type": "m.room.member",
                                "content": {"membership": "leave"},
                            }
                        ]
                    }
                }
            }
        }
    }

    assert MatrixAdapter._without_stale_left_room_invites(sync_data) is sync_data


@pytest.mark.asyncio
async def test_dispatch_sync_filters_stale_invite_before_mautrix():
    adapter = _make_adapter()
    adapter._client = MagicMock()
    adapter._client.handle_sync.return_value = []
    sync_data = {
        "rooms": {
            "leave": {
                "!old:example.org": {
                    "timeline": {
                        "events": [
                            {
                                "type": "m.room.member",
                                "content": {"membership": "invite"},
                            },
                            {
                                "type": "m.room.member",
                                "content": {"membership": "leave"},
                            },
                        ]
                    }
                }
            },
            "invite": {
                "!current:example.org": {
                    "invite_state": {
                        "events": [
                            {
                                "type": "m.room.member",
                                "content": {"membership": "invite"},
                            }
                        ]
                    }
                }
            },
        }
    }

    await adapter._dispatch_sync(sync_data)

    dispatched = adapter._client.handle_sync.call_args.args[0]
    assert dispatched["rooms"]["leave"]["!old:example.org"]["timeline"][
        "events"
    ] == [
        {"type": "m.room.member", "content": {"membership": "leave"}}
    ]
    assert "!current:example.org" in dispatched["rooms"]["invite"]
