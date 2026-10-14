"""Cron delivery and a real Matrix reply must continue the same isolated transcript."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cron.scheduler_delivery import _deliver_result
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.session import SessionStore
from gateway.conversation import conversation_origin
from plugins.platforms.matrix.adapter import MatrixAdapter


@pytest.fixture
def adapter(tmp_path):
    config = PlatformConfig(enabled=True, token="test", extra={
        "homeserver": "https://example.org", "user_id": "@bot:example.org",
        "encryption": False,
    })
    adapter = MatrixAdapter(config)
    adapter._client = MagicMock()
    adapter._client.send_message_event = AsyncMock(side_effect=["$root", "$brief"])
    adapter._client.get_state_event = AsyncMock(return_value={})
    adapter._client.state_store.get_members = AsyncMock(return_value=["bot", "alice", "bob"])
    adapter._get_display_name = AsyncMock(return_value="Alice")
    adapter._background_read_receipt = MagicMock()
    adapter._require_mention = False
    adapter._auto_thread = False
    adapter._dm_auto_thread = False
    adapter._matrix_session_scope = "room"
    gateway = GatewayConfig(platforms={Platform.MATRIX: config}, thread_sessions_per_user=True)
    adapter._session_store = SessionStore(tmp_path / "sessions", gateway)
    adapter.config.extra["thread_sessions_per_user"] = True
    return adapter, gateway


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["dm", "group"])
@pytest.mark.parametrize("legacy", [False, True, "normalized"])
async def test_delivered_brief_and_actual_reply_share_only_the_intended_lane(adapter, kind, legacy):
    adapter, gateway = adapter
    room = "!room:example.org"
    adapter._dm_rooms = {room: kind == "dm"}
    origin = {"platform": "matrix", "chat_id": room, "user_id": "@alice:example.org"}
    if not legacy:
        origin["chat_type"] = kind
    if legacy == "normalized":
        from gateway.session import SessionSource
        source = SessionSource(Platform.MATRIX, room, chat_type=kind, user_id=origin["user_id"])
        origin["conversation"] = conversation_origin(source, thread_id=None)
    job = {"id": "digest", "name": "digest", "deliver": "origin", "origin": origin,
           "attach_to_session": True}
    with patch("gateway.config.load_gateway_config", return_value=gateway):
        error = await asyncio.to_thread(_deliver_result, job, "private brief", {Platform.MATRIX: adapter},
                                        asyncio.get_running_loop())
    assert error is None
    assert adapter._client.send_message_event.await_count == 2

    async def reply(room_id=room, thread="$root", sender="@alice:example.org"):
        event = await adapter._build_inbound_event(
            room_id, sender, "$reply", "continue", {},
            {"rel_type": "m.thread", "event_id": thread})
        assert event is not None
        return adapter._session_store.get_or_create_session(event.source)

    entry = await reply()
    transcript = adapter._session_store.load_transcript(entry.session_id)
    assert any("private brief" in message.get("content", "") for message in transcript)
    for other in [await reply(thread="$other"), await reply(room_id="!other:example.org")]:
        assert other.session_id != entry.session_id
        assert not adapter._session_store.load_transcript(other.session_id)
    if kind == "group":
        other = await reply(sender="@bob:example.org")
        assert other.session_id != entry.session_id
        assert not adapter._session_store.load_transcript(other.session_id)


@pytest.mark.asyncio
async def test_matrix_membership_does_not_change_conversation_kind(adapter):
    adapter, _ = adapter
    room = "!room:example.org"
    adapter._dm_rooms = {room: False}
    adapter._client.state_store.get_members.return_value = ["bot", "alice"]
    assert (await adapter.get_chat_info(room))["type"] == "group"
    adapter._client.state_store.get_members.return_value.append("bob")
    adapter._invalidate_room_identities(room)
    assert (await adapter.get_chat_info(room))["type"] == "group"


@pytest.mark.asyncio
async def test_legacy_matrix_dm_row_cannot_override_current_group_kind(adapter):
    from gateway.session import SessionSource

    adapter, gateway = adapter
    room = "!room:example.org"
    adapter._dm_rooms = {room: False}
    stale = adapter._session_store.get_or_create_session(
        SessionSource(Platform.MATRIX, room, chat_type="dm", user_id="@alice:example.org"))
    adapter._session_store.append_to_transcript(stale.session_id, {"role": "assistant", "content": "old private history"})
    job = {"id": "digest", "deliver": "origin", "attach_to_session": True,
           "origin": {"platform": "matrix", "chat_id": room, "user_id": "@alice:example.org"}}
    with patch("gateway.config.load_gateway_config", return_value=gateway):
        error = await asyncio.to_thread(_deliver_result, job, "group brief", {Platform.MATRIX: adapter},
                                        asyncio.get_running_loop())
    assert error is None
    event = await adapter._build_inbound_event(room, "@alice:example.org", "$reply", "continue", {},
                                             {"rel_type": "m.thread", "event_id": "$root"})
    reply = adapter._session_store.get_or_create_session(event.source)
    assert reply.session_id != stale.session_id
    transcript = adapter._session_store.load_transcript(reply.session_id)
    assert any("group brief" in m.get("content", "") for m in transcript)
    assert not any("old private history" in m.get("content", "") for m in transcript)
