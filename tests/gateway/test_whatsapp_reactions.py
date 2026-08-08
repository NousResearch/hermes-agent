"""Tests for WhatsApp emoji reactions (send_message action='react'/'unreact').

Covers the adapter → bridge `/react` dispatch, the "default to the message the
agent is replying to" behavior, and WhatsApp's empty-emoji unreact.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tests.gateway.test_whatsapp_formatting import _AsyncCM, _make_adapter


@pytest.fixture(autouse=True)
def _whatsapp_open_optin(monkeypatch):
    """Opt into WhatsApp allow-all so inbound dispatch reaches the event builder.

    The adapter fails closed on ``dm_policy: open`` without this (SECURITY.md
    2.6), which would otherwise drop the message before it is recorded as the
    chat's last inbound.
    """
    monkeypatch.setenv("WHATSAPP_ALLOW_ALL_USERS", "true")


def _ok_response(adapter, status=200, text=""):
    resp = MagicMock(status=status)
    resp.json = AsyncMock(return_value={"success": True})
    resp.text = AsyncMock(return_value=text)
    adapter._http_session.post = MagicMock(return_value=_AsyncCM(resp))
    return resp


@pytest.mark.asyncio
async def test_add_reaction_posts_to_bridge_react_endpoint():
    adapter = _make_adapter()
    _ok_response(adapter)

    result = await adapter.add_reaction("15551234567", "👍", message_id="msg-1")

    assert result == {"success": True, "message_id": "msg-1"}
    call = adapter._http_session.post.call_args
    assert call.args[0] == "http://127.0.0.1:3000/react"
    assert call.kwargs["json"] == {
        "chatId": "15551234567@s.whatsapp.net",
        "messageId": "msg-1",
        "emoji": "👍",
        "fromMe": False,
    }


@pytest.mark.asyncio
async def test_add_reaction_defaults_to_last_inbound_message():
    adapter = _make_adapter()
    await adapter._build_message_event({
        "messageId": "inbound-9",
        "chatId": "15551234567@s.whatsapp.net",
        "senderId": "15551234567@s.whatsapp.net",
        "senderName": "Tester",
        "chatName": "Tester",
        "isGroup": False,
        "body": "ship it",
        "hasMedia": False,
        "mediaUrls": [],
    })
    _ok_response(adapter)

    result = await adapter.add_reaction("15551234567", "❤️")

    assert result == {"success": True, "message_id": "inbound-9"}
    assert adapter._http_session.post.call_args.kwargs["json"]["messageId"] == "inbound-9"


@pytest.mark.asyncio
async def test_remove_reaction_sends_empty_emoji():
    adapter = _make_adapter()
    adapter._record_last_inbound("15551234567@s.whatsapp.net", "inbound-9")
    _ok_response(adapter)

    result = await adapter.remove_reaction("15551234567")

    assert result == {"success": True, "message_id": "inbound-9"}
    # WhatsApp models an unreact as re-reacting with an empty emoji.
    assert adapter._http_session.post.call_args.kwargs["json"]["emoji"] == ""


@pytest.mark.asyncio
async def test_react_by_phone_finds_message_recorded_under_lid():
    """The bridge surfaces a DM as a LID; a react addressed by phone must still
    resolve it — both forms canonicalize to the same identity."""
    aliases = {"195794724511942@lid": "94768741618", "94768741618": "94768741618"}

    adapter = _make_adapter()
    _ok_response(adapter)

    with patch(
        "plugins.platforms.whatsapp.adapter.canonical_whatsapp_identifier",
        side_effect=lambda cid: aliases.get(cid, cid),
    ):
        adapter._record_last_inbound("195794724511942@lid", "inbound-lid")
        result = await adapter.add_reaction("94768741618", "👍")

    assert result == {"success": True, "message_id": "inbound-lid"}
    assert adapter._http_session.post.call_args.kwargs["json"]["messageId"] == "inbound-lid"


@pytest.mark.asyncio
async def test_reacting_to_owner_message_sets_from_me():
    """fromOwner messages are our own (fromMe=true); the reaction key must match
    or WhatsApp resolves it to no message at all."""
    adapter = _make_adapter()
    await adapter._build_message_event({
        "messageId": "owner-1",
        "chatId": "15551234567@s.whatsapp.net",
        "senderId": "15551234567@s.whatsapp.net",
        "senderName": "Owner",
        "chatName": "Owner",
        "isGroup": False,
        "body": "handling this myself",
        "hasMedia": False,
        "mediaUrls": [],
        "fromOwner": True,
    })
    _ok_response(adapter)

    result = await adapter.add_reaction("15551234567", "👍")

    assert result == {"success": True, "message_id": "owner-1"}
    assert adapter._http_session.post.call_args.kwargs["json"]["fromMe"] is True


@pytest.mark.asyncio
async def test_inbound_reaction_does_not_become_the_react_target():
    """A reaction is not itself a reactable message — reacting to one resolves to
    no target at all, so it must not displace the real last inbound message."""
    adapter = _make_adapter()
    base = {
        "chatId": "15551234567@s.whatsapp.net",
        "senderId": "15551234567@s.whatsapp.net",
        "senderName": "Tester",
        "chatName": "Tester",
        "isGroup": False,
        "hasMedia": False,
        "mediaUrls": [],
    }
    await adapter._build_message_event({**base, "messageId": "real-msg", "body": "ship it"})
    await adapter._build_message_event({
        **base,
        "messageId": "reaction-evt",
        "body": "[Reaction: ❤️ to real-msg]",
        "nativeType": "reactionMessage",
    })
    _ok_response(adapter)

    result = await adapter.add_reaction("15551234567", "👍")

    assert result == {"success": True, "message_id": "real-msg"}


@pytest.mark.asyncio
async def test_reacting_to_inbound_message_does_not_set_from_me():
    adapter = _make_adapter()
    adapter._record_last_inbound("15551234567@s.whatsapp.net", "inbound-9")
    _ok_response(adapter)

    await adapter.add_reaction("15551234567", "👍")

    assert adapter._http_session.post.call_args.kwargs["json"]["fromMe"] is False


@pytest.mark.asyncio
async def test_reaction_without_target_or_inbound_message_errors():
    adapter = _make_adapter()
    adapter._http_session.post = MagicMock()

    result = await adapter.add_reaction("15551234567", "👍")

    assert result["success"] is False
    assert "pass message_id" in result["error"]
    adapter._http_session.post.assert_not_called()


@pytest.mark.asyncio
async def test_reaction_surfaces_bridge_error():
    adapter = _make_adapter()
    _ok_response(adapter, status=500, text="Not connected to WhatsApp")

    result = await adapter.add_reaction("15551234567", "👍", message_id="msg-1")

    assert result == {"success": False, "error": "Not connected to WhatsApp"}


@pytest.mark.asyncio
async def test_last_inbound_map_is_bounded():
    adapter = _make_adapter()
    for i in range(adapter._LAST_INBOUND_CHATS_MAX + 25):
        adapter._record_last_inbound(f"1555{i:07d}@s.whatsapp.net", f"msg-{i}")

    assert len(adapter._last_inbound_by_chat) == adapter._LAST_INBOUND_CHATS_MAX
    # Oldest chats evicted, newest retained.
    n = adapter._LAST_INBOUND_CHATS_MAX + 24
    assert adapter._reaction_chat_key("15550000000@s.whatsapp.net") not in (
        adapter._last_inbound_by_chat
    )
    newest = adapter._reaction_chat_key(f"1555{n:07d}@s.whatsapp.net")
    assert adapter._last_inbound_by_chat[newest] == (f"msg-{n}", False)
