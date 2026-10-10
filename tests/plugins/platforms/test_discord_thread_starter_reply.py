"""Real discord.py reply payloads distinguish parent-channel starters from in-thread posts."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

discord = pytest.importorskip("discord")

from gateway.config import PlatformConfig
from plugins.platforms.discord.adapter import DiscordAdapter


def _adapter_channel(parent_type, *, thread=True, mode="first"):
    http = SimpleNamespace(
        send_message=AsyncMock(return_value={"id": "9000"}),
        request=AsyncMock(return_value={"id": "9000"}),
    )
    state = SimpleNamespace(
        http=http, allowed_mentions=None,
        create_message=lambda **kw: SimpleNamespace(id=int(kw["data"]["id"])),
    )
    guild = SimpleNamespace(id=100, get_channel=lambda _id: parent)
    parent = None
    if parent_type is not None:
        parent_cls = discord.ForumChannel if parent_type in (15, 16) else discord.TextChannel
        parent = parent_cls(state=state, guild=guild, data={
            "id": "200", "name": "parent", "type": parent_type, "position": 0,
        })
    if thread:
        channel = discord.Thread(state=state, guild=guild, data={
            "id": "300", "parent_id": "200", "owner_id": "400", "name": "conversation",
            "type": 10 if parent_type == 5 else 11, "message_count": 1, "member_count": 1,
            "thread_metadata": {
                "archived": False, "auto_archive_duration": 1440,
                "archive_timestamp": "2026-01-01T00:00:00+00:00",
            },
        })
    elif parent is not None:
        channel = parent
    else:
        channel = discord.DMChannel(state=state, me=SimpleNamespace(id=400), data={
            "id": "300", "recipients": [],
        })
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="test-token", reply_to_mode=mode))
    adapter._client = SimpleNamespace(
        get_channel=lambda _id: channel, http=http, fetch_channel=AsyncMock(),
    )
    return adapter, channel, http


def _message_payloads(http):
    payloads = []
    for call in http.send_message.await_args_list:
        params = call.kwargs["params"]
        payloads.append(params.payload if params.payload is not None else json.loads(
            next(part["value"] for part in params.multipart if part["name"] == "payload_json"),
        ))
    return payloads


@pytest.mark.asyncio
@pytest.mark.parametrize("parent_type,thread,starter,mode,transport,keep_reference", [
    pytest.param(parent_type, True, True, mode, transport, False,
                 id=f"{transport}-{mode}-{'text' if parent_type == 0 else 'announcement'}")
    for parent_type in (0, 5)
    for mode in ("first", "all")
    for transport in ("text", "voice", "voice-file")
] + [
    pytest.param(0, True, False, "first", "text", True, id="follow-up"),
    pytest.param(5, True, False, "all", "text", True, id="announcement-follow-up"),
    pytest.param(15, True, True, "first", "text", True, id="forum"),
    pytest.param(16, True, True, "first", "text", True, id="media"),
    pytest.param(None, True, True, "first", "text", True, id="uncached-parent"),
    pytest.param(0, False, True, "first", "text", True, id="channel"),
    pytest.param(None, False, True, "first", "text", True, id="dm"),
    pytest.param(15, True, True, "off", "text", False, id="off"),
])
async def test_reply_payloads_preserve_valid_targets_and_delivery_bookkeeping(
    parent_type, thread, starter, mode, transport, keep_reference, tmp_path, monkeypatch,
):
    monkeypatch.setenv("DISCORD_MISSED_MESSAGE_BACKFILL", "true")
    adapter, channel, http = _adapter_channel(parent_type, thread=thread, mode=mode)
    reply_to = str(channel.id if starter else channel.id + 1)
    if transport == "text":
        result = await adapter.send(
            str(channel.id), "x" * (adapter.MAX_MESSAGE_LENGTH + 1), reply_to=reply_to,
            metadata={"notify": True},
        )
        assert result.success, result.error
        payloads = _message_payloads(http)
        assert len(payloads) == 2
        row = adapter._with_discord_recovery_db(lambda conn: conn.execute(
            "SELECT status, replied, response_message_id FROM discord_messages WHERE message_id=?",
            (reply_to,),
        ).fetchone())
        assert tuple(row) == ("responded", 1, result.message_id)
    else:
        audio = tmp_path / "voice.ogg"
        audio.write_bytes(b"OggS" + bytes(64))
        if transport == "voice-file":
            http.request.side_effect = RuntimeError("native voice unavailable")
        result = await adapter.send_voice(str(channel.id), str(audio), reply_to=reply_to)
        assert result.success, result.error
        form = http.request.await_args.kwargs["form"]
        payloads = [json.loads(next(part["value"] for part in form if part["name"] == "payload_json"))]
        payloads.extend(_message_payloads(http))
        assert len(payloads) == (2 if transport == "voice-file" else 1)
    for index, payload in enumerate(payloads):
        if keep_reference and (transport != "text" or mode == "all" or index == 0):
            assert int(payload["message_reference"]["message_id"]) == int(reply_to)
            assert payload["message_reference"].get("channel_id", channel.id) == channel.id
        else:
            assert "message_reference" not in payload
    adapter._client.fetch_channel.assert_not_awaited()
