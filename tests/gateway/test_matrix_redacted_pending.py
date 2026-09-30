"""A Matrix message that its sender redacts before its turn starts never reaches the agent."""

import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig

ROOM = "!room1:example.org"
ALICE = "@alice:example.org"


def _make_adapter():
    from plugins.platforms.matrix.adapter import MatrixAdapter

    adapter = MatrixAdapter(PlatformConfig(enabled=True, token="syt_test_token", extra={
        "homeserver": "https://matrix.example.org", "user_id": "@hermes:example.org"}))
    adapter.handle_message = AsyncMock()
    adapter._startup_ts = time.time() - 10
    adapter._background_read_receipt = MagicMock()
    return adapter


def _message(body, event_id, **content):
    return SimpleNamespace(sender=ALICE, event_id=event_id, room_id=ROOM, timestamp=int(time.time() * 1000),
                           content={"body": body, "msgtype": "m.text", **content})


def _redaction(sender, target, *, in_content=False):
    """``in_content``: room version 11 moved ``redacts`` from the event into its content."""
    if in_content:
        return SimpleNamespace(room_id=ROOM, sender=sender, event_id="$redaction", content={"redacts": target})
    return SimpleNamespace(room_id=ROOM, sender=sender, event_id="$redaction", redacts=target, content={})


def _dispatched(adapter):
    return [(call.args[0].message_id, call.args[0].text) for call in adapter.handle_message.await_args_list]


@pytest.mark.asyncio
@pytest.mark.parametrize("redacted_by, in_content, expected", [
    (ALICE, False, [("$one", "first")]),
    (ALICE, True, [("$one", "first")]),
    ("@mallory:example.org", False, [("$one", "first\nsecond")]),
])
async def test_sender_redaction_withdraws_a_batched_message(monkeypatch, redacted_by, in_content, expected):
    monkeypatch.setenv("MATRIX_REQUIRE_MENTION", "false")
    monkeypatch.setenv("MATRIX_AUTO_THREAD", "false")
    adapter = _make_adapter()
    adapter._text_batch_delay_seconds = 60
    await adapter._on_room_message(_message("first", "$one"))
    await adapter._on_room_message(_message("second", "$two"))

    await adapter._on_redaction(_redaction(redacted_by, "$two", in_content=in_content))
    for key in list(adapter._pending_text_batches):
        await adapter._flush_text_batch_now(key)

    assert _dispatched(adapter) == expected


@pytest.mark.asyncio
async def test_redacted_parked_voice_is_not_claimed_by_a_later_mention(monkeypatch):
    """Under require_mention an unmentioned voice waits for its sender's bare @mention. If the
    sender redacts the voice first, the mention is answered on its own."""
    monkeypatch.delenv("MATRIX_REQUIRE_MENTION", raising=False)
    monkeypatch.setenv("MATRIX_AUTO_THREAD", "false")
    adapter = _make_adapter()
    adapter._text_batch_delay_seconds = 0
    adapter._download_and_cache_media = AsyncMock(return_value="/tmp/voice.ogg")
    await adapter._on_room_message(_message(
        "voice message", "$voice", msgtype="m.audio", url="mxc://example.org/v",
        info={"mimetype": "audio/ogg"}, **{"org.matrix.msc3245.voice": {}, "m.mentions": {}}))

    await adapter._on_redaction(_redaction(ALICE, "$voice"))
    await adapter._on_room_message(_message(
        "@hermes:example.org", "$mention", **{"m.mentions": {"user_ids": ["@hermes:example.org"]}}))

    assert _dispatched(adapter) == [("$mention", "")]
