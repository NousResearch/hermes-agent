"""A Matrix message that its sender redacts before its turn starts never reaches the agent."""

import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
from tests.gateway.test_matrix_sync_resume import (
    batch,
    gateway_intake,
    make_adapter,
    message,
    transport,
)

ROOM = "!room1:example.org"
ALICE = "@alice:example.org"


def _make_adapter():
    from plugins.platforms.matrix.adapter import MatrixAdapter

    adapter = MatrixAdapter(
        PlatformConfig(
            enabled=True,
            token="syt_test_token",
            extra={
                "homeserver": "https://matrix.example.org",
                "user_id": "@hermes:example.org",
            },
        )
    )
    adapter.handle_message = AsyncMock()
    adapter._startup_ts = time.time() - 10
    adapter._background_read_receipt = MagicMock()
    return adapter


def _message(body, event_id, **content):
    return SimpleNamespace(
        sender=ALICE,
        event_id=event_id,
        room_id=ROOM,
        timestamp=int(time.time() * 1000),
        content={"body": body, "msgtype": "m.text", **content},
    )


def _redaction(sender, target, *, in_content=False):
    """``in_content``: room version 11 moved ``redacts`` from the event into its content."""
    if in_content:
        return SimpleNamespace(
            room_id=ROOM,
            sender=sender,
            event_id="$redaction",
            content={"redacts": target},
        )
    return SimpleNamespace(
        room_id=ROOM, sender=sender, event_id="$redaction", redacts=target, content={}
    )


def _dispatched(adapter):
    return [
        (call.args[0].message_id, call.args[0].text)
        for call in adapter.handle_message.await_args_list
    ]


async def _assert_batched_redaction(monkeypatch, redacted_by, in_content, expected):
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


async def _assert_parked_voice_redaction(monkeypatch):
    """Under require_mention an unmentioned voice waits for its sender's bare @mention. If the
    sender redacts the voice first, the mention is answered on its own."""
    monkeypatch.delenv("MATRIX_REQUIRE_MENTION", raising=False)
    monkeypatch.setenv("MATRIX_AUTO_THREAD", "false")
    adapter = _make_adapter()
    adapter._text_batch_delay_seconds = 0
    adapter._download_and_cache_media = AsyncMock(return_value="/tmp/voice.ogg")
    await adapter._on_room_message(
        _message(
            "voice message",
            "$voice",
            msgtype="m.audio",
            url="mxc://example.org/v",
            info={"mimetype": "audio/ogg"},
            **{"org.matrix.msc3245.voice": {}, "m.mentions": {}},
        )
    )

    await adapter._on_redaction(_redaction(ALICE, "$voice"))
    await adapter._on_room_message(
        _message(
            "@hermes:example.org",
            "$mention",
            **{"m.mentions": {"user_ids": ["@hermes:example.org"]}},
        )
    )

    assert _dispatched(adapter) == [("$mention", "")]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "check, args",
    [
        (_assert_batched_redaction, (ALICE, False, [("$one", "first")])),
        (_assert_batched_redaction, (ALICE, True, [("$one", "first")])),
        (
            _assert_batched_redaction,
            ("@mallory:example.org", False, [("$one", "first\nsecond")]),
        ),
        (_assert_parked_voice_redaction, ()),
    ],
)
async def test_redaction_preserves_only_the_authorised_remaining_input(
    monkeypatch, check, args
):
    await check(monkeypatch, *args)


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ["mautrix"], indirect=True)
@pytest.mark.parametrize("targets", [("$two",), ("$one", "$two")])
async def test_redacted_batch_preserves_native_intake_receipts(
    tmp_path, monkeypatch, transport, targets
):
    import asyncio

    from plugins.platforms.matrix.sync_transport import DurableSyncStore

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _, responses = transport
    adapter = make_adapter()
    responses.append(batch("s1"))
    assert await adapter.connect()
    gateway_intake(adapter, {})
    turns = []

    async def model(event, _key):
        turns.append((event.message_id, event.text))

    adapter.set_message_handler(AsyncMock())
    adapter._process_message_background = model
    quiet, release = asyncio.Event(), asyncio.Event()
    real_sleep = asyncio.sleep

    async def batch_clock(delay):
        if delay == adapter._text_batch_delay_seconds:
            quiet.set()
            await release.wait()
            return
        await real_sleep(delay)

    monkeypatch.setattr(asyncio, "sleep", batch_clock)
    client = adapter._client
    durable = DurableSyncStore(tmp_path, "unused", "unused", "unused", "unused")
    durable.path = client.sync_store.path
    try:
        await asyncio.wait_for(
            adapter._absorb_sync(client, batch("s2", message("$one"), message("$two"))),
            timeout=5,
        )
        await asyncio.wait_for(quiet.wait(), timeout=5)
        receipts = dict(adapter._buffered_intakes)
        for target in targets:
            await adapter._on_redaction(
                SimpleNamespace(
                    room_id="!room:example.org",
                    sender="@alice:example.org",
                    event_id="$redaction" + target,
                    redacts=target,
                    content={},
                )
            )
        assert tuple(
            (event_id, receipt.done()) for event_id, receipt in receipts.items()
        ) == (
            ("$one", "$one" in targets),
            ("$two", True),
        )
        release.set()
        await asyncio.wait_for(adapter._sync_checkpoints.settled(), timeout=5)
        await asyncio.gather(*adapter._background_tasks)
        await durable.load()
        assert (
            turns,
            tuple(
                (event_id, receipt.result()) for event_id, receipt in receipts.items()
            ),
            await durable.get_next_batch(),
            adapter._text_batch_intakes,
            adapter._buffered_intakes,
        ) == (
            [] if "$one" in targets else [("$one", "$one")],
            (("$one", True), ("$two", True)),
            "s2",
            {},
            {},
        )
    finally:
        release.set()
        await adapter.disconnect()
