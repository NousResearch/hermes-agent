"""Native text batches defer Matrix checkpoints until durable admission."""

import asyncio
from unittest.mock import AsyncMock

import pytest

from tests.gateway.test_matrix_sync_resume import batch, gateway_intake, make_adapter, message, transport


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ["mautrix"], indirect=True)
@pytest.mark.parametrize("msgtype", ["m.text", "m.emote"])
async def test_native_batch_keeps_cursor_before_unadmitted_event(tmp_path, monkeypatch, transport, msgtype):
    from plugins.platforms.matrix.sync_transport import DurableSyncStore

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _, responses = transport
    adapter = make_adapter()
    responses.append(batch("s1"))
    assert await adapter.connect()
    gateway_intake(adapter, {})
    turns = []

    async def model(event, _key):
        turns.append(event.text)

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
    event = message("$native")
    event["content"]["msgtype"] = msgtype
    try:
        await asyncio.wait_for(adapter._absorb_sync(client, batch("s2", event)), timeout=5)
        await asyncio.wait_for(quiet.wait(), timeout=5)
        await durable.load()
        buffered = (list(turns), await durable.get_next_batch())
        release.set()
        await adapter._sync_checkpoints.settled()
        await asyncio.gather(*adapter._background_tasks)
        await durable.load()
        assert (buffered, turns, await durable.get_next_batch()) == (
            ([], "s1"), ["$native"], "s2",
        )
    finally:
        release.set()
        await adapter.disconnect()
