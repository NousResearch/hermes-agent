"""Matrix ownership survives failed cleanup and drains SDK background work."""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from tests.gateway.test_matrix import _make_adapter
from tests.waiting import async_wait_until


def owner(tmp_path, name):
    adapter = _make_adapter()
    adapter._store_dir = tmp_path / name
    adapter._store_dir.mkdir()
    adapter._client = SimpleNamespace(
        mxid="@bot:example.org", device_id="RESOLVED", api=SimpleNamespace(session=SimpleNamespace(close=AsyncMock())))
    return adapter


@pytest.mark.platforms("posix")
@pytest.mark.asyncio
@pytest.mark.parametrize("failed_resource", ["database", "http"])
async def test_failed_cleanup_keeps_identity_and_store_claims(tmp_path, failed_resource):
    first, contender = owner(tmp_path, "first"), owner(tmp_path, "other")
    first._crypto_db = SimpleNamespace(stop=AsyncMock())
    stop, close = first._crypto_db.stop, first._client.api.session.close
    failing = stop if failed_resource == "database" else close
    failing.side_effect = RuntimeError("synthetic close failure")
    assert first._acquire_matrix_store_lock()
    try:
        with pytest.raises(ExceptionGroup, match="retaining crypto ownership"):
            await first.disconnect()
        stop.assert_awaited_once()
        close.assert_awaited_once()
        assert not contender._acquire_matrix_store_lock()
        await contender.disconnect()
        # A refused contender's cleanup cannot unlock the real owner's device.
        contender._client = SimpleNamespace(mxid="@bot:example.org", device_id="RESOLVED")
        assert not contender._acquire_matrix_store_lock()
        failing.side_effect = None
        await first.disconnect()
        assert contender._acquire_matrix_store_lock()
    finally:
        first._release_matrix_store_lock()
        contender._release_matrix_store_lock()


@pytest.mark.platforms("posix")
@pytest.mark.asyncio
async def test_disconnect_waits_for_sdk_callback_send_and_unwedge(tmp_path):
    from mautrix.client import Client
    from plugins.platforms.matrix.adapter_lifecycle import track_matrix_activity
    from mautrix.api import HTTPAPI
    from mautrix.types import EventType

    adapter = owner(tmp_path, "first")
    session = adapter._client.api.session
    client = Client(mxid="@bot:example.org", device_id="RESOLVED", api=HTTPAPI(base_url="https://matrix.example.org", token="test", client_session=session))
    client.api.session = session
    adapter._client = client
    adapter._track_matrix_client(client)
    adapter._crypto_db = SimpleNamespace(stop=AsyncMock())
    stop = adapter._crypto_db.stop
    entered, unwedge_entered, send_entered, release = (asyncio.Event() for _ in range(4))
    finished = asyncio.Event()
    children = []

    async def unwedge():
        unwedge_entered.set()
        await release.wait()

    async def handler(event):
        entered.set()
        # The pinned SDK spawns _unwedge_session via background_task.create.
        from mautrix.util import background_task
        children.append(background_task.create(track_matrix_activity(adapter, unwedge)()))
        await release.wait()
        finished.set()

    async def send(*args):
        send_entered.set()
        await release.wait()
        return "$event"

    client.send_message_event = send
    client.add_event_handler(EventType.ROOM_MESSAGE, handler)  # wait_sync=False
    assert adapter._acquire_matrix_store_lock()
    sending = closing = None
    try:
        assert client.dispatch_manual_event(EventType.ROOM_MESSAGE, object()) == []
        await asyncio.wait_for(entered.wait(), 3)
        await asyncio.wait_for(unwedge_entered.wait(), 3)
        sending = asyncio.create_task(adapter._send_room_message("!room:example.org", {}))
        await asyncio.wait_for(send_entered.wait(), 3)
        closing = asyncio.create_task(adapter.disconnect())
        assert await async_wait_until(lambda: adapter._closing, timeout=3)
        assert not closing.done()
        stop.assert_not_awaited()
        session.close.assert_not_awaited()
        with pytest.raises(RuntimeError, match="closing"):
            await adapter._send_room_message("!room:example.org", {})
        release.set()
        await asyncio.wait_for(finished.wait(), 3)
        await asyncio.gather(sending, closing, *children)
        stop.assert_awaited_once()
        session.close.assert_awaited_once()
        assert adapter._matrix_store_lock_handle is None
    finally:
        release.set()
        await asyncio.gather(*(t for t in [sending, closing, *children] if t), return_exceptions=True)
        adapter._release_matrix_store_lock()


@pytest.mark.platforms("posix")
def test_device_lock_uses_resolved_identity_across_store_directories(tmp_path):
    first, second, different = (owner(tmp_path, name) for name in ("first", "second", "different"))
    second._homeserver = "https://alias.example.org"
    different._client.device_id = "ANOTHER_DEVICE"
    try:
        assert first._acquire_matrix_store_lock()
        assert not second._acquire_matrix_store_lock()
        assert different._acquire_matrix_store_lock()
        second._release_matrix_store_lock()
        assert not second._acquire_matrix_store_lock()
        first._release_matrix_store_lock()
        assert second._acquire_matrix_store_lock()
    finally:
        for adapter in (first, second, different):
            adapter._release_matrix_store_lock()


@pytest.mark.platforms("posix")
@pytest.mark.asyncio
async def test_failed_ephemeral_owner_outlives_its_caller(tmp_path):
    import gc
    import weakref

    adapter, contender = owner(tmp_path, "first"), owner(tmp_path, "contender")
    adapter._crypto_db = SimpleNamespace(stop=AsyncMock(side_effect=RuntimeError("close failed")))
    reference = weakref.ref(adapter)
    assert adapter._acquire_matrix_store_lock()
    with pytest.raises(ExceptionGroup):
        await adapter.disconnect()
    del adapter
    gc.collect()
    retained = reference()
    assert retained is not None
    try:
        assert not contender._acquire_matrix_store_lock()
        retained._crypto_db.stop.side_effect = None
        await retained.disconnect()
        assert contender._acquire_matrix_store_lock()
    finally:
        retained._release_matrix_store_lock()
        contender._release_matrix_store_lock()
