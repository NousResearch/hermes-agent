"""Focused lifecycle tests for BlueBubbles read receipts."""

import asyncio

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.bluebubbles import BlueBubblesAdapter


def _adapter(**extra):
    return BlueBubblesAdapter(
        PlatformConfig(
            enabled=True,
            extra={
                "server_url": "http://localhost:1234",
                "password": "secret",
                **extra,
            },
        )
    )


@pytest.mark.asyncio
async def test_queued_receipt_survives_helper_cold_boot(monkeypatch):
    adapter = _adapter()
    delivered = asyncio.Event()
    posts = []
    refreshes = 0

    class FakeClient:
        async def post(self, url, timeout):
            posts.append((url, timeout))
            delivered.set()

    async def refresh_helper():
        nonlocal refreshes
        refreshes += 1
        return refreshes > 1

    async def resolve_chat(chat_id):
        return chat_id

    adapter.client = FakeClient()
    monkeypatch.setattr(adapter, "_refresh_helper_state", refresh_helper)
    monkeypatch.setattr(adapter, "_resolve_chat_guid", resolve_chat)
    monkeypatch.setattr(
        "gateway.platforms.bluebubbles._READ_RECEIPT_RETRY_SECONDS", 0
    )

    assert await adapter._queue_read_receipt(
        "iMessage;-;user@example.com",
        "message-1",
        is_group=False,
        admitted=True,
    )
    await asyncio.wait_for(delivered.wait(), timeout=0.2)

    assert refreshes >= 2
    assert len(posts) == 1
    await adapter.cancel_background_tasks()


@pytest.mark.asyncio
async def test_group_receipt_requires_policy_admission(monkeypatch):
    adapter = _adapter(require_mention=True)
    posts = []

    class FakeClient:
        async def post(self, url, timeout):
            posts.append((url, timeout))

    async def helper_ready():
        return True

    async def resolve_chat(chat_id):
        return chat_id

    adapter.client = FakeClient()
    monkeypatch.setattr(adapter, "_refresh_helper_state", helper_ready)
    monkeypatch.setattr(adapter, "_resolve_chat_guid", resolve_chat)

    assert not await adapter._queue_read_receipt(
        "iMessage;+;family-group",
        "message-1",
        is_group=True,
        admitted=False,
    )
    assert not await adapter.mark_read("iMessage;+;family-group")
    await asyncio.sleep(0)

    assert posts == []
    assert adapter._pending_read_receipts == {}
    assert adapter._read_receipt_tasks == {}


@pytest.mark.asyncio
async def test_admitted_group_receipt_is_delivered(monkeypatch):
    adapter = _adapter(require_mention=True)
    delivered = asyncio.Event()
    posts = []

    class FakeClient:
        async def post(self, url, timeout):
            posts.append((url, timeout))
            delivered.set()

    async def helper_ready():
        return True

    async def resolve_chat(chat_id):
        return chat_id

    adapter.client = FakeClient()
    monkeypatch.setattr(adapter, "_refresh_helper_state", helper_ready)
    monkeypatch.setattr(adapter, "_resolve_chat_guid", resolve_chat)

    assert await adapter._queue_read_receipt(
        "iMessage;+;family-group",
        "message-1",
        is_group=True,
        admitted=True,
    )
    await asyncio.wait_for(delivered.wait(), timeout=0.2)

    assert len(posts) == 1
    await adapter.cancel_background_tasks()


@pytest.mark.asyncio
async def test_duplicate_receipt_lifecycle_events_send_once(monkeypatch):
    adapter = _adapter()
    delivered = asyncio.Event()
    posts = []

    class FakeClient:
        async def post(self, url, timeout):
            posts.append((url, timeout))
            delivered.set()

    async def helper_ready():
        return True

    async def resolve_chat(chat_id):
        return chat_id

    adapter.client = FakeClient()
    monkeypatch.setattr(adapter, "_refresh_helper_state", helper_ready)
    monkeypatch.setattr(adapter, "_resolve_chat_guid", resolve_chat)

    first, duplicate = await asyncio.gather(
        adapter._queue_read_receipt(
            "iMessage;-;user@example.com",
            "message-1",
            is_group=False,
            admitted=True,
        ),
        adapter._queue_read_receipt(
            "iMessage;-;user@example.com",
            "message-1",
            is_group=False,
            admitted=True,
        ),
    )
    await asyncio.wait_for(delivered.wait(), timeout=0.2)
    assert first is True
    assert duplicate is False

    assert not await adapter._queue_read_receipt(
        "iMessage;-;user@example.com",
        "message-1",
        is_group=False,
        admitted=True,
    )
    await asyncio.sleep(0)

    assert len(posts) == 1
    await adapter.cancel_background_tasks()


@pytest.mark.asyncio
async def test_receipt_cancellation_prevents_later_send(monkeypatch):
    adapter = _adapter()
    refresh_started = asyncio.Event()
    release_refresh = asyncio.Event()
    posts = []

    class FakeClient:
        async def post(self, url, timeout):
            posts.append((url, timeout))

    async def blocked_refresh():
        refresh_started.set()
        await release_refresh.wait()
        return True

    async def resolve_chat(chat_id):
        return chat_id

    adapter.client = FakeClient()
    monkeypatch.setattr(adapter, "_refresh_helper_state", blocked_refresh)
    monkeypatch.setattr(adapter, "_resolve_chat_guid", resolve_chat)

    assert await adapter._queue_read_receipt(
        "iMessage;-;user@example.com",
        "message-1",
        is_group=False,
        admitted=True,
    )
    await refresh_started.wait()
    await adapter.cancel_background_tasks()
    release_refresh.set()
    await asyncio.sleep(0)

    assert posts == []
    assert adapter._pending_read_receipts == {}
    assert adapter._read_receipt_tasks == {}


@pytest.mark.asyncio
async def test_receipt_cancellation_during_delivery_blocks_completion_callback(
    monkeypatch,
):
    adapter = _adapter()
    delivery_started = asyncio.Event()
    delivery_cancelled = asyncio.Event()
    completed = []

    class FakeClient:
        async def post(self, url, timeout):
            delivery_started.set()
            try:
                await asyncio.Event().wait()
            finally:
                delivery_cancelled.set()
            completed.append((url, timeout))

    async def helper_ready():
        return True

    async def resolve_chat(chat_id):
        return chat_id

    adapter.client = FakeClient()
    monkeypatch.setattr(adapter, "_refresh_helper_state", helper_ready)
    monkeypatch.setattr(adapter, "_resolve_chat_guid", resolve_chat)

    assert await adapter._queue_read_receipt(
        "iMessage;-;user@example.com",
        "message-1",
        is_group=False,
        admitted=True,
    )
    await delivery_started.wait()

    await adapter.cancel_background_tasks()

    assert delivery_cancelled.is_set()
    assert completed == []
    assert adapter._pending_read_receipts == {}
    assert adapter._read_receipt_tasks == {}
    assert adapter._sent_read_receipts == {}
    assert not await adapter._queue_read_receipt(
        "iMessage;-;user@example.com",
        "message-2",
        is_group=False,
        admitted=True,
    )


@pytest.mark.asyncio
async def test_repeated_cleanup_cancels_pending_receipt_retry(monkeypatch):
    adapter = _adapter()
    helper_checked = asyncio.Event()
    posts = []

    class FakeClient:
        async def post(self, url, timeout):
            posts.append((url, timeout))

    async def helper_unavailable():
        helper_checked.set()
        return False

    adapter.client = FakeClient()
    monkeypatch.setattr(adapter, "_refresh_helper_state", helper_unavailable)
    monkeypatch.setattr(
        "gateway.platforms.bluebubbles._READ_RECEIPT_RETRY_SECONDS", 60
    )

    assert await adapter._queue_read_receipt(
        "iMessage;-;user@example.com",
        "message-1",
        is_group=False,
        admitted=True,
    )
    await helper_checked.wait()
    worker = adapter._read_receipt_tasks["iMessage;-;user@example.com"]

    await adapter.cancel_background_tasks()
    await adapter.cancel_background_tasks()

    assert worker.done()
    assert posts == []
    assert adapter._pending_read_receipts == {}
    assert adapter._read_receipt_tasks == {}
    assert adapter._sent_read_receipts == {}
