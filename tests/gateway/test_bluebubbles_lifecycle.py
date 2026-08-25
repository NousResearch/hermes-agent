"""Focused cancellation and retry contracts for BlueBubbles revisions."""

import asyncio

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType
from gateway.platforms.bluebubbles import BlueBubblesAdapter


def _adapter(monkeypatch, **extra):
    monkeypatch.setenv("BLUEBUBBLES_SERVER_URL", "http://localhost:1234")
    monkeypatch.setenv("BLUEBUBBLES_PASSWORD", "secret")
    return BlueBubblesAdapter(
        PlatformConfig(
            enabled=True,
            extra={
                "server_url": "http://localhost:1234",
                "password": "secret",
                "message_revision_wait_seconds": 0,
                "message_retry_max_attempts": 3,
                "message_retry_base_delay_seconds": 0,
                **extra,
            },
        )
    )


def _event(adapter, text="revision"):
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=adapter.build_source(
            chat_id="iMessage;-;user@example.com",
            chat_name="user@example.com",
            chat_type="dm",
            user_id="user@example.com",
        ),
        message_id="stable-guid",
    )


@pytest.mark.asyncio
async def test_superseding_revision_cancels_active_attempt_before_commit(monkeypatch):
    adapter = _adapter(monkeypatch)
    key = ("iMessage;-;user@example.com", "user@example.com", "stable-guid")
    started = asyncio.Event()
    cancelled = asyncio.Event()
    release = asyncio.Event()
    handled = []

    async def handle(event):
        handled.append(event.text)
        if event.text == "old":
            started.set()
            try:
                await release.wait()
            except asyncio.CancelledError:
                cancelled.set()
                raise

    monkeypatch.setattr(adapter, "handle_message", handle)
    adapter._message_revision_serials[key] = 1
    old = asyncio.create_task(
        adapter._handle_reserved_message(_event(adapter, "old"), "old-id", (), key, 1)
    )
    await started.wait()
    adapter._message_revision_serials[key] = 2
    await asyncio.wait_for(
        adapter._handle_reserved_message(
            _event(adapter, "latest"), "latest-id", (), key, 2
        ),
        timeout=0.2,
    )
    await asyncio.wait_for(old, timeout=0.2)

    assert cancelled.is_set()
    assert handled == ["old", "latest"]
    assert "old-id" not in adapter._seen_message_guids
    assert "latest-id" in adapter._seen_message_guids


@pytest.mark.asyncio
async def test_latest_revision_is_not_cancelled_by_stale_cleanup(monkeypatch):
    adapter = _adapter(monkeypatch)
    key = ("iMessage;-;user@example.com", "user@example.com", "stable-guid")
    latest_started = asyncio.Event()
    latest_release = asyncio.Event()
    latest_cancelled = False

    async def handle(event):
        nonlocal latest_cancelled
        if event.text == "latest":
            latest_started.set()
            try:
                await latest_release.wait()
            except asyncio.CancelledError:
                latest_cancelled = True
                raise
        else:
            await asyncio.Event().wait()

    monkeypatch.setattr(adapter, "handle_message", handle)
    adapter._message_revision_serials[key] = 1
    stale = asyncio.create_task(
        adapter._handle_reserved_message(_event(adapter, "stale"), "stale-id", (), key, 1)
    )
    await asyncio.sleep(0)
    adapter._message_revision_serials[key] = 2
    latest = asyncio.create_task(
        adapter._handle_reserved_message(_event(adapter, "latest"), "latest-id", (), key, 2)
    )
    await asyncio.wait_for(latest_started.wait(), timeout=0.2)
    await asyncio.wait_for(stale, timeout=0.2)
    latest_release.set()
    await latest

    assert latest_cancelled is False
    assert "latest-id" in adapter._seen_message_guids


@pytest.mark.asyncio
async def test_transient_failure_recovers_with_same_identity(monkeypatch):
    adapter = _adapter(monkeypatch)
    key = ("iMessage;-;user@example.com", "user@example.com", "stable-guid")
    adapter._message_revision_serials[key] = 1
    attempts = []

    async def handle(event):
        attempts.append(event.message_id)
        if len(attempts) < 3:
            raise ConnectionError("connection reset")

    monkeypatch.setattr(adapter, "handle_message", handle)
    await adapter._handle_reserved_message(_event(adapter), "stable-id", (), key, 1)

    assert attempts == ["stable-guid", "stable-guid", "stable-guid"], (
        "BB-RM-FL-001: transient retries must preserve the inbound identity"
    )
    assert "stable-id" in adapter._seen_message_guids


@pytest.mark.asyncio
async def test_retry_exhaustion_is_terminal_and_duplicate_delivery_is_idempotent(monkeypatch):
    adapter = _adapter(monkeypatch)
    key = ("iMessage;-;user@example.com", "user@example.com", "stable-guid")
    adapter._message_revision_serials[key] = 1
    attempts = 0

    async def handle(_event):
        nonlocal attempts
        attempts += 1
        raise ConnectionError("connection reset")

    monkeypatch.setattr(adapter, "handle_message", handle)
    await adapter._handle_reserved_message(_event(adapter), "stable-id", (), key, 1)
    await adapter._handle_reserved_message(_event(adapter), "stable-id", (), key, 1)

    assert attempts == 3, "BB-RM-FL-002: exhausted retries must remain bounded"
    assert "stable-id" in adapter._terminal_message_identities
    assert "stable-id" not in adapter._seen_message_guids


@pytest.mark.asyncio
async def test_permanent_failure_is_not_retried(monkeypatch):
    adapter = _adapter(monkeypatch)
    key = ("iMessage;-;user@example.com", "user@example.com", "stable-guid")
    adapter._message_revision_serials[key] = 1
    attempts = 0

    async def handle(_event):
        nonlocal attempts
        attempts += 1
        raise ValueError("invalid payload")

    monkeypatch.setattr(adapter, "handle_message", handle)
    await adapter._handle_reserved_message(_event(adapter), "stable-id", (), key, 1)

    assert attempts == 1, "BB-RM-FL-002: permanent failures must not retry"
    assert "stable-id" in adapter._terminal_message_identities
    assert "stable-id" not in adapter._seen_message_guids


@pytest.mark.asyncio
async def test_duplicate_retry_delivery_joins_active_attempt(monkeypatch):
    adapter = _adapter(monkeypatch)
    key = ("iMessage;-;user@example.com", "user@example.com", "stable-guid")
    adapter._message_revision_serials[key] = 1
    started = asyncio.Event()
    release = asyncio.Event()
    attempts = 0

    async def handle(_event):
        nonlocal attempts
        attempts += 1
        started.set()
        await release.wait()

    monkeypatch.setattr(adapter, "handle_message", handle)
    first = asyncio.create_task(
        adapter._handle_reserved_message(_event(adapter), "stable-id", (), key, 1)
    )
    await started.wait()
    duplicate = asyncio.create_task(
        adapter._handle_reserved_message(_event(adapter), "stable-id", (), key, 1)
    )
    await asyncio.sleep(0)
    release.set()
    await asyncio.wait_for(asyncio.gather(first, duplicate), timeout=0.2)

    assert attempts == 1
    assert "stable-id" in adapter._seen_message_guids


@pytest.mark.asyncio
async def test_superseding_revision_cancels_obsolete_retry_backoff(monkeypatch):
    adapter = _adapter(monkeypatch, message_retry_base_delay_seconds=1)
    key = ("iMessage;-;user@example.com", "user@example.com", "stable-guid")
    adapter._message_revision_serials[key] = 1
    failed_once = asyncio.Event()
    attempts = []

    async def handle(event):
        attempts.append(event.text)
        if event.text == "old":
            failed_once.set()
            raise ConnectionError("connection reset")

    monkeypatch.setattr(adapter, "handle_message", handle)
    old = asyncio.create_task(
        adapter._handle_reserved_message(_event(adapter, "old"), "old-id", (), key, 1)
    )
    await failed_once.wait()
    await asyncio.sleep(0)
    adapter._message_revision_serials[key] = 2
    await asyncio.wait_for(
        adapter._handle_reserved_message(
            _event(adapter, "latest"), "latest-id", (), key, 2
        ),
        timeout=0.2,
    )
    await asyncio.wait_for(old, timeout=0.2)

    assert attempts == ["old", "latest"]
    assert "old-id" not in adapter._seen_message_guids
    assert "latest-id" in adapter._seen_message_guids
