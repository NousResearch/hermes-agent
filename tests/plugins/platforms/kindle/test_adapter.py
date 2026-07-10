from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("aiohttp")
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from plugins.platforms.kindle.adapter import KindleAdapter


def _adapter(monkeypatch: pytest.MonkeyPatch, *, timeout: float = 1.0) -> KindleAdapter:
    monkeypatch.setenv("KINDLE_INGEST_TOKEN", "test-token")
    monkeypatch.setenv("KINDLE_REPLY_TIMEOUT", str(timeout))
    monkeypatch.delenv("KINDLE_INSECURE", raising=False)
    return KindleAdapter(PlatformConfig(enabled=True, token="", extra={}))


def _client(adapter: KindleAdapter) -> TestClient:
    app = web.Application()
    app.router.add_post("/ingest", adapter._handle_ingest)
    return TestClient(TestServer(app))


async def _wait_for_pending(adapter: KindleAdapter, chat_id: str) -> None:
    for _ in range(100):
        if chat_id in adapter._pending:
            return
        await asyncio.sleep(0)
    raise AssertionError(f"waiter for {chat_id!r} was not registered")


def _payload(chat_id: str = "scribe-1", text: str = "hello") -> dict[str, str]:
    return {"chat_id": chat_id, "user": "jeff", "text": text}


@pytest.mark.asyncio
async def test_ingest_rejects_invalid_token(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _adapter(monkeypatch)
    async with _client(adapter) as client:
        response = await client.post("/ingest", json=_payload())
        body = await response.json()

    assert response.status == 401
    assert body == {"error": "unauthorized"}
    assert adapter._pending == {}


@pytest.mark.asyncio
async def test_final_notify_delivers_reply_not_preview(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _adapter(monkeypatch)

    async def accept(_event) -> None:
        return None

    monkeypatch.setattr(adapter, "handle_message", accept)
    async with _client(adapter) as client:
        request = asyncio.create_task(
            client.post("/ingest", json=_payload(), headers={"X-Kindle-Token": "test-token"})
        )
        await _wait_for_pending(adapter, "scribe-1")

        preview = await adapter.send("scribe-1", "working", metadata={"notify": False})
        await asyncio.sleep(0)
        assert preview.success is True
        assert preview.message_id is None
        assert request.done() is False

        delivered = await adapter.send("scribe-1", "finished", metadata={"notify": True})
        response = await request
        body = await response.json()

    assert delivered.success is True
    assert response.status == 200
    assert body == {"reply": "finished"}
    assert adapter._pending == {}


@pytest.mark.asyncio
async def test_timeout_removes_waiter(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _adapter(monkeypatch, timeout=0.01)

    async def accept(_event) -> None:
        return None

    monkeypatch.setattr(adapter, "handle_message", accept)
    async with _client(adapter) as client:
        response = await client.post(
            "/ingest", json=_payload(), headers={"X-Kindle-Token": "test-token"}
        )
        body = await response.json()

    assert response.status == 504
    assert body == {"error": "agent timed out"}
    assert adapter._pending == {}


@pytest.mark.asyncio
async def test_disconnect_cancels_request_and_removes_waiter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _adapter(monkeypatch)

    async def accept(_event) -> None:
        return None

    monkeypatch.setattr(adapter, "handle_message", accept)
    async with _client(adapter) as client:
        request = asyncio.create_task(
            client.post("/ingest", json=_payload(), headers={"X-Kindle-Token": "test-token"})
        )
        await _wait_for_pending(adapter, "scribe-1")
        await adapter.disconnect()
        response = await request
        body = await response.json()

    assert response.status == 503
    assert body == {"error": "cancelled"}
    assert adapter._pending == {}


@pytest.mark.asyncio
async def test_overlapping_same_chat_request_is_rejected_without_stealing_waiter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _adapter(monkeypatch)
    dispatched = []

    async def accept(event) -> None:
        dispatched.append(event.text)

    monkeypatch.setattr(adapter, "handle_message", accept)
    async with _client(adapter) as client:
        first = asyncio.create_task(
            client.post(
                "/ingest",
                json=_payload(text="turn A"),
                headers={"X-Kindle-Token": "test-token"},
            )
        )
        await _wait_for_pending(adapter, "scribe-1")

        second = await client.post(
            "/ingest",
            json=_payload(text="turn B"),
            headers={"X-Kindle-Token": "test-token"},
        )
        assert second.status == 409
        assert await second.json() == {"error": "a request for this chat is already in progress"}

        sent = await adapter.send("scribe-1", "reply A", metadata={"notify": True})
        first_response = await first
        first_body = await first_response.json()

    assert sent.success is True
    assert first_response.status == 200
    assert first_body == {"reply": "reply A"}
    assert dispatched == ["turn A"]
    assert adapter._pending == {}
