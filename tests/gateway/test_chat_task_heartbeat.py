import asyncio
from unittest.mock import AsyncMock

import pytest

import gateway.run as gateway_run
from gateway.config import Platform
from gateway.session import SessionSource


class _FakeResponse:
    def __init__(self, status_code: int, payload: dict):
        self.status_code = status_code
        self._payload = payload
        self.text = gateway_run.json.dumps(payload)

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"http {self.status_code}")

    def json(self):
        return self._payload


class _FakeAsyncClient:
    calls = []

    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def request(self, method, url, headers=None, json=None):
        self.calls.append({
            "method": method,
            "url": url,
            "headers": headers,
            "json": json,
        })
        if method == "POST":
            return _FakeResponse(201, {"success": True, "task": {"id": "task-123"}})
        return _FakeResponse(200, {"success": True, "task": {"id": "task-123"}})


@pytest.fixture
def fake_httpx(monkeypatch):
    _FakeAsyncClient.calls = []
    monkeypatch.setattr(gateway_run.httpx, "AsyncClient", _FakeAsyncClient)
    monkeypatch.setenv("OS_AGENT_API_KEY", "test-key")
    return _FakeAsyncClient.calls


def _source() -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="chat-1",
        chat_type="dm",
        user_id="user-1",
    )


@pytest.mark.asyncio
async def test_chat_heartbeat_skips_short_replies(fake_httpx):
    heartbeat = gateway_run._GatewayChatTaskHeartbeat(
        user_message="short reply test",
        source=_source(),
        threshold_seconds=0.05,
        patch_interval_seconds=0.05,
    )
    await heartbeat.start()
    await asyncio.sleep(0.01)
    await heartbeat.finish(cancelled=False)

    assert fake_httpx == []


@pytest.mark.asyncio
async def test_chat_heartbeat_posts_then_marks_done(fake_httpx):
    heartbeat = gateway_run._GatewayChatTaskHeartbeat(
        user_message="x" * 120,
        source=_source(),
        threshold_seconds=0.01,
        patch_interval_seconds=0.05,
    )
    await heartbeat.start()
    await asyncio.sleep(0.02)
    await heartbeat.finish(cancelled=False)

    assert [call["method"] for call in fake_httpx] == ["POST", "PATCH"]
    assert fake_httpx[0]["json"]["title"] == ("x" * 80) + "…"
    assert fake_httpx[0]["json"]["metadata"] == {"kind": "chat"}
    assert fake_httpx[1]["json"] == {"status": "done", "working": False}


@pytest.mark.asyncio
async def test_chat_heartbeat_run_loop_emits_periodic_patch(monkeypatch):
    heartbeat = gateway_run._GatewayChatTaskHeartbeat(
        user_message="long reply test",
        source=_source(),
        threshold_seconds=0.0,
        patch_interval_seconds=0.001,
    )
    heartbeat._create_task = AsyncMock(return_value=True)
    heartbeat._patch_task = AsyncMock(return_value=True)

    wait_calls = {"count": 0}

    async def fake_wait_for(awaitable, timeout):
        wait_calls["count"] += 1
        if wait_calls["count"] == 1:
            if hasattr(awaitable, "close"):
                awaitable.close()
            raise asyncio.TimeoutError()
        if hasattr(awaitable, "close"):
            awaitable.close()
        heartbeat._stop_event.set()
        return True

    monkeypatch.setattr(gateway_run.asyncio, "wait_for", fake_wait_for)

    await heartbeat._run()

    heartbeat._patch_task.assert_awaited_once_with({"working": True}, reason="heartbeat")
