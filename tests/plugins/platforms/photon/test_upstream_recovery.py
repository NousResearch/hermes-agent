"""Photon upstream failure recovery tests.

These tests assert the safety contract for iMessage delivery: upstream failures
mark the adapter unhealthy so the gateway reconnects/restarts the sidecar for
future messages, but the failing outbound operation is not retried.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.base import SendResult
from plugins.platforms.photon.adapter import PhotonAdapter
import plugins.platforms.photon.adapter as photon_adapter


class _FakeResponse:
    def __init__(self, status_code: int, payload: Dict[str, Any] | None = None, text: str = ""):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text

    def json(self) -> Dict[str, Any]:
        return self._payload

    async def aread(self) -> bytes:
        return self.text.encode("utf-8")


class _FakePostClient:
    def __init__(self, response: _FakeResponse, calls: List[Tuple[str, Dict[str, Any]]]):
        self._response = response
        self._calls = calls

    async def __aenter__(self) -> "_FakePostClient":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None

    async def post(self, url: str, json: Dict[str, Any], headers: Dict[str, str]) -> _FakeResponse:
        self._calls.append((url, json))
        return self._response


class _FakeInboundStream:
    def __init__(self, response: _FakeResponse):
        self._response = response

    async def __aenter__(self) -> _FakeResponse:
        return self._response

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None


class _FakeInboundClient:
    def __init__(self, response: _FakeResponse):
        self._response = response

    def stream(self, method: str, url: str, headers: Dict[str, str], timeout: Any = None) -> _FakeInboundStream:
        return _FakeInboundStream(self._response)


@pytest.fixture
def adapter(monkeypatch: pytest.MonkeyPatch) -> PhotonAdapter:
    monkeypatch.setenv("PHOTON_PROJECT_ID", "test-project-id")
    monkeypatch.setenv("PHOTON_PROJECT_SECRET", "test-project-secret")
    cfg = PlatformConfig(enabled=True, token="", extra={})
    instance = PhotonAdapter(cfg)
    instance._http_client = object()  # type: ignore[assignment]
    instance._inbound_running = True
    return instance


def _capture_fatal_notify(adapter: PhotonAdapter) -> List[Tuple[str | None, str | None]]:
    calls: List[Tuple[str | None, str | None]] = []

    async def _notify() -> None:
        calls.append((adapter.fatal_error_code, adapter.fatal_error_message))

    adapter._notify_fatal_error = _notify  # type: ignore[method-assign]
    return calls


@pytest.mark.asyncio
async def test_sidecar_call_upstream_failure_marks_unhealthy_once(
    adapter: PhotonAdapter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    notify_calls = _capture_fatal_notify(adapter)
    post_calls: List[Tuple[str, Dict[str, Any]]] = []
    response = _FakeResponse(
        200,
        {"ok": False, "error": "upstream connection dropped"},
    )
    monkeypatch.setattr(
        photon_adapter.httpx,
        "AsyncClient",
        lambda timeout=30.0: _FakePostClient(response, post_calls),
    )

    with pytest.raises(RuntimeError, match="upstream connection dropped"):
        await adapter._sidecar_call("/send", {"spaceId": "+15555550100", "text": "hi"})
    with pytest.raises(RuntimeError, match="upstream connection dropped"):
        await adapter._sidecar_call("/typing", {"spaceId": "+15555550100"})

    assert len(post_calls) == 2
    assert notify_calls == [("PHOTON_UPSTREAM_DROPPED", adapter.fatal_error_message)]
    assert "upstream connection" in (adapter.fatal_error_message or "")
    assert adapter.fatal_error_retryable is True


@pytest.mark.asyncio
async def test_sidecar_call_non_upstream_error_does_not_mark_unhealthy(
    adapter: PhotonAdapter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    notify_calls = _capture_fatal_notify(adapter)
    response = _FakeResponse(400, text="message not found")
    monkeypatch.setattr(
        photon_adapter.httpx,
        "AsyncClient",
        lambda timeout=30.0: _FakePostClient(response, []),
    )

    with pytest.raises(RuntimeError, match="message not found"):
        await adapter._sidecar_call("/react", {"spaceId": "+1"})

    assert notify_calls == []
    assert adapter.fatal_error_code is None


@pytest.mark.asyncio
async def test_send_with_retry_is_single_shot_and_not_retryable(
    adapter: PhotonAdapter,
) -> None:
    calls: List[Tuple[str, str]] = []

    async def _send(chat_id: str, content: str, reply_to=None, metadata=None) -> SendResult:
        calls.append((chat_id, content))
        return SendResult(success=False, error="Connection dropped", retryable=True)

    adapter.send = _send  # type: ignore[method-assign]

    result = await adapter._send_with_retry("+15555550100", "hello")

    assert calls == [("+15555550100", "hello")]
    assert result.success is False
    assert result.retryable is False


@pytest.mark.asyncio
async def test_sidecar_send_failure_does_not_retry(
    adapter: PhotonAdapter,
) -> None:
    calls: List[Tuple[str, Dict[str, Any]]] = []

    async def _boom(path: str, body: Dict[str, Any]) -> Dict[str, Any]:
        calls.append((path, body))
        raise RuntimeError("upstream connection dropped")

    adapter._sidecar_call = _boom  # type: ignore[method-assign]

    result = await adapter._sidecar_send("+15555550100", "hello")

    assert calls == [("/send", {"spaceId": "+15555550100", "text": "hello", "format": "markdown"})]
    assert result.success is False
    assert result.retryable is False


@pytest.mark.asyncio
async def test_inbound_upstream_failure_marks_unhealthy_and_stops_loop(
    adapter: PhotonAdapter,
) -> None:
    notify_calls = _capture_fatal_notify(adapter)
    adapter._http_client = _FakeInboundClient(
        _FakeResponse(503, text="[upstream] Connection dropped")
    )  # type: ignore[assignment]

    await adapter._inbound_loop()

    assert notify_calls == [("PHOTON_UPSTREAM_DROPPED", adapter.fatal_error_message)]
    assert adapter.fatal_error_retryable is True


def test_sidecar_classifies_upstream_failures() -> None:
    assert PhotonAdapter._is_upstream_failure("[upstream] Connection dropped")
    assert PhotonAdapter._is_upstream_failure("[spectrum.stream] stream interrupted; reconnecting")
    assert PhotonAdapter._is_upstream_failure("photon-sidecar: handler error: ConnectionError")
    assert not PhotonAdapter._is_upstream_failure("photon-sidecar: handler error: invalid JSON body")
