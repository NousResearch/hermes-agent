"""Outbound delivery tests for PhotonAdapter.

Guards against duplicate iMessage bubbles when the sidecar/upstream is slow
or the loopback HTTP client times out with an empty error string.
"""
from __future__ import annotations

from typing import Any, Dict, List
from unittest.mock import AsyncMock

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.base import SendResult
from plugins.platforms.photon.adapter import PhotonAdapter


def _make_adapter(monkeypatch: pytest.MonkeyPatch) -> PhotonAdapter:
    monkeypatch.setenv("PHOTON_PROJECT_ID", "test-project-id")
    monkeypatch.setenv("PHOTON_PROJECT_SECRET", "test-project-secret")
    cfg = PlatformConfig(enabled=True, token="", extra={})
    return PhotonAdapter(cfg)


@pytest.mark.parametrize(
    "error,expected",
    [
        ("", True),
        ("Photon sidecar /send timed out (read timeout)", True),
        ("[upstream] Service temporarily unavailable. Please retry.", True),
        ("Photon sidecar /send returned 500: internal sidecar error", True),
        ("permission denied for chat", False),
    ],
)
def test_is_ambiguous_photon_delivery_error(
    error: str, expected: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = _make_adapter(monkeypatch)
    assert adapter._is_ambiguous_photon_delivery_error(error) is expected


@pytest.mark.asyncio
async def test_send_with_retry_does_not_duplicate_on_empty_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _make_adapter(monkeypatch)
    send_calls: List[str] = []

    async def _fake_send(**kwargs: Any) -> SendResult:
        send_calls.append(kwargs.get("content", ""))
        return SendResult(success=False, error="")

    adapter.send = AsyncMock(side_effect=_fake_send)  # type: ignore[method-assign]

    result = await adapter._send_with_retry("+15555550100", "hello **world**")

    assert result.success is False
    assert len(send_calls) == 1


@pytest.mark.asyncio
async def test_send_with_retry_does_not_plaintext_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _make_adapter(monkeypatch)
    send_calls: List[str] = []

    async def _fake_send(**kwargs: Any) -> SendResult:
        send_calls.append(str(kwargs.get("content", "")))
        return SendResult(
            success=False,
            error="Photon sidecar /send returned 500: internal sidecar error",
        )

    adapter.send = AsyncMock(side_effect=_fake_send)  # type: ignore[method-assign]

    await adapter._send_with_retry("+15555550100", "long reply")

    assert len(send_calls) == 1


@pytest.mark.asyncio
async def test_sidecar_send_timeout_sets_non_empty_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import httpx

    adapter = _make_adapter(monkeypatch)
    adapter._http_client = object()  # type: ignore[assignment]

    async def _raise_timeout(path: str, body: Dict[str, Any]) -> Dict[str, Any]:
        raise httpx.ReadTimeout("")

    adapter._sidecar_call = _raise_timeout  # type: ignore[assignment]

    result = await adapter._sidecar_send("any;-;+15555550100", "hi")

    assert result.success is False
    assert result.retryable is False
    assert result.error
    assert "timed out" in (result.error or "").lower()


@pytest.mark.asyncio
async def test_sidecar_call_uses_longer_timeout_for_send(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import plugins.platforms.photon.adapter as photon_adapter

    adapter = _make_adapter(monkeypatch)
    adapter._http_client = object()  # type: ignore[assignment]

    captured: List[float] = []

    class _Resp:
        status_code = 200

        @staticmethod
        def json() -> Dict[str, Any]:
            return {"ok": True, "messageId": "m-1"}

    class _FakeClient:
        def __init__(self, *a: Any, timeout: float = 30.0, **k: Any):
            captured.append(timeout)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a: Any):
            return False

        async def post(self, url: str, json: Dict[str, Any], headers=None):
            return _Resp()

    monkeypatch.setattr(photon_adapter.httpx, "AsyncClient", _FakeClient)
    monkeypatch.delenv("PHOTON_SIDECAR_SEND_TIMEOUT", raising=False)

    await adapter._sidecar_call("/send", {"spaceId": "x", "text": "y"})
    await adapter._sidecar_call("/typing", {"spaceId": "x", "state": "start"})

    assert captured == [120.0, 30.0]