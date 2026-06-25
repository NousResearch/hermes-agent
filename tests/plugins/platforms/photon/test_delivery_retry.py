"""Delivery retry tests for transient Photon sidecar failures."""
from __future__ import annotations

from typing import Any, Dict

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.base import SendResult
from plugins.platforms.photon.adapter import PhotonAdapter

_TRANSIENT_500 = 'Photon sidecar /send returned 500: {"ok":false,"error":"internal sidecar error"}'


def _make_adapter(monkeypatch: pytest.MonkeyPatch) -> PhotonAdapter:
    monkeypatch.setenv("PHOTON_PROJECT_ID", "test-project-id")
    monkeypatch.setenv("PHOTON_PROJECT_SECRET", "test-project-secret")
    cfg = PlatformConfig(enabled=True, token="", extra={})
    return PhotonAdapter(cfg)


@pytest.mark.asyncio
async def test_sidecar_send_marks_generic_internal_500_retryable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _make_adapter(monkeypatch)

    async def _fake_call(path: str, body: Dict[str, Any]) -> Dict[str, Any]:
        raise RuntimeError(_TRANSIENT_500)

    adapter._sidecar_call = _fake_call  # type: ignore[assignment]

    result = await adapter.send("+15551234567", "hi")

    assert result.success is False
    assert result.retryable is True
    assert result.error == _TRANSIENT_500


@pytest.mark.asyncio
async def test_send_with_retry_retries_photon_internal_500_without_plain_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _make_adapter(monkeypatch)
    sleeps: list[float] = []
    calls = 0

    async def _fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    async def _fake_send(
        chat_id: str,
        content: str,
        reply_to=None,
        metadata=None,
    ) -> SendResult:
        nonlocal calls
        calls += 1
        if calls < 3:
            return SendResult(success=False, error=_TRANSIENT_500)
        return SendResult(success=True, message_id="delivered")

    monkeypatch.setattr("plugins.platforms.photon.adapter.asyncio.sleep", _fake_sleep)
    adapter.send = _fake_send  # type: ignore[assignment]

    result = await adapter._send_with_retry(
        "+15551234567",
        "hi",
        max_retries=2,
        base_delay=2.0,
    )

    assert result.success is True
    assert result.message_id == "delivered"
    assert calls == 3
    assert sleeps == [2.0, 4.0]
