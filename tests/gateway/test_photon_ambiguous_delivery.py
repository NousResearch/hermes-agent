from __future__ import annotations

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.base import SendResult
from plugins.platforms.photon.adapter import PhotonAdapter


@pytest.mark.asyncio
async def test_photon_ambiguous_delivery_error_skips_retry(monkeypatch):
    monkeypatch.setenv("PHOTON_MARKDOWN", "false")
    adapter = PhotonAdapter(PlatformConfig(enabled=True, token="test-token"))
    calls: list[str] = []

    async def send_once(
        chat_id: str,
        content: str,
        reply_to: str | None = None,
        metadata: object | None = None,
    ) -> SendResult:
        calls.append(content)
        return SendResult(
            success=False,
            error="sidecar /send returned 500: remote refused stream reset",
            retryable=True,
        )

    adapter.send = send_once

    result = await adapter._send_with_retry(
        chat_id="chat-1",
        content="hello",
        max_retries=2,
        base_delay=0,
    )

    assert result.success is False
    assert len(calls) == 1
