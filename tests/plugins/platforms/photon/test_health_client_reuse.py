"""Tests for photon health monitoring client reuse (RFC #96778).

Ensures _sidecar_call reuses self._http_client when available,
avoiding TIME_WAIT socket exhaustion from frequent /healthz polling.
"""
from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from gateway.config import PlatformConfig
from plugins.platforms.photon.adapter import PhotonAdapter


def _make_adapter(monkeypatch: pytest.MonkeyPatch, **extra: Any) -> PhotonAdapter:
    monkeypatch.setenv("PHOTON_PROJECT_ID", "test-project-id")
    monkeypatch.setenv("PHOTON_PROJECT_SECRET", "test-project-secret")
    cfg = PlatformConfig(enabled=True, token="", extra=dict(extra))
    return PhotonAdapter(cfg)


@pytest.mark.asyncio
async def test_sidecar_call_reuses_http_client(monkeypatch: pytest.MonkeyPatch) -> None:
    a = _make_adapter(monkeypatch)
    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"ok": True, "stream": {"ok": True}}
    mock_client.post.return_value = mock_resp
    a._http_client = mock_client

    data = await a._sidecar_call("/healthz", {})
    assert data["ok"] is True
    assert mock_client.post.called


@pytest.mark.asyncio
async def test_sidecar_call_falls_back_without_http_client(monkeypatch: pytest.MonkeyPatch) -> None:
    a = _make_adapter(monkeypatch)
    a._http_client = None

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"ok": True, "stream": {"ok": True}}

    with monkeypatch.context() as m:
        mock_fresh = AsyncMock()
        mock_fresh.post.return_value = mock_resp
        mock_ctx = AsyncMock()
        mock_ctx.__aenter__.return_value = mock_fresh
        mock_ctx.__aexit__.return_value = None
        m.setattr(httpx, "AsyncClient", lambda *args, **kwargs: mock_ctx)

        data = await a._sidecar_call("/healthz", {})
        assert data["ok"] is True
        assert mock_fresh.post.called
