"""Standalone cron send tests for Wechaty."""
from __future__ import annotations

from typing import Any

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.wechaty import adapter as wechaty_adapter
from plugins.platforms.wechaty.adapter import _standalone_send


@pytest.mark.asyncio
async def test_standalone_send_requires_sidecar_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not wechaty_adapter.HTTPX_AVAILABLE:
        pytest.skip("httpx not installed")
    monkeypatch.delenv("WECHATY_SIDECAR_TOKEN", raising=False)
    pconfig = PlatformConfig(enabled=True, extra={})
    result = await _standalone_send(pconfig, "contact:abc", "hi")
    assert "error" in result
    assert "WECHATY_SIDECAR_TOKEN" in result["error"]


@pytest.mark.asyncio
async def test_standalone_send_success(monkeypatch: pytest.MonkeyPatch) -> None:
    if not wechaty_adapter.HTTPX_AVAILABLE:
        pytest.skip("httpx not installed")
    monkeypatch.setenv("WECHATY_SIDECAR_TOKEN", "test-token")

    class _FakeClient:
        def __init__(self, *a: Any, **k: Any) -> None:
            pass

        async def __aenter__(self) -> "_FakeClient":
            return self

        async def __aexit__(self, *a: Any) -> bool:
            return False

        async def post(self, url: str, **kwargs: Any) -> Any:
            class _Resp:
                status_code = 200

                @staticmethod
                def json() -> dict:
                    return {"ok": True}

            return _Resp()

    monkeypatch.setattr(wechaty_adapter.httpx, "AsyncClient", _FakeClient)
    pconfig = PlatformConfig(enabled=True, extra={})
    result = await _standalone_send(pconfig, "contact:abc", "hello")
    assert result == {"success": True}
