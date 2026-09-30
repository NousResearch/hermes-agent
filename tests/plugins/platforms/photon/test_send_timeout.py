"""Outbound send timeout tests for PhotonAdapter.

``/send``, ``/send-richlink`` and ``/send-attachment`` use
``send_timeout_seconds`` (env ``PHOTON_SEND_TIMEOUT_SECONDS``, default 90s).
These tests stub ``_sidecar_call`` and record the timeout each send passes.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.photon.adapter import PhotonAdapter

_MD = "**bold** and `code`"


def _make_adapter(monkeypatch: pytest.MonkeyPatch) -> PhotonAdapter:
    monkeypatch.setenv("PHOTON_PROJECT_ID", "test-project-id")
    monkeypatch.setenv("PHOTON_PROJECT_SECRET", "test-project-secret")
    cfg = PlatformConfig(enabled=True, token="", extra={})
    return PhotonAdapter(cfg)


def _capture_timeouts(adapter: PhotonAdapter) -> List[float]:
    seen: List[float] = []

    async def _fake_call(
        path: str, body: Dict[str, Any], timeout: float = 30.0
    ) -> Dict[str, Any]:
        seen.append(timeout)
        return {"ok": True, "messageId": "msg-123"}

    adapter._sidecar_call = _fake_call  # type: ignore[assignment]
    return seen


@pytest.mark.asyncio
async def test_sidecar_send_default_timeout_outlasts_cron_delivery_wait(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The default /send timeout must exceed the cron live-delivery wait (60s).

    Otherwise a send still in flight fails here first, and the cron standalone
    fallback resends it, delivering the message twice.
    """
    monkeypatch.delenv("PHOTON_SEND_TIMEOUT_SECONDS", raising=False)
    adapter = _make_adapter(monkeypatch)
    seen = _capture_timeouts(adapter)

    await adapter.send("+15551234567", _MD)

    assert seen == [90.0]


@pytest.mark.asyncio
async def test_sidecar_send_timeout_honours_env_var(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PHOTON_SEND_TIMEOUT_SECONDS", "120")
    adapter = _make_adapter(monkeypatch)
    seen = _capture_timeouts(adapter)

    await adapter.send("+15551234567", _MD)

    assert seen == [120.0]

    monkeypatch.setenv("PHOTON_SEND_TIMEOUT_SECONDS", "0")
    assert _make_adapter(monkeypatch)._send_timeout == 90.0


@pytest.mark.asyncio
async def test_richlink_send_uses_send_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PHOTON_SEND_TIMEOUT_SECONDS", "120")
    adapter = _make_adapter(monkeypatch)
    seen: List[Tuple[str, float]] = []

    async def _fake_call(
        path: str, body: Dict[str, Any], timeout: float = 30.0
    ) -> Dict[str, Any]:
        seen.append((path, timeout))
        return {"ok": True, "messageId": "msg-123"}

    adapter._sidecar_call = _fake_call  # type: ignore[assignment]

    await adapter.send("+15551234567", "https://example.com/article")

    assert seen == [("/send-richlink", 120.0)]


class _FakeResponse:
    status_code = 200

    def json(self) -> Dict[str, Any]:
        return {"ok": True, "messageId": "msg-standalone"}


def _record_standalone_client(monkeypatch: pytest.MonkeyPatch) -> List[float]:
    """Replace httpx.AsyncClient in the adapter module and record its timeout."""
    import plugins.platforms.photon.adapter as photon

    timeouts: List[float] = []

    class _FakeClient:
        def __init__(self, *, timeout: float, **_: Any) -> None:
            timeouts.append(timeout)

        async def __aenter__(self) -> "_FakeClient":
            return self

        async def __aexit__(self, *exc: Any) -> None:
            return None

        async def post(self, url: str, **_: Any) -> _FakeResponse:
            return _FakeResponse()

    monkeypatch.setattr(photon.httpx, "AsyncClient", _FakeClient)
    real_secret = photon._get_scoped_secret
    monkeypatch.setattr(
        photon, "_get_scoped_secret",
        lambda name, *a, **k: "sidecar-token" if name == "PHOTON_SIDECAR_TOKEN" else real_secret(name, *a, **k),
    )
    return timeouts


@pytest.mark.asyncio
async def test_standalone_send_uses_the_send_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """Cron without a live gateway sends through _standalone_send; it must honour the setting too."""
    from plugins.platforms.photon.adapter import _standalone_send

    timeouts = _record_standalone_client(monkeypatch)

    monkeypatch.delenv("PHOTON_SEND_TIMEOUT_SECONDS", raising=False)
    result = await _standalone_send(PlatformConfig(enabled=True, extra={}), "+15551234567", "hello")
    assert result["success"] is True
    assert timeouts == [90.0]

    monkeypatch.setenv("PHOTON_SEND_TIMEOUT_SECONDS", "120")
    await _standalone_send(PlatformConfig(enabled=True, extra={}), "+15551234567", "hello")
    assert timeouts[-1] == 120.0

    monkeypatch.delenv("PHOTON_SEND_TIMEOUT_SECONDS")
    await _standalone_send(
        PlatformConfig(enabled=True, extra={"send_timeout_seconds": 75}), "+15551234567", "hello")
    assert timeouts[-1] == 75.0


@pytest.mark.parametrize("raw", ["0", "-5", "inf", "nan", "soon"])
def test_unusable_send_timeout_falls_back_to_default(monkeypatch: pytest.MonkeyPatch, raw: str) -> None:
    from plugins.platforms.photon.adapter import _resolve_send_timeout

    monkeypatch.setenv("PHOTON_SEND_TIMEOUT_SECONDS", raw)
    assert _resolve_send_timeout({}) == 90.0
