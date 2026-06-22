"""Regression tests for the Photon platform plugin."""

from __future__ import annotations

import types
from pathlib import Path

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.photon import adapter as photon


class _FakeResponse:
    def __init__(self, status_code: int, payload: dict | None = None, text: str = ""):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text

    def json(self):
        return self._payload


class _FakeAsyncClient:
    calls: list[tuple[str, dict]] = []
    responses: list[_FakeResponse] = []

    def __init__(self, timeout=None):
        self.timeout = timeout

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def post(self, url, json=None, headers=None):  # noqa: A002 - mirrors httpx API
        self.calls.append((url, json or {}))
        if not self.responses:
            raise AssertionError("no fake response queued")
        return self.responses.pop(0)


@pytest.mark.asyncio
async def test_photon_standalone_send_retries_transient_sidecar_500(monkeypatch):
    """Transient Photon/Spectrum connection failures should be retried."""

    monkeypatch.setattr(photon, "HTTPX_AVAILABLE", True)
    monkeypatch.setattr(photon, "httpx", types.SimpleNamespace(AsyncClient=_FakeAsyncClient))
    monkeypatch.setenv("PHOTON_SIDECAR_TOKEN", "token")
    monkeypatch.setattr(photon.asyncio, "sleep", lambda _delay: _noop_sleep())
    _FakeAsyncClient.calls = []
    _FakeAsyncClient.responses = [
        _FakeResponse(500, {"ok": False}, '{"ok":false,"error":"Connection dropped"}'),
        _FakeResponse(200, {"ok": True, "messageId": "msg-ok"}),
    ]

    result = await photon._standalone_send(
        PlatformConfig(extra={}),
        "any;-;+15551234567",
        "hello",
    )

    assert result == {"success": True, "message_id": "msg-ok"}
    assert len(_FakeAsyncClient.calls) == 2
    assert _FakeAsyncClient.calls[0][1]["spaceId"] == "any;-;+15551234567"


@pytest.mark.asyncio
async def test_photon_standalone_send_uses_sidecar_retryable_flag(monkeypatch):
    monkeypatch.setattr(photon, "HTTPX_AVAILABLE", True)
    monkeypatch.setattr(photon, "httpx", types.SimpleNamespace(AsyncClient=_FakeAsyncClient))
    monkeypatch.setenv("PHOTON_SIDECAR_TOKEN", "token")
    monkeypatch.setattr(photon.asyncio, "sleep", lambda _delay: _noop_sleep())
    _FakeAsyncClient.calls = []
    _FakeAsyncClient.responses = [
        _FakeResponse(
            500,
            {"ok": False, "error": "Photon upstream connection dropped", "retryable": True},
            '{"ok":false,"error":"Photon upstream connection dropped","retryable":true}',
        ),
        _FakeResponse(200, {"ok": True, "messageId": "msg-ok"}),
    ]

    result = await photon._standalone_send(
        PlatformConfig(extra={}),
        "any;-;+15551234567",
        "hello",
    )

    assert result == {"success": True, "message_id": "msg-ok"}
    assert len(_FakeAsyncClient.calls) == 2


@pytest.mark.asyncio
async def test_photon_standalone_send_does_not_retry_non_retryable(monkeypatch):
    monkeypatch.setattr(photon, "HTTPX_AVAILABLE", True)
    monkeypatch.setattr(photon, "httpx", types.SimpleNamespace(AsyncClient=_FakeAsyncClient))
    monkeypatch.setenv("PHOTON_SIDECAR_TOKEN", "token")
    monkeypatch.setattr(photon.asyncio, "sleep", lambda _delay: _noop_sleep())
    _FakeAsyncClient.calls = []
    _FakeAsyncClient.responses = [
        _FakeResponse(
            500,
            {"ok": False, "error": "target not allowed for this Photon project", "retryable": False},
            '{"ok":false,"error":"target not allowed for this Photon project","retryable":false}',
        ),
        _FakeResponse(200, {"ok": True, "messageId": "should-not-send"}),
    ]

    result = await photon._standalone_send(
        PlatformConfig(extra={}),
        "+15551234567",
        "hello",
    )

    assert "target not allowed" in result["error"]
    assert len(_FakeAsyncClient.calls) == 1


@pytest.mark.asyncio
async def test_sidecar_call_preserves_retryable_error_classification(monkeypatch):
    monkeypatch.setattr(photon, "httpx", types.SimpleNamespace(AsyncClient=_FakeAsyncClient))
    _FakeAsyncClient.calls = []
    _FakeAsyncClient.responses = [
        _FakeResponse(
            500,
            {
                "ok": False,
                "code": "upstream_transient",
                "error": "Photon upstream connection dropped; retrying may succeed",
                "retryable": True,
            },
        )
    ]
    adapter = photon.PhotonAdapter(PlatformConfig(extra={}))
    adapter._http_client = object()  # type: ignore[assignment]

    with pytest.raises(photon.PhotonSidecarError) as exc_info:
        await adapter._sidecar_call("/send", {"spaceId": "any;-;+15551234567", "text": "hello"})

    assert exc_info.value.retryable is True
    assert exc_info.value.code == "upstream_transient"
    assert "connection dropped" in str(exc_info.value)


def test_photon_sidecar_resolves_dm_space_before_phone_create():
    """DM space ids must resolve by `space.get(id)` before phone create().

    Spectrum can reject phone-based DM creation as not allowed even when replying
    to the already-established `any;-;+E164` DM space is valid.
    """

    source = Path("plugins/platforms/photon/sidecar/index.mjs").read_text()
    dm_get = source.index("photon-sidecar: DM space.get failed")
    phone_create = source.index("photon-sidecar: phone->DM space.create failed")
    assert dm_get < phone_create


def test_photon_sidecar_returns_sanitized_error_classifications():
    source = Path("plugins/platforms/photon/sidecar/index.mjs").read_text()
    assert "function classifySidecarError" in source
    assert 'code: "target_not_allowed"' in source
    assert 'code: "invalid_credentials"' in source
    assert 'code: "upstream_transient"' in source
    assert "e && e.stack ? e.stack" in source  # logs keep detail server-side
    assert "...classifySidecarError(error)" in source  # API returns only classification


async def _noop_sleep():
    return None
