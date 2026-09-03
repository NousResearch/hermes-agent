"""Rich-link handling tests for PhotonAdapter.

Photon's spectrum-ts SDK exposes a ``richlink()`` content builder for native
URL previews. Hermes routes URL-only outbound messages to the sidecar's
rich-link endpoint and preserves inbound rich-link URLs when Spectrum emits
that content type.
"""
from __future__ import annotations

import base64
from typing import Any, Dict, List, Tuple

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType
from plugins.platforms.photon import adapter as photon_adapter
from plugins.platforms.photon.adapter import PhotonAdapter

_URL = "https://example.com/article"


def _make_adapter(monkeypatch: pytest.MonkeyPatch) -> PhotonAdapter:
    monkeypatch.setenv("PHOTON_PROJECT_ID", "test-project-id")
    monkeypatch.setenv("PHOTON_PROJECT_SECRET", "test-project-secret")
    cfg = PlatformConfig(enabled=True, token="", extra={})
    return PhotonAdapter(cfg)


def _capture_sidecar(adapter: PhotonAdapter) -> List[Tuple[str, Dict[str, Any]]]:
    calls: List[Tuple[str, Dict[str, Any]]] = []

    async def _fake_call(path: str, body: Dict[str, Any]) -> Dict[str, Any]:
        calls.append((path, body))
        return {"ok": True, "messageId": "msg-123"}

    adapter._sidecar_call = _fake_call  # type: ignore[assignment]
    return calls


def _capture_inbound(
    adapter: PhotonAdapter, monkeypatch: pytest.MonkeyPatch
) -> List[MessageEvent]:
    captured: List[MessageEvent] = []

    async def fake_handle(event: MessageEvent) -> None:
        captured.append(event)

    monkeypatch.setattr(adapter, "handle_message", fake_handle)
    return captured


def _dm_event(content: Dict[str, Any], msg_id: str = "spc-msg-rich") -> Dict[str, Any]:
    return {
        "messageId": msg_id,
        "platform": "iMessage",
        "space": {"id": "+155****4567", "type": "dm", "phone": "+155****4567"},
        "sender": {"id": "+155****4567"},
        "content": content,
        "timestamp": "2026-05-14T19:06:32.000Z",
    }


@pytest.mark.asyncio
async def test_url_only_send_routes_to_richlink_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PHOTON_MARKDOWN", raising=False)
    adapter = _make_adapter(monkeypatch)
    calls = _capture_sidecar(adapter)

    result = await adapter.send("+155****4567", _URL)

    assert result.success is True
    assert calls == [("/send-richlink", {"spaceId": "+155****4567", "url": _URL})]


@pytest.mark.asyncio
async def test_mixed_prose_url_stays_on_markdown_send(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PHOTON_MARKDOWN", raising=False)
    adapter = _make_adapter(monkeypatch)
    calls = _capture_sidecar(adapter)

    await adapter.send("+155****4567", f"Read this: {_URL}")

    path, body = calls[0]
    assert path == "/send"
    assert body["format"] == "markdown"
    assert body["text"] == f"Read this: {_URL}"


@pytest.mark.asyncio
async def test_malformed_url_like_send_stays_on_markdown_send(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PHOTON_MARKDOWN", raising=False)
    adapter = _make_adapter(monkeypatch)
    calls = _capture_sidecar(adapter)

    await adapter.send("+155****4567", "http://[::1")

    path, body = calls[0]
    assert path == "/send"
    assert body["format"] == "markdown"
    assert body["text"] == "http://[::1"


@pytest.mark.asyncio
async def test_markdown_link_stays_on_markdown_send(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PHOTON_MARKDOWN", raising=False)
    adapter = _make_adapter(monkeypatch)
    calls = _capture_sidecar(adapter)

    await adapter.send("+155****4567", f"[Read this]({_URL})")

    path, body = calls[0]
    assert path == "/send"
    assert body["format"] == "markdown"


@pytest.mark.asyncio
async def test_direct_url_only_send_falls_back_to_plain_send(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PHOTON_MARKDOWN", raising=False)
    adapter = _make_adapter(monkeypatch)
    calls: List[Tuple[str, Dict[str, Any]]] = []

    async def _fake_call(path: str, body: Dict[str, Any]) -> Dict[str, Any]:
        calls.append((path, body))
        if path == "/send-richlink":
            raise RuntimeError("richlink unsupported")
        return {"ok": True, "messageId": "plain-msg"}

    adapter._sidecar_call = _fake_call  # type: ignore[assignment]

    result = await adapter.send("+155****4567", _URL)

    assert result.success is True
    assert result.message_id == "plain-msg"
    assert calls == [
        ("/send-richlink", {"spaceId": "+155****4567", "url": _URL}),
        ("/send", {"spaceId": "+155****4567", "text": _URL}),
    ]


@pytest.mark.asyncio
async def test_target_not_allowed_falls_back_and_reports_error_class(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A target_not_allowed rich-link failure degrades to plain text, and the
    returned SendResult carries the error class so callers (and the fallback
    path) can distinguish a plan limitation from a transient fault (#97305)."""
    monkeypatch.delenv("PHOTON_MARKDOWN", raising=False)
    adapter = _make_adapter(monkeypatch)
    calls: List[Tuple[str, Dict[str, Any]]] = []

    async def _fake_call(path: str, body: Dict[str, Any]) -> Dict[str, Any]:
        calls.append((path, body))
        if path == "/send-richlink":
            raise photon_adapter.PhotonSidecarError(
                path="/send-richlink",
                status_code=500,
                error="Target not allowed for this project",
                error_class="target_not_allowed",
                retryable=False,
            )
        return {"ok": True, "messageId": "plain-msg"}

    adapter._sidecar_call = _fake_call  # type: ignore[assignment]

    result = await adapter.send("+155****4567", _URL)

    assert result.success is True
    assert result.message_id == "plain-msg"
    assert calls == [
        ("/send-richlink", {"spaceId": "+155****4567", "url": _URL}),
        ("/send", {"spaceId": "+155****4567", "text": _URL}),
    ]
    # The richlink attempt itself reported the structured error class.
    probe = await adapter._sidecar_send_richlink("+155****4567", _URL)
    assert probe.success is False
    assert probe.raw_response == {"error_class": "target_not_allowed", "retryable": False}
    assert probe.retryable is False


@pytest.mark.asyncio
async def test_standalone_url_only_send_routes_to_richlink_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PHOTON_MARKDOWN", raising=False)
    monkeypatch.setenv("PHOTON_SIDECAR_TOKEN", "tok")
    posted: List[Tuple[str, Dict[str, Any]]] = []

    class _Resp:
        status_code = 200

        @staticmethod
        def json() -> Dict[str, Any]:
            return {"ok": True, "messageId": "m-9"}

    class _FakeClient:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url: str, json: Dict[str, Any], headers=None):
            posted.append((url, json))
            return _Resp()

    monkeypatch.setattr(photon_adapter.httpx, "AsyncClient", _FakeClient)

    cfg = PlatformConfig(enabled=True, token="", extra={})
    result = await photon_adapter._standalone_send(cfg, "+155****4567", _URL)

    assert result.get("success") is True
    assert posted == [
        (
            "http://127.0.0.1:8789/send-richlink",
            {"spaceId": "+155****4567", "url": _URL},
        )
    ]


@pytest.mark.asyncio
async def test_inbound_richlink_dispatches_url_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _make_adapter(monkeypatch)
    captured = _capture_inbound(adapter, monkeypatch)
    event = _dm_event({"type": "richlink", "url": _URL})

    await adapter._dispatch_inbound(event)

    assert len(captured) == 1
    assert captured[0].text == _URL
    assert captured[0].message_type == MessageType.TEXT
    assert captured[0].raw_message["content"] == {"type": "richlink", "url": _URL}


_PNG_1X1_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYPhf"
    "DwAChwGA60e6kgAAAABJRU5ErkJggg=="
)


def _preview_attachment(
    name: str = "preview.pluginPayloadAttachment",
    mime_type: str = "image/png",
) -> Dict[str, Any]:
    raw = base64.b64decode(_PNG_1X1_B64)
    return {
        "type": "attachment",
        "name": name,
        "mimeType": mime_type,
        "size": len(raw),
        "data": _PNG_1X1_B64,
        "encoding": "base64",
    }


def _preview_attachment_by_id(
    attachment_id: str = "doc_123.pluginPayloadAttachment",
) -> Dict[str, Any]:
    payload = _preview_attachment(name="")
    payload["id"] = attachment_id
    payload["name"] = None
    return payload


