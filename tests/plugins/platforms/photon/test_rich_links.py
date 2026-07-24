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
async def test_url_only_send_trims_surrounding_whitespace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PHOTON_MARKDOWN", raising=False)
    adapter = _make_adapter(monkeypatch)
    calls = _capture_sidecar(adapter)

    await adapter.send("+155****4567", f"  {_URL}\n")

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
async def test_markdown_disabled_keeps_url_on_plain_send(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PHOTON_MARKDOWN", "false")
    adapter = _make_adapter(monkeypatch)
    calls = _capture_sidecar(adapter)

    await adapter.send("+155****4567", _URL)

    assert calls == [("/send", {"spaceId": "+155****4567", "text": _URL})]


@pytest.mark.asyncio
async def test_url_only_send_fallback_bypasses_richlink_endpoint(
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

    result = await adapter._send_with_retry("+155****4567", _URL, base_delay=0)

    assert result.success is True
    assert result.message_id == "plain-msg"
    assert calls == [
        ("/send-richlink", {"spaceId": "+155****4567", "url": _URL}),
        ("/send", {"spaceId": "+155****4567", "text": _URL}),
    ]


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
async def test_url_only_retry_exhaustion_falls_back_to_plain_send(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PHOTON_MARKDOWN", raising=False)
    adapter = _make_adapter(monkeypatch)
    calls: List[Tuple[str, Dict[str, Any]]] = []

    async def _fake_call(path: str, body: Dict[str, Any]) -> Dict[str, Any]:
        calls.append((path, body))
        if path == "/send-richlink":
            raise RuntimeError("upstream unavailable")
        return {"ok": True, "messageId": "plain-msg"}

    adapter._sidecar_call = _fake_call  # type: ignore[assignment]

    result = await adapter._send_with_retry(
        "+155****4567", _URL, max_retries=1, base_delay=0
    )

    assert result.success is True
    assert result.message_id == "plain-msg"
    assert calls == [
        ("/send-richlink", {"spaceId": "+155****4567", "url": _URL}),
        ("/send", {"spaceId": "+155****4567", "text": _URL}),
    ]


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
async def test_standalone_url_only_send_falls_back_to_plain_send(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PHOTON_MARKDOWN", raising=False)
    monkeypatch.setenv("PHOTON_SIDECAR_TOKEN", "tok")
    posted: List[Tuple[str, Dict[str, Any]]] = []

    class _Resp:
        def __init__(self, status_code: int, message_id: str = "m-9"):
            self.status_code = status_code
            self.text = "not found" if status_code != 200 else ""
            self._message_id = message_id

        def json(self) -> Dict[str, Any]:
            return {"ok": True, "messageId": self._message_id}

    class _FakeClient:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url: str, json: Dict[str, Any], headers=None):
            posted.append((url, json))
            if url.endswith("/send-richlink"):
                return _Resp(404)
            return _Resp(200, "plain-msg")

    monkeypatch.setattr(photon_adapter.httpx, "AsyncClient", _FakeClient)

    cfg = PlatformConfig(enabled=True, token="", extra={})
    result = await photon_adapter._standalone_send(cfg, "+155****4567", _URL)

    assert result.get("success") is True
    assert result.get("message_id") == "plain-msg"
    assert posted == [
        (
            "http://127.0.0.1:8789/send-richlink",
            {"spaceId": "+155****4567", "url": _URL},
        ),
        (
            "http://127.0.0.1:8789/send",
            {"spaceId": "+155****4567", "text": _URL},
        ),
    ]


@pytest.mark.asyncio
async def test_standalone_markdown_disabled_keeps_url_on_plain_send(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PHOTON_MARKDOWN", "false")
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
            "http://127.0.0.1:8789/send",
            {"spaceId": "+155****4567", "text": _URL},
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


@pytest.mark.asyncio
async def test_inbound_richlink_preserves_metadata_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _make_adapter(monkeypatch)
    captured = _capture_inbound(adapter, monkeypatch)
    event = _dm_event(
        {
            "type": "richlink",
            "url": _URL,
            "title": "Example Article",
            "summary": "A summary of the article",
        }
    )

    await adapter._dispatch_inbound(event)

    assert len(captured) == 1
    assert captured[0].text == f"Example Article\nA summary of the article\n{_URL}"
    assert captured[0].message_type == MessageType.TEXT


@pytest.mark.asyncio
async def test_inbound_richlink_dedupes_repeated_summary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _make_adapter(monkeypatch)
    captured = _capture_inbound(adapter, monkeypatch)
    event = _dm_event(
        {
            "type": "richlink",
            "url": "https://example.com/dup",
            "title": "Same Text",
            "summary": "Same Text",
        }
    )

    await adapter._dispatch_inbound(event)

    assert len(captured) == 1
    assert captured[0].text == "Same Text\nhttps://example.com/dup"


@pytest.mark.asyncio
async def test_inbound_richlink_without_url_preserves_title(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _make_adapter(monkeypatch)
    captured = _capture_inbound(adapter, monkeypatch)
    event = _dm_event({"type": "richlink", "url": "", "title": "Some Title"})

    await adapter._dispatch_inbound(event)

    assert len(captured) == 1
    assert captured[0].text == "Some Title"


@pytest.mark.asyncio
async def test_malformed_url_like_inbound_text_dispatches_normally(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _make_adapter(monkeypatch)
    captured = _capture_inbound(adapter, monkeypatch)

    await adapter._dispatch_inbound(
        _dm_event({"type": "text", "text": "http://[::1"}, "malformed-url-msg")
    )

    assert len(captured) == 1
    assert captured[0].text == "http://[::1"
    assert captured[0].message_id == "malformed-url-msg"


@pytest.mark.asyncio
async def test_inbound_group_preserves_richlink_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _make_adapter(monkeypatch)
    captured = _capture_inbound(adapter, monkeypatch)
    event = _dm_event(
        {
            "type": "group",
            "items": [
                {"id": "p:0", "content": {"type": "text", "text": "Read this"}},
                {"id": "p:1", "content": {"type": "richlink", "url": _URL}},
            ],
        },
        msg_id="spc-msg-rich-group",
    )

    await adapter._dispatch_inbound(event)

    assert len(captured) == 1
    assert captured[0].text == f"Read this\n{_URL}"
    assert captured[0].message_type == MessageType.TEXT


@pytest.mark.asyncio
async def test_inbound_group_preserves_richlink_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _make_adapter(monkeypatch)
    captured = _capture_inbound(adapter, monkeypatch)
    event = _dm_event(
        {
            "type": "group",
            "items": [
                {"id": "p:0", "content": {"type": "text", "text": "Read this"}},
                {
                    "id": "p:1",
                    "content": {
                        "type": "richlink",
                        "url": _URL,
                        "title": "Example Article",
                        "summary": "A summary of the article",
                    },
                },
            ],
        },
        msg_id="spc-msg-rich-group-metadata",
    )

    await adapter._dispatch_inbound(event)

    assert len(captured) == 1
    assert captured[0].text == f"Read this\nExample Article\nA summary of the article\n{_URL}"
    assert captured[0].message_type == MessageType.TEXT


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


@pytest.mark.asyncio
async def test_inbound_url_preview_attachment_is_coalesced(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _make_adapter(monkeypatch)
    captured = _capture_inbound(adapter, monkeypatch)

    await adapter._dispatch_inbound(_dm_event({"type": "text", "text": _URL}, "url-msg"))
    await adapter._dispatch_inbound(_dm_event(_preview_attachment(), "preview-msg"))

    assert len(captured) == 1
    assert captured[0].text == _URL
    assert captured[0].message_id == "url-msg"


@pytest.mark.asyncio
async def test_inbound_url_preview_attachment_id_only_is_coalesced(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _make_adapter(monkeypatch)
    captured = _capture_inbound(adapter, monkeypatch)

    await adapter._dispatch_inbound(_dm_event({"type": "text", "text": _URL}, "url-msg"))
    await adapter._dispatch_inbound(
        _dm_event(_preview_attachment_by_id("doc_5f418810.pluginPayloadAttachment"), "preview-msg")
    )

    assert len(captured) == 1
    assert captured[0].text == _URL


@pytest.mark.asyncio
async def test_inbound_url_preview_octet_stream_plugin_payload_is_coalesced(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _make_adapter(monkeypatch)
    captured = _capture_inbound(adapter, monkeypatch)

    await adapter._dispatch_inbound(_dm_event({"type": "text", "text": _URL}, "url-msg"))
    await adapter._dispatch_inbound(
        _dm_event(
            _preview_attachment(
                "8DBFD7DD-97E6-40DA-BBBD-8B920E36951D.pluginPayloadAttachment",
                mime_type="application/octet-stream",
            ),
            "preview-doc-msg",
        )
    )

    assert len(captured) == 1
    assert captured[0].text == _URL


@pytest.mark.asyncio
async def test_inbound_grouped_url_preview_attachments_are_coalesced(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _make_adapter(monkeypatch)
    captured = _capture_inbound(adapter, monkeypatch)

    await adapter._dispatch_inbound(_dm_event({"type": "richlink", "url": _URL}, "url-msg"))
    await adapter._dispatch_inbound(
        _dm_event(
            {
                "type": "group",
                "items": [
                    {"id": "p:0", "content": _preview_attachment("wide.pluginPayloadAttachment")},
                    {"id": "p:1", "content": _preview_attachment("icon.pluginPayloadAttachment")},
                ],
            },
            "preview-group",
        )
    )

    assert len(captured) == 1
    assert captured[0].text == _URL


@pytest.mark.asyncio
async def test_inbound_richlink_metadata_preview_attachment_is_coalesced(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _make_adapter(monkeypatch)
    captured = _capture_inbound(adapter, monkeypatch)

    await adapter._dispatch_inbound(
        _dm_event(
            {
                "type": "richlink",
                "url": _URL,
                "title": "Example Article",
                "summary": "A summary of the article",
            },
            "rich-metadata-msg",
        )
    )
    await adapter._dispatch_inbound(_dm_event(_preview_attachment(), "preview-msg"))

    assert len(captured) == 1
    assert captured[0].text == f"Example Article\nA summary of the article\n{_URL}"
    assert captured[0].message_id == "rich-metadata-msg"
