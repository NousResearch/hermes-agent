"""Rich-link handling tests for PhotonAdapter.

Photon's spectrum-ts SDK exposes a ``richlink()`` content builder for native
URL previews. Hermes routes URL-only outbound messages to the sidecar's
rich-link endpoint and preserves inbound rich-link URLs when Spectrum emits
that content type.
"""
from __future__ import annotations

import base64
from unittest.mock import AsyncMock
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
async def test_blank_line_paragraphs_and_trailing_urls_send_as_separate_bubbles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PHOTON_MARKDOWN", raising=False)
    adapter = _make_adapter(monkeypatch)
    calls = _capture_sidecar(adapter)

    result = await adapter.send(
        "+155****4567",
        f"I found it.\n\nHere is the source:\n{_URL}\nhttps://example.com/second",
    )

    assert result.success is True
    assert calls == [
        ("/send", {"spaceId": "+155****4567", "text": "I found it.", "format": "markdown"}),
        ("/send", {"spaceId": "+155****4567", "text": "Here is the source:", "format": "markdown"}),
        ("/send-richlink", {"spaceId": "+155****4567", "url": _URL}),
        ("/send-richlink", {"spaceId": "+155****4567", "url": "https://example.com/second"}),
    ]


@pytest.mark.asyncio
async def test_bare_url_between_sections_becomes_a_native_preview_card(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PHOTON_MARKDOWN", raising=False)
    adapter = _make_adapter(monkeypatch)
    calls = _capture_sidecar(adapter)

    await adapter.send(
        "+155****4567",
        f"Top pick:\n{_URL}\nWhy it wins.\n\nTwo practical things:\n- Start early.",
    )

    assert calls == [
        ("/send", {"spaceId": "+155****4567", "text": "Top pick:", "format": "markdown"}),
        ("/send-richlink", {"spaceId": "+155****4567", "url": _URL}),
        ("/send", {"spaceId": "+155****4567", "text": "Why it wins.", "format": "markdown"}),
        (
            "/send",
            {
                "spaceId": "+155****4567",
                "text": "Two practical things:\n- Start early.",
                "format": "markdown",
            },
        ),
    ]


@pytest.mark.asyncio
async def test_url_only_line_inside_fence_stays_code_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PHOTON_MARKDOWN", raising=False)
    adapter = _make_adapter(monkeypatch)
    calls = _capture_sidecar(adapter)

    await adapter.send("+155****4567", f"```text\n{_URL}\n```")

    assert calls == [
        (
            "/send",
            {
                "spaceId": "+155****4567",
                "text": f"```text\n{_URL}\n```",
                "format": "markdown",
            },
        )
    ]


@pytest.mark.asyncio
async def test_blank_lines_inside_fenced_code_stay_in_one_bubble(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PHOTON_MARKDOWN", raising=False)
    adapter = _make_adapter(monkeypatch)
    calls = _capture_sidecar(adapter)

    await adapter.send(
        "+155****4567",
        "Before.\n\n```python\nfirst = 1\n\nsecond = 2\n```\n\nAfter.",
    )

    assert [body.get("text") for path, body in calls if path == "/send"] == [
        "Before.",
        "```python\nfirst = 1\n\nsecond = 2\n```",
        "After.",
    ]


@pytest.mark.asyncio
async def test_partial_bubble_failure_retries_only_failed_bubble(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _make_adapter(monkeypatch)
    sleep = AsyncMock()
    monkeypatch.setattr(photon_adapter.asyncio, "sleep", sleep)
    adapter._sidecar_send = AsyncMock(side_effect=[
        photon_adapter.SendResult(success=True, message_id="m-1"),
        photon_adapter.SendResult(success=False, error="connection reset", retryable=True),
        photon_adapter.SendResult(success=True, message_id="m-2"),
    ])

    result = await adapter._send_with_retry("+155****4567", "First.\n\nSecond.")

    assert result.success is True
    assert [call.args[1] for call in adapter._sidecar_send.await_args_list] == [
        "First.", "Second.", "Second.",
    ]
    sleep.assert_awaited_once_with(2.0)


@pytest.mark.asyncio
async def test_exhausted_partial_bubble_failure_is_not_replayed_from_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _make_adapter(monkeypatch)
    monkeypatch.setattr(photon_adapter.asyncio, "sleep", AsyncMock())
    failure = photon_adapter.SendResult(
        success=False,
        error="connection reset; retryable=false",
        retryable=False,
    )
    adapter._sidecar_send = AsyncMock(side_effect=[
        photon_adapter.SendResult(success=True, message_id="m-1"), failure, failure,
    ])

    result = await adapter._send_with_retry("+155****4567", "First.\n\nSecond.")

    assert result.success is False
    assert adapter._sidecar_send.await_count == 3
    assert result.raw_response["partial_bubble_delivery"] is True
    assert result.raw_response["message_ids"] == ["m-1"]
    assert result.raw_response["delivered_prefix"] == "First."
    assert result.raw_response["last_message_id"] == "m-1"
    assert result.raw_response["undelivered_content"] == "Second."
    assert result.message_id == "m-1"


@pytest.mark.asyncio
async def test_bubble_send_returns_all_message_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _make_adapter(monkeypatch)
    adapter._sidecar_send = AsyncMock(side_effect=[
        photon_adapter.SendResult(success=True, message_id="m-1"),
        photon_adapter.SendResult(success=True, message_id="m-2"),
    ])

    result = await adapter.send("+155****4567", "First.\n\nSecond.")

    assert result.success is True
    assert result.message_id == "m-2"
    assert result.continuation_message_ids == ("m-1",)


@pytest.mark.asyncio
async def test_null_message_id_still_counts_as_partial_delivery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _make_adapter(monkeypatch)
    adapter._sidecar_send = AsyncMock(side_effect=[
        photon_adapter.SendResult(success=True, message_id=None),
        photon_adapter.SendResult(success=False, error="rejected"),
    ])

    result = await adapter.send("+155****4567", "First.\n\nSecond.")

    assert result.success is False
    assert result.raw_response["partial_bubble_delivery"] is True
    assert result.raw_response["delivered_bubbles"] == 1
    assert result.raw_response["delivered_prefix"] == "First."
    assert result.raw_response["last_message_id"] is None
    assert result.raw_response["undelivered_content"] == "Second."


@pytest.mark.asyncio
async def test_markdown_disabled_splits_before_stripping_fenced_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PHOTON_MARKDOWN", "false")
    adapter = _make_adapter(monkeypatch)
    calls = _capture_sidecar(adapter)

    await adapter._send_with_retry(
        "+155****4567",
        "Before.\n\n```python\nfirst = 1\n\nsecond = 2\n```\n\nAfter.",
    )

    assert [body["text"] for path, body in calls if path == "/send"] == [
        "Before.",
        "first = 1\n\nsecond = 2",
        "After.",
    ]
    assert all("format" not in body for path, body in calls if path == "/send")


@pytest.mark.asyncio
async def test_markdown_disabled_uses_plain_send_for_standalone_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PHOTON_MARKDOWN", "false")
    adapter = _make_adapter(monkeypatch)
    calls = _capture_sidecar(adapter)

    await adapter._send_with_retry("+155****4567", _URL)

    assert calls == [("/send", {"spaceId": "+155****4567", "text": _URL})]


@pytest.mark.asyncio
async def test_long_paragraph_is_split_before_sidecar_send(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _make_adapter(monkeypatch)
    monkeypatch.setattr(adapter, "MAX_MESSAGE_LENGTH", 80)
    next_id = 0

    async def _send(*_args, **_kwargs):
        nonlocal next_id
        next_id += 1
        return photon_adapter.SendResult(success=True, message_id=f"m-{next_id}")

    adapter._sidecar_send = AsyncMock(side_effect=_send)

    result = await adapter.send("+155****4567", "word " * 28)

    assert result.success is True
    sent = [call.args[1] for call in adapter._sidecar_send.await_args_list]
    assert len(sent) > 1
    assert all(len(chunk) <= adapter.MAX_MESSAGE_LENGTH for chunk in sent)


@pytest.mark.asyncio
async def test_partial_send_honors_zero_retry_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _make_adapter(monkeypatch)
    adapter._sidecar_send = AsyncMock(side_effect=[
        photon_adapter.SendResult(success=True, message_id="m-1"),
        photon_adapter.SendResult(success=False, error="connection reset", retryable=True),
        photon_adapter.SendResult(success=False, error="plain fallback rejected"),
    ])

    result = await adapter._send_with_retry(
        "+155****4567", "First.\n\nSecond.", max_retries=0)

    assert result.success is False
    assert adapter._sidecar_send.await_count == 3
    assert result.raw_response["undelivered_content"] == "Second."


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


