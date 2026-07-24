"""Outbound-media tests for PhotonAdapter.

Photon ships outbound attachments via spectrum-ts' ``attachment()`` /
``voice()`` content builders, reached through the Node sidecar's
``/send-attachment`` endpoint. These tests stub ``_sidecar_call`` so we
can assert the endpoint + body shape each ``send_*`` override produces
without spawning Node or binding ports.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.photon import adapter as photon_adapter
from plugins.platforms.photon.adapter import PhotonAdapter
from plugins.platforms.photon.state import PhotonStateStore


def _make_adapter(monkeypatch: pytest.MonkeyPatch) -> PhotonAdapter:
    monkeypatch.setenv("PHOTON_PROJECT_ID", "test-project-id")
    monkeypatch.setenv("PHOTON_PROJECT_SECRET", "test-project-secret")
    monkeypatch.delenv("PHOTON_WEBHOOK_SECRET", raising=False)
    cfg = PlatformConfig(enabled=True, token="", extra={})
    return PhotonAdapter(cfg)


def _capture_sidecar(adapter: PhotonAdapter) -> List[Tuple[str, Dict[str, Any]]]:
    """Replace ``_sidecar_call`` with a recorder that returns a fixed id."""
    calls: List[Tuple[str, Dict[str, Any]]] = []

    async def _fake_call(path: str, body: Dict[str, Any]) -> Dict[str, Any]:
        calls.append((path, body))
        return {"ok": True, "messageId": "msg-123"}

    adapter._sidecar_call = _fake_call  # type: ignore[assignment]
    return calls


@pytest.fixture()
def real_file(tmp_path) -> str:
    p = tmp_path / "photo.jpg"
    p.write_bytes(b"\xff\xd8\xff\xe0fake-jpeg")
    return str(p)


def _patch_safe_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make path validation a passthrough so tmp files outside the cache pass."""
    monkeypatch.setattr(
        PhotonAdapter,
        "validate_media_delivery_path",
        staticmethod(lambda p: p if os.path.exists(p) else None),
    )


@pytest.mark.asyncio
async def test_send_image_file_hits_attachment_endpoint(
    monkeypatch: pytest.MonkeyPatch, real_file: str
) -> None:
    _patch_safe_path(monkeypatch)
    adapter = _make_adapter(monkeypatch)
    calls = _capture_sidecar(adapter)

    result = await adapter.send_image_file(
        "any;-;+15551234567", real_file, caption="look"
    )

    assert result.success is True
    assert result.message_id == "msg-123"
    assert len(calls) == 1
    path, body = calls[0]
    assert path == "/send-attachment"
    assert body["spaceId"] == "any;-;+15551234567"
    assert body["path"] == real_file
    assert body["kind"] == "attachment"
    assert body["caption"] == "look"
    assert body["mimeType"] == "image/jpeg"  # inferred from .jpg


@pytest.mark.asyncio
async def test_send_voice_marks_kind_voice(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    _patch_safe_path(monkeypatch)
    audio = tmp_path / "note.m4a"
    audio.write_bytes(b"fake-audio")
    adapter = _make_adapter(monkeypatch)
    calls = _capture_sidecar(adapter)

    result = await adapter.send_voice("any;-;+1", str(audio))

    assert result.success is True
    path, body = calls[0]
    assert path == "/send-attachment"
    assert body["kind"] == "voice"


@pytest.mark.asyncio
async def test_send_document_passes_filename(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    _patch_safe_path(monkeypatch)
    doc = tmp_path / "report.pdf"
    doc.write_bytes(b"%PDF-1.4 fake")
    adapter = _make_adapter(monkeypatch)
    calls = _capture_sidecar(adapter)

    await adapter.send_document("any;-;+1", str(doc), file_name="Q3.pdf")

    _, body = calls[0]
    assert body["kind"] == "attachment"
    assert body["name"] == "Q3.pdf"
    assert body["mimeType"] == "application/pdf"


@pytest.mark.asyncio
async def test_send_video_passes_through(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    _patch_safe_path(monkeypatch)
    vid = tmp_path / "clip.mp4"
    vid.write_bytes(b"fake-mp4")
    adapter = _make_adapter(monkeypatch)
    calls = _capture_sidecar(adapter)

    await adapter.send_video("any;+;groupguid", str(vid), caption="watch")

    _, body = calls[0]
    assert body["kind"] == "attachment"
    assert body["caption"] == "watch"


@pytest.mark.asyncio
async def test_send_image_url_caches_then_sends_attachment(
    monkeypatch: pytest.MonkeyPatch, real_file: str
) -> None:
    _patch_safe_path(monkeypatch)
    adapter = _make_adapter(monkeypatch)
    calls = _capture_sidecar(adapter)

    async def _fake_cache(url: str, *a, **k) -> str:
        assert url == "https://example.com/cat.jpg"
        return real_file

    import gateway.platforms.base as base_mod

    monkeypatch.setattr(base_mod, "cache_image_from_url", _fake_cache)

    result = await adapter.send_image(
        "any;-;+1", "https://example.com/cat.jpg", caption="cat"
    )

    assert result.success is True
    path, body = calls[0]
    assert path == "/send-attachment"
    assert body["path"] == real_file
    assert body["caption"] == "cat"


@pytest.mark.asyncio
async def test_send_image_url_fetch_failure_falls_back_to_text(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = _make_adapter(monkeypatch)
    calls = _capture_sidecar(adapter)

    async def _boom(url: str, *a, **k) -> str:
        raise RuntimeError("network down")

    import gateway.platforms.base as base_mod

    monkeypatch.setattr(base_mod, "cache_image_from_url", _boom)

    result = await adapter.send_image(
        "any;-;+1", "https://example.com/cat.jpg", caption="cat"
    )

    # Fallback path: base send_image() routes to send() → /send (text).
    assert result.success is True
    assert calls[0][0] == "/send"
    assert "https://example.com/cat.jpg" in calls[0][1]["text"]


@pytest.mark.asyncio
async def test_send_attachment_rejects_unsafe_path(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    # Default validation (no passthrough patch) should reject a nonexistent /
    # traversal path, returning a failed SendResult without calling the sidecar.
    monkeypatch.setattr(
        PhotonAdapter,
        "validate_media_delivery_path",
        staticmethod(lambda p: None),
    )
    adapter = _make_adapter(monkeypatch)
    calls = _capture_sidecar(adapter)

    result = await adapter.send_image_file("any;-;+1", "/etc/passwd")

    assert result.success is False
    assert "unsafe" in (result.error or "")
    assert calls == []  # never reached the sidecar


@pytest.mark.asyncio
async def test_standalone_send_text_then_attachments(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    _patch_safe_path(monkeypatch)
    img = tmp_path / "a.png"
    img.write_bytes(b"\x89PNG fake")
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
    result = await photon_adapter._standalone_send(
        cfg,
        "any;-;+15551234567",
        "hello",
        media_files=[(str(img), False)],
    )

    assert result.get("success") is True
    # First call is the text /send, second is /send-attachment.
    assert posted[0][0].endswith("/send")
    assert posted[0][1]["text"] == "hello"
    assert posted[1][0].endswith("/send-attachment")
    assert posted[1][1]["path"] == str(img)
    assert posted[1][1]["kind"] == "attachment"
    assert posted[1][1]["mimeType"] == "image/png"
    state = PhotonStateStore().load()
    assert state["sent_messages"]["m-9"]["chat_key"] == "+15551234567"
    assert state["sent_messages"]["m-9"]["kind"] == "attachment"
