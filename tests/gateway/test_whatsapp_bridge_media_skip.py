"""Regression: the WhatsApp bridge must not hand ingress a non-local media entry.

``_collect_bridge_media`` previously (1) kept the original https URL in place when the
image/voice cacher raised, and (2) passed DOCUMENT/VIDEO URLs straight through the final
``else``. Ingress admission (``gateway.session_ingress_media``) requires a local regular
file and raises ``RuntimeStoreError('invalid_params')`` otherwise — dropping the WHOLE
message, text included (#142487). The offending entry must be skipped instead so the
message text still reaches the agent, the way Signal skips a failed attachment.
"""

from unittest.mock import AsyncMock

import pytest

from gateway.platforms.base import Platform
from gateway.platforms.event import MessageType
from plugins.platforms.whatsapp.adapter import WhatsAppAdapter


def _make_adapter() -> WhatsAppAdapter:
    adapter = WhatsAppAdapter.__new__(WhatsAppAdapter)
    adapter.platform = Platform.WHATSAPP
    return adapter


@pytest.mark.asyncio
async def test_cache_failure_skips_entry(monkeypatch):
    """A cacher exception must not leave the original https URL in media_urls."""
    monkeypatch.setattr(
        "plugins.platforms.whatsapp.adapter.cache_image_from_url",
        AsyncMock(side_effect=Exception("download failed")),
    )
    adapter = _make_adapter()

    urls, mimes = await adapter._collect_bridge_media(
        {"mediaUrls": ["https://mmg.whatsapp.net/v/t62.7118-24/abc.jpg"]},
        MessageType.PHOTO,
    )

    assert urls == [] and mimes == []


@pytest.mark.asyncio
async def test_remote_document_url_is_skipped(monkeypatch):
    """A DOCUMENT https URL has no cacher path; it must be skipped, not passed through."""
    adapter = _make_adapter()

    urls, mimes = await adapter._collect_bridge_media(
        {"mediaUrls": ["https://mmg.whatsapp.net/v/t62.7118-24/report.pdf"]},
        MessageType.DOCUMENT,
    )

    assert urls == [] and mimes == []


@pytest.mark.asyncio
async def test_remote_video_url_is_skipped(monkeypatch):
    """A VIDEO https URL previously fell into the passthrough else-branch."""
    adapter = _make_adapter()

    urls, mimes = await adapter._collect_bridge_media(
        {"mediaUrls": ["https://mmg.whatsapp.net/v/t62.7118-24/clip.mp4"]},
        MessageType.VIDEO,
    )

    assert urls == [] and mimes == []


@pytest.mark.asyncio
async def test_cached_photo_still_accepted(monkeypatch):
    """A successful cache still yields the local path (no behaviour regression)."""
    monkeypatch.setattr(
        "plugins.platforms.whatsapp.adapter.cache_image_from_url",
        AsyncMock(return_value="/tmp/hermes/cache/images/img_abc.jpg"),
    )
    adapter = _make_adapter()

    urls, mimes = await adapter._collect_bridge_media(
        {"mediaUrls": ["https://mmg.whatsapp.net/v/t62.7118-24/abc.jpg"]},
        MessageType.PHOTO,
    )

    assert urls == ["/tmp/hermes/cache/images/img_abc.jpg"]
    assert mimes == ["image/jpeg"]


@pytest.mark.asyncio
async def test_allowed_bridge_local_path_still_accepted(monkeypatch, tmp_path):
    """A bridge-written local absolute path inside the cache dir still passes (no regression)."""
    monkeypatch.setattr(
        "plugins.platforms.whatsapp.adapter._is_allowed_bridge_path", lambda p: True
    )
    adapter = _make_adapter()
    # tmp_path, not "/tmp/...": the branch under test is gated on os.path.isabs, which is False
    # for a POSIX literal on Windows.
    local_audio = str(tmp_path / "aud_xyz.ogg")

    urls, mimes = await adapter._collect_bridge_media(
        {"mediaUrls": [local_audio]}, MessageType.VOICE
    )

    assert urls == [local_audio]
    assert mimes == ["audio/ogg"]
