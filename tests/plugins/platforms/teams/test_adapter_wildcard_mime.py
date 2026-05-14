"""Tests for wildcard / missing MIME subtypes on inbound BF attachments.

Teams sometimes posts attachments with ``content_type="image/*"`` (literal
asterisk) — particularly when a user pastes an image inline rather than
attaching a typed file. Naively splitting that on ``"/"`` produced an extension
of ``".*"`` which then leaked into the cache filename
(``img_xxx.*``) and broke downstream tooling that opens files by extension.

Fix: when the subtype is empty, ``"*"``, or otherwise unrecognised, sniff the
fetched bytes via magic numbers and pick a sensible extension. Fall back to
the kind's default (``.jpg`` / ``.ogg`` / ``.mp4``) only as a last resort.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.teams.adapter import (
    TeamsAdapter,
    _resolve_media_ext,
)


# ---------------------------------------------------------------------------
# _resolve_media_ext — magic-byte sniffer
# ---------------------------------------------------------------------------

# Magic-byte fixtures. Only the prefix needs to match the sniffer — payload
# beyond the header is irrelevant for these unit tests.
_PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32
_JPG = b"\xff\xd8\xff\xe0" + b"\x00" * 32
_GIF87 = b"GIF87a" + b"\x00" * 32
_GIF89 = b"GIF89a" + b"\x00" * 32
_WEBP = b"RIFF" + b"\x00\x00\x00\x00" + b"WEBP" + b"\x00" * 32
_OGG = b"OggS" + b"\x00" * 32
_MP3_ID3 = b"ID3" + b"\x00" * 32
_MP3_FRAME = b"\xff\xfb" + b"\x00" * 32
_WAV = b"RIFF" + b"\x00\x00\x00\x00" + b"WAVE" + b"\x00" * 32
_FLAC = b"fLaC" + b"\x00" * 32
_MP4 = b"\x00\x00\x00\x20ftypisom" + b"\x00" * 32
_MOV = b"\x00\x00\x00\x14ftypqt  " + b"\x00" * 32
_WEBM = b"\x1aE\xdf\xa3" + b"\x00" * 32


@pytest.mark.parametrize(
    "subtype,data,kind,expected",
    [
        # explicit, well-known subtypes pass straight through
        ("png", b"", "image", ".png"),
        ("jpeg", b"", "image", ".jpg"),  # jpeg → jpg normalisation
        ("gif", b"", "image", ".gif"),
        ("webp", b"", "image", ".webp"),
        ("ogg", b"", "audio", ".ogg"),
        ("mp3", b"", "audio", ".mp3"),
        ("mp4", b"", "video", ".mp4"),

        # wildcard / empty subtype → sniff bytes
        ("*", _PNG, "image", ".png"),
        ("*", _JPG, "image", ".jpg"),
        ("*", _GIF87, "image", ".gif"),
        ("*", _GIF89, "image", ".gif"),
        ("*", _WEBP, "image", ".webp"),
        ("", _PNG, "image", ".png"),
        ("*", _OGG, "audio", ".ogg"),
        ("*", _MP3_ID3, "audio", ".mp3"),
        ("*", _MP3_FRAME, "audio", ".mp3"),
        ("*", _WAV, "audio", ".wav"),
        ("*", _FLAC, "audio", ".flac"),
        ("*", _MP4, "video", ".mp4"),
        ("*", _MOV, "video", ".mov"),
        ("*", _WEBM, "video", ".webm"),

        # last-resort defaults when sniff also fails
        ("*", b"\x00\x00\x00\x00\x00", "image", ".jpg"),
        ("*", b"\x00\x00\x00\x00\x00", "audio", ".ogg"),
        ("*", b"\x00\x00\x00\x00\x00", "video", ".mp4"),
        ("*", b"", "image", ".jpg"),
    ],
)
def test_resolve_media_ext(subtype, data, kind, expected):
    assert _resolve_media_ext(subtype, data, kind) == expected


def test_resolve_media_ext_strips_codec_params():
    # MIME params like ``image/jpeg; codecs=...`` would arrive after the caller
    # already split on ``";"``; we still defend against accidental leakage.
    assert _resolve_media_ext("jpeg ", b"", "image") == ".jpg"


def test_resolve_media_ext_unknown_subtype_falls_through_to_default():
    # Unknown but non-wildcard subtype: trust it for image/audio (codecs vary
    # too widely to whitelist) — caller validated category via prefix.
    assert _resolve_media_ext("heic", b"", "image") == ".heic"


# ---------------------------------------------------------------------------
# Adapter fixture (mirrors test_adapter_inbound_bf_auth.py)
# ---------------------------------------------------------------------------


_BF_URL = (
    "https://smba.trafficmanager.net/amer/def57894-de93-412b-8ce2-87faabe44453/"
    "v3/attachments/0-eus-d6-0d3d6109e0e45c3a5a11f8cd7c3731bf/views/original"
)


@pytest.fixture
def adapter(monkeypatch) -> TeamsAdapter:
    for var in ("TEAMS_CLIENT_ID", "TEAMS_CLIENT_SECRET", "TEAMS_TENANT_ID"):
        monkeypatch.delenv(var, raising=False)
    cfg = PlatformConfig(
        enabled=True,
        extra={
            "client_id": "fake-client",
            "client_secret": "fake-secret",
            "tenant_id": "fake-tenant",
        },
    )
    a = TeamsAdapter(cfg)
    fake_app = MagicMock()
    fake_token = MagicMock()
    fake_token.__str__ = lambda self: "FAKE.JWT.TOKEN"  # type: ignore[assignment]
    fake_app._get_bot_token = AsyncMock(return_value=fake_token)
    a._app = fake_app
    return a


def _make_activity(attachments):
    activity = MagicMock()
    activity.text = ""
    activity.attachments = attachments
    activity.id = "ACT-1"
    activity.channel_data = None
    conv = MagicMock()
    conv.id = "chat-id"
    conv.name = ""
    conv.conversation_type = "personal"
    conv.tenant_id = "tenant-1"
    activity.conversation = conv
    from_acct = MagicMock()
    from_acct.aad_object_id = "user-1"
    from_acct.id = "user-1"
    from_acct.name = "Test User"
    activity.from_ = from_acct
    return activity


def _make_attachment(content_type, content_url, name=None):
    a = MagicMock()
    a.content_type = content_type
    a.contentType = content_type
    a.content_url = content_url
    a.contentUrl = content_url
    a.name = name
    a.content = None
    return a


# ---------------------------------------------------------------------------
# Wiring tests — _on_message must NOT cache files with literal '*' extension
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_image_wildcard_mime_with_png_bytes_sniffs_to_png(adapter, monkeypatch):
    """Regression: content_type='image/*' + PNG bytes used to cache as 'img_xxx.*'."""
    fetch_spy = AsyncMock(return_value=_PNG)
    adapter._fetch_bf_attachment_bytes = fetch_spy  # type: ignore[method-assign]

    captured = {}

    def fake_cache_bytes(data, ext):
        captured["data"] = data
        captured["ext"] = ext
        return f"/cache/img.{ext.lstrip('.')}"

    import gateway.platforms.base as base
    monkeypatch.setattr(base, "cache_image_from_bytes", fake_cache_bytes)

    ctx = MagicMock()
    ctx.activity = _make_activity([_make_attachment("image/*", _BF_URL)])
    adapter.send = AsyncMock()  # type: ignore[method-assign]

    await adapter._on_message(ctx)

    assert captured["ext"] == ".png", f"expected .png, got {captured.get('ext')!r}"
    assert "*" not in captured["ext"]


@pytest.mark.asyncio
async def test_image_wildcard_mime_with_jpeg_bytes_sniffs_to_jpg(adapter, monkeypatch):
    fetch_spy = AsyncMock(return_value=_JPG)
    adapter._fetch_bf_attachment_bytes = fetch_spy  # type: ignore[method-assign]

    captured = {}

    def fake_cache_bytes(data, ext):
        captured["ext"] = ext
        return "/cache/img.jpg"

    import gateway.platforms.base as base
    monkeypatch.setattr(base, "cache_image_from_bytes", fake_cache_bytes)

    ctx = MagicMock()
    ctx.activity = _make_activity([_make_attachment("image/*", _BF_URL)])
    adapter.send = AsyncMock()  # type: ignore[method-assign]

    await adapter._on_message(ctx)

    assert captured["ext"] == ".jpg"


@pytest.mark.asyncio
async def test_audio_wildcard_mime_with_ogg_bytes_sniffs_to_ogg(adapter, monkeypatch):
    fetch_spy = AsyncMock(return_value=_OGG)
    adapter._fetch_bf_attachment_bytes = fetch_spy  # type: ignore[method-assign]

    captured = {}

    def fake_cache_bytes(data, ext):
        captured["ext"] = ext
        return "/cache/voice.ogg"

    import gateway.platforms.base as base
    monkeypatch.setattr(base, "cache_audio_from_bytes", fake_cache_bytes)

    ctx = MagicMock()
    ctx.activity = _make_activity([_make_attachment("audio/*", _BF_URL)])
    adapter.send = AsyncMock()  # type: ignore[method-assign]

    await adapter._on_message(ctx)

    assert captured["ext"] == ".ogg"
    assert "*" not in captured["ext"]


@pytest.mark.asyncio
async def test_video_wildcard_mime_with_mp4_bytes_sniffs_to_mp4(adapter, monkeypatch):
    fetch_spy = AsyncMock(return_value=_MP4)
    adapter._fetch_bf_attachment_bytes = fetch_spy  # type: ignore[method-assign]

    captured = {}

    def fake_cache_bytes(data, ext):
        captured["ext"] = ext
        return "/cache/clip.mp4"

    import gateway.platforms.base as base
    monkeypatch.setattr(base, "cache_video_from_bytes", fake_cache_bytes)

    ctx = MagicMock()
    ctx.activity = _make_activity([_make_attachment("video/*", _BF_URL)])
    adapter.send = AsyncMock()  # type: ignore[method-assign]

    await adapter._on_message(ctx)

    assert captured["ext"] == ".mp4"
    assert "*" not in captured["ext"]


@pytest.mark.asyncio
async def test_image_wildcard_mime_unsniffable_falls_back_to_jpg(adapter, monkeypatch):
    """When sniffer can't ID the bytes, default to ``.jpg`` — never ``.*``."""
    fetch_spy = AsyncMock(return_value=b"\x00\x00\x00\x00garbage")
    adapter._fetch_bf_attachment_bytes = fetch_spy  # type: ignore[method-assign]

    captured = {}

    def fake_cache_bytes(data, ext):
        captured["ext"] = ext
        return "/cache/img.jpg"

    import gateway.platforms.base as base
    monkeypatch.setattr(base, "cache_image_from_bytes", fake_cache_bytes)

    ctx = MagicMock()
    ctx.activity = _make_activity([_make_attachment("image/*", _BF_URL)])
    adapter.send = AsyncMock()  # type: ignore[method-assign]

    await adapter._on_message(ctx)

    assert captured["ext"] == ".jpg"
    assert "*" not in captured["ext"]
