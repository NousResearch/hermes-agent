"""Tests for Bot Framework attachment auth (Bearer token) on inbound media.

Bot Framework attachment URLs (``https://smba.trafficmanager.net/.../v3/
attachments/.../views/original``) require an ``Authorization: Bearer ...``
header — without one the service returns 401 Unauthorized. The shared
``cache_image_from_url`` / ``cache_audio_from_url`` / ``cache_video_from_url``
helpers do not (and should not) carry per-platform auth.

The adapter therefore detects BF-hosted URLs by host, fetches the bytes
ourselves with the SDK-managed bearer token (``self._app._get_bot_token``)
and routes raw bytes into the ``cache_*_from_bytes`` helpers. Only the
Teams adapter knows how to mint that token, so the fix lives here, not in
``gateway/platforms/base.py``.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.teams.adapter import (
    TeamsAdapter,
    _is_bf_attachment_url,
)


# ---------------------------------------------------------------------------
# Pure unit tests — _is_bf_attachment_url
# ---------------------------------------------------------------------------


def test_is_bf_attachment_url_recognises_smba_trafficmanager():
    url = (
        "https://smba.trafficmanager.net/amer/def57894-de93-412b-8ce2-87faabe44453/"
        "v3/attachments/0-eus-d6-0d3d6109e0e45c3a5a11f8cd7c3731bf/views/original"
    )
    assert _is_bf_attachment_url(url) is True


def test_is_bf_attachment_url_rejects_graph_url():
    url = (
        "https://graph.microsoft.com/v1.0/teams/T/channels/C/messages/M/"
        "hostedContents/abc/$value"
    )
    assert _is_bf_attachment_url(url) is False


def test_is_bf_attachment_url_rejects_sharepoint_url():
    url = "https://contoso.sharepoint.com/sites/X/Documents/file.pdf?tempauth=Y"
    assert _is_bf_attachment_url(url) is False


def test_is_bf_attachment_url_handles_empty_or_none():
    assert _is_bf_attachment_url("") is False
    assert _is_bf_attachment_url(None) is False  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Adapter fixture — minimal, no connect()
# ---------------------------------------------------------------------------


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
    # Stand in for the SDK-built App with its token machinery.
    fake_app = MagicMock()
    fake_token = MagicMock()
    fake_token.__str__ = lambda self: "FAKE.JWT.TOKEN"  # type: ignore[assignment]
    fake_app._get_bot_token = AsyncMock(return_value=fake_token)
    a._app = fake_app
    return a


_BF_URL = (
    "https://smba.trafficmanager.net/amer/def57894-de93-412b-8ce2-87faabe44453/"
    "v3/attachments/0-eus-d6-0d3d6109e0e45c3a5a11f8cd7c3731bf/views/original"
)


# ---------------------------------------------------------------------------
# _fetch_bf_attachment_bytes
# ---------------------------------------------------------------------------


class _FakeResp:
    def __init__(self, status: int, body: bytes = b""):
        self.status = status
        self._body = body

    async def read(self) -> bytes:
        return self._body

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def raise_for_status(self):
        if self.status >= 400:
            from aiohttp import ClientResponseError
            from aiohttp.client_reqrep import RequestInfo
            from yarl import URL

            raise ClientResponseError(
                request_info=RequestInfo(URL(_BF_URL), "GET", {}, URL(_BF_URL)),
                history=(),
                status=self.status,
                message="err",
            )


class _FakeSession:
    """aiohttp.ClientSession stand-in capturing the headers used on get()."""

    def __init__(self, response: _FakeResp):
        self._response = response
        self.captured_headers: dict | None = None
        self.captured_url: str | None = None

    def get(self, url, headers=None):
        self.captured_url = url
        self.captured_headers = headers or {}
        return self._response

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


@pytest.mark.asyncio
async def test_fetch_bf_attachment_bytes_attaches_bearer_header(adapter, monkeypatch):
    fake_resp = _FakeResp(200, body=b"PNGDATA")
    fake_session = _FakeSession(fake_resp)

    import aiohttp

    monkeypatch.setattr(
        aiohttp, "ClientSession", lambda *a, **kw: fake_session
    )

    data = await adapter._fetch_bf_attachment_bytes(_BF_URL)
    assert data == b"PNGDATA"
    assert fake_session.captured_url == _BF_URL
    assert fake_session.captured_headers is not None
    assert fake_session.captured_headers.get("Authorization") == "Bearer FAKE.JWT.TOKEN"
    adapter._app._get_bot_token.assert_awaited_once()


@pytest.mark.asyncio
async def test_fetch_bf_attachment_bytes_returns_none_on_401(adapter, monkeypatch):
    fake_resp = _FakeResp(401)
    fake_session = _FakeSession(fake_resp)
    import aiohttp

    monkeypatch.setattr(aiohttp, "ClientSession", lambda *a, **kw: fake_session)

    data = await adapter._fetch_bf_attachment_bytes(_BF_URL)
    assert data is None


@pytest.mark.asyncio
async def test_fetch_bf_attachment_bytes_returns_none_when_no_app(monkeypatch):
    cfg = PlatformConfig(
        enabled=True,
        extra={
            "client_id": "fake-client",
            "client_secret": "fake-secret",
            "tenant_id": "fake-tenant",
        },
    )
    a = TeamsAdapter(cfg)
    # _app is None here — nothing to mint a token from.
    assert a._app is None
    out = await a._fetch_bf_attachment_bytes(_BF_URL)
    assert out is None


# ---------------------------------------------------------------------------
# Wiring tests — image / audio / video paths in _on_message attachment loop
# ---------------------------------------------------------------------------


def _make_activity(attachments):
    """Build a minimal MessageActivity-shaped object for _on_message."""
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


@pytest.mark.asyncio
async def test_image_branch_uses_bf_fetch_when_url_is_bf(adapter, monkeypatch):
    """BF image URL should go through _fetch_bf_attachment_bytes + cache_image_from_bytes."""
    fetch_spy = AsyncMock(return_value=b"PNGBYTES")
    adapter._fetch_bf_attachment_bytes = fetch_spy  # type: ignore[method-assign]

    captured = {}

    def fake_cache_bytes(data, ext):
        captured["data"] = data
        captured["ext"] = ext
        return "/cache/img.jpg"

    import gateway.platforms.base as base
    monkeypatch.setattr(base, "cache_image_from_bytes", fake_cache_bytes)

    # cache_image_from_url MUST NOT be called for BF URLs.
    bad_url_helper = AsyncMock(side_effect=AssertionError("should not call cache_image_from_url for BF urls"))
    import plugins.platforms.teams.adapter as adapter_mod
    monkeypatch.setattr(adapter_mod, "cache_image_from_url", bad_url_helper)

    ctx = MagicMock()
    ctx.activity = _make_activity([_make_attachment("image/png", _BF_URL, name="paste.png")])

    # Stub send() so on_message's downstream path doesn't blow up
    adapter.send = AsyncMock()  # type: ignore[method-assign]

    await adapter._on_message(ctx)

    fetch_spy.assert_awaited_once_with(_BF_URL)
    assert captured["data"] == b"PNGBYTES"


@pytest.mark.asyncio
async def test_image_branch_uses_url_helper_when_url_is_not_bf(adapter, monkeypatch):
    """Non-BF image URL should keep using cache_image_from_url unchanged."""
    fetch_spy = AsyncMock(side_effect=AssertionError("should not BF-fetch for non-BF urls"))
    adapter._fetch_bf_attachment_bytes = fetch_spy  # type: ignore[method-assign]

    url_helper = AsyncMock(return_value="/cache/img.jpg")
    import plugins.platforms.teams.adapter as adapter_mod
    monkeypatch.setattr(adapter_mod, "cache_image_from_url", url_helper)

    ctx = MagicMock()
    ctx.activity = _make_activity([
        _make_attachment("image/png", "https://example.com/foo.png", name="foo.png")
    ])
    adapter.send = AsyncMock()  # type: ignore[method-assign]

    await adapter._on_message(ctx)

    url_helper.assert_awaited_once_with("https://example.com/foo.png")
    fetch_spy.assert_not_awaited()


@pytest.mark.asyncio
async def test_audio_branch_uses_bf_fetch_when_url_is_bf(adapter, monkeypatch):
    fetch_spy = AsyncMock(return_value=b"OGGDATA")
    adapter._fetch_bf_attachment_bytes = fetch_spy  # type: ignore[method-assign]

    captured = {}

    def fake_cache_bytes(data, ext):
        captured["data"] = data
        captured["ext"] = ext
        return "/cache/voice.ogg"

    import gateway.platforms.base as base
    monkeypatch.setattr(base, "cache_audio_from_bytes", fake_cache_bytes)

    import plugins.platforms.teams.adapter as adapter_mod
    bad_url_helper = AsyncMock(side_effect=AssertionError("should not call cache_audio_from_url for BF urls"))
    monkeypatch.setattr(adapter_mod, "cache_audio_from_url", bad_url_helper)

    ctx = MagicMock()
    ctx.activity = _make_activity([_make_attachment("audio/ogg", _BF_URL, name="voice.ogg")])
    adapter.send = AsyncMock()  # type: ignore[method-assign]

    await adapter._on_message(ctx)

    fetch_spy.assert_awaited_once_with(_BF_URL)
    assert captured["data"] == b"OGGDATA"
    assert captured["ext"] == ".ogg"


@pytest.mark.asyncio
async def test_video_branch_uses_bf_fetch_when_url_is_bf(adapter, monkeypatch):
    fetch_spy = AsyncMock(return_value=b"MP4DATA")
    adapter._fetch_bf_attachment_bytes = fetch_spy  # type: ignore[method-assign]

    captured = {}

    def fake_cache_bytes(data, ext):
        captured["data"] = data
        captured["ext"] = ext
        return "/cache/clip.mp4"

    import gateway.platforms.base as base
    monkeypatch.setattr(base, "cache_video_from_bytes", fake_cache_bytes)

    import plugins.platforms.teams.adapter as adapter_mod
    bad_url_helper = AsyncMock(side_effect=AssertionError("should not call cache_video_from_url for BF urls"))
    monkeypatch.setattr(adapter_mod, "cache_video_from_url", bad_url_helper)

    ctx = MagicMock()
    ctx.activity = _make_activity([_make_attachment("video/mp4", _BF_URL, name="clip.mp4")])
    adapter.send = AsyncMock()  # type: ignore[method-assign]

    await adapter._on_message(ctx)

    fetch_spy.assert_awaited_once_with(_BF_URL)
    assert captured["data"] == b"MP4DATA"
    assert captured["ext"] == ".mp4"
