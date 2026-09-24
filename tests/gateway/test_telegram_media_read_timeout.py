"""Every Telegram media upload passes the long media read timeout.

Telegram processes an uploaded photo/video/document server-side before it
answers, so the wait for the response is unrelated to how fast the bytes went
out and can outlast the short read timeout the rest of the Bot API is tuned
for. ``_MEDIA_SEND_READ_TIMEOUT`` is the longer budget; a media send that
misses it fails on a slow upload and falls back to posting the bare URL as
text, so the picture never arrives as a picture.

``send_image`` was the gap: both of its ``send_photo`` calls -- the URL send
and the byte-upload fallback that handles up to 10MB -- went out on the short
timeout. Both are driven for real below and asserted on the ``read_timeout``
that actually reaches the Bot API.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from gateway.config import PlatformConfig  # noqa: E402
from plugins.platforms.telegram import adapter as tg  # noqa: E402
from plugins.platforms.telegram.adapter import TelegramAdapter  # noqa: E402


@pytest.fixture
def adapter():
    a = TelegramAdapter(PlatformConfig(enabled=True, token="fake-token"))
    a._bot = MagicMock()
    a._metadata_thread_id = lambda metadata: None
    a._thread_kwargs_for_send = lambda *args, **kwargs: {}
    a._notification_kwargs = lambda metadata: {}
    a._reply_to_message_id_for_send = lambda *args, **kwargs: None

    async def _direct(fn, payload, *args, **kwargs):
        return await fn(**payload)

    a._send_with_dm_topic_reply_anchor_retry = _direct
    return a


def _stub_download(monkeypatch, size: int, captured_headers: dict | None = None):
    """Make the fallback's SSRF-safe client return ``size`` bytes.

    ``captured_headers``, when given, is filled in with whatever headers the
    fallback's ``client.get()`` call sends, so a test can assert on them.
    """

    class _Resp:
        content = b"x" * size

        def raise_for_status(self):
            return None

    class _Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def get(self, url, headers=None, **kwargs):
            if captured_headers is not None:
                captured_headers.update(headers or {})
            return _Resp()

    import tools.url_safety as url_safety

    monkeypatch.setattr(url_safety, "create_ssrf_safe_async_client", lambda **kw: _Client())


@pytest.mark.asyncio
async def test_send_image_url_path_uses_media_read_timeout(adapter):
    calls = []

    async def _photo(**kwargs):
        calls.append(kwargs)
        msg = MagicMock()
        msg.message_id = 1
        return msg

    adapter._bot.send_photo = AsyncMock(side_effect=_photo)

    result = await adapter.send_image("123", "https://example.com/pic.png", caption="hi")

    assert result.success
    assert calls[0]["read_timeout"] == tg._MEDIA_SEND_READ_TIMEOUT


@pytest.mark.asyncio
async def test_send_image_upload_fallback_uses_media_read_timeout(adapter, monkeypatch):
    """The >5MB fallback is the slowest send in the file — it needs it most."""
    _stub_download(monkeypatch, 8 * 1024 * 1024)
    calls = []

    async def _photo(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise RuntimeError("Photo too big for a URL send")
        msg = MagicMock()
        msg.message_id = 2
        return msg

    adapter._bot.send_photo = AsyncMock(side_effect=_photo)

    result = await adapter.send_image("123", "https://example.com/big.png", caption="hi")

    assert result.success
    assert len(calls) == 2, "expected the byte-upload fallback to run"
    upload = calls[1]
    assert isinstance(upload["photo"], (bytes, bytearray))
    assert upload["read_timeout"] == tg._MEDIA_SEND_READ_TIMEOUT


@pytest.mark.asyncio
async def test_send_image_upload_fallback_sends_identifying_user_agent(adapter, monkeypatch):
    """The byte-upload fallback must not download with the bare httpx default
    UA: bot-detecting CDNs (upload.wikimedia.org confirmed) serve
    ``python-httpx/x`` a 403 while the same URL works from Telegram's own
    fetch and from a browser seconds earlier (#89260). The fallback must
    send the same identifying UA already used for direct downloads
    elsewhere (``tools.url_safety.DEFAULT_USER_AGENT``), not a browser-spoof
    string — Wikimedia's UA policy asks clients to identify themselves
    rather than impersonate a browser.
    """
    from tools.url_safety import DEFAULT_USER_AGENT

    headers: dict = {}
    _stub_download(monkeypatch, 8 * 1024 * 1024, captured_headers=headers)
    calls = []

    async def _photo(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise RuntimeError("Photo too big for a URL send")
        msg = MagicMock()
        msg.message_id = 4
        return msg

    adapter._bot.send_photo = AsyncMock(side_effect=_photo)

    result = await adapter.send_image(
        "123", "https://upload.wikimedia.org/wikipedia/commons/x/x.jpg", caption="hi"
    )

    assert result.success and len(calls) == 2
    assert headers.get("User-Agent") == DEFAULT_USER_AGENT
    assert "python-httpx" not in headers.get("User-Agent", "")
