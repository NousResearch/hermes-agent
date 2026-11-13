"""A native Telegram media upload repeats only a failure Telegram provably never posted (#62079).

PTB wraps every httpx transport error as ``NetworkError(...) from <httpx error>``; Telegram's own
5xx JSON answer is an unchained ``NetworkError('Bad Gateway')``. A connect failure and that 5xx
answer are repeated as the same request; an error after the upload left (read reset, timeout) may
already be in the chat, so it is never resent in any shape and never reported as a hard failure.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from hermes_agent.gateway.config import PlatformConfig
from hermes_agent.gateway.platforms.base import BasePlatformAdapter, SendResult
from hermes_agent.plugins.platforms.telegram.adapter import TelegramAdapter


def _network_error(cause=None, text=None):
    from telegram.error import NetworkError
    err = NetworkError(text or f"httpx.{type(cause).__name__}: {cause}")
    err.__cause__ = cause
    return err


def _connect_error():
    return _network_error(httpx.ConnectError("All connection attempts failed"))


def _bad_gateway():
    return _network_error(text="Bad Gateway")


def _read_error():
    return _network_error(httpx.ReadError("Server disconnected without sending a response."))


@pytest.fixture
def adapter(monkeypatch):
    monkeypatch.setattr("asyncio.sleep", AsyncMock())
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="123456:test-token", extra={}))
    adapter._bot = MagicMock()
    adapter._compress_image_to_jpeg = lambda _path: None
    for name in ("send_document", "send_image_file", "send_video"):
        monkeypatch.setattr(BasePlatformAdapter, name, AsyncMock(
            return_value=SendResult(success=False, error="notice")))
    return adapter


def _uploads(adapter, method, outcomes):
    """Bot method that fails with ``outcomes`` in order, then posts; records the bytes each try read."""
    reads = []

    async def upload(**kwargs):
        reads.append(kwargs[method.removeprefix("send_")].read())
        outcome = outcomes[len(reads) - 1] if len(reads) <= len(outcomes) else None
        if outcome is not None:
            raise outcome
        return SimpleNamespace(message_id=77)

    setattr(adapter._bot, method, upload)
    return reads


@pytest.mark.asyncio
@pytest.mark.parametrize("first", [_connect_error, _bad_gateway], ids=["connect-error", "telegram-502"])
async def test_unsent_upload_is_repeated_and_delivered(adapter, tmp_path, first):
    path = tmp_path / "package.zip"
    path.write_bytes(b"zip-bytes")
    reads = _uploads(adapter, "send_document", [first()])

    result = await adapter.send_document("123", str(path))

    assert result.success is True and result.message_id == "77"
    assert reads == [b"zip-bytes", b"zip-bytes"]  # the same upload again, rewound
    BasePlatformAdapter.send_document.assert_not_awaited()  # no "Couldn't deliver" notice


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [_connect_error, _bad_gateway], ids=["connect-error", "telegram-502-outage"])
async def test_unsent_retries_are_bounded(adapter, tmp_path, error):
    """Telegram's 5xx JSON on every try still means nothing was posted: a definite failure with the
    notice, never reported as maybe-delivered."""
    path = tmp_path / "package.zip"
    path.write_bytes(b"zip-bytes")
    reads = _uploads(adapter, "send_document", [error() for _ in range(5)])

    result = await adapter.send_document("123", str(path))

    assert len(reads) == 3
    assert result.success is False and not (result.raw_response or {}).get("ambiguous")
    BasePlatformAdapter.send_document.assert_awaited_once()  # the notice, once the budget is spent


@pytest.mark.asyncio
@pytest.mark.parametrize("method,call", [
    ("send_document", lambda a, p: a.send_document("123", p)),
    ("send_photo", lambda a, p: a.send_image_file("123", p)),
    ("send_video", lambda a, p: a.send_video("123", p)),
], ids=["document", "photo", "video"])
async def test_upload_lost_after_the_request_left_is_never_resent(adapter, tmp_path, method, call):
    path = tmp_path / ("clip.mp4" if method == "send_video" else "chart.png")
    path.write_bytes(b"media-bytes")
    reads = _uploads(adapter, method, [_read_error()])
    adapter._bot.send_document = adapter._bot.send_document if method == "send_document" else AsyncMock()

    result = await call(adapter, str(path))

    assert len(reads) == 1
    assert result.success is True and result.raw_response.get("ambiguous") is True  # unknown, not failed
    if method != "send_document":
        adapter._bot.send_document.assert_not_awaited()  # no re-upload as a document
    for name in ("send_document", "send_image_file", "send_video"):
        getattr(BasePlatformAdapter, name).assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["send_image", "send_animation"])
async def test_url_media_lost_after_the_request_left_falls_back_to_nothing(adapter, monkeypatch, method):
    """A URL photo whose request may have posted is not downloaded and re-uploaded, and a lost
    animation is not resent as a photo: one request, no notice."""
    calls = []

    async def lost(**kwargs):
        calls.append(kwargs)
        raise _read_error()

    adapter._bot.send_photo = lost
    adapter._bot.send_animation = lost
    monkeypatch.setattr("hermes_agent.tools.url_safety.is_safe_url", lambda _url: True)
    base_send_image = AsyncMock()
    monkeypatch.setattr(BasePlatformAdapter, "send_image", base_send_image)
    monkeypatch.setattr("hermes_agent.tools.url_safety.create_ssrf_safe_async_client", MagicMock(
        side_effect=AssertionError("the photo must not be downloaded for a re-upload")))

    result = await getattr(adapter, method)("123", "https://example.com/a.gif")

    assert len(calls) == 1
    assert result.success is False and result.raw_response.get("delivery_uncertain") is True
    assert result.raw_response.get("failure_notice_sent") is True  # the dispatcher posts no notice
    base_send_image.assert_not_awaited()


@pytest.mark.asyncio
async def test_local_error_before_the_request_is_still_a_failure(adapter, tmp_path, monkeypatch):
    """An OSError raised before any request (the file vanished between check and open) never left:
    it is a plain failure with its notice, not a maybe-delivered upload."""
    path = tmp_path / "package.zip"
    path.write_bytes(b"zip-bytes")
    adapter._bot.send_document = AsyncMock()
    monkeypatch.setattr("builtins.open", MagicMock(side_effect=PermissionError("denied")))

    result = await adapter.send_document("123", str(path))

    assert result.success is False
    adapter._bot.send_document.assert_not_awaited()
    BasePlatformAdapter.send_document.assert_awaited_once()


@pytest.mark.asyncio
async def test_standalone_sender_repeats_unsent_and_never_resends_a_lost_upload(tmp_path, monkeypatch):
    """The out-of-process sender (send_message tool, cron) follows the same policy."""
    from hermes_agent.plugins.platforms.telegram import standalone
    monkeypatch.setattr("asyncio.sleep", AsyncMock())
    path = tmp_path / "package.zip"
    path.write_bytes(b"zip-bytes")
    bot = SimpleNamespace()
    reads = []

    def upload_with(outcomes):
        reads.clear()

        async def upload(**kwargs):
            reads.append(kwargs["document"].read())
            if len(reads) <= len(outcomes):
                raise outcomes[len(reads) - 1]
            return SimpleNamespace(message_id=5)
        bot.send_document = upload

    kw = dict(caption=None, parse_mode=None, has_html=False, thread_kwargs={}, force_document=False)
    upload_with([_connect_error()])
    msg = await standalone._telegram_send_one_media(bot, 123, str(path), False, **kw)
    assert msg.message_id == 5 and reads == [b"zip-bytes", b"zip-bytes"]

    upload_with([_read_error()])
    with pytest.raises(Exception):
        await standalone._telegram_send_one_media(bot, 123, str(path), False, **kw)
    assert len(reads) == 1


@pytest.mark.asyncio
async def test_album_photo_lost_after_the_request_left_is_reported_ambiguous(adapter, monkeypatch):
    """A refused album falls back photo by photo; a photo whose request may have posted is reported
    as possibly delivered (``ambiguous``), never as a failure a caller would answer with a resend."""
    from telegram.error import BadRequest

    calls = []

    async def lost(**kwargs):
        calls.append(kwargs)
        raise _read_error()

    adapter._bot.send_media_group = AsyncMock(side_effect=BadRequest("Wrong file identifier/http url specified"))
    adapter._bot.send_photo = lost
    monkeypatch.setattr("hermes_agent.tools.url_safety.is_safe_url", lambda _url: True)

    result = await adapter.send_multiple_images("123", [("https://example.com/a.png", ""), ("https://example.com/b.png", "")])

    assert len(calls) == 2  # one request per photo, none resent
    assert result.success is True and result.raw_response["ambiguous"] is True
    assert result.raw_response["failed"] == []


@pytest.mark.asyncio
@pytest.mark.parametrize("images", [
    [("https://example.com/a.gif", "")],
    [("https://example.com/a.gif", ""), ("https://example.com/b.png", "")],
])
async def test_a_gif_whose_receipt_was_lost_keeps_the_batch_ambiguous(adapter, monkeypatch, images):
    """A GIF upload that left but got no answer may be in the chat: the batch (GIF-only, or GIF
    beside a still photo that posted) reports it possibly delivered, never a definite failure."""
    calls = []

    async def lost(**kwargs):
        calls.append(kwargs)
        raise _read_error()

    adapter._bot.send_animation = lost
    adapter._bot.send_media_group = AsyncMock(return_value=[SimpleNamespace(message_id=9)])
    adapter._bot.send_photo = AsyncMock(return_value=SimpleNamespace(message_id=9))
    monkeypatch.setattr("hermes_agent.tools.url_safety.is_safe_url", lambda _url: True)

    result = await adapter.send_multiple_images("123", images)

    assert len(calls) == 1  # never resent
    assert result.success is True and result.raw_response["ambiguous"] is True
