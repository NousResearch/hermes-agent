"""A Telegram inbound attachment download survives a transient network blip.

``get_file()`` was called exactly once on every inbound media path, so a single
``TimedOut`` (Telegram answering ``getFile`` late for a freshly uploaded file, or
one dropped connection) lost the voice note / audio / document / photo / sticker
outright — the user got nothing but a "could not be downloaded" note.

``_fetch_telegram_file`` retries the whole fetch (``get_file()`` plus the
download that follows) on a transient network error, and never on
``BadRequest`` (a permanent answer to what was asked, even though
python-telegram-bot makes it a ``NetworkError`` subclass) nor on
``Forbidden``/``InvalidToken`` (not network errors at all).

The end-to-end case below drives the real inbound voice path
(``_handle_media_message`` -> ``_cache_inbound_av``), so it fails on unpatched
``main`` where the download is attempted exactly once.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.event import MessageEvent, MessageType
from plugins.platforms.telegram.adapter import TelegramAdapter

try:
    from telegram.error import BadRequest, Forbidden, InvalidToken, NetworkError, TimedOut
except Exception:  # pragma: no cover — PTB ships with the adapter
    BadRequest = Forbidden = InvalidToken = NetworkError = TimedOut = None


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def sleeps(monkeypatch):
    """Record the retry pauses instead of waiting them out."""
    recorded: list[float] = []

    async def _sleep(delay, *args, **kwargs):
        recorded.append(delay)

    monkeypatch.setattr(asyncio, "sleep", _sleep)
    return recorded


@pytest.fixture
def adapter():
    config = PlatformConfig(enabled=True, token="fake-token")
    a = TelegramAdapter(config)
    a.handle_message = AsyncMock()
    a._is_callback_user_authorized = lambda user_id, **_kw: True
    return a


@pytest.fixture(autouse=True)
def _redirect_cache(tmp_path, monkeypatch):
    """Point the media caches at tmp_path so the tests never touch ~/.hermes."""
    monkeypatch.setattr("gateway.platforms.base.DOCUMENT_CACHE_DIR", tmp_path / "doc_cache")
    monkeypatch.setattr("gateway.platforms.base.VIDEO_CACHE_DIR", tmp_path / "video_cache")
    monkeypatch.setattr("gateway.platforms.base.AUDIO_CACHE_DIR", tmp_path / "audio_cache")


def _make_file_obj(data: bytes = b"payload"):
    """A mock Telegram File whose download returns ``data``."""
    file_obj = AsyncMock()
    file_obj.download_as_bytearray = AsyncMock(return_value=bytearray(data))
    file_obj.file_path = "voice/file_1.oga"
    return file_obj


def _make_source(file_obj=None, failures=None):
    """A mock attachment source.

    ``failures`` is a list of exceptions raised by consecutive ``get_file()`` calls;
    once it is exhausted the source succeeds.
    """
    source = SimpleNamespace()
    source.calls = 0
    pending = list(failures or [])

    async def _get_file(*args, **kwargs):
        source.calls += 1
        if pending:
            raise pending.pop(0)
        return file_obj or _make_file_obj()

    source.get_file = _get_file
    return source


def _timed_out(cause=None):
    """PTB's ``TimedOut`` — what a stalled read actually raises."""
    err = TimedOut()
    if cause is not None:
        err.__cause__ = cause
    return err


# ---------------------------------------------------------------------------
# The end-to-end inbound voice path
# ---------------------------------------------------------------------------


class TestInboundVoiceSurvivesTransientTimeout:
    @pytest.mark.asyncio
    async def test_voice_is_cached_after_one_stalled_fetch(self, adapter, sleeps):
        """The real inbound path must survive one stalled ``getFile``.

        Before the fix this raised out of ``_cache_inbound_av`` on the first
        attempt, the user was told the attachment could not be downloaded, and
        ``event.media_urls`` stayed empty.
        """
        source = _make_source(failures=[_timed_out()])
        msg = SimpleNamespace(
            voice=source,
            audio=None,
            video=None,
            video_note=None,
            document=None,
            sticker=None,
            photo=None,
            reply_text=AsyncMock(),
        )
        event = MessageEvent(text="", message_type=MessageType.VOICE)

        await adapter._cache_inbound_av(msg, event, source, "voice message", "voice", ".ogg", "audio/ogg")

        assert source.calls == 2, "the stalled fetch must be retried"
        assert event.media_urls, "the voice note must be cached, not dropped"
        assert event.media_types == ["audio/ogg"]

    @pytest.mark.asyncio
    async def test_voice_failure_still_surfaces_to_the_user(self, adapter, sleeps):
        """When every attempt fails, the existing user-facing notice must still run."""
        source = _make_source(failures=[_timed_out() for _ in range(10)])
        msg = SimpleNamespace(
            voice=source,
            audio=None,
            video=None,
            video_note=None,
            document=None,
            sticker=None,
            photo=None,
            reply_text=AsyncMock(),
        )
        event = MessageEvent(text="", message_type=MessageType.VOICE)

        await adapter._cache_inbound_av(msg, event, source, "voice message", "voice", ".ogg", "audio/ogg")

        assert source.calls == 3, "retries must be bounded"
        assert not event.media_urls, "nothing may be cached when every attempt fails"
        # The user is told to retry rather than silently losing the attachment.
        assert msg.reply_text.await_count >= 1


# ---------------------------------------------------------------------------
# The classifier
# ---------------------------------------------------------------------------


class TestRetryClassifier:
    @pytest.mark.parametrize(
        "exc",
        [
            pytest.param(TimedOut(), id="TimedOut"),
            pytest.param(NetworkError("httpx.ReadError: boom"), id="NetworkError"),
        ],
    )
    def test_transient_errors_are_retryable(self, exc):
        from plugins.platforms.telegram.adapter import _is_retryable_telegram_fetch_error

        assert _is_retryable_telegram_fetch_error(exc) is True

    def test_httpx_read_timeout_wrapped_in_timed_out_is_retryable(self):
        """PTB hides the httpx class; the cause chain must still match."""
        import httpx

        from plugins.platforms.telegram.adapter import _is_retryable_telegram_fetch_error

        err = TimedOut()
        err.__cause__ = httpx.ReadTimeout("stalled")
        assert _is_retryable_telegram_fetch_error(err) is True

    def test_bare_httpx_read_timeout_is_retryable(self):
        import httpx

        from plugins.platforms.telegram.adapter import _is_retryable_telegram_fetch_error

        assert _is_retryable_telegram_fetch_error(httpx.ReadTimeout("stalled")) is True

    def test_bad_request_is_not_retryable(self):
        """PTB makes ``BadRequest`` a ``NetworkError`` subclass — it must be excluded."""
        from plugins.platforms.telegram.adapter import _is_retryable_telegram_fetch_error

        assert issubclass(BadRequest, NetworkError), "PTB hierarchy assumption"
        assert _is_retryable_telegram_fetch_error(BadRequest("nope")) is False

    def test_forbidden_and_invalid_token_are_not_retryable(self):
        from plugins.platforms.telegram.adapter import _is_retryable_telegram_fetch_error

        assert _is_retryable_telegram_fetch_error(Forbidden("x")) is False
        assert _is_retryable_telegram_fetch_error(InvalidToken("x")) is False

    def test_pool_timeout_is_not_retryable(self):
        """A pool timeout means the request was never sent — the pool is exhausted."""
        import httpx

        from plugins.platforms.telegram.adapter import _is_retryable_telegram_fetch_error

        assert _is_retryable_telegram_fetch_error(httpx.PoolTimeout("no slots")) is False

    def test_unrelated_error_is_not_retryable(self):
        from plugins.platforms.telegram.adapter import _is_retryable_telegram_fetch_error

        assert _is_retryable_telegram_fetch_error(ValueError("bad input")) is False
