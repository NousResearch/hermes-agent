"""Regression tests for Telegram media-caption MarkdownV2 rendering (#32839).

Background
----------
``send_message`` runs the agent's response text through :meth:`format_message`
and sends it with ``parse_mode=MARKDOWN_V2`` so headings, bold, and code spans
render correctly on every Telegram client.  The media senders
(``send_image_file``, ``send_document``, ``send_video``, ``send_image``,
``send_voice``, ``send_animation``, ``send_multiple_images``) historically
forwarded ``caption`` raw with no ``parse_mode``, so markdown attached to a
photo or document (e.g. a research "Detailed Analysis" section sent alongside
a screenshot) showed up as literal ``### Heading`` / ``**bold**`` on mobile.

These tests pin the post-fix behaviour:

* :meth:`_caption_send_kwargs` returns formatted caption + ``MARKDOWN_V2``.
* :meth:`_send_media_with_caption_fallback` falls back to plain text only on
  actual MarkdownV2 parse errors and leaves other failures alone.
* Every media sender propagates the caption through the new pipeline.
* ``send_multiple_images`` propagates ``parse_mode`` into each
  ``InputMediaPhoto`` so album captions render too.
"""

from __future__ import annotations

import asyncio
import sys
from unittest.mock import AsyncMock, MagicMock

import pytest


# ---------------------------------------------------------------------------
# Telegram package mock
# ---------------------------------------------------------------------------


def _ensure_telegram_mock() -> None:
    if "telegram" in sys.modules and hasattr(sys.modules["telegram"], "__file__"):
        return
    mod = MagicMock()
    mod.ext.ContextTypes.DEFAULT_TYPE = type(None)
    mod.constants.ParseMode.MARKDOWN_V2 = "MarkdownV2"
    mod.constants.ChatType.GROUP = "group"
    mod.constants.ChatType.SUPERGROUP = "supergroup"
    mod.constants.ChatType.CHANNEL = "channel"
    mod.constants.ChatType.PRIVATE = "private"
    for name in ("telegram", "telegram.ext", "telegram.constants", "telegram.request"):
        sys.modules.setdefault(name, mod)


_ensure_telegram_mock()

from gateway.config import PlatformConfig  # noqa: E402
from gateway.platforms.telegram import TelegramAdapter  # noqa: E402
from telegram.constants import ParseMode as _ParseMode  # noqa: E402

# In environments where the real ``telegram`` package isn't installed the
# stub above gives us a MagicMock whose ``ParseMode.MARKDOWN_V2`` auto-spawns
# a sentinel.  Comparing against ``_MDV2_SENTINEL`` keeps the assertions
# robust regardless of whether the real SDK or the mock supplies the value.
_MDV2_SENTINEL = _ParseMode.MARKDOWN_V2


def _run(coro):
    return asyncio.run(coro)


@pytest.fixture
def adapter():
    config = PlatformConfig(enabled=True, token="fake-token")
    a = TelegramAdapter(config)
    a._bot = MagicMock()
    return a


# ===========================================================================
# _caption_send_kwargs (pure)
# ===========================================================================


class TestCaptionSendKwargs:
    """Pin the contract of the caption-formatting helper."""

    def test_none_returns_caption_only(self, adapter):
        """Empty caption stays empty and skips parse_mode entirely so
        callers can ``**``-unpack without overriding a default ``parse_mode``.
        """
        assert adapter._caption_send_kwargs(None) == {"caption": None}

    def test_empty_string_returns_caption_only(self, adapter):
        assert adapter._caption_send_kwargs("") == {"caption": None}

    def test_plain_text_formats_and_sets_markdown_v2(self, adapter):
        """Plain ASCII gets the MarkdownV2 parse_mode even when no formatting
        is present — this guarantees Telegram never sees ``parse_mode=None``
        for a non-empty caption that survives :meth:`format_message`.
        """
        out = adapter._caption_send_kwargs("hello world")
        assert out["parse_mode"] == _MDV2_SENTINEL
        assert out["caption"] == "hello world"

    def test_markdown_headers_converted_to_bold(self, adapter):
        """The regression case: a ``###`` heading must become MarkdownV2
        bold so mobile clients render formatting instead of raw markdown.
        """
        out = adapter._caption_send_kwargs("### Detailed Analysis")
        assert out["parse_mode"] == _MDV2_SENTINEL
        assert "*Detailed Analysis*" in out["caption"]
        assert "###" not in out["caption"]

    def test_markdown_bold_converted(self, adapter):
        out = adapter._caption_send_kwargs("**New Feature**")
        assert out["parse_mode"] == _MDV2_SENTINEL
        assert out["caption"] == "*New Feature*"

    def test_raw_caption_truncated_to_1024_chars(self, adapter):
        """Telegram caps captions at 1024 chars — the helper must enforce
        this before formatting so the raw input length is bounded.
        """
        out = adapter._caption_send_kwargs("A" * 2000)
        assert len(out["caption"]) == 1024
        assert out["parse_mode"] == _MDV2_SENTINEL

    def test_inflated_caption_falls_back_to_plain(self, adapter):
        """When MarkdownV2 escape backslashes push the formatted text past
        the 1024 ceiling, the helper drops parse_mode and returns the raw
        (truncated) caption so the send isn't rejected by Telegram.
        """
        # 1000 dots → each becomes ``\.`` after escape ≈ 2000 chars formatted.
        out = adapter._caption_send_kwargs("." * 1000)
        assert out["parse_mode"] is None
        assert out["caption"] == "." * 1000


# ===========================================================================
# _is_markdown_parse_error (pure)
# ===========================================================================


class TestIsMarkdownParseError:
    def test_classic_parse_error_message(self, adapter):
        exc = Exception("Bad Request: can't parse entities: Character '*' is reserved")
        assert adapter._is_markdown_parse_error(exc)

    def test_can_t_find_end_of_entity(self, adapter):
        exc = Exception("Bad Request: can't find end of the entity starting at byte offset 12")
        assert adapter._is_markdown_parse_error(exc)

    def test_unrelated_error_is_not_parse_error(self, adapter):
        exc = Exception("Connection reset by peer")
        assert not adapter._is_markdown_parse_error(exc)

    def test_chat_not_found_is_not_parse_error(self, adapter):
        exc = Exception("Bad Request: chat not found")
        assert not adapter._is_markdown_parse_error(exc)


# ===========================================================================
# _send_media_with_caption_fallback (integration)
# ===========================================================================


class TestSendMediaWithCaptionFallback:
    def test_first_attempt_uses_markdown_v2(self, adapter):
        """Happy path: success on the first attempt with MarkdownV2 set."""
        send_fn = AsyncMock(return_value=MagicMock(message_id=1))
        _run(
            adapter._send_media_with_caption_fallback(
                send_fn,
                {"chat_id": 1},
                caption="### Detailed Analysis",
                metadata=None,
                reply_to_message_id=None,
                media_label="photo",
            )
        )
        send_fn.assert_awaited_once()
        kwargs = send_fn.call_args.kwargs
        assert kwargs["parse_mode"] == _MDV2_SENTINEL
        assert "*Detailed Analysis*" in kwargs["caption"]

    def test_parse_error_falls_back_to_plain(self, adapter):
        """A Bad Request about parse_mode triggers exactly one retry without
        parse_mode so the user still gets the photo, just without formatting.
        """
        calls = []

        async def _flaky(**kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                raise Exception("Bad Request: can't parse entities at byte offset 5")
            return MagicMock(message_id=2)

        _run(
            adapter._send_media_with_caption_fallback(
                _flaky,
                {"chat_id": 1},
                caption="### Detailed Analysis",
                metadata=None,
                reply_to_message_id=None,
                media_label="photo",
            )
        )
        assert len(calls) == 2
        assert calls[0]["parse_mode"] == _MDV2_SENTINEL
        assert calls[1]["parse_mode"] is None
        # The plain-text retry strips MarkdownV2 markers so the reader sees
        # ``Detailed Analysis`` rather than the raw ``*Detailed Analysis*``.
        assert calls[1]["caption"] == "Detailed Analysis"

    def test_non_parse_error_is_not_retried(self, adapter):
        """Network errors must propagate so existing retry logic upstream
        keeps working — only MarkdownV2 parse errors deserve the fallback.
        """
        send_fn = AsyncMock(side_effect=Exception("Connection reset by peer"))
        with pytest.raises(Exception, match="Connection reset"):
            _run(
                adapter._send_media_with_caption_fallback(
                    send_fn,
                    {"chat_id": 1},
                    caption="### Heading",
                    metadata=None,
                    reply_to_message_id=None,
                    media_label="photo",
                )
            )
        send_fn.assert_awaited_once()

    def test_empty_caption_skips_fallback(self, adapter):
        """An empty caption can't trigger MarkdownV2 errors, so no retry
        scaffolding runs even when the send raises something parse-like.
        """
        send_fn = AsyncMock(side_effect=Exception("can't parse"))
        with pytest.raises(Exception, match="can't parse"):
            _run(
                adapter._send_media_with_caption_fallback(
                    send_fn,
                    {"chat_id": 1},
                    caption=None,
                    metadata=None,
                    reply_to_message_id=None,
                    media_label="photo",
                )
            )
        send_fn.assert_awaited_once()


# ===========================================================================
# Per-method wiring
# ===========================================================================


class TestSendImageFileCaptionParseMode:
    def test_sends_with_markdown_v2_caption(self, adapter, tmp_path):
        """End-to-end: a markdown caption on send_image_file reaches
        Telegram with ``parse_mode=MARKDOWN_V2`` and the formatted body.
        """
        img = tmp_path / "shot.png"
        img.write_bytes(b"\x89PNG" + b"\x00" * 50)
        adapter._bot.send_photo = AsyncMock(return_value=MagicMock(message_id=1))

        _run(
            adapter.send_image_file(
                chat_id="12345",
                image_path=str(img),
                caption="### Detailed Analysis\n**New Feature**",
            )
        )

        kwargs = adapter._bot.send_photo.call_args.kwargs
        assert kwargs["parse_mode"] == _MDV2_SENTINEL
        assert "*Detailed Analysis*" in kwargs["caption"]
        assert "*New Feature*" in kwargs["caption"]

    def test_empty_caption_does_not_set_parse_mode(self, adapter, tmp_path):
        """Avoid setting ``parse_mode`` for empty captions so we don't
        confuse python-telegram-bot about a non-existent payload.
        """
        img = tmp_path / "shot.png"
        img.write_bytes(b"\x89PNG" + b"\x00" * 50)
        adapter._bot.send_photo = AsyncMock(return_value=MagicMock(message_id=1))

        _run(adapter.send_image_file(chat_id="12345", image_path=str(img)))

        kwargs = adapter._bot.send_photo.call_args.kwargs
        assert kwargs["caption"] is None
        assert "parse_mode" not in kwargs


class TestSendDocumentCaptionParseMode:
    def test_document_caption_uses_markdown_v2(self, adapter, tmp_path):
        doc = tmp_path / "report.pdf"
        doc.write_bytes(b"%PDF-1.4\n")
        adapter._bot.send_document = AsyncMock(return_value=MagicMock(message_id=1))

        _run(
            adapter.send_document(
                chat_id="12345",
                file_path=str(doc),
                caption="### Summary",
            )
        )

        kwargs = adapter._bot.send_document.call_args.kwargs
        assert kwargs["parse_mode"] == _MDV2_SENTINEL
        assert "*Summary*" in kwargs["caption"]


class TestSendVideoCaptionParseMode:
    def test_video_caption_uses_markdown_v2(self, adapter, tmp_path):
        video = tmp_path / "clip.mp4"
        video.write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 50)
        adapter._bot.send_video = AsyncMock(return_value=MagicMock(message_id=1))

        _run(
            adapter.send_video(
                chat_id="12345",
                video_path=str(video),
                caption="**Demo**",
            )
        )

        kwargs = adapter._bot.send_video.call_args.kwargs
        assert kwargs["parse_mode"] == _MDV2_SENTINEL
        assert kwargs["caption"] == "*Demo*"


class TestSendVoiceCaptionParseMode:
    def test_voice_ogg_caption_uses_markdown_v2(self, adapter, tmp_path):
        audio = tmp_path / "voice.ogg"
        audio.write_bytes(b"OggS" + b"\x00" * 50)
        adapter._bot.send_voice = AsyncMock(return_value=MagicMock(message_id=1))

        _run(
            adapter.send_voice(
                chat_id="12345",
                audio_path=str(audio),
                caption="### Voice Memo",
            )
        )

        kwargs = adapter._bot.send_voice.call_args.kwargs
        assert kwargs["parse_mode"] == _MDV2_SENTINEL
        assert "*Voice Memo*" in kwargs["caption"]

    def test_audio_mp3_caption_uses_markdown_v2(self, adapter, tmp_path):
        audio = tmp_path / "song.mp3"
        audio.write_bytes(b"ID3" + b"\x00" * 50)
        adapter._bot.send_audio = AsyncMock(return_value=MagicMock(message_id=1))

        _run(
            adapter.send_voice(
                chat_id="12345",
                audio_path=str(audio),
                caption="**Track One**",
            )
        )

        kwargs = adapter._bot.send_audio.call_args.kwargs
        assert kwargs["parse_mode"] == _MDV2_SENTINEL
        assert kwargs["caption"] == "*Track One*"


class TestSendAnimationCaptionParseMode:
    def test_animation_caption_uses_markdown_v2(self, adapter):
        adapter._bot.send_animation = AsyncMock(return_value=MagicMock(message_id=1))

        _run(
            adapter.send_animation(
                chat_id="12345",
                animation_url="https://example.com/funny.gif",
                caption="### Reaction",
            )
        )

        kwargs = adapter._bot.send_animation.call_args.kwargs
        assert kwargs["parse_mode"] == _MDV2_SENTINEL
        assert "*Reaction*" in kwargs["caption"]


class TestSendImageCaptionParseMode:
    def test_url_image_caption_uses_markdown_v2(self, adapter):
        adapter._bot.send_photo = AsyncMock(return_value=MagicMock(message_id=1))

        _run(
            adapter.send_image(
                chat_id="12345",
                image_url="https://example.com/img.png",
                caption="**Wow**",
            )
        )

        kwargs = adapter._bot.send_photo.call_args.kwargs
        assert kwargs["parse_mode"] == _MDV2_SENTINEL
        assert kwargs["caption"] == "*Wow*"


class TestSendMultipleImagesCaptionParseMode:
    def test_album_member_captions_set_parse_mode(self, adapter):
        """Albums (``send_media_group``) build :class:`InputMediaPhoto`
        objects per item; each must receive ``parse_mode`` so individual
        captions render the same as standalone photos.
        """
        import telegram

        captured: list[dict] = []

        def _fake_input_media_photo(media, caption=None, parse_mode=None, **_):
            captured.append({"caption": caption, "parse_mode": parse_mode})
            return MagicMock()

        telegram.InputMediaPhoto = MagicMock(side_effect=_fake_input_media_photo)
        adapter._bot.send_media_group = AsyncMock(return_value=[MagicMock(message_id=1)])

        images = [
            ("https://x.com/a.png", "### First"),
            ("https://x.com/b.png", "**Second**"),
        ]
        _run(adapter.send_multiple_images("12345", images))

        assert len(captured) == 2
        assert captured[0]["parse_mode"] == _MDV2_SENTINEL
        assert "*First*" in captured[0]["caption"]
        assert captured[1]["parse_mode"] == _MDV2_SENTINEL
        assert captured[1]["caption"] == "*Second*"

    def test_album_member_without_alt_text_omits_parse_mode(self, adapter):
        """Empty alt-text must not leak a stray ``parse_mode=None`` kwarg
        that would clobber a future Telegram default — keep parity with
        the single-photo path.
        """
        import telegram

        captured: list[dict] = []

        def _fake_input_media_photo(media, caption=None, **kwargs):
            captured.append({"caption": caption, **kwargs})
            return MagicMock()

        telegram.InputMediaPhoto = MagicMock(side_effect=_fake_input_media_photo)
        adapter._bot.send_media_group = AsyncMock(return_value=[MagicMock(message_id=1)])

        images = [("https://x.com/a.png", "")]
        _run(adapter.send_multiple_images("12345", images))

        assert captured == [{"caption": None}]
