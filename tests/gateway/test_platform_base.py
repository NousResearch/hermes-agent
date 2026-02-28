"""Tests for gateway/platforms/base.py — MessageEvent, media extraction, message truncation."""

import os
from unittest.mock import patch

from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
)


# ---------------------------------------------------------------------------
# MessageEvent — command parsing
# ---------------------------------------------------------------------------


class TestMessageEventIsCommand:
    def test_slash_command(self):
        event = MessageEvent(text="/new")
        assert event.is_command() is True

    def test_regular_text(self):
        event = MessageEvent(text="hello world")
        assert event.is_command() is False

    def test_empty_text(self):
        event = MessageEvent(text="")
        assert event.is_command() is False

    def test_slash_only(self):
        event = MessageEvent(text="/")
        assert event.is_command() is True


class TestMessageEventGetCommand:
    def test_simple_command(self):
        event = MessageEvent(text="/new")
        assert event.get_command() == "new"

    def test_command_with_args(self):
        event = MessageEvent(text="/reset session")
        assert event.get_command() == "reset"

    def test_not_a_command(self):
        event = MessageEvent(text="hello")
        assert event.get_command() is None

    def test_command_is_lowercased(self):
        event = MessageEvent(text="/HELP")
        assert event.get_command() == "help"

    def test_slash_only_returns_empty(self):
        event = MessageEvent(text="/")
        assert event.get_command() == ""


class TestMessageEventGetCommandArgs:
    def test_command_with_args(self):
        event = MessageEvent(text="/new session id 123")
        assert event.get_command_args() == "session id 123"

    def test_command_without_args(self):
        event = MessageEvent(text="/new")
        assert event.get_command_args() == ""

    def test_not_a_command_returns_full_text(self):
        event = MessageEvent(text="hello world")
        assert event.get_command_args() == "hello world"


# ---------------------------------------------------------------------------
# extract_images
# ---------------------------------------------------------------------------


class TestExtractImages:
    def test_no_images(self):
        images, cleaned = BasePlatformAdapter.extract_images("Just regular text.")
        assert images == []
        assert cleaned == "Just regular text."

    def test_markdown_image_with_image_ext(self):
        content = "Here is a photo: ![cat](https://example.com/cat.png)"
        images, cleaned = BasePlatformAdapter.extract_images(content)
        assert len(images) == 1
        assert images[0][0] == "https://example.com/cat.png"
        assert images[0][1] == "cat"
        assert "![cat]" not in cleaned

    def test_markdown_image_jpg(self):
        content = "![photo](https://example.com/photo.jpg)"
        images, _ = BasePlatformAdapter.extract_images(content)
        assert len(images) == 1

    def test_markdown_image_jpeg(self):
        content = "![](https://example.com/photo.jpeg)"
        images, _ = BasePlatformAdapter.extract_images(content)
        assert len(images) == 1

    def test_markdown_image_gif(self):
        content = "![anim](https://example.com/anim.gif)"
        images, _ = BasePlatformAdapter.extract_images(content)
        assert len(images) == 1

    def test_markdown_image_webp(self):
        content = "![](https://example.com/img.webp)"
        images, _ = BasePlatformAdapter.extract_images(content)
        assert len(images) == 1

    def test_fal_media_cdn(self):
        content = "![gen](https://fal.media/files/abc123/output.png)"
        images, _ = BasePlatformAdapter.extract_images(content)
        assert len(images) == 1

    def test_fal_cdn_url(self):
        content = "![](https://fal-cdn.example.com/result)"
        images, _ = BasePlatformAdapter.extract_images(content)
        assert len(images) == 1

    def test_replicate_delivery(self):
        content = "![](https://replicate.delivery/pbxt/abc/output)"
        images, _ = BasePlatformAdapter.extract_images(content)
        assert len(images) == 1

    def test_non_image_ext_not_extracted(self):
        """Markdown image with non-image extension should not be extracted."""
        content = "![doc](https://example.com/report.pdf)"
        images, cleaned = BasePlatformAdapter.extract_images(content)
        assert images == []
        assert "![doc]" in cleaned  # Should be preserved

    def test_html_img_tag(self):
        content = 'Check this: <img src="https://example.com/photo.png">'
        images, cleaned = BasePlatformAdapter.extract_images(content)
        assert len(images) == 1
        assert images[0][0] == "https://example.com/photo.png"
        assert images[0][1] == ""  # HTML images have no alt text
        assert "<img" not in cleaned

    def test_html_img_self_closing(self):
        content = '<img src="https://example.com/photo.png"/>'
        images, _ = BasePlatformAdapter.extract_images(content)
        assert len(images) == 1

    def test_html_img_with_closing_tag(self):
        content = '<img src="https://example.com/photo.png"></img>'
        images, _ = BasePlatformAdapter.extract_images(content)
        assert len(images) == 1

    def test_multiple_images(self):
        content = "![a](https://example.com/a.png)\n![b](https://example.com/b.jpg)"
        images, cleaned = BasePlatformAdapter.extract_images(content)
        assert len(images) == 2
        assert "![a]" not in cleaned
        assert "![b]" not in cleaned

    def test_mixed_markdown_and_html(self):
        content = '![cat](https://example.com/cat.png)\n<img src="https://example.com/dog.jpg">'
        images, _ = BasePlatformAdapter.extract_images(content)
        assert len(images) == 2

    def test_cleaned_content_trims_excess_newlines(self):
        content = "Before\n\n![img](https://example.com/img.png)\n\n\n\nAfter"
        _, cleaned = BasePlatformAdapter.extract_images(content)
        assert "\n\n\n" not in cleaned

    def test_non_http_url_not_matched(self):
        content = "![file](file:///local/path.png)"
        images, _ = BasePlatformAdapter.extract_images(content)
        assert images == []

    def test_non_image_link_preserved_when_mixed_with_images(self):
        """Regression: non-image markdown links must not be silently removed
        when the response also contains real images."""
        content = (
            "Here is the image: ![photo](https://fal.media/cat.png)\n"
            "And a doc: ![report](https://example.com/report.pdf)"
        )
        images, cleaned = BasePlatformAdapter.extract_images(content)
        assert len(images) == 1
        assert images[0][0] == "https://fal.media/cat.png"
        # The PDF link must survive in cleaned content
        assert "![report](https://example.com/report.pdf)" in cleaned


# ---------------------------------------------------------------------------
# extract_media
# ---------------------------------------------------------------------------


class TestExtractMedia:
    def test_no_media(self):
        media, cleaned = BasePlatformAdapter.extract_media("Just text.")
        assert media == []
        assert cleaned == "Just text."

    def test_single_media_tag(self):
        content = "MEDIA:/path/to/audio.ogg"
        media, cleaned = BasePlatformAdapter.extract_media(content)
        assert len(media) == 1
        assert media[0][0] == "/path/to/audio.ogg"
        assert media[0][1] is False  # no voice tag

    def test_media_with_voice_directive(self):
        content = "[[audio_as_voice]]\nMEDIA:/path/to/voice.ogg"
        media, cleaned = BasePlatformAdapter.extract_media(content)
        assert len(media) == 1
        assert media[0][0] == "/path/to/voice.ogg"
        assert media[0][1] is True  # voice tag present

    def test_multiple_media_tags(self):
        content = "MEDIA:/a.ogg\nMEDIA:/b.ogg"
        media, _ = BasePlatformAdapter.extract_media(content)
        assert len(media) == 2

    def test_voice_directive_removed_from_content(self):
        content = "[[audio_as_voice]]\nSome text\nMEDIA:/voice.ogg"
        _, cleaned = BasePlatformAdapter.extract_media(content)
        assert "[[audio_as_voice]]" not in cleaned
        assert "MEDIA:" not in cleaned
        assert "Some text" in cleaned

    def test_media_with_text_before(self):
        content = "Here is your audio:\nMEDIA:/output.ogg"
        media, cleaned = BasePlatformAdapter.extract_media(content)
        assert len(media) == 1
        assert "Here is your audio" in cleaned

    def test_cleaned_content_trims_excess_newlines(self):
        content = "Before\n\nMEDIA:/audio.ogg\n\n\n\nAfter"
        _, cleaned = BasePlatformAdapter.extract_media(content)
        assert "\n\n\n" not in cleaned


# ---------------------------------------------------------------------------
# truncate_message
# ---------------------------------------------------------------------------


class TestTruncateMessage:
    def _adapter(self):
        """Create a minimal adapter instance for testing static/instance methods."""
        class StubAdapter(BasePlatformAdapter):
            async def connect(self): return True
            async def disconnect(self): pass
            async def send(self, *a, **kw): pass
            async def get_chat_info(self, *a): return {}

        from gateway.config import Platform, PlatformConfig
        config = PlatformConfig(enabled=True, token="test")
        return StubAdapter(config=config, platform=Platform.TELEGRAM)

    def test_short_message_single_chunk(self):
        adapter = self._adapter()
        chunks = adapter.truncate_message("Hello world", max_length=100)
        assert chunks == ["Hello world"]

    def test_exact_length_single_chunk(self):
        adapter = self._adapter()
        msg = "x" * 100
        chunks = adapter.truncate_message(msg, max_length=100)
        assert chunks == [msg]

    def test_long_message_splits(self):
        adapter = self._adapter()
        msg = "word " * 200  # ~1000 chars
        chunks = adapter.truncate_message(msg, max_length=200)
        assert len(chunks) > 1

    def test_chunks_have_indicators(self):
        adapter = self._adapter()
        msg = "word " * 200
        chunks = adapter.truncate_message(msg, max_length=200)
        assert "(1/" in chunks[0]
        assert f"({len(chunks)}/{len(chunks)})" in chunks[-1]

    def test_code_block_first_chunk_closed(self):
        adapter = self._adapter()
        msg = "Before\n```python\n" + "x = 1\n" * 100 + "```\nAfter"
        chunks = adapter.truncate_message(msg, max_length=300)
        assert len(chunks) > 1
        # First chunk must have a closing fence appended (code block was split)
        first_fences = chunks[0].count("```")
        assert first_fences == 2, "First chunk should have opening + closing fence"

    def test_code_block_language_tag_carried(self):
        adapter = self._adapter()
        msg = "Start\n```javascript\n" + "console.log('x');\n" * 80 + "```\nEnd"
        chunks = adapter.truncate_message(msg, max_length=300)
        if len(chunks) > 1:
            # At least one continuation chunk should reopen with ```javascript
            reopened_with_lang = any(
                "```javascript" in chunk for chunk in chunks[1:]
            )
            assert reopened_with_lang, "No continuation chunk reopened with language tag"

    def test_continuation_chunks_have_balanced_fences(self):
        """Regression: continuation chunks must close reopened code blocks."""
        adapter = self._adapter()
        msg = "Before\n```python\n" + "x = 1\n" * 100 + "```\nAfter"
        chunks = adapter.truncate_message(msg, max_length=300)
        assert len(chunks) > 1
        for i, chunk in enumerate(chunks):
            fence_count = chunk.count("```")
            assert fence_count % 2 == 0, (
                f"Chunk {i} has unbalanced fences ({fence_count})"
            )

    def test_each_chunk_under_max_length(self):
        adapter = self._adapter()
        msg = "word " * 500
        max_len = 200
        chunks = adapter.truncate_message(msg, max_length=max_len)
        for i, chunk in enumerate(chunks):
            assert len(chunk) <= max_len + 20, f"Chunk {i} too long: {len(chunk)} > {max_len}"


# ---------------------------------------------------------------------------
# _get_human_delay
# ---------------------------------------------------------------------------


class TestGetHumanDelay:
    def test_off_mode(self):
        with patch.dict(os.environ, {"HERMES_HUMAN_DELAY_MODE": "off"}):
            assert BasePlatformAdapter._get_human_delay() == 0.0

    def test_default_is_off(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("HERMES_HUMAN_DELAY_MODE", None)
            assert BasePlatformAdapter._get_human_delay() == 0.0

    def test_natural_mode_range(self):
        with patch.dict(os.environ, {"HERMES_HUMAN_DELAY_MODE": "natural"}):
            delay = BasePlatformAdapter._get_human_delay()
            assert 0.8 <= delay <= 2.5

    def test_custom_mode_uses_env_vars(self):
        env = {
            "HERMES_HUMAN_DELAY_MODE": "custom",
            "HERMES_HUMAN_DELAY_MIN_MS": "100",
            "HERMES_HUMAN_DELAY_MAX_MS": "200",
        }
        with patch.dict(os.environ, env):
            delay = BasePlatformAdapter._get_human_delay()
            assert 0.1 <= delay <= 0.2
