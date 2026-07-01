"""Tests for gateway/platforms/base.py — MessageEvent, media extraction, message truncation."""

import os
import time
from unittest.mock import patch

import pytest

from gateway.platforms.base import (
    BasePlatformAdapter,
    GATEWAY_SECRET_CAPTURE_UNSUPPORTED_MESSAGE,
    MessageEvent,
    cache_audio_from_bytes,
    cache_image_from_bytes,
    cache_video_from_bytes,
    safe_url_for_log,
    utf16_len,
    validate_inbound_media_size,
    _log_safe_path,
    _prefix_within_utf16_limit,
)


class TestInboundMediaSizeCap:
    """gateway.max_inbound_media_bytes caps inbound media buffered into RAM (#13145)."""

    _PNG = b"\x89PNG\r\n\x1a\n" + b"x" * 64

    def test_default_cap_is_128_mib(self, monkeypatch):
        # No config override -> default. Patch loader to return empty config.
        import gateway.platforms.base as base
        monkeypatch.setattr(base, "get_inbound_media_max_bytes", lambda: base.DEFAULT_INBOUND_MEDIA_MAX_BYTES)
        assert base.DEFAULT_INBOUND_MEDIA_MAX_BYTES == 128 * 1024 * 1024

    def test_image_bytes_rejected_when_oversized(self, monkeypatch):
        import gateway.platforms.base as base
        monkeypatch.setattr(base, "get_inbound_media_max_bytes", lambda: 16)
        with pytest.raises(ValueError, match="Inbound image payload is too large"):
            cache_image_from_bytes(self._PNG, ext=".png")

    def test_audio_bytes_rejected_when_oversized(self, monkeypatch):
        import gateway.platforms.base as base
        monkeypatch.setattr(base, "get_inbound_media_max_bytes", lambda: 4)
        with pytest.raises(ValueError, match="Inbound audio payload is too large"):
            cache_audio_from_bytes(b"x" * 8, ext=".ogg")

    def test_video_bytes_rejected_when_oversized(self, monkeypatch):
        # Video was the gap in the original report — verify it's covered.
        import gateway.platforms.base as base
        monkeypatch.setattr(base, "get_inbound_media_max_bytes", lambda: 4)
        with pytest.raises(ValueError, match="Inbound video payload is too large"):
            cache_video_from_bytes(b"x" * 8, ext=".mp4")

    def test_legit_image_accepted_under_cap(self, monkeypatch):
        import gateway.platforms.base as base
        monkeypatch.setattr(base, "get_inbound_media_max_bytes", lambda: 128 * 1024 * 1024)
        path = cache_image_from_bytes(self._PNG, ext=".png")
        assert os.path.exists(path)
        assert os.path.getsize(path) == len(self._PNG)

    def test_cap_of_zero_disables_check(self, monkeypatch):
        import gateway.platforms.base as base
        monkeypatch.setattr(base, "get_inbound_media_max_bytes", lambda: 0)
        # A would-be-oversized video passes through when the cap is disabled.
        path = cache_video_from_bytes(b"x" * 5000, ext=".mp4")
        assert os.path.exists(path)

    def test_validate_helper_respects_explicit_max_bytes(self):
        # max_bytes arg overrides the configured cap.
        validate_inbound_media_size(100, media_type="image", max_bytes=200)  # ok
        with pytest.raises(ValueError, match="too large"):
            validate_inbound_media_size(300, media_type="image", max_bytes=200)


class TestSecretCaptureGuidance:
    def test_gateway_secret_capture_message_points_to_local_setup(self):
        message = GATEWAY_SECRET_CAPTURE_UNSUPPORTED_MESSAGE
        assert "local cli" in message.lower()
        assert "~/.hermes/.env" in message


class TestSafeUrlForLog:
    def test_strips_query_fragment_and_userinfo(self):
        url = (
            "https://user:pass@example.com/private/path/image.png"
            "?X-Amz-Signature=supersecret&token=abc#frag"
        )
        result = safe_url_for_log(url)
        assert result == "https://example.com/.../image.png"
        assert "supersecret" not in result
        assert "token=abc" not in result
        assert "user:pass@" not in result

    def test_truncates_long_values(self):
        long_url = "https://example.com/" + ("a" * 300)
        result = safe_url_for_log(long_url, max_len=40)
        assert len(result) == 40
        assert result.endswith("...")

    def test_handles_small_and_non_positive_max_len(self):
        url = "https://example.com/very/long/path/file.png?token=secret"
        assert safe_url_for_log(url, max_len=3) == "..."
        assert safe_url_for_log(url, max_len=2) == ".."
        assert safe_url_for_log(url, max_len=0) == ""


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

    def test_command_with_at_botname(self):
        event = MessageEvent(text="/new@TigerNanoBot")
        assert event.get_command() == "new"

    def test_command_with_at_botname_and_args(self):
        event = MessageEvent(text="/compress@TigerNanoBot")
        assert event.get_command() == "compress"

    def test_command_mixed_case_with_at_botname(self):
        event = MessageEvent(text="/RESET@TigerNanoBot")
        assert event.get_command() == "reset"


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
        assert images[0][0] == "https://example.com/photo.jpg"
        assert images[0][1] == "photo"

    def test_markdown_image_jpeg(self):
        content = "![](https://example.com/photo.jpeg)"
        images, _ = BasePlatformAdapter.extract_images(content)
        assert len(images) == 1
        assert images[0][0] == "https://example.com/photo.jpeg"
        assert images[0][1] == ""

    def test_markdown_image_gif(self):
        content = "![anim](https://example.com/anim.gif)"
        images, _ = BasePlatformAdapter.extract_images(content)
        assert len(images) == 1
        assert images[0][0] == "https://example.com/anim.gif"
        assert images[0][1] == "anim"

    def test_markdown_image_webp(self):
        content = "![](https://example.com/img.webp)"
        images, _ = BasePlatformAdapter.extract_images(content)
        assert len(images) == 1
        assert images[0][0] == "https://example.com/img.webp"
        assert images[0][1] == ""

    def test_fal_media_cdn(self):
        content = "![gen](https://fal.media/files/abc123/output.png)"
        images, _ = BasePlatformAdapter.extract_images(content)
        assert len(images) == 1
        assert images[0][0] == "https://fal.media/files/abc123/output.png"
        assert images[0][1] == "gen"

    def test_fal_cdn_url(self):
        content = "![](https://fal-cdn.example.com/result)"
        images, _ = BasePlatformAdapter.extract_images(content)
        assert len(images) == 1
        assert images[0][0] == "https://fal-cdn.example.com/result"
        assert images[0][1] == ""

    def test_replicate_delivery(self):
        content = "![](https://replicate.delivery/pbxt/abc/output)"
        images, _ = BasePlatformAdapter.extract_images(content)
        assert len(images) == 1
        assert images[0][0] == "https://replicate.delivery/pbxt/abc/output"
        assert images[0][1] == ""

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
        assert images[0][0] == "https://example.com/photo.png"
        assert images[0][1] == ""

    def test_html_img_with_closing_tag(self):
        content = '<img src="https://example.com/photo.png"></img>'
        images, _ = BasePlatformAdapter.extract_images(content)
        assert len(images) == 1
        assert images[0][0] == "https://example.com/photo.png"
        assert images[0][1] == ""

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

    def test_media_tag_allows_optional_whitespace_after_colon(self):
        content = "MEDIA: /path/to/audio.ogg"
        media, cleaned = BasePlatformAdapter.extract_media(content)
        assert media == [("/path/to/audio.ogg", False)]
        assert cleaned == ""

    def test_media_tag_strips_wrapping_quotes_and_backticks(self):
        content = "MEDIA: `/path/to/file.png`\nMEDIA:\"/path/to/file2.png\"\nMEDIA:'/path/to/file3.png'"
        media, cleaned = BasePlatformAdapter.extract_media(content)
        assert media == [
            ("/path/to/file.png", False),
            ("/path/to/file2.png", False),
            ("/path/to/file3.png", False),
        ]
        assert cleaned == ""

    def test_media_tag_supports_quoted_paths_with_spaces(self):
        content = "Here\nMEDIA: '/tmp/my image.png'\nAfter"
        media, cleaned = BasePlatformAdapter.extract_media(content)
        assert media == [("/tmp/my image.png", False)]
        assert "Here" in cleaned
        assert "After" in cleaned

    def test_media_tag_supports_unquoted_flac_paths_with_spaces(self):
        content = "MEDIA:/tmp/Jane Doe/speech.flac"
        media, cleaned = BasePlatformAdapter.extract_media(content)
        assert media == [("/tmp/Jane Doe/speech.flac", False)]
        assert cleaned == ""

    def test_as_document_directive_stripped_from_cleaned_text(self):
        """[[as_document]] is a routing directive — strip it from
        user-visible text just like [[audio_as_voice]]. Callers detect the
        directive on the original content (before extract_media)."""
        content = "Here is your infographic:\n[[as_document]]\nMEDIA:/tmp/x.jpg"
        media, cleaned = BasePlatformAdapter.extract_media(content)
        assert media == [("/tmp/x.jpg", False)]
        assert "[[as_document]]" not in cleaned
        assert "Here is your infographic" in cleaned

    def test_as_document_directive_alone_does_not_attach_voice_flag(self):
        """[[as_document]] is independent of [[audio_as_voice]] — combining
        them in the same response should not entangle the flags."""
        content = "[[as_document]]\nMEDIA:/tmp/x.jpg"
        media, cleaned = BasePlatformAdapter.extract_media(content)
        assert media == [("/tmp/x.jpg", False)]  # voice flag stays False
        assert "[[as_document]]" not in cleaned

    def test_both_directives_can_coexist(self):
        """A response could (rarely) contain both [[audio_as_voice]] for an
        ogg file AND [[as_document]] for an attached image. The voice flag
        propagates per-tuple; [[as_document]] is detected at dispatch."""
        content = "[[audio_as_voice]]\n[[as_document]]\nMEDIA:/tmp/x.ogg"
        media, cleaned = BasePlatformAdapter.extract_media(content)
        # Voice flag is propagated to every media tuple (this matches the
        # existing extract_media contract)
        assert media == [("/tmp/x.ogg", True)]
        # Both directives stripped from cleaned text
        assert "[[audio_as_voice]]" not in cleaned
        assert "[[as_document]]" not in cleaned

    # Windows path support — regression coverage for #34632

    def test_media_tag_windows_backslash_path(self):
        """extract_media should recognise Windows backslash paths."""
        media, cleaned = BasePlatformAdapter.extract_media(
            r"MEDIA:C:\Users\kotsu\file.pdf"
        )
        assert len(media) == 1
        assert media[0][0].endswith("file.pdf")

    def test_media_tag_windows_forward_slash_path(self):
        """extract_media should recognise Windows forward-slash paths."""
        media, cleaned = BasePlatformAdapter.extract_media(
            "MEDIA:C:/Users/kotsu/file.pdf"
        )
        assert len(media) == 1
        assert media[0][0].endswith("file.pdf")

    def test_media_tag_windows_drive_root(self):
        """extract_media should recognise a path at the drive root."""
        media, cleaned = BasePlatformAdapter.extract_media(
            r"MEDIA:D:\report.md"
        )
        assert len(media) == 1
        assert media[0][0].endswith("report.md")

    def test_media_tag_unix_paths_still_work(self):
        """Unix absolute and tilde paths must still extract after Windows change."""
        for content in ["MEDIA:/tmp/audio.ogg", r"MEDIA:~/docs/notes.md"]:
            media, _ = BasePlatformAdapter.extract_media(content)
            assert len(media) == 1, f"Failed for: {content}"

    def test_relative_path_still_ignored(self):
        """Relative Windows-style paths (no drive letter) must not match."""
        media, _ = BasePlatformAdapter.extract_media(
            r"MEDIA:Users\kotsu\file.pdf"
        )
        assert media == []

    # --- Code block / inline code / blockquote false-positive guards (#35695) ---

    def test_media_in_fenced_code_block_ignored(self):
        """MEDIA: inside ``` fenced code blocks must not be extracted."""
        content = "Here is an example:\n```text\nMEDIA:/path/to/example.png\n```\nDone."
        media, cleaned = BasePlatformAdapter.extract_media(content)
        assert media == []
        assert "example" in cleaned.lower()

    def test_media_in_inline_code_ignored(self):
        """MEDIA: inside backtick inline code must not be extracted."""
        content = "Use `MEDIA:/path/to/file.png` in your response."
        media, cleaned = BasePlatformAdapter.extract_media(content)
        assert media == []
        assert "MEDIA:" in cleaned  # preserved as text

    def test_media_in_blockquote_ignored(self):
        """MEDIA: inside a > blockquote must not be extracted."""
        content = "> To send an image, include MEDIA:/path/to/image.jpg\nEnd."
        media, cleaned = BasePlatformAdapter.extract_media(content)
        assert media == []
        assert "End." in cleaned

    def test_media_outside_code_blocks_still_extracted(self):
        """Real MEDIA: tags outside protected regions must still work."""
        content = "MEDIA:/real/file.png\n```code\nMEDIA:/fake/file.png\n```"
        media, _ = BasePlatformAdapter.extract_media(content)
        assert len(media) == 1
        assert media[0][0] == "/real/file.png"

    def test_media_mixed_code_and_prose(self):
        """Real MEDIA: in prose + example in code block: only prose extracted,
        and the code block survives verbatim in the delivered text."""
        content = (
            "Here is your file:\n"
            "MEDIA:/output/report.pdf\n"
            "Example usage:\n"
            "```text\nMEDIA:/example/path.pdf\n```\n"
            "Done."
        )
        media, cleaned = BasePlatformAdapter.extract_media(content)
        assert len(media) == 1
        assert media[0][0] == "/output/report.pdf"
        assert "Done." in cleaned
        # The real tag is stripped from the delivered text...
        assert "MEDIA:/output/report.pdf" not in cleaned
        # ...but the fenced code block (incl. its example MEDIA: line) must
        # survive verbatim — masking is a locator, not a text rewrite.
        assert "```text\nMEDIA:/example/path.pdf\n```" in cleaned

    def test_inline_code_survives_when_real_media_present(self):
        """When a real MEDIA: tag is delivered, an inline-code example in the
        same reply must not be blanked to whitespace."""
        content = "See MEDIA:/r/a.png and `MEDIA:/ex/b.png` inline"
        media, cleaned = BasePlatformAdapter.extract_media(content)
        assert [p for p, _ in media] == ["/r/a.png"]
        assert "`MEDIA:/ex/b.png`" in cleaned


class TestMediaInsideSerializedJson:
    """Regression coverage for #34375 — MEDIA: embedded in serialized JSON
    string values (e.g. a stored previous reply inside a tool result) must not
    be re-delivered as a real attachment, while legitimate MEDIA: tags in prose,
    at line start, indented, or as quoted-path tags keep working.
    """

    def test_media_in_json_value_not_extracted(self):
        content = '{"result": "MEDIA:/tmp/stale.png"}'
        media, _ = BasePlatformAdapter.extract_media(content)
        assert media == [], f"JSON value MEDIA: leaked: {media}"

    def test_media_in_pretty_json_value_not_extracted(self):
        content = '{\n  "tool_result": "MEDIA:/var/old.jpg"\n}'
        media, _ = BasePlatformAdapter.extract_media(content)
        assert media == [], f"pretty JSON MEDIA: leaked: {media}"

    def test_media_in_json_array_not_extracted(self):
        content = '["MEDIA:/a/b.png", "other"]'
        media, _ = BasePlatformAdapter.extract_media(content)
        assert media == [], f"JSON array MEDIA: leaked: {media}"

    def test_media_in_nested_json_value_not_extracted(self):
        content = '{"a":{"b":"see MEDIA:/x/y.pdf here"}}'
        media, _ = BasePlatformAdapter.extract_media(content)
        assert media == [], f"nested JSON MEDIA: leaked: {media}"

    def test_media_in_embedded_serialized_reply_not_extracted(self):
        """A serialized tool result that embeds a prior reply's MEDIA: tag."""
        content = (
            '{"content":"previous reply MEDIA:/Users/ex/.hermes/media/'
            'generated/stale.png and more text"}'
        )
        media, _ = BasePlatformAdapter.extract_media(content)
        assert media == [], f"embedded serialized reply leaked: {media}"

    # --- Legitimate tags must still extract (no regression vs line-start anchor) ---

    def test_media_at_line_start_still_extracted(self):
        media, _ = BasePlatformAdapter.extract_media("MEDIA:/real/file.png")
        assert len(media) == 1 and media[0][0] == "/real/file.png"

    def test_media_after_prose_same_line_still_extracted(self):
        media, _ = BasePlatformAdapter.extract_media(
            "Here is your file: MEDIA:/out/report.pdf"
        )
        assert len(media) == 1 and media[0][0] == "/out/report.pdf"

    def test_media_indented_still_extracted(self):
        media, _ = BasePlatformAdapter.extract_media("  MEDIA:/tmp/x.png")
        assert len(media) == 1 and media[0][0] == "/tmp/x.png"

    def test_quoted_path_media_still_extracted(self):
        """MEDIA:"..." quoted-path form (a real LLM output) is not JSON-masked."""
        media, _ = BasePlatformAdapter.extract_media(
            'MEDIA:"/path/with space/file.png"'
        )
        assert len(media) == 1 and media[0][0] == "/path/with space/file.png"

    def test_tts_two_line_still_extracted(self):
        media, _ = BasePlatformAdapter.extract_media(
            "[[audio_as_voice]]\nMEDIA:/tmp/v.ogg"
        )
        assert len(media) == 1 and media[0][0] == "/tmp/v.ogg"
        assert media[0][1] is True  # voice flag

    # --- cleaned-text invariants: real tags stripped, JSON data kept verbatim ---

    def test_json_embedded_media_kept_verbatim_in_cleaned_text(self):
        """A real tag is delivered+stripped; a JSON-embedded MEDIA: stays as
        literal text (stored data must read back unchanged)."""
        content = 'MEDIA:/real/r.png\nlog: {"old":"MEDIA:/stale/s.png"}'
        media, cleaned = BasePlatformAdapter.extract_media(content)
        assert [p for p, _ in media] == ["/real/r.png"]
        # The JSON-embedded path must survive verbatim — not blanked to spaces.
        assert '{"old":"MEDIA:/stale/s.png"}' in cleaned

    def test_cleaned_text_after_directive_not_truncated(self):
        """Stripping a tag preceded by a [[as_document]] directive must not
        shift offsets and chop the path or trailing text."""
        content = "See [[as_document]] MEDIA:/d/report.pdf now"
        media, cleaned = BasePlatformAdapter.extract_media(content)
        assert [p for p, _ in media] == ["/d/report.pdf"]
        assert "MEDIA:" not in cleaned          # real tag removed
        assert cleaned.endswith("now")          # trailing text intact (not chopped)


class TestMediaExtensionAllowlistParity:
    """Regression coverage for issue #34517 — the MEDIA: extension black hole.

    extract_media used to carry a narrow extension allowlist that omitted
    .md/.json/.yaml/.xml/.html etc., while extract_local_files had a broad one.
    Combined with an unconditional ``MEDIA:\\s*\\S+`` strip at the dispatch
    sites, an unmatched MEDIA: tag for one of those extensions was deleted from
    the body before extract_local_files could pick up the bare path — the file
    was silently dropped. Both extractors now derive from the single
    MEDIA_DELIVERY_EXTS source of truth, and the strip is anchored to that set.
    """

    DROPPED_BEFORE = ["md", "json", "yaml", "yml", "xml", "html", "htm",
                      "tsv", "svg"]

    def test_previously_dropped_extensions_now_extract(self):
        for ext in self.DROPPED_BEFORE:
            path = f"/tmp/report.{ext}"
            media, _ = BasePlatformAdapter.extract_media(f"Here: MEDIA:{path}")
            assert media == [(path, False)], f".{ext} should extract via MEDIA:"

    def test_extract_media_and_local_files_share_one_extension_set(self):
        from gateway.platforms.base import MEDIA_DELIVERY_EXTS
        # Both functions reference MEDIA_DELIVERY_EXTS; assert the documents
        # that motivated the bug are present in the shared set.
        for ext in (".md", ".json", ".yaml", ".yml", ".xml", ".html", ".htm"):
            assert ext in MEDIA_DELIVERY_EXTS

    def test_unknown_extension_not_black_holed_by_cleanup(self):
        """A MEDIA: tag with an unknown extension is NOT stripped from the
        body — it survives so extract_local_files can still see the bare path,
        rather than vanishing entirely (the core of issue #34517)."""
        from gateway.platforms.base import MEDIA_TAG_CLEANUP_RE
        text = "Saved to MEDIA:/tmp/data.weirdext done"
        media, _ = BasePlatformAdapter.extract_media(text)
        assert media == []  # unknown extension is not a deliverable MEDIA tag
        stripped = MEDIA_TAG_CLEANUP_RE.sub("", text)
        assert "/tmp/data.weirdext" in stripped  # path preserved, not dropped

    def test_known_extension_tag_is_stripped_from_body(self):
        from gateway.platforms.base import MEDIA_TAG_CLEANUP_RE
        text = "Here is your report: MEDIA:/tmp/report.md"
        stripped = MEDIA_TAG_CLEANUP_RE.sub("", text).strip()
        assert "MEDIA:" not in stripped
        assert "/tmp/report.md" not in stripped
        assert "Here is your report:" in stripped


class TestExtensionlessMediaDelivery:
    """Regression: MEDIA: tags for extension-less files (Caddyfile, Makefile)."""

    def _patch_allow_root(self, monkeypatch, root):
        monkeypatch.setattr(
            "gateway.platforms.base.MEDIA_DELIVERY_SAFE_ROOTS",
            (str(root),),
        )
        monkeypatch.delenv("HERMES_MEDIA_DELIVERY_STRICT", raising=False)

    def test_extensionless_media_extracted_when_file_validates(self, tmp_path, monkeypatch):
        root = tmp_path / "output"
        root.mkdir()
        caddy = root / "Caddyfile"
        caddy.write_text("localhost {}", encoding="utf-8")
        self._patch_allow_root(monkeypatch, root)

        content = f"Here is your config:\nMEDIA:{caddy}\nDone."
        media, cleaned = BasePlatformAdapter.extract_media(content)
        assert len(media) == 1
        assert media[0][0] == str(caddy.resolve())
        assert "MEDIA:" not in cleaned
        assert "Done." in cleaned

    def test_extensionless_media_left_visible_when_not_on_disk(self, tmp_path, monkeypatch):
        root = tmp_path / "output"
        root.mkdir()
        self._patch_allow_root(monkeypatch, root)

        content = "MEDIA:/nonexistent/Caddyfile"
        media, cleaned = BasePlatformAdapter.extract_media(content)
        assert media == []
        assert "MEDIA:/nonexistent/Caddyfile" in cleaned

    def test_strip_media_directives_for_display_strips_validated_extensionless(
        self, tmp_path, monkeypatch,
    ):
        root = tmp_path / "output"
        root.mkdir()
        caddy = root / "Caddyfile"
        caddy.write_text("x", encoding="utf-8")
        self._patch_allow_root(monkeypatch, root)

        text = f"MEDIA:{caddy}"
        stripped = BasePlatformAdapter.strip_media_directives_for_display(text)
        assert "MEDIA:" not in stripped

    def test_as_document_directive_stripped_without_media_tag(self):
        """[[as_document]] must be stripped even when no MEDIA: tag is present.

        The display/strip guards short-circuit on text containing none of the
        delivery directives; [[as_document]] must be in that guard or it leaks
        to the user as visible text on any image-only response.
        """
        from gateway.platforms.base import _strip_media_directives

        text = "Here is your image. [[as_document]]"
        assert "[[as_document]]" not in _strip_media_directives(text)
        assert "[[as_document]]" not in (
            BasePlatformAdapter.strip_media_directives_for_display(text)
        )


class TestMediaDeliveryPathValidation:
    def _patch_roots(self, monkeypatch, *roots):
        monkeypatch.setattr(
            "gateway.platforms.base.MEDIA_DELIVERY_SAFE_ROOTS",
            tuple(roots),
        )
        # All tests in this class cover strict-mode behavior (allowlist +
        # recency window + denylist). Force strict on so they keep
        # exercising the legacy path even though the public default
        # flipped to off in 2026-05.
        monkeypatch.setenv("HERMES_MEDIA_DELIVERY_STRICT", "1")
        # Disable recency-based trust by default so the original allowlist
        # tests continue to exercise the strict-allowlist path. Tests that
        # specifically cover recency trust re-enable it themselves.
        monkeypatch.setenv("HERMES_MEDIA_TRUST_RECENT_FILES", "0")

    def test_allows_existing_file_inside_safe_root(self, tmp_path, monkeypatch):
        root = tmp_path / "media-cache"
        media_file = root / "voice.ogg"
        media_file.parent.mkdir(parents=True)
        media_file.write_bytes(b"OggS")
        self._patch_roots(monkeypatch, root)

        assert BasePlatformAdapter.validate_media_delivery_path(str(media_file)) == str(media_file.resolve())

    def test_rejects_existing_file_outside_safe_root(self, tmp_path, monkeypatch):
        root = tmp_path / "media-cache"
        root.mkdir()
        secret = tmp_path / "secrets.txt"
        secret.write_text("not for upload")
        self._patch_roots(monkeypatch, root)

        assert BasePlatformAdapter.validate_media_delivery_path(str(secret)) is None

    def test_rejects_symlink_escape_from_safe_root(self, tmp_path, monkeypatch):
        root = tmp_path / "media-cache"
        root.mkdir()
        secret = tmp_path / "outside.png"
        secret.write_bytes(b"secret")
        link = root / "safe-looking.png"
        try:
            link.symlink_to(secret)
        except OSError:
            pytest.skip("symlink creation is unavailable")
        self._patch_roots(monkeypatch, root)

        assert BasePlatformAdapter.validate_media_delivery_path(str(link)) is None

    def test_filter_keeps_safe_media_and_drops_unsafe(self, tmp_path, monkeypatch):
        root = tmp_path / "media-cache"
        safe = root / "speech.ogg"
        unsafe = tmp_path / "outside.ogg"
        safe.parent.mkdir(parents=True)
        safe.write_bytes(b"OggS")
        unsafe.write_bytes(b"OggS")
        self._patch_roots(monkeypatch, root)

        filtered = BasePlatformAdapter.filter_media_delivery_paths([
            (str(unsafe), False),
            (str(safe), True),
        ])

        assert filtered == [(str(safe.resolve()), True)]

    def test_allows_operator_configured_extra_root(self, tmp_path, monkeypatch):
        extra_root = tmp_path / "operator-media"
        media_file = extra_root / "report.pdf"
        media_file.parent.mkdir(parents=True)
        media_file.write_bytes(b"%PDF-1.4")
        self._patch_roots(monkeypatch)
        monkeypatch.setenv("HERMES_MEDIA_ALLOW_DIRS", str(extra_root))

        assert BasePlatformAdapter.validate_media_delivery_path(str(media_file)) == str(media_file.resolve())

    def test_recency_trust_allows_freshly_produced_file(self, tmp_path, monkeypatch):
        """A PDF the agent just wrote to /tmp should be deliverable.

        Covers the natural case: agent runs ``pandoc -o /tmp/report.pdf`` or
        ``write_file('/home/user/report.pdf', ...)`` and asks the gateway to
        send the result. With recency trust on, fresh files outside the cache
        allowlist are accepted because the file's mtime is within the window.
        """
        self._patch_roots(monkeypatch)  # zero cache allowlist
        monkeypatch.delenv("HERMES_MEDIA_ALLOW_DIRS", raising=False)
        monkeypatch.setenv("HERMES_MEDIA_TRUST_RECENT_FILES", "1")
        monkeypatch.setenv("HERMES_MEDIA_TRUST_RECENT_SECONDS", "600")

        fresh = tmp_path / "scratch" / "report.pdf"
        fresh.parent.mkdir(parents=True)
        fresh.write_bytes(b"%PDF-1.4")

        assert BasePlatformAdapter.validate_media_delivery_path(str(fresh)) == str(fresh.resolve())

    def test_recency_trust_rejects_old_file(self, tmp_path, monkeypatch):
        """A pre-existing host file (~/.bashrc, /etc/passwd shape) is rejected.

        Recency trust is the load-bearing anti-injection signal: prompt-injected
        paths point at files that have existed for days or months, well outside
        the trust window.
        """
        self._patch_roots(monkeypatch)
        monkeypatch.delenv("HERMES_MEDIA_ALLOW_DIRS", raising=False)
        monkeypatch.setenv("HERMES_MEDIA_TRUST_RECENT_FILES", "1")
        monkeypatch.setenv("HERMES_MEDIA_TRUST_RECENT_SECONDS", "60")

        stale = tmp_path / "stale.pdf"
        stale.write_bytes(b"%PDF-1.4")
        old_mtime = time.time() - 7200  # 2 hours ago
        os.utime(stale, (old_mtime, old_mtime))

        assert BasePlatformAdapter.validate_media_delivery_path(str(stale)) is None

    def test_recency_trust_disabled_falls_back_to_pure_allowlist(self, tmp_path, monkeypatch):
        """Setting trust_recent_files=false reverts to pre-existing strict behavior."""
        self._patch_roots(monkeypatch)
        monkeypatch.delenv("HERMES_MEDIA_ALLOW_DIRS", raising=False)
        monkeypatch.setenv("HERMES_MEDIA_TRUST_RECENT_FILES", "0")

        fresh = tmp_path / "report.pdf"
        fresh.write_bytes(b"%PDF-1.4")  # mtime = now

        assert BasePlatformAdapter.validate_media_delivery_path(str(fresh)) is None

    def test_recency_trust_denies_system_paths_even_when_fresh(self, tmp_path, monkeypatch):
        """A freshly-touched file under /etc must NOT be uploaded.

        Belt-and-braces: even if an attacker rewrites the file's mtime
        (e.g. via a separately compromised tool result that touches a system
        file), the denylist refuses to deliver paths under /etc, /proc, /sys,
        ~/.ssh, ~/.aws, etc.
        """
        self._patch_roots(monkeypatch)
        monkeypatch.delenv("HERMES_MEDIA_ALLOW_DIRS", raising=False)
        monkeypatch.setenv("HERMES_MEDIA_TRUST_RECENT_FILES", "1")
        monkeypatch.setenv("HERMES_MEDIA_TRUST_RECENT_SECONDS", "600")

        # Simulate $HOME so ~/.ssh resolves into our tmp dir.
        fake_home = tmp_path / "home"
        ssh_dir = fake_home / ".ssh"
        ssh_dir.mkdir(parents=True)
        secret = ssh_dir / "id_rsa.txt"
        secret.write_bytes(b"-----BEGIN ...")  # mtime = now
        monkeypatch.setenv("HOME", str(fake_home))

        assert BasePlatformAdapter.validate_media_delivery_path(str(secret)) is None

    def test_recency_trust_allows_pdf_in_project_dir(self, tmp_path, monkeypatch):
        """The motivating case: agent produces a PDF in a project directory.

        Reproduces the Discord-PDF-not-delivered bug. Before recency trust,
        files outside ~/.hermes/cache/* were silently dropped, leaving the
        user with a raw filepath in chat instead of an attachment.
        """
        self._patch_roots(monkeypatch)
        monkeypatch.delenv("HERMES_MEDIA_ALLOW_DIRS", raising=False)
        monkeypatch.setenv("HERMES_MEDIA_TRUST_RECENT_FILES", "1")
        monkeypatch.setenv("HERMES_MEDIA_TRUST_RECENT_SECONDS", "600")

        project = tmp_path / "my-project"
        report = project / "build" / "weekly-report.pdf"
        report.parent.mkdir(parents=True)
        report.write_bytes(b"%PDF-1.4")

        assert BasePlatformAdapter.validate_media_delivery_path(str(report)) == str(report.resolve())

    def test_filter_keeps_recently_produced_files(self, tmp_path, monkeypatch):
        """End-to-end: filter_local_delivery_paths routes a fresh PDF through."""
        self._patch_roots(monkeypatch)
        monkeypatch.delenv("HERMES_MEDIA_ALLOW_DIRS", raising=False)
        monkeypatch.setenv("HERMES_MEDIA_TRUST_RECENT_FILES", "1")
        monkeypatch.setenv("HERMES_MEDIA_TRUST_RECENT_SECONDS", "600")

        fresh = tmp_path / "report.pdf"
        fresh.write_bytes(b"%PDF-1.4")

        out = BasePlatformAdapter.filter_local_delivery_paths([str(fresh)])
        assert out == [str(fresh.resolve())]


class TestMediaDeliveryDefaultMode:
    """Default (non-strict) mode — denylist gates delivery, nothing else.

    Symmetric with inbound delivery: Telegram/Discord/Slack accept any
    document type the user uploads, and the agent can hand back any file
    that isn't a credential. Strict mode is opt-in for operators running
    public-facing gateways.
    """

    def _patch_roots(self, monkeypatch, *roots):
        # Empty cache allowlist so the only positive path through
        # validate_media_delivery_path in these tests is the
        # default-mode "anything not denied" branch.
        monkeypatch.setattr(
            "gateway.platforms.base.MEDIA_DELIVERY_SAFE_ROOTS",
            tuple(roots),
        )
        # Pin strict OFF — the public default. Tests that exercise the
        # strict path live in TestMediaDeliveryPathValidation.
        monkeypatch.delenv("HERMES_MEDIA_DELIVERY_STRICT", raising=False)
        monkeypatch.delenv("HERMES_MEDIA_ALLOW_DIRS", raising=False)

    def test_accepts_stale_file_outside_allowlist(self, tmp_path, monkeypatch):
        """The motivating case — agent says ``MEDIA:/home/user/notes.md``
        for an .md it has been working with for hours. Strict mode would
        reject this (outside allowlist, outside recency window). Default
        mode delivers it.
        """
        self._patch_roots(monkeypatch)

        notes = tmp_path / "notes.md"
        notes.write_text("# Old notes\n")
        old_mtime = time.time() - 7200  # 2 hours ago — far outside any window
        os.utime(notes, (old_mtime, old_mtime))

        assert BasePlatformAdapter.validate_media_delivery_path(str(notes)) == str(notes.resolve())

    def test_accepts_any_extension_not_on_denylist(self, tmp_path, monkeypatch):
        """No extension allowlist — .md, .txt, .json, .py all deliver."""
        self._patch_roots(monkeypatch)

        for name in ("report.md", "log.txt", "data.json", "script.py", "blob.bin"):
            f = tmp_path / name
            f.write_bytes(b"x")
            assert BasePlatformAdapter.validate_media_delivery_path(str(f)) == str(f.resolve())

    def test_denylist_still_blocks_credentials(self, tmp_path, monkeypatch):
        """Default mode is permissive but not naive — credential paths
        remain blocked. Simulate $HOME so ~/.ssh resolves into tmp_path.
        """
        self._patch_roots(monkeypatch)

        fake_home = tmp_path / "home"
        ssh_dir = fake_home / ".ssh"
        ssh_dir.mkdir(parents=True)
        secret = ssh_dir / "id_rsa"
        secret.write_bytes(b"-----BEGIN ...")
        monkeypatch.setenv("HOME", str(fake_home))

        assert BasePlatformAdapter.validate_media_delivery_path(str(secret)) is None

    def test_denylist_blocks_system_prefixes(self, tmp_path, monkeypatch):
        """Files under /etc, /proc, /sys, /root, /boot, /var/{log,lib,run}
        are denied. We construct the test by patching the denylist root
        to a tmp dir so we don't need to read /etc.
        """
        self._patch_roots(monkeypatch)

        fake_etc = tmp_path / "fake-etc"
        fake_etc.mkdir()
        secret = fake_etc / "shadow"
        secret.write_bytes(b"root:!:0:0::/root:/bin/sh")

        monkeypatch.setattr(
            "gateway.platforms.base._MEDIA_DELIVERY_DENIED_PREFIXES",
            (str(fake_etc),),
        )

        assert BasePlatformAdapter.validate_media_delivery_path(str(secret)) is None

    def test_denylist_blocks_hermes_credentials(self, tmp_path, monkeypatch):
        """~/.hermes/.env and ~/.hermes/auth.json stay blocked even in
        default mode. They live under $HOME (not the system prefix list)
        so this exercises the home-relative denied paths.
        """
        self._patch_roots(monkeypatch)

        fake_home = tmp_path / "home"
        hermes_dir = fake_home / ".hermes"
        hermes_dir.mkdir(parents=True)
        env_file = hermes_dir / ".env"
        env_file.write_text("OPENAI_API_KEY=sk-...")
        monkeypatch.setenv("HOME", str(fake_home))
        monkeypatch.setattr(
            "gateway.platforms.base._HERMES_HOME",
            hermes_dir,
        )

        assert BasePlatformAdapter.validate_media_delivery_path(str(env_file)) is None

    @pytest.mark.parametrize(
        "rel",
        [
            "mcp-tokens/github.json",
            "mcp-tokens/github.client.json",
            "mcp-tokens/github.meta.json",
        ],
    )
    def test_denylist_blocks_mcp_oauth_tokens(self, tmp_path, monkeypatch, rel):
        """Live MCP OAuth tokens/client creds under ~/.hermes/mcp-tokens/ must
        never deliver as native media — same exfil class as auth.json/.env.
        Sibling to the pairing/ directory denylist entry.
        """
        self._patch_roots(monkeypatch)

        fake_home = tmp_path / "home"
        hermes_dir = fake_home / ".hermes"
        (hermes_dir / "mcp-tokens").mkdir(parents=True)
        secret = hermes_dir / rel
        secret.write_text('{"access_token": "live-bearer-abc123"}')
        monkeypatch.setenv("HOME", str(fake_home))
        monkeypatch.setattr(
            "gateway.platforms.base._HERMES_HOME",
            hermes_dir,
        )
        monkeypatch.setattr(
            "gateway.platforms.base._HERMES_ROOT",
            hermes_dir,
        )

        assert BasePlatformAdapter.validate_media_delivery_path(str(secret)) is None

    def test_denylist_blocks_hermes_config_in_active_profile(self, tmp_path, monkeypatch):
        """The active profile config stays blocked in default mode."""
        self._patch_roots(monkeypatch)

        fake_home = tmp_path / "home"
        hermes_dir = fake_home / ".hermes"
        hermes_dir.mkdir(parents=True)
        config_file = hermes_dir / "config.yaml"
        config_file.write_text("model:\n  provider: openai\n")
        monkeypatch.setenv("HOME", str(fake_home))
        monkeypatch.setattr(
            "gateway.platforms.base._HERMES_HOME",
            hermes_dir,
        )

        assert BasePlatformAdapter.validate_media_delivery_path(str(config_file)) is None

    def test_denylist_blocks_shared_hermes_root_config_for_profiles(self, tmp_path, monkeypatch):
        """Profile-mode gateways must still block the shared Hermes root config."""
        self._patch_roots(monkeypatch)

        fake_home = tmp_path / "home"
        profile_home = fake_home / ".hermes" / "profiles" / "work"
        profile_home.mkdir(parents=True)
        hermes_root = fake_home / ".hermes"
        config_file = hermes_root / "config.yaml"
        config_file.write_text("profiles:\n  active: work\n")
        monkeypatch.setenv("HOME", str(fake_home))
        monkeypatch.setattr(
            "gateway.platforms.base._HERMES_HOME",
            profile_home,
        )
        monkeypatch.setattr(
            "gateway.platforms.base._HERMES_ROOT",
            hermes_root,
        )

        assert BasePlatformAdapter.validate_media_delivery_path(str(config_file)) is None

    def test_denylist_blocks_google_token_default_mode(self, tmp_path, monkeypatch):
        """Integration credentials at the HERMES_HOME root (google_token.json)
        must never be deliverable, even though they aren't the historically
        enumerated .env/auth.json/config.yaml files. Regression for a
        refreshed google_token.json being auto-attached to a Slack reply
        (#50912).
        """
        self._patch_roots(monkeypatch)

        fake_home = tmp_path / "home"
        hermes_dir = fake_home / ".hermes"
        hermes_dir.mkdir(parents=True)
        token = hermes_dir / "google_token.json"
        token.write_text('{"access_token": "***", "refresh_token": "***"}')
        monkeypatch.setenv("HOME", str(fake_home))
        monkeypatch.setattr("gateway.platforms.base._HERMES_HOME", hermes_dir)
        monkeypatch.setattr("gateway.platforms.base._HERMES_ROOT", hermes_dir)

        assert BasePlatformAdapter.validate_media_delivery_path(str(token)) is None

    def test_denylist_blocks_google_token_even_when_freshly_refreshed(self, tmp_path, monkeypatch):
        """The exploit was that the Google integration rewrites
        google_token.json every turn, bumping its mtime to ~now, so the
        strict-mode recency window (trust_recent_files) kept re-trusting it
        and it re-sent on every reply. An explicit denylist entry must win
        over recency trust.
        """
        self._patch_roots(monkeypatch)  # zero cache allowlist, strict mode on
        monkeypatch.setenv("HERMES_MEDIA_TRUST_RECENT_FILES", "1")
        monkeypatch.setenv("HERMES_MEDIA_TRUST_RECENT_SECONDS", "600")

        fake_home = tmp_path / "home"
        hermes_dir = fake_home / ".hermes"
        hermes_dir.mkdir(parents=True)
        token = hermes_dir / "google_token.json"
        token.write_text('{"access_token": "***"}')  # mtime = now → "recent"
        monkeypatch.setenv("HOME", str(fake_home))
        monkeypatch.setattr("gateway.platforms.base._HERMES_HOME", hermes_dir)
        monkeypatch.setattr("gateway.platforms.base._HERMES_ROOT", hermes_dir)

        assert BasePlatformAdapter.validate_media_delivery_path(str(token)) is None

    def test_denylist_blocks_pairing_directory_contents(self, tmp_path, monkeypatch):
        """Files under ~/.hermes/pairing/ (platform pairing tokens) are
        credential material and must not be deliverable.
        """
        self._patch_roots(monkeypatch)

        fake_home = tmp_path / "home"
        hermes_dir = fake_home / ".hermes"
        pairing = hermes_dir / "pairing"
        pairing.mkdir(parents=True)
        token = pairing / "telegram-approved.json"
        token.write_text('{"approved": ["123"]}')
        monkeypatch.setenv("HOME", str(fake_home))
        monkeypatch.setattr("gateway.platforms.base._HERMES_HOME", hermes_dir)
        monkeypatch.setattr("gateway.platforms.base._HERMES_ROOT", hermes_dir)

        assert BasePlatformAdapter.validate_media_delivery_path(str(token)) is None

    def test_hermes_cache_still_delivers_under_denied_home(self, tmp_path, monkeypatch):
        """The targeted credential denylist must not break legitimate cache
        deliveries: a generated artifact under the allowlisted cache root is
        matched before the denylist and still delivers.
        """
        fake_home = tmp_path / "home"
        hermes_dir = fake_home / ".hermes"
        cache_dir = hermes_dir / "cache" / "documents"
        cache_dir.mkdir(parents=True)
        artifact = cache_dir / "report.pdf"
        artifact.write_bytes(b"%PDF-1.4")
        self._patch_roots(monkeypatch, cache_dir)
        monkeypatch.setenv("HOME", str(fake_home))
        monkeypatch.setattr("gateway.platforms.base._HERMES_HOME", hermes_dir)
        monkeypatch.setattr("gateway.platforms.base._HERMES_ROOT", hermes_dir)

        assert BasePlatformAdapter.validate_media_delivery_path(str(artifact)) == str(artifact.resolve())

    def test_denylist_blocks_non_cache_file_under_hermes_home(self, tmp_path, monkeypatch):
        """A non-credential file the agent wrote directly under ~/.hermes
        (not in a cache subdir) is still deliverable via recency trust — we
        did NOT blanket-deny the tree (per #32090/#34425). This guards against
        accidentally re-introducing the rejected whole-tree deny.
        """
        self._patch_roots(monkeypatch)  # strict mode on
        monkeypatch.setenv("HERMES_MEDIA_TRUST_RECENT_FILES", "1")
        monkeypatch.setenv("HERMES_MEDIA_TRUST_RECENT_SECONDS", "600")

        fake_home = tmp_path / "home"
        hermes_dir = fake_home / ".hermes"
        hermes_dir.mkdir(parents=True)
        artifact = hermes_dir / "adhoc_report.pdf"
        artifact.write_bytes(b"%PDF-1.4")  # fresh mtime
        monkeypatch.setenv("HOME", str(fake_home))
        monkeypatch.setattr("gateway.platforms.base._HERMES_HOME", hermes_dir)
        monkeypatch.setattr("gateway.platforms.base._HERMES_ROOT", hermes_dir)

        assert BasePlatformAdapter.validate_media_delivery_path(str(artifact)) == str(artifact.resolve())

    def test_strict_mode_envvar_restores_legacy_behavior(self, tmp_path, monkeypatch):
        """Setting HERMES_MEDIA_DELIVERY_STRICT=1 reactivates the older
        allowlist+recency logic. A stale file outside the allowlist is
        rejected.
        """
        self._patch_roots(monkeypatch)
        monkeypatch.setenv("HERMES_MEDIA_DELIVERY_STRICT", "1")
        monkeypatch.setenv("HERMES_MEDIA_TRUST_RECENT_FILES", "0")

        stale = tmp_path / "old.pdf"
        stale.write_bytes(b"%PDF-1.4")
        old_mtime = time.time() - 7200
        os.utime(stale, (old_mtime, old_mtime))

        assert BasePlatformAdapter.validate_media_delivery_path(str(stale)) is None

    def test_strict_mode_truthy_aliases(self, monkeypatch, tmp_path):
        """``HERMES_MEDIA_DELIVERY_STRICT=true|yes|on|1`` all enable strict mode."""
        self._patch_roots(monkeypatch)
        from gateway.platforms.base import _media_delivery_strict_mode

        for raw in ("1", "true", "TRUE", "yes", "on"):
            monkeypatch.setenv("HERMES_MEDIA_DELIVERY_STRICT", raw)
            assert _media_delivery_strict_mode() is True

        for raw in ("0", "false", "no", "off", ""):
            monkeypatch.setenv("HERMES_MEDIA_DELIVERY_STRICT", raw)
            assert _media_delivery_strict_mode() is False

    def test_filter_passes_default_files_through(self, tmp_path, monkeypatch):
        """End-to-end: filter_local_delivery_paths accepts a stale .md in
        default mode where strict mode would drop it.
        """
        self._patch_roots(monkeypatch)

        notes = tmp_path / "notes.md"
        notes.write_text("# old\n")
        os.utime(notes, (time.time() - 86400, time.time() - 86400))

        out = BasePlatformAdapter.filter_local_delivery_paths([str(notes)])
        assert out == [str(notes.resolve())]

    def test_root_home_deliverable_is_accepted(self, tmp_path, monkeypatch):
        """The motivating bug (#38106): a root-run gateway has ``$HOME=/root``,
        which is on the system-prefix denylist. A plain deliverable the agent
        produced in its working dir (``/root/work/proposal.docx``) must still
        deliver — the home itself is not a credential location.
        """
        self._patch_roots(monkeypatch)

        fake_home = tmp_path / "root"
        workdir = fake_home / "work"
        workdir.mkdir(parents=True)
        doc = workdir / "proposal.docx"
        doc.write_bytes(b"PK\x03\x04")
        monkeypatch.setenv("HOME", str(fake_home))
        # $HOME is itself on the denied-prefix list, mirroring /root.
        monkeypatch.setattr(
            "gateway.platforms.base._MEDIA_DELIVERY_DENIED_PREFIXES",
            (str(fake_home),),
        )

        assert (
            BasePlatformAdapter.validate_media_delivery_path(str(doc))
            == str(doc.resolve())
        )

    def test_root_home_credential_subdir_still_blocked(self, tmp_path, monkeypatch):
        """The $HOME exception must NOT un-block credential sub-dirs inside
        home. ``/root/.ssh/id_rsa`` stays denied because ``~/.ssh`` is a
        separate, more-specific denylist entry — even when $HOME is itself a
        denied prefix.
        """
        self._patch_roots(monkeypatch)

        fake_home = tmp_path / "root"
        ssh_dir = fake_home / ".ssh"
        ssh_dir.mkdir(parents=True)
        key = ssh_dir / "id_rsa"
        key.write_bytes(b"-----BEGIN OPENSSH PRIVATE KEY-----")
        monkeypatch.setenv("HOME", str(fake_home))
        monkeypatch.setattr(
            "gateway.platforms.base._MEDIA_DELIVERY_DENIED_PREFIXES",
            (str(fake_home),),
        )

        assert BasePlatformAdapter.validate_media_delivery_path(str(key)) is None

    def test_root_home_hermes_env_still_blocked(self, tmp_path, monkeypatch):
        """``~/.hermes/.env`` stays blocked under the $HOME exception — it is a
        more-specific denied path, not reachable just because home is allowed.
        """
        self._patch_roots(monkeypatch)

        fake_home = tmp_path / "root"
        hermes_dir = fake_home / ".hermes"
        hermes_dir.mkdir(parents=True)
        env_file = hermes_dir / ".env"
        env_file.write_text("OPENROUTER_API_KEY=sk-...")
        monkeypatch.setenv("HOME", str(fake_home))
        monkeypatch.setattr(
            "gateway.platforms.base._MEDIA_DELIVERY_DENIED_PREFIXES",
            (str(fake_home),),
        )
        monkeypatch.setattr("gateway.platforms.base._HERMES_HOME", hermes_dir)

        assert BasePlatformAdapter.validate_media_delivery_path(str(env_file)) is None

    def test_profile_scoped_cache_delivers_under_symlinked_root(self, tmp_path, monkeypatch):
        """Reopened #31733: a profile gateway whose HERMES_HOME is symlinked
        under a denied prefix (e.g. /opt/data -> /root/.hermes) emits
        profile-scoped paths (``<root>/profiles/<name>/cache/images/x.png``)
        that resolve under ``/root``. ``$HOME`` is NOT that prefix, so the
        root-home exception doesn't fire, and the top-level cache allowlist
        doesn't cover the profile subdir — the file was silently dropped.
        Per-profile cache roots must be allowlisted so it delivers.
        """
        self._patch_roots(monkeypatch)  # strict on, zero top-level cache roots

        # Stand-in for the literal /root deny prefix in the deployment.
        denied_root = tmp_path / "root"
        hermes_root = denied_root / ".hermes"
        prof_cache = hermes_root / "profiles" / "myprof" / "cache" / "images"
        prof_cache.mkdir(parents=True)
        image = prof_cache / "gen.png"
        image.write_bytes(b"\x89PNG\r\n\x1a\n")

        # $HOME is NOT the denied prefix (mirrors HOME=/opt/data/home).
        fake_home = tmp_path / "opt" / "data" / "home"
        fake_home.mkdir(parents=True)
        monkeypatch.setenv("HOME", str(fake_home))
        monkeypatch.setattr(
            "gateway.platforms.base._MEDIA_DELIVERY_DENIED_PREFIXES",
            (str(denied_root),),
        )
        monkeypatch.setattr(
            "gateway.platforms.base._HERMES_ROOT", hermes_root
        )

        assert (
            BasePlatformAdapter.validate_media_delivery_path(str(image))
            == str(image.resolve())
        )

    def test_profile_scoped_credential_still_blocked_under_root(self, tmp_path, monkeypatch):
        """The profile-cache allowlist must not un-block a credential sitting
        directly in the profile dir (``profiles/<name>/auth.json``): it's not
        under a cache subdir, so the credential denylist still rejects it.
        """
        self._patch_roots(monkeypatch)

        denied_root = tmp_path / "root"
        hermes_root = denied_root / ".hermes"
        prof_dir = hermes_root / "profiles" / "myprof"
        prof_dir.mkdir(parents=True)
        cred = prof_dir / "auth.json"
        cred.write_text("{}")

        fake_home = tmp_path / "opt" / "data" / "home"
        fake_home.mkdir(parents=True)
        monkeypatch.setenv("HOME", str(fake_home))
        monkeypatch.setattr(
            "gateway.platforms.base._MEDIA_DELIVERY_DENIED_PREFIXES",
            (str(denied_root),),
        )
        monkeypatch.setattr(
            "gateway.platforms.base._HERMES_ROOT", hermes_root
        )

        assert BasePlatformAdapter.validate_media_delivery_path(str(cred)) is None

    def test_other_users_home_still_blocked_for_nonroot(self, tmp_path, monkeypatch):
        """The exception only un-blocks the *running user's own* home. A
        non-root gateway ($HOME=/home/me) must not deliver another user's home
        (``/root/...``) — that prefix stays denied because it isn't $HOME.
        """
        self._patch_roots(monkeypatch)

        my_home = tmp_path / "home" / "me"
        my_home.mkdir(parents=True)
        other_home = tmp_path / "root"
        other_home.mkdir()
        other_file = other_home / "secret.docx"
        other_file.write_bytes(b"PK\x03\x04")
        monkeypatch.setenv("HOME", str(my_home))
        # Both my home and the other home are denied prefixes; only my home is
        # the running user's $HOME, so the other home must stay blocked.
        monkeypatch.setattr(
            "gateway.platforms.base._MEDIA_DELIVERY_DENIED_PREFIXES",
            (str(my_home), str(other_home)),
        )

        assert (
            BasePlatformAdapter.validate_media_delivery_path(str(other_file)) is None
        )

    def test_root_home_workdir_symlink_to_credential_blocked(self, tmp_path, monkeypatch):
        """A symlink in the workdir pointing at a credential is rejected on its
        resolved target, even under the $HOME exception.
        """
        self._patch_roots(monkeypatch)

        fake_home = tmp_path / "root"
        ssh_dir = fake_home / ".ssh"
        ssh_dir.mkdir(parents=True)
        key = ssh_dir / "id_rsa"
        key.write_bytes(b"-----BEGIN OPENSSH PRIVATE KEY-----")
        workdir = fake_home / "work"
        workdir.mkdir()
        link = workdir / "innocent.pdf"
        link.symlink_to(key)
        monkeypatch.setenv("HOME", str(fake_home))
        monkeypatch.setattr(
            "gateway.platforms.base._MEDIA_DELIVERY_DENIED_PREFIXES",
            (str(fake_home),),
        )

        assert BasePlatformAdapter.validate_media_delivery_path(str(link)) is None


# ---------------------------------------------------------------------------
# should_send_media_as_audio
# ---------------------------------------------------------------------------

class TestShouldSendMediaAsAudio:
    """Audio-routing policy shared by gateway + scheduler + send_message."""

    def test_unknown_extension_returns_false(self):
        from gateway.platforms.base import should_send_media_as_audio
        assert should_send_media_as_audio(None, ".png") is False
        assert should_send_media_as_audio("telegram", ".pdf") is False

    def test_non_telegram_platforms_route_all_audio(self):
        from gateway.platforms.base import should_send_media_as_audio
        for ext in (".mp3", ".m4a", ".wav", ".flac", ".ogg", ".opus"):
            assert should_send_media_as_audio("discord", ext) is True
            assert should_send_media_as_audio("slack", ext) is True

    def test_telegram_mp3_and_m4a_route_to_audio(self):
        from gateway.platforms.base import should_send_media_as_audio
        assert should_send_media_as_audio("telegram", ".mp3") is True
        assert should_send_media_as_audio("telegram", ".m4a") is True

    def test_telegram_wav_and_flac_fall_through_to_document(self):
        from gateway.platforms.base import should_send_media_as_audio
        assert should_send_media_as_audio("telegram", ".wav") is False
        assert should_send_media_as_audio("telegram", ".flac") is False

    def test_telegram_ogg_opus_only_when_voice_flagged(self):
        from gateway.platforms.base import should_send_media_as_audio
        assert should_send_media_as_audio("telegram", ".ogg", is_voice=True) is True
        assert should_send_media_as_audio("telegram", ".opus", is_voice=True) is True
        assert should_send_media_as_audio("telegram", ".ogg") is False
        assert should_send_media_as_audio("telegram", ".opus") is False

    def test_accepts_platform_enum(self):
        from gateway.config import Platform
        from gateway.platforms.base import should_send_media_as_audio
        assert should_send_media_as_audio(Platform.TELEGRAM, ".mp3") is True
        assert should_send_media_as_audio(Platform.TELEGRAM, ".flac") is False
        assert should_send_media_as_audio(Platform.DISCORD, ".flac") is True


# ---------------------------------------------------------------------------
# truncate_message
# ---------------------------------------------------------------------------


class TestTruncateMessage:
    def _adapter(self):
        """Create a minimal adapter instance for testing static/instance methods."""

        class StubAdapter(BasePlatformAdapter):
            async def connect(self, *, is_reconnect: bool = False):
                return True

            async def disconnect(self):
                pass

            async def send(self, *a, **kw):
                pass

            async def get_chat_info(self, *a):
                return {}

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
        # Verify all original content is preserved across chunks
        reassembled = "".join(chunks)
        # Strip chunk indicators like (1/N) to get raw content
        for word in msg.strip().split():
            assert word in reassembled, f"Word '{word}' lost during truncation"

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
            reopened_with_lang = any("```javascript" in chunk for chunk in chunks[1:])
            assert reopened_with_lang, (
                "No continuation chunk reopened with language tag"
            )

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
            assert len(chunk) <= max_len + 20, (
                f"Chunk {i} too long: {len(chunk)} > {max_len}"
            )


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

    def test_natural_mode_ignores_malformed_custom_env_vars(self):
        env = {
            "HERMES_HUMAN_DELAY_MODE": "natural",
            "HERMES_HUMAN_DELAY_MIN_MS": "oops",
            "HERMES_HUMAN_DELAY_MAX_MS": "still-bad",
        }
        with patch.dict(os.environ, env):
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

    def test_custom_mode_tolerates_malformed_env_vars(self):
        env = {
            "HERMES_HUMAN_DELAY_MODE": "custom",
            "HERMES_HUMAN_DELAY_MIN_MS": "oops",
            "HERMES_HUMAN_DELAY_MAX_MS": "still-bad",
        }
        with patch.dict(os.environ, env):
            # falls back to the custom-mode defaults instead of crashing
            delay = BasePlatformAdapter._get_human_delay()
            assert 0.8 <= delay <= 2.5


# ---------------------------------------------------------------------------
# utf16_len / _prefix_within_utf16_limit / truncate_message with len_fn
# ---------------------------------------------------------------------------
# Ported from nearai/ironclaw#2304 — Telegram counts message length in UTF-16
# code units, not Unicode code-points.  Astral-plane characters (emoji, CJK
# Extension B) are surrogate pairs: 1 Python char but 2 UTF-16 units.


class TestUtf16Len:
    """Verify the UTF-16 length helper."""

    def test_ascii(self):
        assert utf16_len("hello") == 5

    def test_bmp_cjk(self):
        # CJK ideographs in the BMP are 1 code unit each
        assert utf16_len("你好") == 2

    def test_emoji_surrogate_pair(self):
        # 😀 (U+1F600) is outside BMP → 2 UTF-16 code units
        assert utf16_len("😀") == 2

    def test_mixed(self):
        # "hi😀" = 2 + 2 = 4 UTF-16 units
        assert utf16_len("hi😀") == 4

    def test_musical_symbol(self):
        # 𝄞 (U+1D11E) — Musical Symbol G Clef, surrogate pair
        assert utf16_len("𝄞") == 2

    def test_empty(self):
        assert utf16_len("") == 0


class TestPrefixWithinUtf16Limit:
    """Verify UTF-16-aware prefix truncation."""

    def test_fits_entirely(self):
        assert _prefix_within_utf16_limit("hello", 10) == "hello"

    def test_ascii_truncation(self):
        result = _prefix_within_utf16_limit("hello world", 5)
        assert result == "hello"
        assert utf16_len(result) <= 5

    def test_does_not_split_surrogate_pair(self):
        # "a😀b" = 1 + 2 + 1 = 4 UTF-16 units; limit 2 should give "a"
        result = _prefix_within_utf16_limit("a😀b", 2)
        assert result == "a"
        assert utf16_len(result) <= 2

    def test_emoji_at_limit(self):
        # "😀" = 2 UTF-16 units; limit 2 should include it
        result = _prefix_within_utf16_limit("😀x", 2)
        assert result == "😀"

    def test_all_emoji(self):
        msg = "😀" * 10  # 20 UTF-16 units
        result = _prefix_within_utf16_limit(msg, 6)
        assert result == "😀😀😀"
        assert utf16_len(result) == 6

    def test_empty(self):
        assert _prefix_within_utf16_limit("", 5) == ""


class TestTruncateMessageUtf16:
    """Verify truncate_message respects UTF-16 lengths when len_fn=utf16_len."""

    def test_short_emoji_message_no_split(self):
        """A short message under the UTF-16 limit should not be split."""
        msg = "Hello 😀 world"
        chunks = BasePlatformAdapter.truncate_message(msg, 4096, len_fn=utf16_len)
        assert len(chunks) == 1
        assert chunks[0] == msg

    def test_emoji_near_limit_triggers_split(self):
        """A message at 4096 codepoints but >4096 UTF-16 units must split."""
        # 2049 emoji = 2049 codepoints but 4098 UTF-16 units → exceeds 4096
        msg = "😀" * 2049
        assert len(msg) == 2049  # Python len sees 2049 chars
        assert utf16_len(msg) == 4098  # but it's 4098 UTF-16 units

        # Without UTF-16 awareness, this would NOT split (2049 < 4096)
        chunks_naive = BasePlatformAdapter.truncate_message(msg, 4096)
        assert len(chunks_naive) == 1, "Without len_fn, no split expected"

        # With UTF-16 awareness, it MUST split
        chunks = BasePlatformAdapter.truncate_message(msg, 4096, len_fn=utf16_len)
        assert len(chunks) > 1, "With utf16_len, message should be split"

        # Each chunk must fit within the UTF-16 limit
        for i, chunk in enumerate(chunks):
            assert utf16_len(chunk) <= 4096, (
                f"Chunk {i} exceeds 4096 UTF-16 units: {utf16_len(chunk)}"
            )

    def test_each_utf16_chunk_within_limit(self):
        """All chunks produced with utf16_len must fit the limit."""
        # Mix of BMP and astral-plane characters
        msg = ("Hello 😀 world 🎵 test 𝄞 " * 200).strip()
        max_len = 200
        chunks = BasePlatformAdapter.truncate_message(msg, max_len, len_fn=utf16_len)
        for i, chunk in enumerate(chunks):
            u16_len = utf16_len(chunk)
            assert u16_len <= max_len + 20, (
                f"Chunk {i} UTF-16 length {u16_len} exceeds {max_len}"
            )

    def test_all_content_preserved(self):
        """Splitting with utf16_len must not lose content."""
        words = ["emoji😀", "music🎵", "cjk你好", "plain"] * 100
        msg = " ".join(words)
        chunks = BasePlatformAdapter.truncate_message(msg, 200, len_fn=utf16_len)
        reassembled = " ".join(chunks)
        for word in words:
            assert word in reassembled, f"Word '{word}' lost during UTF-16 split"

    def test_code_blocks_preserved_with_utf16(self):
        """Code block fence handling should work with utf16_len too."""
        msg = "Before\n```python\n" + "x = '😀'\n" * 200 + "```\nAfter"
        chunks = BasePlatformAdapter.truncate_message(msg, 300, len_fn=utf16_len)
        assert len(chunks) > 1
        # Each chunk should have balanced fences
        for i, chunk in enumerate(chunks):
            fence_count = chunk.count("```")
            assert fence_count % 2 == 0, (
                f"Chunk {i} has unbalanced fences ({fence_count})"
            )


class TestProxyKwargsForAiohttp:
    """Verify proxy_kwargs_for_aiohttp routes all schemes through ProxyConnector."""

    def test_none_returns_empty(self):
        from gateway.platforms.base import proxy_kwargs_for_aiohttp

        sess_kw, req_kw = proxy_kwargs_for_aiohttp(None)
        assert sess_kw == {}
        assert req_kw == {}

    def test_http_proxy_uses_connector_when_aiohttp_socks_available(self):
        pytest.importorskip("aiohttp_socks")
        from unittest.mock import MagicMock
        from gateway.platforms.base import proxy_kwargs_for_aiohttp

        sentinel = MagicMock(name="ProxyConnector")
        with patch("aiohttp_socks.ProxyConnector.from_url", return_value=sentinel):
            sess_kw, req_kw = proxy_kwargs_for_aiohttp("http://proxy:8080")
        assert sess_kw.get("connector") is sentinel, (
            "HTTP proxy must use ProxyConnector so libraries that don't "
            "forward per-request proxy= kwargs still route through the proxy"
        )
        assert req_kw == {}

    def test_socks_proxy_uses_connector(self):
        pytest.importorskip("aiohttp_socks")
        from unittest.mock import MagicMock
        from gateway.platforms.base import proxy_kwargs_for_aiohttp

        sentinel = MagicMock(name="ProxyConnector")
        with patch("aiohttp_socks.ProxyConnector.from_url", return_value=sentinel):
            sess_kw, req_kw = proxy_kwargs_for_aiohttp("socks5://proxy:1080")
        assert sess_kw.get("connector") is sentinel
        assert req_kw == {}

    def test_http_proxy_falls_back_without_aiohttp_socks(self):
        from gateway.platforms.base import proxy_kwargs_for_aiohttp

        with patch.dict("sys.modules", {"aiohttp_socks": None}):
            sess_kw, req_kw = proxy_kwargs_for_aiohttp("http://proxy:8080")
            assert sess_kw == {}
            assert req_kw == {"proxy": "http://proxy:8080"}


class TestMediaDeliveryDiagnosability:
    """Diagnosable rejection logging + crafted-path robustness (#33251)."""

    def test_rejected_path_appears_in_log(self, tmp_path, caplog):
        outside = tmp_path / "outside.ogg"
        outside.write_bytes(b"OggS")
        with patch.dict(os.environ, {"HERMES_MEDIA_DELIVERY_STRICT": "1",
                                     "HERMES_MEDIA_TRUST_RECENT_FILES": "0"}), \
                patch("gateway.platforms.base.MEDIA_DELIVERY_SAFE_ROOTS", ()):
            with caplog.at_level("WARNING"):
                out = BasePlatformAdapter.filter_media_delivery_paths([(str(outside), False)])
        assert out == []
        # The dropped path must be in the log so operators can diagnose it.
        assert str(outside) in caplog.text

    def test_crafted_null_path_does_not_abort_batch(self, tmp_path, monkeypatch):
        """One crafted ~\\x00 path must not drop every other attachment."""
        good = tmp_path / "good.png"
        good.write_bytes(b"\x89PNG")
        monkeypatch.setenv("HERMES_MEDIA_DELIVERY_STRICT", "0")
        out = BasePlatformAdapter.filter_media_delivery_paths([
            ("~\x00evil.png", False),
            (str(good), False),
        ])
        assert out == [(str(good.resolve()), False)]

    def test_extract_media_tolerates_crafted_null_path(self):
        """extract_media must not raise on a crafted ~\\x00 MEDIA tag."""
        content = "here\nMEDIA:`~\x00evil.png`\ntrailing"
        # Must not raise ValueError("embedded null byte").
        media, cleaned = BasePlatformAdapter.extract_media(content)
        assert all("\x00" not in p for p, _ in media)

    def test_log_safe_path_neutralises_line_breaks(self):
        forged = "/tmp/a.png\nWARNING forged second line"
        assert "\n" not in _log_safe_path(forged)
        # Unicode separators that split log lines are also neutralised.
        for sep in ("\u2028", "\u2029", "\x85"):
            assert sep not in _log_safe_path(f"/tmp/a{sep}b.png")

    def test_canonical_cache_roots_present(self):
        from gateway.platforms.base import MEDIA_DELIVERY_SAFE_ROOTS
        roots = {str(r) for r in MEDIA_DELIVERY_SAFE_ROOTS}
        assert any(r.endswith("cache/images") for r in roots)
        assert any(r.endswith("cache/documents") for r in roots)
        # Legacy layout still present.
        assert any(r.endswith("image_cache") for r in roots)


# ---------------------------------------------------------------------------
# Media-send fallback must not leak host filesystem paths into chat
# ---------------------------------------------------------------------------


class _CapturingAdapter(BasePlatformAdapter):
    """Minimal concrete BasePlatformAdapter that records what send() sees.

    The four media-send fallbacks (send_voice, send_video, send_document,
    send_image_file) historically forwarded their *_path argument into the
    chat text. That argument is a host filesystem path inside the Hermes
    cache, so any subclass that fell back to super() — like the Telegram
    adapter on a rejected video — would leak the host's directory layout
    into the user's chat.
    """

    def __init__(self):
        from gateway.config import Platform, PlatformConfig
        super().__init__(PlatformConfig(enabled=True), Platform.TELEGRAM)
        self.sent: list[dict] = []

    async def connect(self) -> bool:  # pragma: no cover - not exercised
        return True

    async def disconnect(self) -> None:  # pragma: no cover - not exercised
        return None

    async def get_chat_info(self, chat_id):  # pragma: no cover - not exercised
        return {"name": chat_id, "type": "dm"}

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        from gateway.platforms.base import SendResult
        self.sent.append({
            "chat_id": chat_id,
            "content": content,
            "reply_to": reply_to,
            "metadata": metadata,
        })
        return SendResult(success=True, message_id="m1")


class TestMediaFallbackDoesNotLeakHostPath:
    """Regression: the four base-class media fallbacks must not echo *_path.

    Telegram, Discord, and Slack adapters all fall back to these base
    implementations on native-send failure. When they did, the user saw
    a chat message like ``🎬 Video: /home/.../hermes/cache/video/abc.mp4``
    — a host filesystem path with no actionable information.
    """

    SENSITIVE_PATH = "/home/jayne/.hermes/cache/media/sensitive_host_path_abc123.bin"

    @pytest.mark.asyncio
    async def test_send_voice_fallback_omits_audio_path(self):
        adapter = _CapturingAdapter()
        result = await adapter.send_voice(chat_id="123", audio_path=self.SENSITIVE_PATH)
        assert result.success
        assert len(adapter.sent) == 1
        sent_text = adapter.sent[0]["content"]
        assert self.SENSITIVE_PATH not in sent_text
        assert "/home/" not in sent_text
        assert "audio" in sent_text.lower()

    @pytest.mark.asyncio
    async def test_send_video_fallback_omits_video_path(self):
        adapter = _CapturingAdapter()
        result = await adapter.send_video(chat_id="123", video_path=self.SENSITIVE_PATH)
        assert result.success
        sent_text = adapter.sent[0]["content"]
        assert self.SENSITIVE_PATH not in sent_text
        assert "/home/" not in sent_text
        assert "video" in sent_text.lower()

    @pytest.mark.asyncio
    async def test_send_document_fallback_omits_file_path(self):
        adapter = _CapturingAdapter()
        result = await adapter.send_document(chat_id="123", file_path=self.SENSITIVE_PATH)
        assert result.success
        sent_text = adapter.sent[0]["content"]
        assert self.SENSITIVE_PATH not in sent_text
        assert "/home/" not in sent_text
        assert "file" in sent_text.lower()

    @pytest.mark.asyncio
    async def test_send_document_fallback_includes_explicit_filename_only(self):
        """A caller-supplied file_name is user-facing and may be shown — but
        the host file_path argument must still be suppressed."""
        adapter = _CapturingAdapter()
        result = await adapter.send_document(
            chat_id="123",
            file_path=self.SENSITIVE_PATH,
            file_name="report.pdf",
        )
        assert result.success
        sent_text = adapter.sent[0]["content"]
        assert self.SENSITIVE_PATH not in sent_text
        assert "/home/" not in sent_text
        assert "report.pdf" in sent_text

    @pytest.mark.asyncio
    async def test_send_image_file_fallback_omits_image_path(self):
        adapter = _CapturingAdapter()
        result = await adapter.send_image_file(chat_id="123", image_path=self.SENSITIVE_PATH)
        assert result.success
        sent_text = adapter.sent[0]["content"]
        assert self.SENSITIVE_PATH not in sent_text
        assert "/home/" not in sent_text
        assert "image" in sent_text.lower()

    @pytest.mark.asyncio
    async def test_caption_is_preserved_in_fallback(self):
        """The user-supplied caption is still shown — only the path is suppressed."""
        adapter = _CapturingAdapter()
        await adapter.send_video(
            chat_id="123",
            video_path=self.SENSITIVE_PATH,
            caption="Here's the daily summary.",
        )
        sent_text = adapter.sent[0]["content"]
        assert "Here's the daily summary." in sent_text
        assert self.SENSITIVE_PATH not in sent_text


class TestDockerPathTranslation:
    """Shared Docker container→host path translation (used by send_message and the
    gateway delivery loop). The mount table is stubbed so tests never shell out to
    docker; container_id=None disables the docker-cp fallback, exercising pure
    bind-mount mapping."""

    def test_media_paths_passthrough_when_no_docker_env(self):
        mf = [("/workspace/a.png", False)]
        with patch.object(BasePlatformAdapter, "_get_docker_mount_table", return_value=([], None)):
            assert BasePlatformAdapter.translate_docker_media_paths(mf) == mf

    def test_media_path_mapped_via_bind_mount(self):
        mounts = [("/workspace", "/host/sandbox/workspace")]
        with patch.object(BasePlatformAdapter, "_get_docker_mount_table", return_value=(mounts, None)):
            out = BasePlatformAdapter.translate_docker_media_paths([("/workspace/img.png", True)])
        assert out == [("/host/sandbox/workspace/img.png", True)]

    def test_longest_prefix_wins(self):
        # _get_docker_mount_table returns mounts pre-sorted longest-first.
        mounts = [("/workspace/out", "/host/out"), ("/workspace", "/host/ws")]
        with patch.object(BasePlatformAdapter, "_get_docker_mount_table", return_value=(mounts, None)):
            out = BasePlatformAdapter.translate_docker_media_paths([("/workspace/out/f.mp3", False)])
        assert out == [("/host/out/f.mp3", False)]

    def test_exact_destination_match(self):
        mounts = [("/data/file.bin", "/host/data/file.bin")]
        with patch.object(BasePlatformAdapter, "_get_docker_mount_table", return_value=(mounts, None)):
            out = BasePlatformAdapter.translate_docker_media_paths([("/data/file.bin", False)])
        assert out == [("/host/data/file.bin", False)]

    def test_unmapped_path_passthrough_without_container(self):
        # No matching mount and container_id=None → docker cp skipped, path unchanged.
        with patch.object(BasePlatformAdapter, "_get_docker_mount_table",
                          return_value=([("/workspace", "/host/ws")], None)):
            out = BasePlatformAdapter.translate_docker_media_paths([("/tmp/x.png", False)])
        assert out == [("/tmp/x.png", False)]

    def test_local_paths_mapped(self):
        mounts = [("/workspace", "/host/ws")]
        with patch.object(BasePlatformAdapter, "_get_docker_mount_table", return_value=(mounts, None)):
            out = BasePlatformAdapter.translate_docker_local_paths(["/workspace/doc.pdf", "/elsewhere/y"])
        assert out == ["/host/ws/doc.pdf", "/elsewhere/y"]

    def test_empty_inputs_are_noops(self):
        # Early return before any docker call.
        assert BasePlatformAdapter.translate_docker_media_paths([]) == []
        assert BasePlatformAdapter.translate_docker_local_paths([]) == []

    def test_extraction_uses_exec_cat_not_cp(self):
        # docker cp fails against the overlayfs storage driver on some hosts;
        # exec+cat must be used instead for the last-resort extraction path.
        container_path = "/tmp/song_deadbeef01.mp3"
        dest_path = os.path.join("/tmp", os.path.basename(container_path))
        captured = {}

        def fake_run(cmd, **kwargs):
            captured["cmd"] = cmd
            kwargs["stdout"].write(b"fake-audio-bytes")
            return type("R", (), {"returncode": 0})()

        try:
            with patch("subprocess.run", side_effect=fake_run), \
                 patch.object(BasePlatformAdapter, "_get_docker_mount_table",
                              return_value=([], "abc123")):
                out = BasePlatformAdapter.translate_docker_local_paths([container_path])
            assert out == [dest_path]
            assert captured["cmd"][:3] == ["docker", "exec", "abc123"]
            assert captured["cmd"][3] == "cat"
            assert "cp" not in captured["cmd"]
        finally:
            if os.path.exists(dest_path):
                os.unlink(dest_path)

    def test_extraction_preserves_original_basename(self):
        # Regression: the old implementation renamed extracted files to
        # hermes_docker_media_<md5hash>.<ext>, which some platform adapters
        # then used as a fallback display title instead of the real filename.
        container_path = "/tmp/my_generated_track.mp3"
        dest_path = os.path.join("/tmp", "my_generated_track.mp3")

        def fake_run(cmd, **kwargs):
            kwargs["stdout"].write(b"fake-audio-bytes")
            return type("R", (), {"returncode": 0})()

        try:
            with patch("subprocess.run", side_effect=fake_run), \
                 patch.object(BasePlatformAdapter, "_get_docker_mount_table",
                              return_value=([], "abc123")):
                out = BasePlatformAdapter.translate_docker_local_paths([container_path])
            assert out == [dest_path]
            assert "hermes_docker_media_" not in out[0]
        finally:
            if os.path.exists(dest_path):
                os.unlink(dest_path)

    def test_extraction_failure_falls_back_to_original_path_and_cleans_up(self):
        container_path = "/tmp/missing_in_container.mp3"
        dest_path = os.path.join("/tmp", os.path.basename(container_path))

        def fake_run(cmd, **kwargs):
            return type("R", (), {"returncode": 1})()

        with patch("subprocess.run", side_effect=fake_run), \
             patch.object(BasePlatformAdapter, "_get_docker_mount_table",
                          return_value=([], "abc123")):
            out = BasePlatformAdapter.translate_docker_local_paths([container_path])
        assert out == [container_path]
        assert not os.path.exists(dest_path)
