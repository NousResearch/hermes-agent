"""
Tests for the opt-in ``MEDIA_CAPTION`` directive.

``MEDIA_CAPTION:{"path": ..., "caption": ..., "type": "image"|"video"}``
lets a tool request that a local image/video be delivered as a native
media bubble with the caption attached to the bubble itself, instead of
the caption landing as a separate text message. The directive is opt-in:
plain ``MEDIA:`` tags and markdown images (``![alt](url)``) are untouched,
so existing flows are unaffected.

Coverage:
- ``extract_captioned_media`` parses valid directives and rejects unsafe /
  malformed / type-mismatched ones.
- ``split_captioned_media_text`` partitions the surrounding text into the
  opening (before the first directive) and closing (after the last) so the
  gateway can preserve ``[opening] [media+caption ...] [closing]`` order.
- ``strip_media_directives_for_display`` hides the raw directive from the
  streamed text.
"""

import pytest

from gateway.platforms.base import BasePlatformAdapter


# A 1x1 PNG so validate_media_delivery_path() sees a real, non-empty file.
_PNG_BYTES = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01"
    b"\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
)


@pytest.fixture
def image_file(tmp_path):
    p = tmp_path / "planta.png"
    p.write_bytes(_PNG_BYTES)
    return str(p)


@pytest.fixture
def second_image_file(tmp_path):
    p = tmp_path / "cozinha.jpg"
    p.write_bytes(_PNG_BYTES)
    return str(p)


class TestExtractCaptionedMedia:
    def test_no_directive_is_noop(self):
        text = "Just a regular reply with no media."
        items, cleaned = BasePlatformAdapter.extract_captioned_media(text)
        assert items == []
        assert cleaned == text

    def test_single_image_directive(self, image_file):
        text = (
            "Here is the material:\n\n"
            'MEDIA_CAPTION:{"path": "%s", "caption": "Floor plan", "type": "image"}'
            % image_file
        )
        items, cleaned = BasePlatformAdapter.extract_captioned_media(text)
        assert len(items) == 1
        assert items[0]["type"] == "image"
        assert items[0]["path"].endswith("planta.png")
        assert items[0]["caption"] == "Floor plan"
        # The raw directive must not survive into the display text.
        assert "MEDIA_CAPTION" not in cleaned
        assert "Here is the material:" in cleaned

    def test_multiple_directives_preserve_order(self, image_file, second_image_file):
        text = (
            'MEDIA_CAPTION:{"path": "%s", "caption": "First", "type": "image"}\n\n'
            'MEDIA_CAPTION:{"path": "%s", "caption": "Second", "type": "image"}'
            % (image_file, second_image_file)
        )
        items, _ = BasePlatformAdapter.extract_captioned_media(text)
        assert [i["caption"] for i in items] == ["First", "Second"]

    def test_missing_caption_yields_none(self, image_file):
        text = 'MEDIA_CAPTION:{"path": "%s", "type": "image"}' % image_file
        items, _ = BasePlatformAdapter.extract_captioned_media(text)
        assert len(items) == 1
        assert items[0]["caption"] is None

    def test_type_must_match_extension(self, image_file):
        # Declared as video but the file is a .png -> rejected.
        text = 'MEDIA_CAPTION:{"path": "%s", "caption": "x", "type": "video"}' % image_file
        items, cleaned = BasePlatformAdapter.extract_captioned_media(text)
        assert items == []
        assert cleaned == text

    def test_unknown_type_rejected(self, tmp_path):
        doc = tmp_path / "contract.pdf"
        doc.write_bytes(b"%PDF-1.4 fake")
        text = 'MEDIA_CAPTION:{"path": "%s", "caption": "x", "type": "document"}' % str(doc)
        items, _ = BasePlatformAdapter.extract_captioned_media(text)
        assert items == []

    def test_nonexistent_path_rejected(self):
        text = 'MEDIA_CAPTION:{"path": "/tmp/does-not-exist-1234.png", "caption": "x", "type": "image"}'
        items, _ = BasePlatformAdapter.extract_captioned_media(text)
        assert items == []

    def test_malformed_json_ignored(self):
        text = 'MEDIA_CAPTION:{"path": "/tmp/a.png", "caption": '
        items, cleaned = BasePlatformAdapter.extract_captioned_media(text)
        assert items == []
        assert cleaned == text

    def test_markdown_image_is_not_intercepted(self):
        # The legacy markdown-image path stays opt-out.
        text = "![cat](https://example.com/cat.png)"
        items, cleaned = BasePlatformAdapter.extract_captioned_media(text)
        assert items == []
        assert cleaned == text


class TestSplitCaptionedMediaText:
    def test_no_directive_returns_empty_pair(self):
        assert BasePlatformAdapter.split_captioned_media_text("plain text") == ("", "")

    def test_opening_and_closing_are_partitioned(self, image_file, second_image_file):
        text = (
            "Here is the material:\n\n"
            'MEDIA_CAPTION:{"path": "%s", "caption": "First", "type": "image"}\n\n'
            'MEDIA_CAPTION:{"path": "%s", "caption": "Second", "type": "image"}\n\n'
            "Anything else?"
        ) % (image_file, second_image_file)
        opening, closing = BasePlatformAdapter.split_captioned_media_text(text)
        assert opening == "Here is the material:"
        assert closing == "Anything else?"
        # Neither segment leaks a raw directive.
        assert "MEDIA_CAPTION" not in opening
        assert "MEDIA_CAPTION" not in closing


class TestStripForDisplay:
    def test_directive_removed_from_display(self, image_file):
        text = (
            "Here is the material:\n\n"
            'MEDIA_CAPTION:{"path": "%s", "caption": "Floor plan", "type": "image"}\n\n'
            "Anything else?"
        ) % image_file
        cleaned = BasePlatformAdapter.strip_media_directives_for_display(text)
        assert "MEDIA_CAPTION" not in cleaned
        assert "Here is the material:" in cleaned
        assert "Anything else?" in cleaned
