"""Tests for MEDIA_TAG_CLEANUP_RE regex matching behavior (#63632)."""


class TestMediaTagCleanup:
    """Tests for MEDIA_TAG_CLEANUP_RE regex matching behavior."""

    def test_media_tag_with_directive_glued_to_extension(self):
        """Regression: MEDIA:<path>[[as_document]] must match when directive is glued
        directly to the extension without whitespace (#63632).

        The fix adds `\\[` to the lookahead character class in MEDIA_TAG_CLEANUP_RE.
        """
        from gateway.platforms.base import MEDIA_TAG_CLEANUP_RE

        # Issue case: [[as_document]] glued directly to .xlsx
        text = "Готово. MEDIA:/home/hermes/report.xlsx[[as_document]]"
        assert MEDIA_TAG_CLEANUP_RE.search(text) is not None
        stripped = MEDIA_TAG_CLEANUP_RE.sub("", text)
        assert "MEDIA:" not in stripped
        assert "/home/hermes/report.xlsx" not in stripped

        # Same with whitespace (should still work)
        text_with_space = "Готово. MEDIA:/home/hermes/report.xlsx [[as_document]]"
        assert MEDIA_TAG_CLEANUP_RE.search(text_with_space) is not None
        stripped = MEDIA_TAG_CLEANUP_RE.sub("", text_with_space)
        assert "MEDIA:" not in stripped
        assert "/home/hermes/report.xlsx" not in stripped

        # Other directives ([[as_image]]) should also work
        text_image = "Done. MEDIA:/tmp/chart.png[[as_image]]"
        assert MEDIA_TAG_CLEANUP_RE.search(text_image) is not None
        stripped = MEDIA_TAG_CLEANUP_RE.sub("", text_image)
        assert "MEDIA:" not in stripped
        assert "/tmp/chart.png" not in stripped


