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


class TestMediaTagCjkTerminators:
    """#88038: CJK full-width punctuation after a MEDIA path is a valid
    terminator, exactly like its ASCII counterparts. Chinese-language agent
    output naturally writes ``MEDIA:D:\\...\\zhibao.pdf（782.6 KB）`` — the
    ASCII-only lookahead silently dropped the attachment."""

    def test_full_width_suffixes_terminate_the_path(self):
        from gateway.platforms.base import MEDIA_TAG_CLEANUP_RE

        cases = [
            "MEDIA:D:/workspace/out/report.pdf（782.6 KB）",
            "MEDIA:D:/workspace/out/report.pdf：内容",
            "MEDIA:D:/workspace/out/report.pdf。",
            "MEDIA:D:/workspace/out/report.pdf，下一条",
            "MEDIA:D:/workspace/out/report.pdf；",
            "MEDIA:D:/workspace/out/report.pdf！",
            "MEDIA:D:/workspace/out/report.pdf？",
            "MEDIA:D:/workspace/out/report.pdf、",
            "MEDIA:D:/workspace/out/report.pdf”",
            "MEDIA:D:/workspace/out/report.pdf’",
        ]
        for text in cases:
            m = MEDIA_TAG_CLEANUP_RE.search(text)
            assert m is not None, text
            assert m.group("path").endswith(".pdf"), (text, m.group("path"))

    def test_real_chinese_cron_delivery_line(self):
        from gateway.platforms.base import MEDIA_TAG_CLEANUP_RE

        text = (
            "## 交付物\n\n- **PDF 早报**："
            "MEDIA:D:/workspace/zaobao/output/早报_2026-08-16.pdf（782.6 KB）"
        )
        m = MEDIA_TAG_CLEANUP_RE.search(text)
        assert m is not None
        assert m.group("path") == "D:/workspace/zaobao/output/早报_2026-08-16.pdf"

    def test_ascii_terminators_still_work(self):
        from gateway.platforms.base import MEDIA_TAG_CLEANUP_RE

        for text in (
            "MEDIA:/tmp/report.pdf",
            "MEDIA:/tmp/report.pdf 782 KB",
            "MEDIA:/tmp/report.pdf, next",
        ):
            m = MEDIA_TAG_CLEANUP_RE.search(text)
            assert m is not None and m.group("path").endswith(".pdf"), text

    def test_adjacent_tags_still_split(self):
        """#68773 guard: the widened class must not let tags glue together."""
        from gateway.platforms.base import MEDIA_TAG_CLEANUP_RE

        found = MEDIA_TAG_CLEANUP_RE.findall("MEDIA:/a.pngMEDIA:/b.png")
        assert list(found) == ["/a.png", "/b.png"]

    def test_extensionless_variant_accepts_full_width_colon(self):
        from gateway.platforms.base import MEDIA_EXTENSIONLESS_TAG_RE

        m = MEDIA_EXTENSIONLESS_TAG_RE.search("MEDIA:/tmp/Caddyfile：内容")
        assert m is not None and m.group("path") == "/tmp/Caddyfile"


