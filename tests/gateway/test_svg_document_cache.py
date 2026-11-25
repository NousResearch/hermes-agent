"""
SVG attachments must reach the document cache.

SVG is an ``image/*`` MIME but XML text, so the raster image cache rejects it
(``Refusing to cache non-image data``) and ``cache_media_bytes`` used to return
``None`` — the attachment was dropped before the agent ever saw it.

Reported per platform: Slack (#131738 — "SVG attachments are rejected as
non-image data"), Desktop (#157056 — generic "turn failed" card), Telegram
("Document '<name>.svg' could not be cached", adapter.py).

The fix is in the shared classifier, so these assertions hold for every adapter
that routes bytes through ``cache_media_bytes``.
"""

from pathlib import Path

import pytest

from gateway.platforms import base

SVG_BYTES = (
    b'<?xml version="1.0" encoding="UTF-8"?>'
    b'<svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 12 12">'
    b'<rect width="12" height="12" fill="#FF401F"/></svg>'
)
# Smallest viable PNG (1x1) — enough for the image path's magic-byte validation.
PNG_BYTES = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000"
    "002af15c890000000a49444154789c63000100000500010d0a2db400"
    "00000049454e44ae426082"
)


@pytest.fixture(autouse=True)
def _redirect_caches(tmp_path, monkeypatch):
    """Keep every cache write inside tmp_path."""
    monkeypatch.setattr(base, "DOCUMENT_CACHE_DIR", tmp_path / "documents")
    monkeypatch.setattr(base, "IMAGE_CACHE_DIR", tmp_path / "images")


class TestSvgCachedAsDocument:
    """An SVG never takes the raster image branch."""

    def test_svg_mime_lands_in_document_cache(self):
        cached = base.cache_media_bytes(
            SVG_BYTES, filename="logo.svg", mime_type="image/svg+xml"
        )
        assert cached is not None, "an SVG must not be dropped by the image branch"
        assert cached.kind == "document"
        path = Path(cached.path)
        assert path.exists()
        assert path.read_bytes() == SVG_BYTES, "vector source must survive byte-identically"

    def test_svg_extension_without_mime_is_a_document(self):
        cached = base.cache_media_bytes(SVG_BYTES, filename="logo.svg")
        assert cached is not None
        assert cached.kind == "document"

    def test_svg_mime_without_filename_is_a_document(self):
        cached = base.cache_media_bytes(SVG_BYTES, mime_type="image/svg+xml")
        assert cached is not None
        assert cached.kind == "document"


class TestRasterPathUnchanged:
    """The skip is SVG-only: real bitmaps still use the image cache."""

    def test_png_is_still_an_image(self):
        cached = base.cache_media_bytes(PNG_BYTES, filename="shot.png", mime_type="image/png")
        assert cached is not None
        assert cached.kind == "image"

    def test_png_without_filename_is_still_an_image(self):
        cached = base.cache_media_bytes(PNG_BYTES, mime_type="image/png")
        assert cached is not None
        assert cached.kind == "image"


class TestSvgBookkeeping:
    """SVG must be a first-class document type, not only a flipped branch."""

    def test_document_type_table_knows_svg(self):
        assert base.SUPPORTED_DOCUMENT_TYPES[".svg"] == "image/svg+xml"

    def test_svg_source_is_text_injectable(self):
        assert ".svg" in base._TEXT_INJECT_EXTENSIONS

    def test_cached_svg_reports_its_mime(self):
        cached = base.cache_media_bytes(SVG_BYTES, filename="logo.svg")
        assert cached is not None
        assert cached.media_type == "image/svg+xml"


class TestTelegramPairingClassifier:
    """``is_image_document`` gates the Telegram download path; SVG must opt out."""

    class _Doc:
        def __init__(self, file_name, mime_type):
            self.file_name = file_name
            self.mime_type = mime_type

    @staticmethod
    def _classifier():
        from plugins.platforms.telegram.telegram_text_photo_pairing import is_image_document

        return is_image_document

    def test_svg_document_is_not_an_image_document(self):
        assert self._classifier()(self._Doc("logo.svg", "image/svg+xml")) is False

    def test_svg_mime_only_is_not_an_image_document(self):
        assert self._classifier()(self._Doc("logo", "image/svg+xml")) is False

    def test_png_document_is_still_an_image_document(self):
        assert self._classifier()(self._Doc("shot.png", "image/png")) is True
