"""Contract tests for gateway.platforms.media_cache — the shared mime↔ext
dispatch — plus per-adapter parity spot-checks that hardcode each adapter's
HISTORICAL (pre-refactor) mappings as the contract.

If any of these fail, an adapter's downloaded-media filenames changed —
that's a behavioral regression, not a test to update casually.
"""

import pytest

from gateway.platforms.media_cache import (
    cache_media_bytes,
    ext_for_mime,
    mime_for_ext,
)

# ---------------------------------------------------------------------------
# Shared table contract
# ---------------------------------------------------------------------------

class TestSharedTable:

    def test_stage_gating(self):
        # use_defaults=False skips the shared table.
        assert ext_for_mime(
            "audio/ogg", use_defaults=False, use_mimetypes=False, fallback=".x"
        ) == ".x"
        # use_mimetypes=False skips the mimetypes fallback.
        assert ext_for_mime("image/bmp", use_mimetypes=False) is None

    def test_mime_for_ext_fallback_and_case(self):
        assert mime_for_ext(".JPG") == "image/jpeg"
        assert mime_for_ext(".unknown") == "application/octet-stream"
        assert mime_for_ext(".unknown", fallback="x/y") == "x/y"
        assert mime_for_ext(".pdf", overrides={".pdf": "custom/pdf"}) == "custom/pdf"

# ---------------------------------------------------------------------------
# cache_media_bytes dispatch
# ---------------------------------------------------------------------------

class TestCacheMediaBytes:
    PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16

    def test_image_dispatch(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            "gateway.platforms.base.get_image_cache_dir", lambda: tmp_path
        )
        path = cache_media_bytes(self.PNG, "image/png")
        assert path.endswith(".png")

    def test_document_dispatch_uses_filename_hint(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            "gateway.platforms.base.get_document_cache_dir", lambda: tmp_path
        )
        path = cache_media_bytes(b"%PDF-1.4", "application/pdf",
                                 filename_hint="report.pdf")
        assert path.endswith("_report.pdf")

# ---------------------------------------------------------------------------
# Per-adapter parity: HISTORICAL mappings hardcoded as the contract
# ---------------------------------------------------------------------------

class TestWhatsAppCloudParity:
    """Historical _ext_for_mime: overrides → mimetypes → None."""

    CASES = {
        # Pinned overrides (Meta-sent types the STT pipeline needs pinned).
        "audio/ogg": ".ogg",         # NOT mimetypes' .oga
        "audio/x-opus+ogg": ".ogg",
        "audio/opus": ".ogg",
        "audio/mp4": ".m4a",
        "audio/x-m4a": ".m4a",
        "image/jpeg": ".jpg",        # NOT the legacy .jpe
    }

    @pytest.mark.parametrize("mime,expected", sorted(CASES.items()))
    def test_pinned_overrides(self, mime, expected):
        from gateway.platforms.whatsapp_cloud import _ext_for_mime
        assert _ext_for_mime(mime) == expected
