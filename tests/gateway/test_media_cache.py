"""Contract tests for gateway.platforms.media_cache — the shared mime↔ext
dispatch — plus per-adapter parity spot-checks that hardcode each adapter's
HISTORICAL (pre-refactor) mappings as the contract.

If any of these fail, an adapter's downloaded-media filenames changed —
that's a behavioral regression, not a test to update casually.
"""

import mimetypes

import pytest

from gateway.platforms.media_cache import (
    DEFAULT_EXT_TO_MIME,
    DEFAULT_MIME_TO_EXT,
    cache_media_bytes,
    ext_for_mime,
    mime_for_ext,
)


# ---------------------------------------------------------------------------
# Shared table contract
# ---------------------------------------------------------------------------

class TestSharedTable:
    def test_defaults_resolve(self):
        for mime, ext in DEFAULT_MIME_TO_EXT.items():
            assert ext_for_mime(mime) == ext

    def test_overrides_always_win(self):
        # Every override is honored even when the default table or
        # mimetypes disagree.
        assert ext_for_mime("image/heic", overrides={"image/heic": ".jpg"}) == ".jpg"
        assert ext_for_mime("audio/ogg", overrides={"audio/ogg": ".weird"}) == ".weird"
        assert ext_for_mime("image/jpeg", overrides={"image/jpeg": ".jpeg"}) == ".jpeg"

    def test_mime_parameters_stripped(self):
        assert ext_for_mime("audio/ogg; codecs=opus") == ".ogg"
        assert ext_for_mime("IMAGE/JPEG; charset=binary") == ".jpg"

    def test_unknown_mime_falls_back_to_mimetypes_then_fallback(self):
        # Known to mimetypes but not our table.
        assert ext_for_mime("image/bmp") == mimetypes.guess_extension("image/bmp")
        # Unknown everywhere → explicit fallback.
        assert ext_for_mime("application/x-no-such-type", fallback=".bin") == ".bin"
        assert ext_for_mime("application/x-no-such-type") is None

    def test_empty_mime_returns_fallback(self):
        assert ext_for_mime("") is None
        assert ext_for_mime("", fallback=".bin") == ".bin"

    def test_stage_gating(self):
        # use_defaults=False skips the shared table.
        assert ext_for_mime(
            "audio/ogg", use_defaults=False, use_mimetypes=False, fallback=".x"
        ) == ".x"
        # use_mimetypes=False skips the mimetypes fallback.
        assert ext_for_mime("image/bmp", use_mimetypes=False) is None

    def test_inverse_map_consistent_with_forward(self):
        # Round-trip: every inverse entry's mime maps forward to an ext
        # whose inverse is the same mime (canonical closure).
        for ext, mime in DEFAULT_EXT_TO_MIME.items():
            fwd_ext = ext_for_mime(mime)
            assert fwd_ext is not None
            assert mime_for_ext(fwd_ext) == mime

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

    def test_audio_dispatch(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            "gateway.platforms.base.get_audio_cache_dir", lambda: tmp_path
        )
        path = cache_media_bytes(b"RIFF\x00\x00\x00\x00WAVEfmt ", "audio/wav")
        assert path.endswith(".wav")

    def test_document_dispatch_uses_filename_hint(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            "gateway.platforms.base.get_document_cache_dir", lambda: tmp_path
        )
        path = cache_media_bytes(b"%PDF-1.4", "application/pdf",
                                 filename_hint="report.pdf")
        assert path.endswith("_report.pdf")

    def test_document_dispatch_generates_name(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            "gateway.platforms.base.get_document_cache_dir", lambda: tmp_path
        )
        path = cache_media_bytes(b"%PDF-1.4", "application/pdf")
        assert path.endswith(".pdf")

    def test_kind_hint_forces_cache(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            "gateway.platforms.base.get_document_cache_dir", lambda: tmp_path
        )
        # Image mime but explicit document hint → document cache.
        path = cache_media_bytes(self.PNG, "image/png", kind_hint="document",
                                 filename_hint="pic.png")
        assert path.endswith("_pic.png")

    def test_ext_overrides_threaded(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            "gateway.platforms.base.get_image_cache_dir", lambda: tmp_path
        )
        path = cache_media_bytes(
            self.PNG, "image/png", ext_overrides={"image/png": ".png2"}
        )
        assert path.endswith(".png2")


# ---------------------------------------------------------------------------
# Per-adapter parity: HISTORICAL mappings hardcoded as the contract
# ---------------------------------------------------------------------------

class TestBlueBubblesParity:
    """Historical closed maps from bluebubbles._download_attachment."""

    IMAGE_CASES = {
        "image/jpeg": ".jpg",
        "image/png": ".png",
        "image/gif": ".gif",
        "image/webp": ".webp",
        "image/heic": ".jpg",   # historically coerced to .jpg
        "image/heif": ".jpg",   # historically coerced to .jpg
        "image/tiff": ".jpg",   # historically coerced to .jpg
        "image/bmp": ".jpg",    # unlisted → historical .jpg fallback
    }
    AUDIO_CASES = {
        "audio/mp3": ".mp3",
        "audio/mpeg": ".mp3",
        "audio/ogg": ".ogg",
        "audio/wav": ".wav",
        "audio/x-caf": ".mp3",  # historically coerced to .mp3
        "audio/mp4": ".m4a",
        "audio/aac": ".m4a",    # historically .m4a (NOT .aac)
        "audio/flac": ".mp3",   # unlisted → historical .mp3 fallback
    }

    @pytest.mark.parametrize("mime,expected", sorted(IMAGE_CASES.items()))
    def test_image_map(self, mime, expected):
        from gateway.platforms.bluebubbles import _BLUEBUBBLES_IMAGE_EXT_OVERRIDES
        got = ext_for_mime(
            mime,
            overrides=_BLUEBUBBLES_IMAGE_EXT_OVERRIDES,
            use_defaults=False,
            use_mimetypes=False,
            fallback=".jpg",
        )
        assert got == expected

    @pytest.mark.parametrize("mime,expected", sorted(AUDIO_CASES.items()))
    def test_audio_map(self, mime, expected):
        from gateway.platforms.bluebubbles import _BLUEBUBBLES_AUDIO_EXT_OVERRIDES
        got = ext_for_mime(
            mime,
            overrides=_BLUEBUBBLES_AUDIO_EXT_OVERRIDES,
            use_defaults=False,
            use_mimetypes=False,
            fallback=".mp3",
        )
        assert got == expected


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

    def test_unpinned_falls_to_mimetypes(self):
        from gateway.platforms.whatsapp_cloud import _ext_for_mime
        assert _ext_for_mime("application/pdf") == mimetypes.guess_extension(
            "application/pdf"
        )

    def test_unknown_returns_none(self):
        from gateway.platforms.whatsapp_cloud import _ext_for_mime
        assert _ext_for_mime("application/x-no-such-type") is None
        assert _ext_for_mime("") is None

    def test_parameters_stripped(self):
        from gateway.platforms.whatsapp_cloud import _ext_for_mime
        assert _ext_for_mime("audio/ogg; codecs=opus") == ".ogg"


class TestSignalParity:
    """Historical _EXT_TO_MIME table from signal.py, verbatim."""

    HISTORICAL = {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
        ".gif": "image/gif", ".webp": "image/webp",
        ".ogg": "audio/ogg", ".mp3": "audio/mpeg", ".wav": "audio/wav",
        ".m4a": "audio/mp4", ".aac": "audio/aac",
        ".mp4": "video/mp4", ".pdf": "application/pdf",
        ".zip": "application/zip",
    }

    @pytest.mark.parametrize("ext,expected", sorted(HISTORICAL.items()))
    def test_table(self, ext, expected):
        from gateway.platforms.signal import _ext_to_mime
        assert _ext_to_mime(ext) == expected
        assert _ext_to_mime(ext.upper()) == expected

    def test_unknown_ext(self):
        from gateway.platforms.signal import _ext_to_mime
        assert _ext_to_mime(".xyz") == "application/octet-stream"

    def test_shared_table_matches_historical_verbatim(self):
        assert DEFAULT_EXT_TO_MIME == self.HISTORICAL


class TestQQBotParity:
    """Historical qqbot image path: mimetypes.guess_extension or '.jpg'."""

    @pytest.mark.parametrize("mime", [
        "image/jpeg", "image/png", "image/gif", "image/webp", "image/bmp",
    ])
    def test_trusts_mimetypes(self, mime):
        historical = mimetypes.guess_extension(mime) or ".jpg"
        got = ext_for_mime(
            mime, use_defaults=False, use_mimetypes=True, fallback=".jpg"
        ) or ".jpg"
        assert got == historical

    def test_unknown_image_mime_falls_back_to_jpg(self):
        got = ext_for_mime(
            "image/x-no-such-type",
            use_defaults=False, use_mimetypes=True, fallback=".jpg",
        ) or ".jpg"
        assert got == ".jpg"
