"""Tests for HEIC/HEIF image handling (utils.transcode_heic_to_jpeg and
the cache/data-URL choke points that use it).

HEIC is the default iPhone camera format; vision APIs (Anthropic, OpenAI)
reject ``image/heic`` payloads, so Hermes transcodes to JPEG at three choke
points: gateway image cache, agent data-URL builder, and the vision tool.
These tests use a synthetic in-memory HEIC header for detection tests and
mock the transcoder for wiring tests — no decoder dependency in CI.
"""

import base64
from pathlib import Path
from unittest import mock

import pytest

from utils import HEIC_FTYP_BRANDS, looks_like_heic, transcode_heic_to_jpeg

# Minimal ISO-BMFF prefix carrying the 'heic' brand — enough for magic-byte
# detection without a real (decodable) image payload.
FAKE_HEIC = b"\x00\x00\x00\x24ftypheic\x00\x00\x00\x00mif1MiPr" + b"\x00" * 64
JPEG_BYTES = b"\xff\xd8\xff\xe0" + b"\x00" * 32
PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32


class TestLooksLikeHeic:
    def test_detects_heic_brand(self):
        assert looks_like_heic(FAKE_HEIC) is True

    @pytest.mark.parametrize("brand", sorted(HEIC_FTYP_BRANDS))
    def test_detects_all_brands(self, brand):
        data = b"\x00\x00\x00\x24ftyp" + brand + b"\x00" * 16
        assert looks_like_heic(data) is True

    def test_rejects_jpeg(self):
        assert looks_like_heic(JPEG_BYTES) is False

    def test_rejects_png(self):
        assert looks_like_heic(PNG_BYTES) is False

    def test_rejects_short_data(self):
        assert looks_like_heic(b"\x00\x00") is False

    def test_rejects_mp4_brand(self):
        # mp42 is ISO-BMFF too but not a HEIF brand.
        assert looks_like_heic(b"\x00\x00\x00\x24ftypmp42" + b"\x00" * 16) is False


class TestTranscodeHeicToJpeg:
    def test_non_heic_returns_none(self):
        assert transcode_heic_to_jpeg(JPEG_BYTES) is None

    def test_returns_none_when_no_decoder(self):
        """With pillow-heif missing and sips absent, falls back to None."""
        with mock.patch("shutil.which", return_value=None), \
             mock.patch.dict("sys.modules", {"pillow_heif": None}):
            assert transcode_heic_to_jpeg(FAKE_HEIC) is None

    def test_real_transcode_if_decoder_available(self):
        """End-to-end conversion when a real decoder exists on this host."""
        import shutil
        try:
            import pillow_heif  # noqa: F401
            have_decoder = True
        except ImportError:
            have_decoder = shutil.which("sips") is not None
        if not have_decoder:
            pytest.skip("no HEIC decoder on this host")
        # Build a real HEIC via Pillow→sips only on macOS; otherwise skip.
        if not shutil.which("sips"):
            pytest.skip("sips unavailable to build fixture")
        from PIL import Image
        import subprocess, tempfile
        with tempfile.TemporaryDirectory() as td:
            png = Path(td) / "src.png"
            heic = Path(td) / "src.heic"
            Image.new("RGB", (32, 24), (10, 200, 100)).save(png)
            subprocess.run(
                ["sips", "-s", "format", "heic", str(png), "--out", str(heic)],
                check=True, capture_output=True,
            )
            out = transcode_heic_to_jpeg(heic.read_bytes())
        assert out is not None
        assert out[:3] == b"\xff\xd8\xff"  # JPEG magic


class TestCacheImageFromBytes:
    def test_heic_accepted_and_transcoded(self, tmp_path, monkeypatch):
        from gateway.platforms import base as gw_base

        monkeypatch.setattr(gw_base, "get_image_cache_dir", lambda: tmp_path)
        with mock.patch.object(
            gw_base, "transcode_heic_to_jpeg", return_value=JPEG_BYTES,
        ) as transcoder:
            path = gw_base.cache_image_from_bytes(FAKE_HEIC, ".heic")
        transcoder.assert_called_once()
        assert path.endswith(".jpg")
        assert Path(path).read_bytes() == JPEG_BYTES

    def test_heic_kept_when_no_decoder(self, tmp_path, monkeypatch):
        """No decoder → original bytes cached (pre-fix behaviour, no regression)."""
        from gateway.platforms import base as gw_base

        monkeypatch.setattr(gw_base, "get_image_cache_dir", lambda: tmp_path)
        with mock.patch.object(
            gw_base, "transcode_heic_to_jpeg", return_value=None,
        ):
            path = gw_base.cache_image_from_bytes(FAKE_HEIC, ".heic")
        assert path.endswith(".heic")
        assert Path(path).read_bytes() == FAKE_HEIC

    def test_non_image_still_rejected(self):
        from gateway.platforms.base import cache_image_from_bytes

        with pytest.raises(ValueError):
            cache_image_from_bytes(b"<html>error page</html>", ".jpg")

    def test_jpeg_passthrough_untouched(self, tmp_path, monkeypatch):
        from gateway.platforms import base as gw_base

        monkeypatch.setattr(gw_base, "get_image_cache_dir", lambda: tmp_path)
        path = gw_base.cache_image_from_bytes(JPEG_BYTES, ".jpg")
        assert Path(path).read_bytes() == JPEG_BYTES


class TestFileToDataUrl:
    def test_heic_transcoded_to_jpeg_data_url(self, tmp_path):
        from agent import image_routing

        heic_file = tmp_path / "photo.heic"
        heic_file.write_bytes(FAKE_HEIC)
        with mock.patch(
            "utils.transcode_heic_to_jpeg", return_value=JPEG_BYTES,
        ):
            url = image_routing._file_to_data_url(heic_file)
        assert url is not None
        assert url.startswith("data:image/jpeg;base64,")
        assert base64.b64decode(url.split(",", 1)[1]) == JPEG_BYTES

    def test_heic_unchanged_when_no_decoder(self, tmp_path):
        from agent import image_routing

        heic_file = tmp_path / "photo.heic"
        heic_file.write_bytes(FAKE_HEIC)
        with mock.patch(
            "utils.transcode_heic_to_jpeg", return_value=None,
        ):
            url = image_routing._file_to_data_url(heic_file)
        assert url is not None
        assert url.startswith("data:image/heic;base64,")

    def test_jpeg_not_routed_through_transcoder(self, tmp_path):
        from agent import image_routing

        jpg_file = tmp_path / "photo.jpg"
        jpg_file.write_bytes(JPEG_BYTES)
        with mock.patch(
            "utils.transcode_heic_to_jpeg",
        ) as transcoder:
            url = image_routing._file_to_data_url(jpg_file)
        transcoder.assert_not_called()
        assert url is not None and url.startswith("data:image/jpeg;base64,")


class TestVisionToolHeic:
    def test_detect_image_mime_type_heic(self, tmp_path):
        from tools.vision_tools import _detect_image_mime_type

        f = tmp_path / "img.heic"
        f.write_bytes(FAKE_HEIC)
        assert _detect_image_mime_type(f) == "image/heic"

    def test_maybe_transcode_heic_swaps_path_and_mime(self, tmp_path, monkeypatch):
        from tools import vision_tools

        f = tmp_path / "img.heic"
        f.write_bytes(FAKE_HEIC)
        monkeypatch.setattr(
            vision_tools, "get_hermes_dir", lambda *a, **k: tmp_path,
        )
        with mock.patch(
            "utils.transcode_heic_to_jpeg", return_value=JPEG_BYTES,
        ):
            new_path, new_mime = vision_tools._maybe_transcode_heic(
                f, "image/heic",
            )
        assert new_mime == "image/jpeg"
        assert new_path != f
        assert new_path.read_bytes() == JPEG_BYTES

    def test_maybe_transcode_noop_for_other_mimes(self, tmp_path):
        from tools.vision_tools import _maybe_transcode_heic

        f = tmp_path / "img.png"
        f.write_bytes(PNG_BYTES)
        assert _maybe_transcode_heic(f, "image/png") == (f, "image/png")
