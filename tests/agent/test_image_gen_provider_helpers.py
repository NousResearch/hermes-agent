"""Tests for shared helpers in ``agent.image_gen_provider``.

Today this module only covers :func:`normalize_reference_image`, which was
introduced alongside reference-image (image-to-image) support for the
``openai-codex`` provider. Other providers can reuse the helper as they
gain reference-image support.
"""

from __future__ import annotations

import base64
from pathlib import Path

import pytest

from agent.image_gen_provider import normalize_reference_image


# 1×1 transparent PNG
_PNG_HEX = (
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4"
    "890000000d49444154789c6300010000000500010d0a2db40000000049454e44"
    "ae426082"
)


def _png_bytes() -> bytes:
    return bytes.fromhex(_PNG_HEX)


class TestNormalizeReferenceImage:
    def test_none_returns_none(self):
        assert normalize_reference_image(None) is None

    def test_empty_string_returns_none(self):
        assert normalize_reference_image("") is None
        assert normalize_reference_image("   ") is None

    def test_unsupported_type_returns_none(self):
        assert normalize_reference_image(12345) is None
        assert normalize_reference_image({"path": "/tmp/x.png"}) is None

    def test_data_url_passes_through(self):
        url = "data:image/png;base64,iVBORw0KGgo="
        block = normalize_reference_image(url)
        assert block == {"type": "input_image", "image_url": url}

    def test_http_url_passes_through(self):
        url = "https://example.com/cat.png"
        assert normalize_reference_image(url) == {
            "type": "input_image",
            "image_url": url,
        }

    def test_http_insecure_url_passes_through(self):
        url = "http://example.com/cat.png"
        assert normalize_reference_image(url) == {
            "type": "input_image",
            "image_url": url,
        }

    def test_missing_file_returns_none(self, tmp_path):
        block = normalize_reference_image(str(tmp_path / "does-not-exist.png"))
        assert block is None

    def test_existing_png_path_str(self, tmp_path):
        f = tmp_path / "cat.png"
        f.write_bytes(_png_bytes())
        block = normalize_reference_image(str(f))
        assert block is not None
        assert block["type"] == "input_image"
        expected_b64 = base64.b64encode(_png_bytes()).decode()
        assert block["image_url"] == f"data:image/png;base64,{expected_b64}"

    def test_existing_path_object(self, tmp_path):
        f = tmp_path / "cat.png"
        f.write_bytes(_png_bytes())
        block = normalize_reference_image(Path(f))
        assert block is not None
        assert block["image_url"].startswith("data:image/png;base64,")

    def test_jpeg_suffix_uses_image_jpeg_mime(self, tmp_path):
        f = tmp_path / "cat.jpg"
        f.write_bytes(_png_bytes())  # bytes don't matter for the mime mapping
        block = normalize_reference_image(str(f))
        assert block is not None
        assert block["image_url"].startswith("data:image/jpeg;base64,")

    @pytest.mark.parametrize(
        "suffix,expected_mime",
        [
            (".png", "image/png"),
            (".jpg", "image/jpeg"),
            (".jpeg", "image/jpeg"),
            (".webp", "image/webp"),
            (".gif", "image/gif"),
            (".PNG", "image/png"),
            (".JPG", "image/jpeg"),
        ],
    )
    def test_mime_inferred_from_suffix(self, tmp_path, suffix, expected_mime):
        f = tmp_path / f"ref{suffix}"
        f.write_bytes(_png_bytes())
        block = normalize_reference_image(str(f))
        assert block is not None
        assert block["image_url"].startswith(f"data:{expected_mime};base64,")

    def test_unknown_suffix_falls_back_to_png_mime(self, tmp_path):
        f = tmp_path / "ref.bin"
        f.write_bytes(_png_bytes())
        block = normalize_reference_image(str(f))
        assert block is not None
        assert block["image_url"].startswith("data:image/png;base64,")

    def test_user_home_path_expansion(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        f = tmp_path / "ref.png"
        f.write_bytes(_png_bytes())
        block = normalize_reference_image("~/ref.png")
        assert block is not None
        assert block["image_url"].startswith("data:image/png;base64,")
