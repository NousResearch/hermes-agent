"""Tests for the image-edit contract on agent/image_gen_provider.py.

Covers the optional ``edit`` capability added to :class:`ImageGenProvider`
and the reusable local-image -> data-URI helper used by edit-capable
providers (xAI today).
"""

from __future__ import annotations

import base64

import pytest

from agent.image_gen_provider import (
    ImageGenProvider,
    is_data_uri,
    is_http_url,
    local_image_to_data_uri,
)


# A minimal concrete provider that implements ONLY the abstract surface
# (name + generate). It deliberately does NOT override supports_edit/edit so
# we exercise the base-class defaults — this is the generate-only backend case.
class _GenOnlyProvider(ImageGenProvider):
    @property
    def name(self) -> str:
        return "genonly"

    def generate(self, prompt, aspect_ratio="landscape", **kwargs):
        return {"success": True, "image": "/tmp/x.png"}


class TestEditCapabilityDefault:
    def test_supports_edit_defaults_false(self):
        assert _GenOnlyProvider().supports_edit() is False

    def test_edit_default_returns_unsupported_capability(self):
        result = _GenOnlyProvider().edit(prompt="make it blue", image="/tmp/in.png")
        assert result["success"] is False
        assert result["image"] is None
        assert result["error_type"] == "unsupported_capability"
        assert result["provider"] == "genonly"
        # The error should be human-actionable, naming the provider.
        assert "genonly" in result["error"]

    def test_edit_default_does_not_crash_on_url(self):
        result = _GenOnlyProvider().edit(
            prompt="x", image="https://example.com/a.png"
        )
        assert result["error_type"] == "unsupported_capability"


class TestImageRefPredicates:
    def test_is_http_url(self):
        assert is_http_url("http://example.com/a.png") is True
        assert is_http_url("https://example.com/a.png") is True
        assert is_http_url("/tmp/a.png") is False
        assert is_http_url("data:image/png;base64,AAAA") is False

    def test_is_data_uri(self):
        assert is_data_uri("data:image/png;base64,AAAA") is True
        assert is_data_uri("https://example.com/a.png") is False
        assert is_data_uri("/tmp/a.png") is False


# Smallest byte sequences that pass the magic-byte sniff for each type.
_PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32
_JPEG_BYTES = b"\xff\xd8\xff\xe0" + b"\x00" * 32
_WEBP_BYTES = b"RIFF" + b"\x00\x00\x00\x00" + b"WEBP" + b"\x00" * 16


class TestLocalImageToDataUri:
    def test_valid_png(self, tmp_path):
        p = tmp_path / "in.png"
        p.write_bytes(_PNG_BYTES)
        uri = local_image_to_data_uri(str(p))
        assert uri.startswith("data:image/png;base64,")
        # Round-trips back to the original bytes.
        b64 = uri.split(",", 1)[1]
        assert base64.b64decode(b64) == _PNG_BYTES

    def test_valid_jpeg(self, tmp_path):
        p = tmp_path / "in.jpg"
        p.write_bytes(_JPEG_BYTES)
        uri = local_image_to_data_uri(str(p))
        assert uri.startswith("data:image/jpeg;base64,")

    def test_valid_webp(self, tmp_path):
        p = tmp_path / "in.webp"
        p.write_bytes(_WEBP_BYTES)
        uri = local_image_to_data_uri(str(p))
        assert uri.startswith("data:image/webp;base64,")

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(ValueError, match="does not exist"):
            local_image_to_data_uri(str(tmp_path / "nope.png"))

    def test_directory_raises(self, tmp_path):
        with pytest.raises(ValueError, match="not a regular file"):
            local_image_to_data_uri(str(tmp_path))

    def test_unsupported_extension_raises(self, tmp_path):
        p = tmp_path / "in.txt"
        p.write_bytes(_PNG_BYTES)  # content is fine; extension is not
        with pytest.raises(ValueError, match="Unsupported image type"):
            local_image_to_data_uri(str(p))

    def test_empty_file_raises(self, tmp_path):
        p = tmp_path / "empty.png"
        p.write_bytes(b"")
        with pytest.raises(ValueError, match="empty"):
            local_image_to_data_uri(str(p))

    def test_content_not_an_image_raises(self, tmp_path):
        # Right extension, wrong bytes — must not be base64-encoded and sent.
        p = tmp_path / "fake.png"
        p.write_bytes(b"this is not an image, it is arbitrary file content")
        with pytest.raises(ValueError, match="not a recognized image"):
            local_image_to_data_uri(str(p))

    def test_oversize_file_raises(self, tmp_path):
        p = tmp_path / "big.png"
        p.write_bytes(_PNG_BYTES + b"\x00" * 1024)
        with pytest.raises(ValueError, match="exceeds"):
            local_image_to_data_uri(str(p), max_bytes=64)
