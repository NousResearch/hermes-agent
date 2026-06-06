"""Tests for safe image reference validation."""

from __future__ import annotations

import base64
from pathlib import Path
from unittest.mock import patch

import pytest

from agent.image_reference import ImageReferenceError, validate_image_reference

PNG_BYTES = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4"
    "890000000d49444154789c6300010000000500010d0a2db40000000049454e44"
    "ae426082"
)
JPEG_BYTES = b"\xff\xd8\xff\xe0" + b"jpeg"
WEBP_BYTES = b"RIFF" + (4).to_bytes(4, "little") + b"WEBP" + b"data"
GIF_BYTES = b"GIF89a" + b"gif"


@pytest.fixture(autouse=True)
def hermes_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    return tmp_path


def _data_url(raw: bytes, mime: str = "image/png") -> str:
    return f"data:{mime};base64,{base64.b64encode(raw).decode()}"


@pytest.mark.parametrize(
    "url",
    ["https://example.test/image.png", "http://example.test/image.png"],
)
def test_allows_http_urls(url):
    ref = validate_image_reference(url)

    assert ref.kind == "url"
    assert ref.value == url
    assert ref.mime_type is None
    assert ref.path is None
    assert ref.bytes_size is None


@pytest.mark.parametrize(
    "raw,mime",
    [
        (PNG_BYTES, "image/png"),
        (JPEG_BYTES, "image/jpeg"),
        (WEBP_BYTES, "image/webp"),
        (GIF_BYTES, "image/gif"),
    ],
)
def test_allows_supported_data_urls(raw, mime):
    ref = validate_image_reference(_data_url(raw, mime))

    assert ref.kind == "data"
    assert ref.mime_type == mime
    assert ref.bytes_size == len(raw)
    assert ref.value == _data_url(raw, mime)


def test_normalizes_whitespace_bearing_data_url():
    compact = base64.b64encode(PNG_BYTES).decode()
    spaced = " \n ".join([compact[:12], compact[12:]])

    ref = validate_image_reference(f"data: image/png ; base64 , {spaced}")

    assert ref.value == f"data:image/png;base64,{compact}"


def test_rejects_fake_data_url_even_with_supported_mime():
    with pytest.raises(ImageReferenceError) as excinfo:
        validate_image_reference(_data_url(b"not an image", "image/png"))

    assert excinfo.value.error_type == "invalid_argument"


def test_rejects_data_url_mime_mismatch():
    with pytest.raises(ImageReferenceError):
        validate_image_reference(_data_url(PNG_BYTES, "image/jpeg"))


def test_rejects_oversized_data_url():
    with pytest.raises(ImageReferenceError) as excinfo:
        validate_image_reference(_data_url(PNG_BYTES), max_bytes=len(PNG_BYTES) - 1)

    assert excinfo.value.error_type == "invalid_argument"
    assert "too large" in str(excinfo.value)


def test_rejects_oversized_data_url_before_decoding():
    oversized = "data:image/png;base64," + ("A" * 1024)

    with patch("agent.image_reference.base64.b64decode", side_effect=AssertionError("decoded")):
        with pytest.raises(ImageReferenceError) as excinfo:
            validate_image_reference(oversized, max_bytes=8)

    assert excinfo.value.error_type == "invalid_argument"
    assert "too large" in str(excinfo.value)


def test_invalid_base64_after_size_precheck_still_reports_invalid():
    with pytest.raises(ImageReferenceError) as excinfo:
        validate_image_reference("data:image/png;base64,!!!!", max_bytes=128)

    assert excinfo.value.error_type == "invalid_argument"
    assert "invalid base64" in str(excinfo.value)


def test_allows_local_file_under_cache_images(hermes_home):
    path = hermes_home / "cache" / "images" / "source.png"
    path.parent.mkdir(parents=True)
    path.write_bytes(PNG_BYTES)

    ref = validate_image_reference(str(path))

    assert ref.kind == "file"
    assert ref.value == str(path.resolve())
    assert ref.path == path.resolve()
    assert ref.mime_type == "image/png"
    assert ref.bytes_size == len(PNG_BYTES)


def test_local_file_validation_reads_only_header_not_entire_file(hermes_home, monkeypatch):
    path = hermes_home / "cache" / "images" / "source.png"
    path.parent.mkdir(parents=True)
    path.write_bytes(PNG_BYTES + (b"x" * 1024))
    monkeypatch.setattr(Path, "read_bytes", lambda self: (_ for _ in ()).throw(AssertionError("read whole file")))

    ref = validate_image_reference(str(path), max_bytes=2048)

    assert ref.kind == "file"
    assert ref.mime_type == "image/png"
    assert ref.bytes_size == len(PNG_BYTES) + 1024


def test_allows_local_file_under_legacy_image_cache(hermes_home):
    path = hermes_home / "image_cache" / "source.gif"
    path.parent.mkdir(parents=True)
    path.write_bytes(GIF_BYTES)

    ref = validate_image_reference(str(path))

    assert ref.kind == "file"
    assert ref.mime_type == "image/gif"


def test_symlink_must_resolve_inside_allowed_cache(hermes_home, tmp_path):
    outside = tmp_path / "outside.png"
    outside.write_bytes(PNG_BYTES)
    link = hermes_home / "cache" / "images" / "link.png"
    link.parent.mkdir(parents=True)
    link.symlink_to(outside)

    with pytest.raises(ImageReferenceError) as excinfo:
        validate_image_reference(str(link))

    assert excinfo.value.error_type == "invalid_argument"


def test_rejects_arbitrary_local_path(tmp_path):
    path = tmp_path / "source.png"
    path.write_bytes(PNG_BYTES)

    with pytest.raises(ImageReferenceError) as excinfo:
        validate_image_reference(str(path))

    assert excinfo.value.error_type == "invalid_argument"


@pytest.mark.parametrize("ref", ["file:///tmp/source.png", "ftp://example.test/source.png", "s3://bucket/key"])
def test_rejects_unsupported_schemes(ref):
    with pytest.raises(ImageReferenceError) as excinfo:
        validate_image_reference(ref)

    assert excinfo.value.error_type == "invalid_argument"


def test_missing_file_raises_not_found(hermes_home):
    with pytest.raises(ImageReferenceError) as excinfo:
        validate_image_reference(str(hermes_home / "cache" / "images" / "missing.png"))

    assert excinfo.value.error_type == "not_found"


def test_rejects_local_non_image_file(hermes_home):
    path = hermes_home / "cache" / "images" / "source.txt"
    path.parent.mkdir(parents=True)
    path.write_text("not an image")

    with pytest.raises(ImageReferenceError) as excinfo:
        validate_image_reference(str(path))

    assert excinfo.value.error_type == "invalid_argument"
