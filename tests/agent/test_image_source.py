"""Tests for the shared image-source resolver.

Covers the security-critical, provider-neutral validation every image-gen
backend shares: scheme classification, the read denylist, magic-byte sniffing
for local files, and base64 / mime validation for ``data:`` URIs.
"""

import base64

import pytest

from agent.image_source import SourceKind, resolve_image_source

_PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16
_GIF = b"GIF89a" + b"\x00" * 16


def _data_uri(raw: bytes, mime: str = "image/png") -> str:
    return f"data:{mime};base64," + base64.b64encode(raw).decode("ascii")


class TestRemoteSources:
    @pytest.mark.parametrize(
        "url",
        [
            "http://example.com/cat.png",
            "https://example.com/cat.png?sig=abc",
            "HTTPS://EXAMPLE.COM/CAT.PNG",
        ],
    )
    def test_remote_url_passes_through(self, url):
        src = resolve_image_source(url)

        assert src.kind is SourceKind.REMOTE
        assert src.value == url.strip()
        assert src.as_url_or_inline() == url.strip()

    def test_remote_read_bytes_raises(self):
        src = resolve_image_source("https://example.com/cat.png")

        with pytest.raises(ValueError, match="remote URL"):
            src.read_bytes()


class TestEmptyReference:
    @pytest.mark.parametrize("ref", ["", "   ", "\n\t"])
    def test_empty_ref_raises(self, ref):
        with pytest.raises(ValueError, match="Empty source image reference"):
            resolve_image_source(ref)


class TestDataUriSources:
    def test_image_data_uri_accepted(self):
        uri = _data_uri(_PNG)
        src = resolve_image_source(uri)

        assert src.kind is SourceKind.DATA_URI
        assert src.mime == "image/png"
        assert src.as_url_or_inline() == uri
        assert src.read_bytes() == _PNG

    def test_wrapped_base64_data_uri_accepted(self):
        body = base64.b64encode(_PNG).decode("ascii")
        wrapped = body[:8] + "\n" + body[8:]
        src = resolve_image_source(f"data:image/png;base64,{wrapped}")

        assert src.kind is SourceKind.DATA_URI
        assert src.mime == "image/png"

    def test_non_image_data_uri_rejected(self):
        with pytest.raises(ValueError, match="not an image"):
            resolve_image_source("data:text/plain;base64," + base64.b64encode(b"hi").decode())

    def test_non_base64_data_uri_rejected(self):
        with pytest.raises(ValueError, match="must be base64-encoded"):
            resolve_image_source("data:image/png,not-base64-data")

    def test_lying_mime_label_rejected(self):
        # Declares image/png but the bytes are not an image.
        uri = "data:image/png;base64," + base64.b64encode(b"this is plain text, not an image").decode()
        with pytest.raises(ValueError, match="not a recognised image file"):
            resolve_image_source(uri)

    def test_malformed_base64_rejected(self):
        with pytest.raises(ValueError, match="malformed base64"):
            resolve_image_source("data:image/png;base64,@@@not base64@@@")


class TestLocalSources:
    def test_local_image_resolved(self, tmp_path):
        path = tmp_path / "cat.png"
        path.write_bytes(_PNG)

        src = resolve_image_source(str(path))

        assert src.kind is SourceKind.LOCAL
        assert src.path == str(path)
        assert src.mime == "image/png"
        assert src.read_bytes() == _PNG
        assert src.as_url_or_inline() == _data_uri(_PNG, "image/png")

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(ValueError, match="Source image not found"):
            resolve_image_source(str(tmp_path / "nope.png"))

    def test_non_image_file_rejected(self, tmp_path):
        path = tmp_path / "notes.txt"
        path.write_text("just some text, definitely not an image")

        with pytest.raises(ValueError, match="not a recognised image file"):
            resolve_image_source(str(path))

    def test_denylisted_file_rejected(self, tmp_path):
        # A .env whose bytes happen to look like a PNG must still be refused.
        env_file = tmp_path / ".env"
        env_file.write_bytes(_PNG)

        with pytest.raises(ValueError, match="Access denied"):
            resolve_image_source(str(env_file))


class TestFileUriSources:
    def test_file_uri_resolves_local(self, tmp_path):
        path = tmp_path / "cat.gif"
        path.write_bytes(_GIF)

        src = resolve_image_source(f"file://{path}")

        assert src.kind is SourceKind.LOCAL
        assert src.mime == "image/gif"

    def test_remote_file_uri_host_rejected(self):
        with pytest.raises(ValueError, match="Unsupported remote file:// host"):
            resolve_image_source("file://server/share/cat.png")


class TestPackaging:
    def test_filename_from_local_path(self, tmp_path):
        path = tmp_path / "portrait.png"
        path.write_bytes(_PNG)

        src = resolve_image_source(str(path))

        assert src.filename() == "portrait.png"

    def test_filename_from_data_uri_mime(self):
        src = resolve_image_source(_data_uri(_GIF, "image/gif"))

        assert src.filename() == "image.gif"

    def test_path_property_raises_for_non_local(self):
        src = resolve_image_source("https://example.com/cat.png")

        with pytest.raises(ValueError, match="not a local file"):
            _ = src.path


class TestBackendConstraints:
    """``max_bytes`` and ``accepted_mimes`` let a backend narrow the resolver to
    what its own API will take, so an input it would reject server-side fails
    locally with an actionable message instead."""

    _BMP = b"BM" + b"\x00" * 24

    def test_oversized_local_rejected(self, tmp_path):
        path = tmp_path / "big.png"
        path.write_bytes(_PNG + b"\x00" * 4096)

        with pytest.raises(ValueError, match="exceeds the .*MB limit"):
            resolve_image_source(str(path), max_bytes=1024)

    def test_oversized_data_uri_rejected(self):
        with pytest.raises(ValueError, match="exceeds the .*MB limit"):
            resolve_image_source(_data_uri(_PNG + b"\x00" * 4096), max_bytes=1024)

    @pytest.mark.parametrize("raw,mime", [(_PNG, "image/png"), (_GIF, "image/gif")])
    def test_within_cap_accepted(self, tmp_path, raw, mime):
        path = tmp_path / "small.img"
        path.write_bytes(raw)

        assert resolve_image_source(str(path), max_bytes=1024).mime == mime
        assert resolve_image_source(_data_uri(raw, mime), max_bytes=1024).mime == mime

    def test_local_outside_accepted_mimes_rejected(self, tmp_path):
        path = tmp_path / "logo.bmp"
        path.write_bytes(self._BMP)

        with pytest.raises(ValueError, match="not supported here"):
            resolve_image_source(str(path), accepted_mimes=frozenset({"image/png"}))

    def test_data_uri_outside_accepted_mimes_rejected(self):
        with pytest.raises(ValueError, match="not supported here"):
            resolve_image_source(
                _data_uri(self._BMP, "image/bmp"),
                accepted_mimes=frozenset({"image/png"}),
            )

    def test_accepted_mime_passes(self, tmp_path):
        path = tmp_path / "ok.png"
        path.write_bytes(_PNG)

        src = resolve_image_source(
            str(path), accepted_mimes=frozenset({"image/png", "image/gif"})
        )

        assert src.mime == "image/png"

    def test_remote_is_not_constrained(self):
        """A REMOTE URL is fetched by the backend, not here, so neither the cap
        nor the allowlist can be applied to it."""
        src = resolve_image_source(
            "https://example.com/huge.bmp",
            max_bytes=1,
            accepted_mimes=frozenset({"image/png"}),
        )

        assert src.kind is SourceKind.REMOTE
