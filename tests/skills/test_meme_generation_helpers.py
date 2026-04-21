"""Tests for optional-skills/creative/meme-generation helper scripts."""

import struct
import sys
import zlib
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "optional-skills" / "creative" / "meme-generation" / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import imgflip_download
import meme_caption


def _png_bytes(width: int, height: int, pixel_fn) -> bytes:
    """Create a small RGBA PNG without external dependencies."""
    raw = bytearray()
    for y in range(height):
        raw.append(0)
        for x in range(width):
            r, g, b, a = pixel_fn(x, y)
            raw.extend([r, g, b, a])
    compressed = zlib.compress(bytes(raw), level=9)

    def chunk(kind: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + kind
            + data
            + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
        )

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0))
        + chunk(b"IDAT", compressed)
        + chunk(b"IEND", b"")
    )


def _png_size(path: Path) -> tuple[int, int]:
    data = path.read_bytes()
    return struct.unpack(">II", data[16:24])


class TestImgflipDownload:
    def test_resolve_template_matches_imgflip_name(self, monkeypatch):
        monkeypatch.setattr(
            imgflip_download,
            "fetch_imgflip_templates",
            lambda: [
                {"id": "505705955", "name": "Absolute Cinema", "url": "https://i.imgflip.com/8d317n.png", "box_count": 1},
                {"id": "345v97", "name": "Woman Yelling at Cat", "url": "https://i.imgflip.com/345v97.jpg", "box_count": 2},
            ],
        )

        template = imgflip_download.resolve_template("absolute cinema")
        assert template["name"] == "Absolute Cinema"
        assert template["url"] == "https://i.imgflip.com/8d317n.png"
        assert template["source"] == "imgflip"

    def test_download_template_writes_image_file(self, tmp_path, monkeypatch):
        payload = _png_bytes(64, 64, lambda x, y: (0, 0, 255, 255))

        monkeypatch.setattr(
            imgflip_download,
            "fetch_imgflip_templates",
            lambda: [{"id": "abc", "name": "Test Meme", "url": "https://example.com/test.png", "box_count": 2}],
        )
        monkeypatch.setattr(imgflip_download, "_fetch_url", lambda url, timeout=15: payload)

        out = tmp_path / "template.png"
        result = imgflip_download.download_template("test meme", out)

        assert Path(result) == out
        assert out.exists()
        assert _png_size(out) == (64, 64)


class TestMemeCaption:
    def test_caption_image_with_trim_padding(self, tmp_path, monkeypatch):
        source = tmp_path / "source.png"
        output = tmp_path / "captioned.png"

        payload = _png_bytes(
            400,
            240,
            lambda x, y: (255, 255, 255, 255) if not (70 <= x < 330 and 70 <= y < 170) else (240, 60, 60, 255),
        )
        source.write_bytes(payload)

        monkeypatch.setattr(meme_caption.generate_meme, "find_font", lambda size: meme_caption.generate_meme.ImageFont.load_default())

        result = meme_caption.caption_image(
            source,
            output,
            top_text="TOP TEXT",
            bottom_text="BOTTOM TEXT",
            trim_padding=True,
        )

        assert Path(result) == output
        assert output.exists()
        assert _png_size(output)[0] < 400
        assert _png_size(output)[1] <= 240
