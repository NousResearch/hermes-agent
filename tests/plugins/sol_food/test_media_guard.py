"""Media guard: every bound at limit-1 / limit / limit+1, fail closed."""

import struct

import pytest

from plugins.sol_food.limits import FOOD_IMAGE_MAX_BYTES
from plugins.sol_food.media_guard import (
    MediaRejected,
    predownload_check,
    probe_image,
    validate_image_bytes,
)


def png_bytes(width: int, height: int, pad: int = 0) -> bytes:
    header = (
        b"\x89PNG\r\n\x1a\n"
        + struct.pack(">I", 13)
        + b"IHDR"
        + struct.pack(">II", width, height)
        + b"\x08\x02\x00\x00\x00"
    )
    return header + b"\x00" * pad


def jpeg_bytes(width: int, height: int) -> bytes:
    return (
        b"\xff\xd8"
        + b"\xff\xc0"
        + struct.pack(">H", 11)
        + b"\x08"
        + struct.pack(">HH", height, width)
        + b"\x03\x01\x11\x00"
    )


def webp_vp8x_bytes(width: int, height: int) -> bytes:
    payload = (
        b"WEBP"
        + b"VP8X"
        + struct.pack("<I", 10)
        + b"\x00\x00\x00\x00"
        + (width - 1).to_bytes(3, "little")
        + (height - 1).to_bytes(3, "little")
    )
    return b"RIFF" + struct.pack("<I", len(payload)) + payload


class TestPredownload:
    def test_album_fails_closed(self):
        with pytest.raises(MediaRejected) as excinfo:
            predownload_check(
                file_size=10, width=10, height=10, media_group_id="g1"
            )
        assert excinfo.value.reason_code == "food_media_album_rejected"

    def test_album_checked_before_size(self):
        # Even an oversize album update reports the album rejection: it
        # fails before anything else is considered.
        with pytest.raises(MediaRejected) as excinfo:
            predownload_check(
                file_size=FOOD_IMAGE_MAX_BYTES + 1,
                width=None,
                height=None,
                media_group_id="g1",
            )
        assert excinfo.value.reason_code == "food_media_album_rejected"

    @pytest.mark.parametrize(
        "size,ok",
        [
            (FOOD_IMAGE_MAX_BYTES - 1, True),
            (FOOD_IMAGE_MAX_BYTES, True),
            (FOOD_IMAGE_MAX_BYTES + 1, False),
        ],
    )
    def test_advertised_size_boundary(self, size, ok):
        if ok:
            predownload_check(
                file_size=size, width=None, height=None, media_group_id=None
            )
        else:
            with pytest.raises(MediaRejected):
                predownload_check(
                    file_size=size, width=None, height=None, media_group_id=None
                )

    @pytest.mark.parametrize(
        "width,height,ok",
        [
            (7899, 1, True),
            (7900, 1, True),
            (7901, 1, False),
            (1, 7900, True),
            (1, 7901, False),
            (6400, 5000, True),  # 32,000,000 pixels exactly
            (6400, 5001, False),  # 32,006,400 pixels
        ],
    )
    def test_advertised_dimension_boundary(self, width, height, ok):
        if ok:
            predownload_check(
                file_size=None, width=width, height=height, media_group_id=None
            )
        else:
            with pytest.raises(MediaRejected):
                predownload_check(
                    file_size=None, width=width, height=height, media_group_id=None
                )

    def test_unknown_metadata_defers(self):
        # None metadata is not trusted as OK — it merely defers to the
        # post-download validation (which is mandatory).
        predownload_check(file_size=None, width=None, height=None, media_group_id=None)


class TestProbe:
    def test_png(self):
        probe = probe_image(png_bytes(640, 480))
        assert (probe.format, probe.width, probe.height) == ("png", 640, 480)

    def test_jpeg(self):
        probe = probe_image(jpeg_bytes(800, 600))
        assert (probe.format, probe.width, probe.height) == ("jpeg", 800, 600)

    def test_webp_vp8x(self):
        probe = probe_image(webp_vp8x_bytes(320, 240))
        assert (probe.format, probe.width, probe.height) == ("webp", 320, 240)

    def test_gif_not_probed(self):
        assert probe_image(b"GIF89a" + b"\x00" * 32) is None


class TestValidateBytes:
    @pytest.mark.parametrize(
        "pad,ok",
        [
            (FOOD_IMAGE_MAX_BYTES - 30 - 1, True),   # total = limit-1
            (FOOD_IMAGE_MAX_BYTES - 30, True),       # total = limit
            (FOOD_IMAGE_MAX_BYTES - 30 + 1, False),  # total = limit+1
        ],
    )
    def test_byte_ceiling_boundary(self, pad, ok):
        data = png_bytes(10, 10, pad=pad + (30 - len(png_bytes(10, 10))))
        assert len(data) in (
            FOOD_IMAGE_MAX_BYTES - 1,
            FOOD_IMAGE_MAX_BYTES,
            FOOD_IMAGE_MAX_BYTES + 1,
        )
        if ok:
            assert validate_image_bytes(data).format == "png"
        else:
            with pytest.raises(MediaRejected) as excinfo:
                validate_image_bytes(data)
            assert excinfo.value.reason_code == "food_media_too_large"

    @pytest.mark.parametrize("builder", [png_bytes, jpeg_bytes, webp_vp8x_bytes])
    @pytest.mark.parametrize(
        "width,height,ok",
        [
            (7899, 2, True),
            (7900, 2, True),
            (7901, 2, False),
            (6400, 5000, True),
            (6400, 5001, False),
        ],
    )
    def test_dimension_boundary_from_headers(self, builder, width, height, ok):
        data = builder(width, height)
        if ok:
            validate_image_bytes(data)
        else:
            with pytest.raises(MediaRejected) as excinfo:
                validate_image_bytes(data)
            assert excinfo.value.reason_code == "food_media_bad_dimensions"

    def test_unsupported_format_fails_closed(self):
        with pytest.raises(MediaRejected) as excinfo:
            validate_image_bytes(b"GIF89a" + b"\x00" * 64)
        assert excinfo.value.reason_code == "food_media_bad_format"

    def test_empty_fails_closed(self):
        with pytest.raises(MediaRejected):
            validate_image_bytes(b"")

    def test_truncated_png_header_fails_closed(self):
        data = png_bytes(100, 100)[:20]
        with pytest.raises(MediaRejected) as excinfo:
            validate_image_bytes(data)
        assert excinfo.value.reason_code == "food_media_header_unparseable"

    def test_zero_dimension_fails_closed(self):
        with pytest.raises(MediaRejected) as excinfo:
            validate_image_bytes(png_bytes(0, 100))
        assert excinfo.value.reason_code == "food_media_bad_dimensions"

    def test_no_full_decode_needed(self):
        # A "decompression bomb" shaped file (huge declared dimensions,
        # tiny actual data) is rejected purely from the header — no
        # decoder ever runs in this module.
        with pytest.raises(MediaRejected) as excinfo:
            validate_image_bytes(png_bytes(50000, 50000))
        assert excinfo.value.reason_code == "food_media_bad_dimensions"
