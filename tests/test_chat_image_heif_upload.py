"""Chat image upload must accept an iPhone HEIC paste instead of rejecting it.

A macOS clipboard paste of a photo from an iPhone arrives as HEIC bytes behind an
``image/png`` data URL. The upload path sniffs magic bytes, finds no PNG signature,
and used to answer 400 "Unsupported image type" — so the image never reached the
vision pipeline. HEIF is transcoded to PNG on the way in, and camera-sized photos
are capped so a 12 MP shot does not arrive as a ~7 MB PNG.
"""

import base64
import io

import pytest
from PIL import Image

pytest.importorskip("pillow_heif")

from hermes_cli.web_models import ChatImageUpload
from hermes_cli.web_routers.files import _decode_chat_image_upload


def _heic_bytes(size: tuple[int, int] = (300, 200)) -> bytes:
    import pillow_heif

    pillow_heif.register_heif_opener()
    buf = io.BytesIO()
    Image.new("RGB", size, (120, 90, 40)).save(buf, format="HEIF")
    return buf.getvalue()


def _data_url(data: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(data).decode()


def test_heic_paste_is_transcoded_to_png():
    data, mime, ext = _decode_chat_image_upload(
        ChatImageUpload(data_url=_data_url(_heic_bytes()))
    )

    assert (mime, ext) == ("image/png", ".png")
    assert data.startswith(b"\x89PNG\r\n\x1a\n")
    with Image.open(io.BytesIO(data)) as image:
        assert image.size == (300, 200)


def test_camera_sized_heic_is_capped_but_keeps_aspect_ratio():
    data, _, _ = _decode_chat_image_upload(
        ChatImageUpload(data_url=_data_url(_heic_bytes((4000, 3000))))
    )

    with Image.open(io.BytesIO(data)) as image:
        assert max(image.size) <= 2048
        assert image.size[0] > image.size[1]


def test_png_paste_passes_through_untouched():
    buf = io.BytesIO()
    Image.new("RGB", (64, 48), (0, 0, 0)).save(buf, format="PNG")
    raw = buf.getvalue()

    data, mime, ext = _decode_chat_image_upload(
        ChatImageUpload(data_url=_data_url(raw))
    )

    assert (mime, ext) == ("image/png", ".png")
    assert data == raw


def test_unrecognised_bytes_still_raise():
    from fastapi import HTTPException

    with pytest.raises(HTTPException):
        _decode_chat_image_upload(
            ChatImageUpload(data_url=_data_url(b"not an image at all"))
        )
