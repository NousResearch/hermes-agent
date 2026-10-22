"""Inbound image names survive byte, URL and unified media cache paths."""

import re
import stat
from contextlib import asynccontextmanager
from pathlib import Path

import pytest

from gateway.platforms import base


PNG = b"\x89PNG\r\n\x1a\n" + b"x" * 64


@pytest.mark.parametrize(
    ("filename", "stem"),
    [
        ("Screenshot 2026-09-15 at 11.12.26.png", "Screenshot-2026-09-15-at-11.12.26"),
        ("--- Screen : 01 !!!.png", "Screen-01"),
        ("../../outside.png", "outside"),
        ("a" * 80 + ".png", "a" * 64),
        (None, ""),
        (123, ""),
        ("!!!.png", ""),
    ],
)
def test_image_cache_filename(filename, stem, tmp_path, monkeypatch):
    monkeypatch.setattr(base, "IMAGE_CACHE_DIR", tmp_path)
    path = Path(base.cache_image_from_bytes(PNG, ext=".png", filename=filename))

    suffix = "_" + re.escape(stem) if stem else ""
    assert re.fullmatch(r"img_[0-9a-f]{12}" + suffix + r"\.png", path.name)
    assert path.parent == tmp_path
    assert path.read_bytes() == PNG


@pytest.mark.platforms("posix")
def test_named_image_retains_private_file_permissions(tmp_path, monkeypatch):
    monkeypatch.setattr(base, "IMAGE_CACHE_DIR", tmp_path)
    path = Path(base.cache_image_from_bytes(PNG, ext=".png", filename="private.png"))

    assert stat.S_IMODE(path.stat().st_mode) == 0o600


@pytest.mark.asyncio
async def test_unified_media_cache_preserves_image_name(tmp_path, monkeypatch):
    monkeypatch.setattr(base, "IMAGE_CACHE_DIR", tmp_path)
    cached = await base.cache_media_bytes_async(
        PNG, filename="Quarterly chart.png", mime_type="image/png"
    )

    path = Path(cached.path)
    assert re.fullmatch(r"img_[0-9a-f]{12}_Quarterly-chart\.png", path.name)
    assert path.read_bytes() == PNG
    assert cached.kind == "image"
    assert cached.media_type == "image/png"


def test_mime_dispatch_preserves_image_name(tmp_path, monkeypatch):
    from gateway.platforms.media_cache import cache_media_bytes

    monkeypatch.setattr(base, "IMAGE_CACHE_DIR", tmp_path)
    path = Path(cache_media_bytes(PNG, "image/png", filename_hint="Chart 01.png"))

    assert re.fullmatch(r"img_[0-9a-f]{12}_Chart-01\.png", path.name)
    assert path.read_bytes() == PNG


@pytest.mark.asyncio
@pytest.mark.parametrize("filename", [None, "URL chart.png"])
async def test_url_cache_preserves_optional_image_name(filename, tmp_path, monkeypatch):
    import httpx
    from tools import url_safety

    @asynccontextmanager
    async def fake_client(**kwargs):
        transport = httpx.MockTransport(
            lambda request: httpx.Response(200, content=PNG, request=request)
        )
        async with httpx.AsyncClient(transport=transport, **kwargs) as client:
            yield client

    monkeypatch.setattr(base, "IMAGE_CACHE_DIR", tmp_path)
    monkeypatch.setattr(url_safety, "is_safe_url", lambda url: True)
    monkeypatch.setattr(url_safety, "create_ssrf_safe_async_client", fake_client)
    path = Path(await base.cache_image_from_url(
        "https://example.com/image.png", ext=".png", filename=filename
    ))

    suffix = "_URL-chart" if filename else ""
    assert re.fullmatch(r"img_[0-9a-f]{12}" + suffix + r"\.png", path.name)
    assert path.read_bytes() == PNG
