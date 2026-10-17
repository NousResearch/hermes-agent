"""API-server clients can read stored images with the credential they already hold."""
import base64
import os

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter

KEY = "k" * 32
AUTH = {"Authorization": f"Bearer {KEY}"}
PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


async def _client(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": KEY}))
    app = web.Application()
    for method, path, handler in adapter._http_route_table():
        app.router.add_route(method, path, handler)
    return TestClient(TestServer(app))


@pytest.mark.asyncio
async def test_serves_an_image_under_the_cache_folder(tmp_path, monkeypatch):
    image = tmp_path / "cache" / "documents" / "native-inputs" / "abc" / "api_1.png"
    image.parent.mkdir(parents=True)
    image.write_bytes(PNG)
    async with await _client(tmp_path, monkeypatch) as client:
        res = await client.get("/api/media", params={"path": str(image)}, headers=AUTH)
        assert res.status == 200
        data_url = (await res.json())["data_url"]
    assert data_url.startswith("data:image/png;base64,")
    assert base64.b64decode(data_url.split(",", 1)[1]) == PNG


@pytest.mark.asyncio
async def test_requires_the_api_key(tmp_path, monkeypatch):
    image = tmp_path / "cache" / "a.png"
    image.parent.mkdir(parents=True)
    image.write_bytes(PNG)
    async with await _client(tmp_path, monkeypatch) as client:
        res = await client.get("/api/media", params={"path": str(image)})
        assert res.status in (401, 403)


@pytest.mark.asyncio
async def test_refuses_files_outside_the_media_folders(tmp_path, monkeypatch):
    outside = tmp_path / "secrets" / "leak.png"
    outside.parent.mkdir(parents=True)
    outside.write_bytes(PNG)
    async with await _client(tmp_path, monkeypatch) as client:
        res = await client.get("/api/media", params={"path": str(outside)}, headers=AUTH)
        assert res.status == 403


@pytest.mark.asyncio
async def test_refuses_a_symlink_that_escapes_the_media_folders(tmp_path, monkeypatch):
    outside = tmp_path / "secrets" / "leak.png"
    outside.parent.mkdir(parents=True)
    outside.write_bytes(PNG)
    link = tmp_path / "cache" / "alias.png"
    link.parent.mkdir(parents=True)
    try:
        os.symlink(outside, link)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks are not available on this filesystem")
    async with await _client(tmp_path, monkeypatch) as client:
        res = await client.get("/api/media", params={"path": str(link)}, headers=AUTH)
        assert res.status == 403


@pytest.mark.asyncio
async def test_refuses_non_image_files_missing_files_and_oversized_files(tmp_path, monkeypatch):
    note = tmp_path / "cache" / "notes.txt"
    note.parent.mkdir(parents=True)
    note.write_text("x")
    huge = tmp_path / "cache" / "huge.png"
    huge.write_bytes(b"x" * (25 * 1024 * 1024 + 1))
    async with await _client(tmp_path, monkeypatch) as client:
        assert (await client.get("/api/media", params={"path": str(note)}, headers=AUTH)).status == 415
        missing = tmp_path / "cache" / "gone.png"
        assert (await client.get("/api/media", params={"path": str(missing)}, headers=AUTH)).status == 404
        assert (await client.get("/api/media", headers=AUTH)).status == 400
        assert (await client.get("/api/media", params={"path": str(huge)}, headers=AUTH)).status == 413


@pytest.mark.asyncio
async def test_capabilities_advertise_the_media_route(tmp_path, monkeypatch):
    async with await _client(tmp_path, monkeypatch) as client:
        res = await client.get("/v1/capabilities", headers=AUTH)
        body = await res.json()
    assert res.status == 200
    assert body["features"]["media_api"] is True
    assert body["endpoints"]["media_read"] == {"method": "GET", "path": "/api/media"}
