"""HTTP-level tests for GET/PUT /api/soul on the API server adapter.

Contract under test (spec 2026-11-09-miri-soul-direct-editor-design.md):
- GET 200 soul doc with object marker/sha256/bytes/updated_at/path;
- GET 404 with soul-JSON marker body (create-from-nothing discrimination);
- PUT 200 with backup name + prompt-refresh epoch; epoch DB failure
  degrades to 200 + null epoch + warnings (the file is saved);
- PUT 400/409/409/503/500 mapped to the contract's error codes;
- 401 auth enforcement; route table carries /p/<profile> mirrors;
- the audit log line never contains the persona content.
"""

import logging
from types import SimpleNamespace

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from gateway.config import PlatformConfig
from gateway.platforms import api_server_soul
from gateway.platforms.api_server import APIServerAdapter, cors_middleware

_MOD = "gateway.platforms.api_server"


def _make_adapter(api_key: str = "") -> APIServerAdapter:
    extra = {}
    if api_key:
        extra["key"] = api_key
    return APIServerAdapter(PlatformConfig(enabled=True, extra=extra))


def _create_app(adapter: APIServerAdapter) -> web.Application:
    app = web.Application(middlewares=[cors_middleware])
    app["api_server_adapter"] = adapter
    app.router.add_get("/api/soul", adapter._handle_get_soul)
    app.router.add_put("/api/soul", adapter._handle_put_soul)
    return app


@pytest.fixture()
def soul_home(tmp_path):
    # write_bytes: Windows write_text would convert \n -> \r\n and break the
    # byte-exact assertions.
    (tmp_path / "SOUL.md").write_bytes("#Хто ти\nЯ Мірі.\r\n".encode("utf-8"))
    token = set_hermes_home_override(tmp_path)
    try:
        yield tmp_path
    finally:
        reset_hermes_home_override(token)


def _epoch_db(epoch=7, boom=False):
    def request_prompt_refresh(session_id=None):
        if boom:
            raise RuntimeError("state.db locked")
        return epoch
    return SimpleNamespace(request_prompt_refresh=request_prompt_refresh)


def _auth(api_key=""):
    return {"Authorization": f"Bearer {api_key}"} if api_key else {}


@pytest.mark.asyncio
async def test_get_soul_200_fields(soul_home):
    adapter = _make_adapter()
    async with TestClient(TestServer(_create_app(adapter))) as client:
        resp = await client.get("/api/soul")
        assert resp.status == 200
        body = await resp.json()
        assert body["object"] == "hermes.soul"
        assert body["content"] == (soul_home / "SOUL.md").read_bytes().decode("utf-8")
        assert body["content"] == "#Хто ти\nЯ Мірі.\r\n"
        assert body["sha256"] == api_server_soul.soul_sha256(
            (soul_home / "SOUL.md").read_bytes())
        assert body["bytes"] == (soul_home / "SOUL.md").stat().st_size
        assert body["path"] == str(soul_home / "SOUL.md")
        from datetime import datetime
        datetime.fromisoformat(body["updated_at"])


@pytest.mark.asyncio
async def test_get_soul_404_json_marker(soul_home):
    # Build the adapter while SOUL.md exists: constructing it runs config load,
    # whose _ensure_default_soul_md seeds the default persona into a home with
    # no soul. Real-world order: server up (seeded), soul deleted at runtime.
    adapter = _make_adapter()
    (soul_home / "SOUL.md").unlink()
    async with TestClient(TestServer(_create_app(adapter))) as client:
        resp = await client.get("/api/soul")
        assert resp.status == 404
        body = await resp.json()
        assert body["object"] == "hermes.soul"
        assert body["error"] == "soul_not_found"


@pytest.mark.asyncio
async def test_get_soul_500_json_on_non_utf8(soul_home):
    # An externally-dropped non-UTF-8 SOUL.md must stay inside the contract
    # table: JSON 500 soul_read_failed, never aiohttp's plain-text 500.
    adapter = _make_adapter()
    (soul_home / "SOUL.md").write_bytes(b"\xff\xfe\x00#\x00bad\x00")
    async with TestClient(TestServer(_create_app(adapter))) as client:
        resp = await client.get("/api/soul")
        assert resp.status == 500
        assert resp.content_type == "application/json"
        body = await resp.json()
        assert body["object"] == "hermes.soul"
        assert body["error"] == "soul_read_failed"


@pytest.mark.asyncio
async def test_put_soul_200_backup_epoch(soul_home):
    adapter = _make_adapter()
    adapter._ensure_session_db = lambda: _epoch_db(epoch=7)
    async with TestClient(TestServer(_create_app(adapter))) as client:
        current = api_server_soul.soul_sha256((soul_home / "SOUL.md").read_bytes())
        resp = await client.put("/api/soul", json={
            "content": "#Хто ти\nНова Мірі.", "base_sha256": current})
        assert resp.status == 200
        body = await resp.json()
        assert body["object"] == "hermes.soul"
        assert body["sha256"] == api_server_soul.soul_sha256(
            "#Хто ти\nНова Мірі.".encode("utf-8"))
        assert body["prompt_refresh_epoch"] == 7
        assert body["warnings"] == []
        assert body["backup"] and (soul_home / body["backup"]).exists()
        assert (soul_home / "SOUL.md").read_text(encoding="utf-8") == "#Хто ти\nНова Мірі."


@pytest.mark.asyncio
async def test_put_soul_epoch_failure_200_null_epoch(soul_home):
    # The persona is saved; the refresh registration failed -> honest 200.
    adapter = _make_adapter()
    adapter._ensure_session_db = lambda: _epoch_db(boom=True)
    async with TestClient(TestServer(_create_app(adapter))) as client:
        resp = await client.put("/api/soul", json={"content": "навіть без епохи"})
        assert resp.status == 200
        body = await resp.json()
        assert body["prompt_refresh_epoch"] is None
        assert body["warnings"] == ["prompt_refresh_failed"]
        assert (soul_home / "SOUL.md").read_text(encoding="utf-8") == "навіть без епохи"


@pytest.mark.asyncio
async def test_put_400_invalid_content(soul_home):
    adapter = _make_adapter()
    adapter._ensure_session_db = lambda: _epoch_db()
    async with TestClient(TestServer(_create_app(adapter))) as client:
        resp = await client.put("/api/soul", json={"content": "   "})
        assert resp.status == 400
        assert (await resp.json())["error"] == "invalid_content"
        # Non-string base_sha256 in a real JSON body is a client bug -> 400,
        # never an accidental 409 that would invite a force-overwrite.
        resp = await client.put("/api/soul", json={"content": "x", "base_sha256": 123})
        assert resp.status == 400
        assert (await resp.json())["error"] == "invalid_content"


@pytest.mark.asyncio
async def test_put_409_soul_changed_since_read(soul_home):
    adapter = _make_adapter()
    adapter._ensure_session_db = lambda: _epoch_db()
    async with TestClient(TestServer(_create_app(adapter))) as client:
        resp = await client.put("/api/soul", json={
            "content": "мій варіант", "base_sha256": "deadbeef"})
        assert resp.status == 409
        body = await resp.json()
        assert body["error"] == "soul_changed_since_read"
        assert body["current_sha256"] == api_server_soul.soul_sha256(
            (soul_home / "SOUL.md").read_bytes())


@pytest.mark.asyncio
async def test_put_409_soul_missing(soul_home):
    adapter = _make_adapter()
    adapter._ensure_session_db = lambda: _epoch_db()
    (soul_home / "SOUL.md").unlink()  # after adapter build (see 404 test: seed-on-load)
    async with TestClient(TestServer(_create_app(adapter))) as client:
        resp = await client.put("/api/soul", json={
            "content": "novyi", "base_sha256": "ab12cd34"})
        assert resp.status == 409
        body = await resp.json()
        assert body["error"] == "soul_missing"
        assert body["current_sha256"] is None
        assert not (soul_home / "SOUL.md").exists()  # refused, not created


@pytest.mark.asyncio
async def test_put_503_backup_collision(soul_home, monkeypatch):
    monkeypatch.setattr(api_server_soul, "_backup_stamp", lambda: "20260101-000000")
    adapter = _make_adapter()
    adapter._ensure_session_db = lambda: _epoch_db()
    async with TestClient(TestServer(_create_app(adapter))) as client:
        statuses = []
        for i in range(11):  # 1 base name + 9 suffixes (-2..-10) succeed, 11th collides
            resp = await client.put("/api/soul", json={"content": f"v{i}"})
            statuses.append(resp.status)
        assert statuses[:10] == [200] * 10
        assert statuses[10] == 503
        body = await (await client.put("/api/soul", json={"content": "v11"})).json()
        assert body["error"] == "backup_collision"


@pytest.mark.asyncio
async def test_put_500_write_failed(soul_home, monkeypatch):
    raw = (soul_home / "SOUL.md").read_bytes()

    def boom(src, dst, *a, **kw):
        raise OSError(13, "AV lock (simulated)")

    monkeypatch.setattr(api_server_soul.os, "replace", boom)
    adapter = _make_adapter()
    adapter._ensure_session_db = lambda: _epoch_db()
    async with TestClient(TestServer(_create_app(adapter))) as client:
        resp = await client.put("/api/soul", json={"content": "зіпсована"})
        assert resp.status == 500
        assert (await resp.json())["error"] == "soul_write_failed"
        assert (soul_home / "SOUL.md").read_bytes() == raw


@pytest.mark.asyncio
async def test_auth_401(soul_home):
    adapter = _make_adapter(api_key="sekret")
    async with TestClient(TestServer(_create_app(adapter))) as client:
        resp = await client.get("/api/soul")
        assert resp.status == 401
        resp = await client.put("/api/soul", json={"content": "x"})
        assert resp.status == 401
        resp = await client.get("/api/soul", headers=_auth("sekret"))
        assert resp.status == 200


def test_route_table_has_soul_routes():
    adapter = _make_adapter()
    routes = {(m, p) for m, p, _h in adapter._http_route_table()}
    assert ("GET", "/api/soul") in routes
    assert ("PUT", "/api/soul") in routes


@pytest.mark.asyncio
async def test_audit_line_has_no_content(soul_home, caplog):
    adapter = _make_adapter()
    adapter._ensure_session_db = lambda: _epoch_db()
    with caplog.at_level(logging.INFO, logger=_MOD):
        async with TestClient(TestServer(_create_app(adapter))) as client:
            resp = await client.put("/api/soul", json={
                "content": "секретний-вміст-не-в-лозі Маргарита-маркер"})
            assert resp.status == 200
    lines = [r.getMessage() for r in caplog.records if "/api/soul" in r.getMessage()]
    assert lines, "expected an audit line for PUT /api/soul"
    for line in lines:
        assert "Маргарита-маркер" not in line
        assert "секретний-вміст" not in line
