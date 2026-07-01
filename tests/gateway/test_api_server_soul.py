"""
Tests for the /api/soul SOUL.md read/write endpoints on the API server adapter.

Exercise the real file read/write against a temp HERMES_HOME:
- GET present / absent
- PUT writes + reports reload_required, backs the prior file up
- PUT validation: empty -> 400, oversized -> 400
- ?profile=default -> 400 (refused by name)
- auth: 401 when API_SERVER_KEY is set and the bearer is missing
"""
from functools import partial

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter, cors_middleware
from gateway.platforms import soul_api


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    return tmp_path


def _make_adapter(api_key: str = "") -> APIServerAdapter:
    extra = {"key": api_key} if api_key else {}
    return APIServerAdapter(PlatformConfig(enabled=True, extra=extra))


def _app(adapter: APIServerAdapter) -> web.Application:
    app = web.Application(middlewares=[cors_middleware])
    app["api_server_adapter"] = adapter
    app.router.add_get("/api/soul", partial(soul_api.handle_get_soul, adapter))
    app.router.add_put("/api/soul", partial(soul_api.handle_put_soul, adapter))
    return app


class TestRead:
    @pytest.mark.asyncio
    async def test_get_absent(self, home):
        adapter = _make_adapter()
        # Adapter construction seeds a default SOUL.md (managed-mode); drop it so
        # this case actually exercises the absent-file path.
        (home / "SOUL.md").unlink(missing_ok=True)
        async with TestClient(TestServer(_app(adapter))) as cli:
            resp = await cli.get("/api/soul")
            assert resp.status == 200
            data = await resp.json()
            assert data["exists"] is False and data["soul"] == ""

    @pytest.mark.asyncio
    async def test_get_present(self, home):
        (home / "SOUL.md").write_text("# SOUL\n\nI am here.\n", encoding="utf-8")
        adapter = _make_adapter()
        async with TestClient(TestServer(_app(adapter))) as cli:
            resp = await cli.get("/api/soul")
            data = await resp.json()
            assert data["exists"] is True
            assert "I am here." in data["soul"]


class TestWrite:
    @pytest.mark.asyncio
    async def test_put_creates_and_flags_reload(self, home):
        adapter = _make_adapter()
        # Adapter construction seeds a default SOUL.md; drop it so this exercises
        # the first-write (no prior file ⇒ no backup) path.
        (home / "SOUL.md").unlink(missing_ok=True)
        async with TestClient(TestServer(_app(adapter))) as cli:
            resp = await cli.put("/api/soul", json={"soul": "# SOUL\n\nv1\n"})
            assert resp.status == 200
            data = await resp.json()
            assert data["ok"] is True and data["reload_required"] is True
            assert data["backup"] is None              # nothing to back up first time
        assert (home / "SOUL.md").read_text(encoding="utf-8") == "# SOUL\n\nv1\n"

    @pytest.mark.asyncio
    async def test_put_backs_up_prior(self, home):
        (home / "SOUL.md").write_text("OLD", encoding="utf-8")
        adapter = _make_adapter()
        async with TestClient(TestServer(_app(adapter))) as cli:
            resp = await cli.put("/api/soul", json={"soul": "NEW"})
            data = await resp.json()
            assert data["backup"] is not None
        from pathlib import Path
        assert Path(data["backup"]).read_text(encoding="utf-8") == "OLD"
        assert (home / "SOUL.md").read_text(encoding="utf-8") == "NEW"

    @pytest.mark.asyncio
    async def test_put_empty_rejected(self, home):
        adapter = _make_adapter()
        async with TestClient(TestServer(_app(adapter))) as cli:
            resp = await cli.put("/api/soul", json={"soul": "   "})
            assert resp.status == 400

    @pytest.mark.asyncio
    async def test_put_oversized_rejected(self, home):
        adapter = _make_adapter()
        big = "x" * (soul_api._MAX_SOUL_BYTES + 1)
        async with TestClient(TestServer(_app(adapter))) as cli:
            resp = await cli.put("/api/soul", json={"soul": big})
            assert resp.status == 400


class TestGuards:
    @pytest.mark.asyncio
    async def test_profile_default_by_name_refused(self, home):
        adapter = _make_adapter()
        async with TestClient(TestServer(_app(adapter))) as cli:
            resp = await cli.get("/api/soul?profile=default")
            assert resp.status == 400

    @pytest.mark.asyncio
    async def test_auth_required(self, home):
        adapter = _make_adapter(api_key="secret")
        async with TestClient(TestServer(_app(adapter))) as cli:
            resp = await cli.get("/api/soul")
            assert resp.status == 401
            ok = await cli.get("/api/soul", headers={"Authorization": "Bearer secret"})
            assert ok.status == 200
