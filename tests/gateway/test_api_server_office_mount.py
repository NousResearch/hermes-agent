"""Digital Office mounted on the Gateway API server (aiohttp)."""

from __future__ import annotations

import pytest

pytest.importorskip("aiohttp", reason="aiohttp not installed")
pytest.importorskip("httpx", reason="httpx not installed")


@pytest.fixture
def office_app_instance(monkeypatch, tmp_path):
    from hermes_office import gateway_http as gh

    gh.get_office_app.cache_clear()
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    (tmp_path / ".hermes").mkdir(parents=True)
    from aiohttp import web

    from hermes_office.gateway_http import register_digital_office_routes

    app = web.Application()
    assert register_digital_office_routes(app) is True
    try:
        yield app
    finally:
        gh.get_office_app.cache_clear()


@pytest.mark.asyncio
async def test_api_health_probe(office_app_instance):
    from aiohttp.test_utils import TestClient, TestServer

    async with TestClient(TestServer(office_app_instance)) as client:
        async with client.get("/api/health") as resp:
            assert resp.status == 200
            data = await resp.json()
            assert data.get("status") == "ok"


@pytest.mark.asyncio
async def test_api_office_health_bridge(office_app_instance):
    from aiohttp.test_utils import TestClient, TestServer

    async with TestClient(TestServer(office_app_instance)) as client:
        async with client.get("/api/office/health") as resp:
            assert resp.status == 200
            data = await resp.json()
            assert data.get("ok") is True
            assert "version" in data


@pytest.mark.asyncio
async def test_office_spa_root_contains_root_div(office_app_instance):
    from aiohttp.test_utils import TestClient, TestServer

    async with TestClient(TestServer(office_app_instance)) as client:
        async with client.get("/office/") as resp:
            if resp.status == 503:
                pytest.skip("office frontend dist not built")
            assert resp.status == 200
            text = await resp.text()
            assert 'id="root"' in text
