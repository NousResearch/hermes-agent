"""
Tests for the config endpoints on the API server adapter.

Covers:
- GET /api/config/persona  — read SOUL.md (returns "" when missing)
- PUT /api/config/persona  — write SOUL.md, validate type/length
- GET /api/config/toolsets — read toolsets list from config.yaml
- PUT /api/config/toolsets — write toolsets list, validate items + de-dup
- Auth enforcement (401 when key configured)
- Bad JSON body handling
"""

from pathlib import Path
from unittest.mock import patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter, cors_middleware


def _make_adapter(api_key: str = "") -> APIServerAdapter:
    extra = {}
    if api_key:
        extra["key"] = api_key
    return APIServerAdapter(PlatformConfig(enabled=True, extra=extra))


def _create_app(adapter: APIServerAdapter) -> web.Application:
    app = web.Application(middlewares=[cors_middleware])
    app["api_server_adapter"] = adapter
    app.router.add_get("/api/config/persona", adapter._handle_get_persona)
    app.router.add_put("/api/config/persona", adapter._handle_set_persona)
    app.router.add_get("/api/config/toolsets", adapter._handle_get_toolsets)
    app.router.add_put("/api/config/toolsets", adapter._handle_set_toolsets)
    return app


@pytest.fixture
def adapter():
    return _make_adapter()


@pytest.fixture
def auth_adapter():
    return _make_adapter(api_key="sk-secret")


# ---------------------------------------------------------------------------
# Persona — GET
# ---------------------------------------------------------------------------

class TestGetPersona:
    @pytest.mark.asyncio
    async def test_returns_empty_string_when_missing(self, adapter, tmp_path):
        soul_path = tmp_path / "SOUL.md"  # does not exist
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch("hermes_constants.get_hermes_home", return_value=tmp_path):
                resp = await cli.get("/api/config/persona")
                assert resp.status == 200
                data = await resp.json()
                assert data == {"persona": ""}
        assert not soul_path.exists()

    @pytest.mark.asyncio
    async def test_returns_file_contents(self, adapter, tmp_path):
        soul_path = tmp_path / "SOUL.md"
        soul_path.write_text("You are Hermes.", encoding="utf-8")
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch("hermes_constants.get_hermes_home", return_value=tmp_path):
                resp = await cli.get("/api/config/persona")
                assert resp.status == 200
                data = await resp.json()
                assert data == {"persona": "You are Hermes."}


# ---------------------------------------------------------------------------
# Persona — PUT
# ---------------------------------------------------------------------------

class TestSetPersona:
    @pytest.mark.asyncio
    async def test_writes_file(self, adapter, tmp_path):
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch("hermes_constants.get_hermes_home", return_value=tmp_path):
                resp = await cli.put(
                    "/api/config/persona", json={"persona": "Hello world"},
                )
                assert resp.status == 200
                data = await resp.json()
                assert data == {"ok": True, "persona": "Hello world"}
                assert (tmp_path / "SOUL.md").read_text() == "Hello world"

    @pytest.mark.asyncio
    async def test_creates_parent_dir(self, adapter, tmp_path):
        nested = tmp_path / "profiles" / "p1"  # does not exist yet
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch("hermes_constants.get_hermes_home", return_value=nested):
                resp = await cli.put("/api/config/persona", json={"persona": "x"})
                assert resp.status == 200
                assert (nested / "SOUL.md").read_text() == "x"

    @pytest.mark.asyncio
    async def test_rejects_non_string(self, adapter, tmp_path):
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch("hermes_constants.get_hermes_home", return_value=tmp_path):
                resp = await cli.put(
                    "/api/config/persona", json={"persona": 123},
                )
                assert resp.status == 400
                data = await resp.json()
                assert "must be a string" in data["error"]

    @pytest.mark.asyncio
    async def test_rejects_too_long(self, adapter, tmp_path):
        app = _create_app(adapter)
        too_long = "x" * 60_000
        async with TestClient(TestServer(app)) as cli:
            with patch("hermes_constants.get_hermes_home", return_value=tmp_path):
                resp = await cli.put(
                    "/api/config/persona", json={"persona": too_long},
                )
                assert resp.status == 400
                data = await resp.json()
                assert "≤" in data["error"]

    @pytest.mark.asyncio
    async def test_rejects_bad_json(self, adapter, tmp_path):
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch("hermes_constants.get_hermes_home", return_value=tmp_path):
                resp = await cli.put(
                    "/api/config/persona",
                    data="not json",
                    headers={"Content-Type": "application/json"},
                )
                assert resp.status == 400
                data = await resp.json()
                assert data["error"] == "Invalid JSON body"


# ---------------------------------------------------------------------------
# Toolsets — GET
# ---------------------------------------------------------------------------

class TestGetToolsets:
    @pytest.mark.asyncio
    async def test_returns_list_from_config(self, adapter):
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch(
                "hermes_cli.config.load_config",
                return_value={"toolsets": ["hermes-cli", "web", "browser"]},
            ):
                resp = await cli.get("/api/config/toolsets")
                assert resp.status == 200
                data = await resp.json()
                assert data == {"toolsets": ["hermes-cli", "web", "browser"]}

    @pytest.mark.asyncio
    async def test_returns_empty_when_missing(self, adapter):
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch(
                "hermes_cli.config.load_config", return_value={},
            ):
                resp = await cli.get("/api/config/toolsets")
                assert resp.status == 200
                data = await resp.json()
                assert data == {"toolsets": []}

    @pytest.mark.asyncio
    async def test_returns_empty_when_not_a_list(self, adapter):
        # Config has ``toolsets: "everything"`` — bad shape on disk; coerce to []
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch(
                "hermes_cli.config.load_config",
                return_value={"toolsets": "everything"},
            ):
                resp = await cli.get("/api/config/toolsets")
                assert resp.status == 200
                data = await resp.json()
                assert data == {"toolsets": []}


# ---------------------------------------------------------------------------
# Toolsets — PUT
# ---------------------------------------------------------------------------

class TestSetToolsets:
    @pytest.mark.asyncio
    async def test_writes_list(self, adapter):
        app = _create_app(adapter)
        captured = {}

        def fake_save(cfg):
            captured["cfg"] = cfg

        async with TestClient(TestServer(app)) as cli:
            with patch(
                "hermes_cli.config.load_config",
                return_value={"toolsets": ["hermes-cli"], "model": "x"},
            ), patch(
                "hermes_cli.config.save_config", side_effect=fake_save,
            ):
                resp = await cli.put(
                    "/api/config/toolsets", json={"toolsets": ["web", "browser"]},
                )
                assert resp.status == 200
                data = await resp.json()
                assert data == {"ok": True, "toolsets": ["web", "browser"]}
                assert captured["cfg"]["toolsets"] == ["web", "browser"]
                assert captured["cfg"]["model"] == "x"  # other keys preserved

    @pytest.mark.asyncio
    async def test_dedupes_preserving_order(self, adapter):
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch(
                "hermes_cli.config.load_config", return_value={},
            ), patch(
                "hermes_cli.config.save_config",
            ):
                resp = await cli.put(
                    "/api/config/toolsets",
                    json={"toolsets": ["web", "browser", "web", "file"]},
                )
                assert resp.status == 200
                data = await resp.json()
                assert data["toolsets"] == ["web", "browser", "file"]

    @pytest.mark.asyncio
    async def test_rejects_non_list(self, adapter):
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.put(
                "/api/config/toolsets", json={"toolsets": "web,browser"},
            )
            assert resp.status == 400

    @pytest.mark.asyncio
    async def test_rejects_non_string_items(self, adapter):
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.put(
                "/api/config/toolsets", json={"toolsets": ["web", 5]},
            )
            assert resp.status == 400

    @pytest.mark.asyncio
    async def test_rejects_bad_chars(self, adapter):
        # Path-traversal / injection guard — keys are constrained to [a-zA-Z0-9_-]
        app = _create_app(adapter)
        for bad in ["web/file", "../etc", "with space", "semi;colon"]:
            async with TestClient(TestServer(app)) as cli:
                resp = await cli.put(
                    "/api/config/toolsets", json={"toolsets": [bad]},
                )
                assert resp.status == 400, f"expected reject for {bad!r}"

    @pytest.mark.asyncio
    async def test_rejects_bad_json(self, adapter):
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.put(
                "/api/config/toolsets",
                data="not json",
                headers={"Content-Type": "application/json"},
            )
            assert resp.status == 400


# ---------------------------------------------------------------------------
# Auth enforcement
# ---------------------------------------------------------------------------

class TestAuthEnforced:
    @pytest.mark.asyncio
    async def test_persona_get_unauth(self, auth_adapter, tmp_path):
        app = _create_app(auth_adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/api/config/persona")
            assert resp.status == 401

    @pytest.mark.asyncio
    async def test_persona_put_unauth(self, auth_adapter, tmp_path):
        app = _create_app(auth_adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.put(
                "/api/config/persona", json={"persona": "x"},
            )
            assert resp.status == 401

    @pytest.mark.asyncio
    async def test_toolsets_get_unauth(self, auth_adapter):
        app = _create_app(auth_adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/api/config/toolsets")
            assert resp.status == 401

    @pytest.mark.asyncio
    async def test_toolsets_put_unauth(self, auth_adapter):
        app = _create_app(auth_adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.put(
                "/api/config/toolsets", json={"toolsets": ["web"]},
            )
            assert resp.status == 401

    @pytest.mark.asyncio
    async def test_persona_get_with_correct_key(self, auth_adapter, tmp_path):
        soul_path = tmp_path / "SOUL.md"
        soul_path.write_text("hi", encoding="utf-8")
        app = _create_app(auth_adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch("hermes_constants.get_hermes_home", return_value=tmp_path):
                resp = await cli.get(
                    "/api/config/persona",
                    headers={"Authorization": "Bearer sk-secret"},
                )
                assert resp.status == 200
                assert (await resp.json()) == {"persona": "hi"}
