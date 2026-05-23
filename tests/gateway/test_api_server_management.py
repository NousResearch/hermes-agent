"""Tests for remote management endpoints on the API server adapter."""

from pathlib import Path

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter, cors_middleware
from hermes_state import SessionDB


def _make_adapter(api_key: str = "") -> APIServerAdapter:
    extra = {"key": api_key} if api_key else {}
    return APIServerAdapter(PlatformConfig(enabled=True, extra=extra))


def _create_app(adapter: APIServerAdapter) -> web.Application:
    app = web.Application(middlewares=[cors_middleware])
    app["api_server_adapter"] = adapter
    app.router.add_get("/v1/capabilities", adapter._handle_capabilities)
    app.router.add_get("/api/memory", adapter._handle_read_memory)
    app.router.add_post("/api/memory/entries", adapter._handle_add_memory_entry)
    app.router.add_put("/api/memory/entries/{index}", adapter._handle_update_memory_entry)
    app.router.add_delete("/api/memory/entries/{index}", adapter._handle_delete_memory_entry)
    app.router.add_put("/api/memory/user", adapter._handle_write_user_profile)
    app.router.add_get("/api/sessions", adapter._handle_list_sessions)
    app.router.add_get("/api/sessions/search", adapter._handle_search_sessions)
    app.router.add_get("/api/skills/content", adapter._handle_skill_content)
    return app


class TestCapabilities:
    @pytest.mark.asyncio
    async def test_capabilities_advertise_remote_management(self):
        app = _create_app(_make_adapter())
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/v1/capabilities")
            assert resp.status == 200
            data = await resp.json()

        assert data["features"]["remote_sessions"] is True
        assert data["features"]["remote_profiles"] is True
        assert data["features"]["remote_memory"] is True
        assert data["features"]["remote_persona"] is True
        assert data["features"]["remote_skills"] is True
        assert data["features"]["remote_toolsets"] is True
        assert data["endpoints"]["sessions"]["path"] == "/api/sessions"
        assert data["endpoints"]["profile_soul"]["path"] == "/api/profiles/{name}/soul"


class TestSessionsApi:
    @pytest.mark.asyncio
    async def test_list_sessions_can_filter_by_source(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        db = SessionDB(tmp_path / "state.db")
        try:
            db.create_session("api-session", "api_server", model="gpt-test")
            db.append_message("api-session", "user", "desktop chat")
            db.create_session("cron-session", "cron", model="gpt-test")
            db.append_message("cron-session", "user", "background job")
        finally:
            db.close()

        app = _create_app(_make_adapter())
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/api/sessions", params={"source": "api_server"})
            assert resp.status == 200
            data = await resp.json()

        assert [session["id"] for session in data["sessions"]] == ["api-session"]
        assert data["sessions"][0]["source"] == "api_server"

    @pytest.mark.asyncio
    async def test_search_sessions_can_filter_by_source(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        db = SessionDB(tmp_path / "state.db")
        try:
            db.create_session("api-session", "api_server", model="gpt-test")
            db.append_message("api-session", "user", "find unique desktop phrase")
            db.create_session("cron-session", "cron", model="gpt-test")
            db.append_message("cron-session", "user", "find unique cron phrase")
        finally:
            db.close()

        app = _create_app(_make_adapter())
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get(
                "/api/sessions/search",
                params={"q": "unique", "source": "api_server"},
            )
            assert resp.status == 200
            data = await resp.json()

        assert [result["session_id"] for result in data["results"]] == ["api-session"]


class TestMemoryApi:
    @pytest.mark.asyncio
    async def test_memory_read_and_edit_roundtrip(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        app = _create_app(_make_adapter())

        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post("/api/memory/entries", json={"content": "first fact"})
            assert resp.status == 200
            resp = await cli.post("/api/memory/entries", json={"content": "second fact"})
            assert resp.status == 200

            resp = await cli.get("/api/memory")
            assert resp.status == 200
            data = await resp.json()
            assert [entry["content"] for entry in data["memory"]["entries"]] == [
                "first fact",
                "second fact",
            ]

            resp = await cli.put("/api/memory/entries/0", json={"content": "updated fact"})
            assert resp.status == 200
            resp = await cli.delete("/api/memory/entries/1")
            assert resp.status == 200
            resp = await cli.put("/api/memory/user", json={"content": "user profile"})
            assert resp.status == 200

            resp = await cli.get("/api/memory")
            assert resp.status == 200
            data = await resp.json()

        # MEMORY.md entries (user-added structural data) survive the
        # sanitiser — only the identity blocks are redacted.
        assert [entry["content"] for entry in data["memory"]["entries"]] == ["updated fact"]
        # USER.md content is the identity block. After the sanitiser
        # was introduced, the Remote Management API hard-redacts it
        # on the wire regardless of who's holding the Bearer token.
        # The PUT round-trips into the file on disk (next assertion
        # below) — the Remote read just doesn't echo it back.
        from gateway.platforms.api_sanitiser import REDACTED

        assert data["user"]["content"] == REDACTED
        # Sanity-check the write side: the file on disk did get
        # written with the real value. The sanitiser is wire-only.
        user_file_path = tmp_path / "memories" / "USER.md"
        assert user_file_path.exists()
        assert "user profile" in user_file_path.read_text()

    @pytest.mark.asyncio
    async def test_memory_requires_auth_when_api_key_configured(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        app = _create_app(_make_adapter(api_key="sk-secret"))

        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/api/memory")
            assert resp.status == 401
            resp = await cli.get("/api/memory", headers={"Authorization": "Bearer sk-secret"})
            assert resp.status == 200


class TestSkillContentApi:
    @pytest.mark.asyncio
    async def test_skill_content_rejects_paths_outside_skill_roots(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
        outside = tmp_path / "not-a-skill"
        outside.mkdir()
        (outside / "SKILL.md").write_text("secret", encoding="utf-8")
        app = _create_app(_make_adapter())

        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/api/skills/content", params={"path": str(outside)})
            assert resp.status == 403

    @pytest.mark.asyncio
    async def test_skill_content_reads_installed_skill(self, tmp_path, monkeypatch):
        hermes_home = tmp_path / ".hermes"
        skill_dir = hermes_home / "skills" / "productivity" / "focus"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text("---\nname: focus\n---\nBody", encoding="utf-8")
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        app = _create_app(_make_adapter())

        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/api/skills/content", params={"path": str(skill_dir)})
            assert resp.status == 200
            data = await resp.json()

        assert data["content"].startswith("---")
        assert Path(data["path"]) == skill_dir
