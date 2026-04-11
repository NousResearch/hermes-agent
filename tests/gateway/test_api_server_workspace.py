import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter, cors_middleware, security_headers_middleware


class _FakeSessionDB:
    def __init__(self, sessions, total_count):
        self._sessions = sessions
        self._total_count = total_count

    def list_sessions_rich(self, source=None, limit=50, offset=0):
        items = self._sessions
        if source is not None:
            items = [item for item in items if item.get("source") == source]
        return items[offset: offset + limit]

    def count_sessions(self, source=None):
        if source is None:
            return self._total_count
        return len([item for item in self._sessions if item.get("source") == source])


class _TitleConflictSessionDB:
    def create_session(self, session_id, source, model=""):
        return session_id

    def set_session_title(self, session_id, title):
        raise ValueError("title already in use")

    def get_session(self, session_id):
        return {"id": session_id, "title": "existing"}


def _make_adapter(api_key: str = "") -> APIServerAdapter:
    extra = {}
    if api_key:
        extra["key"] = api_key
    return APIServerAdapter(PlatformConfig(enabled=True, extra=extra))


def _create_workspace_app(adapter: APIServerAdapter) -> web.Application:
    mws = [mw for mw in (cors_middleware, security_headers_middleware) if mw is not None]
    app = web.Application(middlewares=mws)
    app["api_server_adapter"] = adapter
    app.router.add_get("/api/sessions", adapter._handle_list_sessions)
    app.router.add_post("/api/sessions", adapter._handle_create_session)
    app.router.add_get("/api/skills/{name}", adapter._handle_get_skill)
    return app


class TestWorkspaceSessionsAPI:
    @pytest.mark.asyncio
    async def test_list_sessions_total_matches_source_filter(self):
        adapter = _make_adapter()
        adapter._session_db = _FakeSessionDB(
            sessions=[
                {"id": "ws_1", "source": "workspace", "title": "A"},
                {"id": "ws_2", "source": "workspace", "title": "B"},
                {"id": "tg_1", "source": "telegram", "title": "C"},
            ],
            total_count=3,
        )
        app = _create_workspace_app(adapter)

        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/api/sessions?source=workspace")
            assert resp.status == 200
            data = await resp.json()
            assert [item["id"] for item in data["items"]] == ["ws_1", "ws_2"]
            assert data["total"] == 2

    @pytest.mark.asyncio
    async def test_create_session_returns_conflict_for_duplicate_title(self):
        adapter = _make_adapter()
        adapter._session_db = _TitleConflictSessionDB()
        app = _create_workspace_app(adapter)

        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/api/sessions",
                json={"title": "Duplicate title", "model": "test-model"},
            )
            assert resp.status == 409
            data = await resp.json()
            assert data["code"] == "title_conflict"
            assert "already in use" in data["error"]


class TestWorkspaceSkillsAPI:
    @pytest.mark.asyncio
    async def test_get_skill_requires_category_when_name_is_ambiguous(self):
        adapter = _make_adapter()
        app = _create_workspace_app(adapter)

        with tempfile.TemporaryDirectory() as tmpdir:
            skills_dir = Path(tmpdir)
            (skills_dir / "devops" / "deploy-skill").mkdir(parents=True)
            (skills_dir / "frontend" / "deploy-skill").mkdir(parents=True)
            (skills_dir / "devops" / "deploy-skill" / "SKILL.md").write_text("# DevOps skill\n")
            (skills_dir / "frontend" / "deploy-skill" / "SKILL.md").write_text("# Frontend skill\n")

            adapter._get_skills_dir = MagicMock(return_value=skills_dir)

            async with TestClient(TestServer(app)) as cli:
                resp = await cli.get("/api/skills/deploy-skill")
                assert resp.status == 409
                data = await resp.json()
                assert data["code"] == "ambiguous_skill_name"
                assert sorted(data["categories"]) == ["devops", "frontend"]

                resp = await cli.get("/api/skills/deploy-skill?category=frontend")
                assert resp.status == 200
                data = await resp.json()
                assert data["category"] == "frontend"
                assert "Frontend skill" in data["content"]
