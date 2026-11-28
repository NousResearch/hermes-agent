"""GET /v1/skills contract: the handler lists skills through the real find_all_skills."""

from unittest.mock import patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter


def _skills_app() -> web.Application:
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={}))
    app = web.Application()
    app["api_server_adapter"] = adapter
    app.router.add_get("/v1/skills", adapter._handle_skills)
    return app


@pytest.mark.asyncio
async def test_skills_returns_list_envelope():
    fake_skills = [
        {"name": "github", "description": "GitHub workflow skill", "category": "github"},
        {"name": "ascii-art", "description": "ASCII art generation", "category": "creative"},
    ]
    # autospec: a call with a kwarg the real signature lacks fails here, not in production.
    with patch("tools.skills_tool.find_all_skills", return_value=list(fake_skills), autospec=True):
        async with TestClient(TestServer(_skills_app())) as cli:
            resp = await cli.get("/v1/skills")
            assert resp.status == 200
            data = await resp.json()
    assert data["object"] == "list"
    assert sorted(s["name"] for s in data["data"]) == ["ascii-art", "github"]
    for entry in data["data"]:
        assert set(entry.keys()) >= {"name", "description", "category"}


@pytest.mark.asyncio
async def test_skills_end_to_end_without_mock(tmp_path, monkeypatch):
    skill_dir = tmp_path / "skills" / "demo-skill"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: demo-skill\ndescription: Demo skill for GET /v1/skills contract test\n---\n\n# Demo Skill\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    async with TestClient(TestServer(_skills_app())) as cli:
        resp = await cli.get("/v1/skills")
        assert resp.status == 200
        data = await resp.json()
    assert data["object"] == "list"
    by_name = {entry["name"]: entry for entry in data["data"]}
    assert by_name["demo-skill"]["description"] == "Demo skill for GET /v1/skills contract test"
