"""GET /v1/skills must enumerate the real skill catalog.

The handler calls ``tools.skills_tool._find_all_skills`` directly; a test that patches that
function with a MagicMock accepts any keyword, so a call-signature drift between the two
modules only shows up as a 500 in production. These tests drive the real function against
the temp ``HERMES_HOME`` the test conftest provides.
"""

import os
from pathlib import Path

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter


def _write_skill(skills_root: Path, category: str, name: str, description: str) -> None:
    skill_dir = skills_root / category / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {description}\n---\n\n# {name}\n\nBody.\n",
        encoding="utf-8",
    )


def _create_app(adapter: APIServerAdapter) -> web.Application:
    """Minimal app exposing only the route under test (no key configured → auth passes)."""
    app = web.Application()
    app.router.add_get("/v1/skills", adapter._handle_skills)
    return app


@pytest.mark.asyncio
async def test_skills_endpoint_lists_installed_skills_from_real_catalog():
    skills_root = Path(os.environ["HERMES_HOME"]) / "skills"
    _write_skill(skills_root, "devops", "zeta-deploy", "Deploy things")
    _write_skill(skills_root, "creative", "alpha-art", "Make art")

    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    async with TestClient(TestServer(_create_app(adapter))) as cli:
        response = await cli.get("/v1/skills")
        body = await response.json()

    assert response.status == 200, body
    assert body["object"] == "list"
    by_name = {entry["name"]: entry for entry in body["data"]}
    assert {"alpha-art", "zeta-deploy"} <= set(by_name)
    assert by_name["alpha-art"]["description"] == "Make art"
    assert by_name["alpha-art"]["category"] == "creative"
