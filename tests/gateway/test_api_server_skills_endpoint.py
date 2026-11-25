"""GET /v1/skills must call the skills catalog with the signature it declares.

Regression test: the handler passed ``include_editorial=True`` to
``find_all_skills`` after the function stopped accepting it. The TypeError raised
on the call was swallowed by the handler's ``except Exception``, so the endpoint
answered 500 ("Failed to enumerate skills") while the mocked test in
``test_api_server.py`` stayed green — a mock without ``autospec`` accepts any
keyword. Binding the real signature makes the same mistake fail the endpoint.
"""

from unittest.mock import patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter


def _make_adapter() -> APIServerAdapter:
    return APIServerAdapter(PlatformConfig(enabled=True, extra={}))


def _create_app(adapter: APIServerAdapter) -> web.Application:
    app = web.Application()
    app["api_server_adapter"] = adapter
    app.router.add_get("/v1/skills", adapter._handle_skills)
    return app


class TestSkillsEndpointSignature:
    @pytest.mark.asyncio
    async def test_skills_enumerates_with_the_function_declared_signature(self):
        fake_skills = [
            {"name": "github", "description": "GitHub workflow skill", "category": "github"},
        ]
        with patch(
            "tools.skills_tool.find_all_skills",
            autospec=True,
            return_value=list(fake_skills),
        ) as enumerate_skills:
            app = _create_app(_make_adapter())
            async with TestClient(TestServer(app)) as cli:
                resp = await cli.get("/v1/skills")
                assert resp.status == 200, await resp.text()
                data = await resp.json()
        assert [s["name"] for s in data["data"]] == ["github"]
        enumerate_skills.assert_called_once_with(skip_disabled=False)
