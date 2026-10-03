"""Regression test for #120831: GET /v1/skills must not 500 on a
``_find_all_skills()`` signature mismatch.

The old ``TestSkillsEndpoint`` mocks ``tools.skills_tool._find_all_skills``
with a plain MagicMock, which swallows any kwarg — so the handler passing
``include_editorial=True`` (a parameter the real function never had) stayed
green while production returned HTTP 500. This test patches with
``autospec=True`` so the mock enforces the real signature: any kwarg the
handler passes that the callee does not accept raises TypeError inside the
handler, which returns 500 and fails the 200 assertion.
"""

from unittest.mock import patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import (
    APIServerAdapter,
    cors_middleware,
    security_headers_middleware,
)


def _make_adapter() -> APIServerAdapter:
    return APIServerAdapter(PlatformConfig(enabled=True, extra={}))


def _create_app(adapter: APIServerAdapter) -> web.Application:
    mws = [mw for mw in (cors_middleware, security_headers_middleware) if mw is not None]
    app = web.Application(middlewares=mws)
    app["api_server_adapter"] = adapter
    app.router.add_get("/v1/skills", adapter._handle_skills)
    return app


@pytest.fixture
def adapter():
    return _make_adapter()


class TestSkillsEndpointSignature:
    @pytest.mark.asyncio
    async def test_skills_uses_only_supported_find_all_skills_kwargs(self, adapter):
        fake_skills = [
            {"name": "github", "description": "GitHub workflow skill", "category": "github"},
            {"name": "ascii-art", "description": "ASCII art generation", "category": "creative"},
        ]
        with patch(
            "tools.skills_tool._find_all_skills",
            autospec=True,
            return_value=list(fake_skills),
        ):
            app = _create_app(adapter)
            async with TestClient(TestServer(app)) as cli:
                resp = await cli.get("/v1/skills")
                assert resp.status == 200
                data = await resp.json()
                assert data["object"] == "list"
                names = sorted(s["name"] for s in data["data"])
                assert names == ["ascii-art", "github"]
