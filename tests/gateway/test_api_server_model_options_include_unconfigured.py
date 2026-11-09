"""GET /api/model/options must forward ``include_unconfigured`` from the query
string (default true) to ``build_model_options_payload`` instead of hardcoding
it — external API consumers need a way to request the explicit-only view, and the
unconfigured-row path must honor ``model_catalog.excluded_providers``
(#68816)."""

from unittest.mock import patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter


def _make_adapter() -> APIServerAdapter:
    return APIServerAdapter(PlatformConfig(enabled=True))


def _create_app(adapter: APIServerAdapter) -> web.Application:
    """Minimal app exposing only the route under test (no key configured → auth passes)."""
    app = web.Application()
    app.router.add_get("/api/model/options", adapter._handle_model_options)
    return app


async def _captured_include_unconfigured(adapter, query: str) -> tuple[int, bool]:
    app = _create_app(adapter)
    with patch("hermes_cli.inventory.load_picker_context", return_value=object()), \
         patch("hermes_cli.inventory.build_model_options_payload",
               return_value={"providers": []}) as build:
        async with TestClient(TestServer(app)) as cli:
            response = await cli.get(f"/api/model/options{query}")
    return response.status, bool(build.call_args.kwargs.get("include_unconfigured"))


@pytest.mark.asyncio
async def test_default_keeps_unconfigured_rows():
    status, flag = await _captured_include_unconfigured(_make_adapter(), "")
    assert status == 200
    assert flag is True, "default (no query param) must stay include_unconfigured=True"


@pytest.mark.asyncio
async def test_false_query_param_disables_unconfigured_rows():
    status, flag = await _captured_include_unconfigured(
        _make_adapter(), "?include_unconfigured=false")
    assert status == 200
    assert flag is False, "explicit include_unconfigured=false must reach the payload builder"
