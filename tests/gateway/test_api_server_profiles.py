"""GET /v1/profiles — the agents a gateway serves, for a client-side switcher."""

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import GatewayConfig, PlatformConfig
from gateway.platforms.api_server import APIServerAdapter


def _adapter(multiplex: bool, key: str = "") -> APIServerAdapter:
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": key} if key else {}))

    class _Runner:
        config = GatewayConfig(multiplex_profiles=multiplex)

    adapter.gateway_runner = _Runner()
    return adapter


async def _get(adapter: APIServerAdapter, headers=None):
    app = web.Application()
    app.router.add_get("/v1/profiles", adapter._handle_profiles)
    async with TestClient(TestServer(app)) as cli:
        resp = await cli.get("/v1/profiles", headers=headers or {})
        return resp.status, await resp.json()


@pytest.mark.asyncio
async def test_multiplex_lists_every_served_profile_with_its_prefix(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.profiles.profiles_to_serve",
        lambda multiplex, profile_allowlist=None: [
            ("default", "/home/me/.hermes"),
            ("athena", "/home/me/.hermes/profiles/athena"),
            ("atlas", "/home/me/.hermes/profiles/atlas"),
        ],
    )
    status, body = await _get(_adapter(multiplex=True))
    assert status == 200
    assert body["multiplex"] is True
    assert body["data"] == [
        {"id": "default", "object": "hermes.profile", "default": True, "prefix": ""},
        {"id": "athena", "object": "hermes.profile", "default": False, "prefix": "/p/athena"},
        {"id": "atlas", "object": "hermes.profile", "default": False, "prefix": "/p/atlas"},
    ]
    # Profile homes are local paths; they never leave the gateway.
    assert "/home/me" not in str(body)


@pytest.mark.asyncio
async def test_single_profile_gateway_serves_its_profile_at_the_root(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.profiles.profiles_to_serve",
        lambda multiplex, profile_allowlist=None: [("athena", "/x")],
    )
    status, body = await _get(_adapter(multiplex=False))
    assert status == 200
    assert body["data"] == [{"id": "athena", "object": "hermes.profile", "default": True, "prefix": ""}]


@pytest.mark.asyncio
async def test_requires_the_api_key():
    status, _ = await _get(_adapter(multiplex=True, key="sk-secret"))
    assert status == 401
