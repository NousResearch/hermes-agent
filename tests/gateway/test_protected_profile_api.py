"""Protected profile API admission with a shared (non-profile-bound) peer key."""

from __future__ import annotations

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter


@pytest.mark.asyncio
async def test_protected_forge_api_denied_bare_and_own_prefix(tmp_path, monkeypatch):
    (tmp_path / "config.yaml").write_text(
        "bot_mode:\n  invocation_acl:\n    forge: [forge, forge-worker-1, forge-worker-2, forge-reviewer]\n"
    )
    monkeypatch.setattr("hermes_cli.profile_invocation_acl.install_root", lambda: tmp_path)
    monkeypatch.setattr("hermes_cli.profiles.get_active_profile_name", lambda: "forge")
    monkeypatch.setattr("hermes_cli.profiles.profile_matches_home", lambda name, home=None: name == "forge")

    async def handler(request):
        return web.json_response({"called": True})

    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    app = web.Application(middlewares=[adapter._make_profile_prefix_middleware()])
    app.router.add_post("/v1/chat/completions", handler)
    app.router.add_post("/p/{profile}/v1/chat/completions", handler)
    app.router.add_get("/health", handler)
    app.router.add_get("/p/{profile}/health", handler)
    app.router.add_post("/p/{profile}/{tail:.*}", handler)  # shared webhook ingress
    async with TestClient(TestServer(app)) as cli:
        for path in ("/v1/chat/completions", "/p/forge/v1/chat/completions"):
            response = await cli.post(path, json={"messages": [{"role": "user", "content": "x"}]})
            assert response.status == 403
            assert (await response.json())["error"] == "Profile API invocation denied"
        assert (await cli.get("/health")).status == 200
        assert (await cli.get("/p/forge/health")).status == 200
        # Third-party webhook ingress has its own auth and is not peer/API chat.
        assert (await cli.post("/p/forge/webhooks/fixture")).status == 200


@pytest.mark.asyncio
async def test_unprotected_profile_api_still_works(tmp_path, monkeypatch):
    (tmp_path / "config.yaml").write_text(
        "bot_mode:\n  invocation_acl:\n    forge: [forge, forge-worker-1, forge-worker-2, forge-reviewer]\n"
    )
    monkeypatch.setattr("hermes_cli.profile_invocation_acl.install_root", lambda: tmp_path)
    monkeypatch.setattr("hermes_cli.profiles.get_active_profile_name", lambda: "default")

    async def handler(request):
        return web.json_response({"called": True})

    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    app = web.Application(middlewares=[adapter._make_profile_prefix_middleware()])
    app.router.add_post("/v1/chat/completions", handler)
    app.router.add_post("/p/{profile}/v1/chat/completions", handler)
    async with TestClient(TestServer(app)) as cli:
        assert (await cli.post("/v1/chat/completions", json={})).status == 200
        assert (await cli.post("/p/default/v1/chat/completions", json={})).status == 200
        assert (await cli.post("/p/forge/v1/chat/completions", json={})).status == 404
