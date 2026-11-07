"""Plugin-specific profile routes must precede the shared-ingress catch-all."""
from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import aiohttp
import pytest
from aiohttp import web

from gateway.platforms.base import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter


@pytest.mark.asyncio
async def test_connect_wires_native_plugin_routes_before_profile_ingress(monkeypatch):
    from hermes_cli.profiles import get_profile_dir
    from hermes_cli import plugins
    from hermes_cli import profiles
    from gateway.platforms import shared_ingress

    lead_home = get_profile_dir("lead").resolve()
    lead_home.mkdir(parents=True, exist_ok=True)
    profile_key = "lead-profile-api-key-0123456789abcdef"
    (lead_home / ".env").write_text(f"API_SERVER_KEY={profile_key}\n", encoding="utf-8")
    monkeypatch.setattr(profiles, "profiles_to_serve", lambda *, multiplex: [("lead", lead_home)])

    def factory(native, adapter):
        assert native is adapter._app
        async def mesh_route(request):
            denied = adapter._check_auth(request)
            if denied is not None:
                return denied
            from hermes_constants import get_hermes_home as current_home
            return web.json_response({
                "route": "plugin",
                "profile": request.match_info["profile"],
                "home": str(current_home().resolve()),
                "authorization": request.headers.get("Authorization"),
            })
        native.router.add_get("/p/{profile}/mesh/control", mesh_route)

    class PluginRegistry:
        def get_platform_handler_factories(self, platform):
            assert platform == "api_server"
            return [(factory, "mesh-route-test")]

    monkeypatch.setattr(plugins, "get_plugin_manager", lambda: PluginRegistry())

    async def fake_ingress(runner, profile, tail, request, *, scoped):
        from hermes_constants import get_hermes_home as current_home
        return web.json_response({
            "route": "catchall", "profile": profile, "tail": tail,
            "scoped": scoped, "home": str(current_home().resolve()),
        })

    monkeypatch.setattr(shared_ingress, "dispatch_profile_ingress", fake_ingress)

    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={
        "key": "default-api-server-key-0123456789abcdef", "host": "127.0.0.1", "port": 0,
    }))
    # Keep the real API profile-prefix middleware under test, but replace only
    # gateway.run's broader runtime/bootstrap scope, which can perform startup
    # recovery outside this route-order test's isolated HERMES_HOME.
    from hermes_constants import (
        reset_hermes_home_override,
        set_hermes_home_override,
    )
    from agent import secret_scope

    @contextmanager
    def isolated_profile_scope(profile):
        profile_path = str(get_profile_dir(profile))
        home_token = set_hermes_home_override(profile_path)
        secret_token = secret_scope.set_secret_scope(
            {"API_SERVER_KEY": profile_key}, profile_home=profile_path
        )
        try:
            yield
        finally:
            secret_scope.reset_secret_scope(secret_token)
            reset_hermes_home_override(home_token)

    monkeypatch.setattr(
        APIServerAdapter, "_profile_scope", staticmethod(isolated_profile_scope)
    )
    adapter.gateway_runner = SimpleNamespace(config=SimpleNamespace(multiplex_profiles=True))
    assert await adapter.connect()
    try:
        port = adapter._site._server.sockets[0].getsockname()[1]
        base = f"http://127.0.0.1:{port}"
        async with aiohttp.ClientSession() as client:
            wrong = await client.get(base + "/p/lead/mesh/control",
                                     headers={"Authorization": "Bearer wrong"})
            assert wrong.status == 401

            plugin_response = await client.get(
                base + "/p/lead/mesh/control",
                headers={"Authorization": f"Bearer {profile_key}"},
            )
            assert plugin_response.status == 200
            plugin_data = await plugin_response.json()
            assert plugin_data == {
                "route": "plugin", "profile": "lead", "home": str(lead_home),
                "authorization": f"Bearer {profile_key}",
            }

            mirror = await client.get(base + "/p/lead/v1/capabilities",
                                      headers={"Authorization": f"Bearer {profile_key}"})
            assert mirror.status == 200
            assert (await mirror.json())["object"] == "hermes.api_server.capabilities"

            ingress = await client.get(base + "/p/lead/twilio/hook")
            assert ingress.status == 200
            ingress_data = await ingress.json()
            assert ingress_data == {
                "route": "catchall", "profile": "lead", "tail": "twilio/hook",
                "scoped": True, "home": str(lead_home),
            }

            # The native mirror is still registered ahead of the plugin and
            # catch-all routes, and retains the same profile-scoped bearer.
            native_bad_auth = await client.get(base + "/p/lead/v1/capabilities")
            assert native_bad_auth.status == 401
    finally:
        await adapter.disconnect()
