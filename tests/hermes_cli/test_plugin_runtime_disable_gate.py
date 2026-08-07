"""Regression tests for runtime plugin disable gating.

Covers two residual bypasses addressed in the PR:

1. Plugin API routes mounted at startup remain callable even after the
   plugin is added to ``plugins.disabled`` at runtime.  The new
   ``_plugin_api_runtime_gate`` middleware blocks these requests.

2. Bundled plugin assets were served from the unauthenticated
   ``/dashboard-plugins/{name}/{path}`` route even when the bundled
   plugin was in ``plugins.disabled``.  The updated ``serve_plugin_asset``
   now applies the disabled check to bundled plugins too.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch, AsyncMock

import pytest

from hermes_cli import web_server
from hermes_cli.plugin_activation import PluginActivationState


def _activation(*, enabled=(), disabled=(), safe_mode=False):
    return PluginActivationState(
        enabled=frozenset(enabled),
        disabled=frozenset(disabled),
        safe_mode=safe_mode,
    )


def _plugin_api_request(name):
    from starlette.requests import Request

    return Request({
        "type": "http",
        "method": "GET",
        "path": f"/api/plugins/{name}/probe",
        "query_string": b"",
        "headers": [],
        "state": {"token_authenticated": True},
    })


def _ok_call_next():
    from starlette.responses import JSONResponse

    response = JSONResponse({"ok": True})
    return response, AsyncMock(return_value=response)


@pytest.fixture(autouse=True)
def _reset_plugin_cache():
    """Bust the plugin cache before and after each test."""
    mounted_identities = dict(
        web_server._mounted_dashboard_plugin_api_identities
    )
    web_server._dashboard_plugins_cache = None
    web_server._dashboard_plugins_cache_fingerprint = None
    yield
    web_server._dashboard_plugins_cache = None
    web_server._dashboard_plugins_cache_fingerprint = None
    web_server._mounted_dashboard_plugin_api_identities.clear()
    web_server._mounted_dashboard_plugin_api_identities.update(
        mounted_identities
    )


@pytest.fixture
def test_client(monkeypatch, tmp_path):
    """Set up a Starlette TestClient with auth bypassed."""
    try:
        from starlette.testclient import TestClient
    except ImportError:
        pytest.skip("fastapi/starlette not installed")

    from hermes_cli.web_server import app, _SESSION_HEADER_NAME, _SESSION_TOKEN

    # Isolate HERMES_HOME so config reads go to our tmp.
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    (tmp_path / "home").mkdir(parents=True)

    client = TestClient(app)
    client.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN
    return client


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_user_plugin(tmp_path, name="hot"):
    """Create a minimal user plugin with a JS asset."""
    dashboard_dir = tmp_path / "plugins" / name / "dashboard"
    dashboard_dir.mkdir(parents=True)
    dist_dir = dashboard_dir / "dist"
    dist_dir.mkdir()
    (dist_dir / "index.js").write_text("console.log('hello');")
    (dashboard_dir / "manifest.json").write_text(json.dumps({
        "name": name,
        "label": name.title(),
        "entry": "dist/index.js",
    }))
    return dashboard_dir


def _make_bundled_plugin(tmp_path, name="bundledx"):
    """Create a minimal bundled plugin with a JS asset."""
    dashboard_dir = tmp_path / "bundled" / name / "dashboard"
    dashboard_dir.mkdir(parents=True)
    dist_dir = dashboard_dir / "dist"
    dist_dir.mkdir()
    (dist_dir / "index.js").write_text("console.log('bundled');")
    (dashboard_dir / "manifest.json").write_text(json.dumps({
        "name": name,
        "label": name.title(),
        "entry": "dist/index.js",
    }))
    return dashboard_dir


def _make_runtime_plugin(
    root,
    directory,
    *,
    key,
    dashboard_name,
    api_source=None,
):
    """Create a runtime plugin claiming a dashboard route name."""
    plugin_root = root / directory
    dashboard_dir = plugin_root / "dashboard"
    dashboard_dir.mkdir(parents=True)
    (plugin_root / "plugin.yaml").write_text(
        f"name: {key}\nkind: backend\nversion: 1.0.0\n"
    )
    manifest = {
        "name": dashboard_name,
        "label": dashboard_name,
        "entry": "dist/index.js",
    }
    if api_source is not None:
        (dashboard_dir / "api.py").write_text(
            "from fastapi import APIRouter\n"
            "router = APIRouter()\n"
            "@router.get('/probe')\n"
            "async def probe():\n"
            f"    return {{'source': {api_source!r}}}\n"
        )
        manifest["api"] = "api.py"
    (dashboard_dir / "manifest.json").write_text(json.dumps(manifest))
    return dashboard_dir


def _make_project_plugin(project_root, name="project-extension"):
    """Create a dashboard-only project extension with a browser asset."""
    dashboard_dir = project_root / ".hermes" / "plugins" / name / "dashboard"
    dashboard_dir.mkdir(parents=True)
    dist_dir = dashboard_dir / "dist"
    dist_dir.mkdir()
    (dist_dir / "index.js").write_text("console.log('project');")
    (dashboard_dir / "manifest.json").write_text(json.dumps({
        "name": name,
        "label": "Project Extension",
        "entry": "dist/index.js",
    }))
    return dashboard_dir


# ---------------------------------------------------------------------------
# Test 1: Runtime-disabled user plugin API routes return 404
# ---------------------------------------------------------------------------


class TestPluginApiRuntimeGate:
    """After a user plugin is disabled at runtime, its mounted API routes
    must return 404 — not 200 — even though the router was already
    included at startup.  The _plugin_api_runtime_gate middleware enforces
    this at request time."""

    @pytest.mark.asyncio
    async def test_middleware_blocks_disabled_user_plugin(self):
        """Middleware returns 404 for a user plugin added to disabled set."""
        from starlette.requests import Request
        from starlette.responses import JSONResponse

        fake_plugin = {
            "name": "hot",
            "source": "user",
        }

        # Simulate a request to /api/plugins/hot/probe
        scope = {
            "type": "http",
            "method": "GET",
            "path": "/api/plugins/hot/probe",
            "query_string": b"",
            "headers": [],
            "state": {"token_authenticated": True},
        }
        request = Request(scope)

        call_next = AsyncMock(return_value=JSONResponse({"ok": True}))

        with patch.object(web_server, "_get_dashboard_plugins", return_value=[fake_plugin]), \
             patch(
                 "hermes_cli.config.load_plugin_activation_state",
                 return_value=_activation(enabled={"hot"}, disabled={"hot"}),
             ):
            response = await web_server._plugin_api_runtime_gate(request, call_next)

        assert response.status_code == 404
        call_next.assert_not_called()


    @pytest.mark.asyncio
    async def test_middleware_passes_non_plugin_api_routes(self):
        """Middleware ignores non-plugin API routes."""
        from starlette.requests import Request
        from starlette.responses import JSONResponse

        scope = {
            "type": "http",
            "method": "GET",
            "path": "/api/status",
            "query_string": b"",
            "headers": [],
            "state": {"token_authenticated": True},
        }
        request = Request(scope)

        expected_resp = JSONResponse({"ok": True})
        call_next = AsyncMock(return_value=expected_resp)

        response = await web_server._plugin_api_runtime_gate(request, call_next)

        assert response is expected_resp
        call_next.assert_called_once()

    @pytest.mark.asyncio
    async def test_middleware_unknown_plugin_defaults_to_user_blocks(self):
        """Unknown plugin name (not in discovery cache) is treated as user
        plugin and blocked when not enabled."""
        from starlette.requests import Request
        from starlette.responses import JSONResponse

        scope = {
            "type": "http",
            "method": "GET",
            "path": "/api/plugins/unknown/action",
            "query_string": b"",
            "headers": [],
            "state": {"token_authenticated": True},
        }
        request = Request(scope)

        call_next = AsyncMock(return_value=JSONResponse({"ok": True}))

        with patch.object(web_server, "_get_dashboard_plugins", return_value=[]), \
             patch(
                 "hermes_cli.config.load_plugin_activation_state",
                 return_value=_activation(),
             ):
            response = await web_server._plugin_api_runtime_gate(request, call_next)

        assert response.status_code == 404
        call_next.assert_not_called()

    @pytest.mark.asyncio
    async def test_middleware_requires_mounted_canonical_key(
        self,
        tmp_path,
    ):
        """An active route-name claimant cannot borrow another key's mount."""
        dashboard_dir = tmp_path / "plugins" / "key-swap" / "dashboard"
        dashboard_dir.mkdir(parents=True)
        plugin = {
            "name": "key-swap",
            "source": "user",
            "_dir": str(dashboard_dir),
            "_runtime_name": "current-key",
            "_runtime_key": "current-key",
            "_runtime_source": "user",
            "_runtime_kind": "standalone",
            "_runtime_managed": False,
        }
        request = _plugin_api_request("key-swap")
        _expected, call_next = _ok_call_next()

        with patch.object(
            web_server,
            "_get_dashboard_plugins",
            return_value=[plugin],
        ), patch.object(
            web_server,
            "_discover_dashboard_runtime_entries",
            return_value=[],
        ), patch.object(
            web_server,
            "_load_dashboard_plugin_activation_state",
            return_value=_activation(enabled={"current-key"}),
        ), patch.dict(
            web_server._mounted_dashboard_plugin_api_identities,
            {"key-swap": ("mounted-key", dashboard_dir.resolve().parent)},
        ):
            response = await web_server._plugin_api_runtime_gate(
                request,
                call_next,
            )

        assert response.status_code == 404
        call_next.assert_not_called()

    @pytest.mark.asyncio
    async def test_middleware_fails_closed_when_winner_discovery_fails(
        self,
        tmp_path,
    ):
        """A stale router must not run when its current winner is unknown."""
        dashboard_dir = tmp_path / "plugins" / "unknown-winner" / "dashboard"
        dashboard_dir.mkdir(parents=True)
        request = _plugin_api_request("unknown-winner")
        _expected, call_next = _ok_call_next()
        identity = ("unknown-winner", dashboard_dir.resolve().parent)

        with patch.object(
            web_server,
            "_get_dashboard_plugins",
            side_effect=RuntimeError("discovery unavailable"),
        ), patch.dict(
            web_server._mounted_dashboard_plugin_api_identities,
            {"unknown-winner": identity},
        ):
            response = await web_server._plugin_api_runtime_gate(
                request,
                call_next,
            )

        assert response.status_code == 404
        call_next.assert_not_called()

    @pytest.mark.asyncio
    async def test_dashboard_only_manifest_change_invalidates_runtime_gate(
        self,
        tmp_path,
        monkeypatch,
    ):
        """Cached dashboard metadata cannot keep an old API identity alive."""
        home = tmp_path / "home"
        home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(home))
        dashboard_dir = _make_user_plugin(home, "manifest-swap")
        api_file = dashboard_dir / "api.py"
        api_file.write_text(
            "from fastapi import APIRouter\nrouter = APIRouter()\n"
        )
        manifest_file = dashboard_dir / "manifest.json"
        manifest = json.loads(manifest_file.read_text())
        manifest["api"] = "api.py"
        manifest_file.write_text(json.dumps(manifest))

        activation = _activation(enabled={"manifest-swap"})
        request = _plugin_api_request("manifest-swap")
        expected, call_next = _ok_call_next()

        with patch(
            "hermes_cli.config.load_plugin_activation_state",
            return_value=activation,
        ):
            plugins = web_server._get_dashboard_plugins(force_rescan=True)
            mounted = next(
                plugin
                for plugin in plugins
                if plugin["name"] == "manifest-swap"
            )
            mounted_identity = web_server._dashboard_plugin_api_identity(
                mounted,
                [],
            )
            assert mounted_identity is not None

            with patch.dict(
                web_server._mounted_dashboard_plugin_api_identities,
                {"manifest-swap": mounted_identity},
            ):
                response = await web_server._plugin_api_runtime_gate(
                    request,
                    call_next,
                )
                assert response is expected
                call_next.assert_awaited_once()

                # Keep activation unchanged and mutate only the manifest.
                # Its new canonical name must invalidate the cached winner.
                manifest["name"] = "renamed-plugin"
                manifest_file.write_text(json.dumps(manifest))
                call_next.reset_mock()

                response = await web_server._plugin_api_runtime_gate(
                    request,
                    call_next,
                )

        assert response.status_code == 404
        call_next.assert_not_called()

    def test_bundled_fallback_cannot_authorize_mounted_user_override(
        self,
        test_client,
        tmp_path,
    ):
        """A fallback with the same namespace must not unlock a stale router."""
        bundled = tmp_path / "bundled"
        user_plugins = tmp_path / "home" / "plugins"
        runtime_key = "shared-runtime"
        route_name = "winner-swap"
        _make_runtime_plugin(
            bundled,
            "bundled-copy",
            key=runtime_key,
            dashboard_name=route_name,
            api_source="bundled",
        )
        _make_runtime_plugin(
            user_plugins,
            "user-copy",
            key=runtime_key,
            dashboard_name=route_name,
            api_source="user",
        )
        state = {
            "activation": _activation(enabled={runtime_key}),
        }
        original_routes = list(web_server.app.router.routes)
        module_name = f"hermes_dashboard_plugin_{route_name}"

        try:
            with patch(
                "hermes_cli.plugins.get_bundled_plugins_dir",
                return_value=bundled,
            ), patch(
                "hermes_cli.config.load_plugin_activation_state",
                side_effect=lambda: state["activation"],
            ):
                initial = web_server._get_dashboard_plugins(
                    force_rescan=True
                )
                mounted = next(p for p in initial if p["name"] == route_name)
                assert mounted["source"] == "user"
                web_server._mount_plugin_api_routes()

                new_routes = [
                    route
                    for route in web_server.app.router.routes
                    if route not in original_routes
                ]
                for route in new_routes:
                    web_server.app.router.routes.remove(route)
                web_server.app.router.routes[0:0] = new_routes

                response = test_client.get(
                    f"/api/plugins/{route_name}/probe"
                )
                assert response.status_code == 200
                assert response.json() == {"source": "user"}

                # Removing the user opt-in reselects the bundled candidate,
                # but FastAPI still points this path at the mounted user
                # handler until restart.
                state["activation"] = _activation()
                current = web_server._get_dashboard_plugins()
                winner = next(p for p in current if p["name"] == route_name)
                assert winner["source"] == "bundled"

                response = test_client.get(
                    f"/api/plugins/{route_name}/probe"
                )
                assert response.status_code == 404
                assert response.json() == {"detail": "Plugin not found"}
        finally:
            web_server.app.router.routes[:] = original_routes
            sys.modules.pop(module_name, None)


class TestDashboardRouteNameCollision:
    @pytest.mark.asyncio
    async def test_distinct_canonical_plugins_sharing_route_fail_closed(
        self,
        tmp_path,
        monkeypatch,
    ):
        bundled = tmp_path / "bundled"
        home = tmp_path / "home"
        bundled.mkdir()
        home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(home))
        _make_runtime_plugin(
            bundled,
            "one",
            key="runtime-one",
            dashboard_name="shared-route",
        )
        _make_runtime_plugin(
            bundled,
            "two",
            key="runtime-two",
            dashboard_name="shared-route",
        )

        with patch(
            "hermes_cli.plugins.get_bundled_plugins_dir",
            return_value=bundled,
        ):
            plugins = web_server._get_dashboard_plugins(force_rescan=True)
            claimants = [p for p in plugins if p["name"] == "shared-route"]
            assert len(claimants) == 2
            assert all(p.get("_route_name_collision") for p in claimants)
            assert all(
                web_server._dashboard_plugin_status(p) == "not enabled"
                for p in claimants
            )
        request = _plugin_api_request("shared-route")
        _expected, call_next = _ok_call_next()
        with patch.object(
            web_server,
            "_get_dashboard_plugins",
            return_value=claimants,
        ):
            response = await web_server._plugin_api_runtime_gate(request, call_next)

        assert response.status_code == 404
        call_next.assert_not_called()


# ---------------------------------------------------------------------------
# Test 2: Disabled bundled plugin assets return 404
# ---------------------------------------------------------------------------


class TestBundledPluginAssetGate:
    """Bundled plugins in ``plugins.disabled`` must have their static
    assets blocked — not just hidden from the listing endpoint."""

    def test_bundled_asset_returns_404_when_disabled(self, test_client, tmp_path, monkeypatch):
        """A disabled bundled plugin's JS asset must return 404."""
        plugin_dir = _make_bundled_plugin(tmp_path, "bundledx")

        fake_plugin = {
            "name": "bundledx",
            "label": "Bundledx",
            "source": "bundled",
            "entry": "dist/index.js",
            "_dir": str(plugin_dir),
        }

        with patch.object(web_server, "_get_dashboard_plugins", return_value=[fake_plugin]):
            # Sanity: asset is served when not disabled.
            with patch(
                "hermes_cli.config.load_plugin_activation_state",
                return_value=_activation(),
            ):
                resp = test_client.get("/dashboard-plugins/bundledx/dist/index.js")
                assert resp.status_code == 200, (
                    "Sanity: bundled plugin asset should be served when not disabled"
                )

            # Disable it.
            with patch(
                "hermes_cli.config.load_plugin_activation_state",
                return_value=_activation(disabled={"bundledx"}),
            ):
                resp = test_client.get("/dashboard-plugins/bundledx/dist/index.js")
                assert resp.status_code == 404, (
                    "Disabled bundled plugin asset must return 404"
                )

    def test_bundled_asset_served_when_not_disabled(self, test_client, tmp_path, monkeypatch):
        """Bundled plugin assets are served normally when not in disabled set."""
        plugin_dir = _make_bundled_plugin(tmp_path, "goodbundled")

        fake_plugin = {
            "name": "goodbundled",
            "label": "Good Bundled",
            "source": "bundled",
            "entry": "dist/index.js",
            "_dir": str(plugin_dir),
        }

        with patch.object(web_server, "_get_dashboard_plugins", return_value=[fake_plugin]):
            with patch(
                "hermes_cli.config.load_plugin_activation_state",
                return_value=_activation(),
            ):
                resp = test_client.get("/dashboard-plugins/goodbundled/dist/index.js")
                assert resp.status_code == 200


@pytest.mark.parametrize(
    ("grant", "expected"),
    (("dashboard-label", "not enabled"), ("web/runtime-key", "enabled")),
)
def test_dashboard_display_name_cannot_replace_canonical_runtime_key(
    tmp_path, grant, expected
):
    plugin_root = tmp_path / "plugins" / "web" / "runtime-key"
    dashboard_dir = plugin_root / "dashboard"
    dashboard_dir.mkdir(parents=True)
    plugin = {"name": "dashboard-label", "source": "user", "_dir": str(dashboard_dir)}
    runtime_entries = [(
        "runtime-name", "1.0.0", "", "user", plugin_root,
        "web/runtime-key", "standalone",
    )]
    assert web_server._dashboard_plugin_status(
        plugin, runtime_entries, _activation(enabled={grant})
    ) == expected


def test_dashboard_only_fallback_honors_safe_mode():
    plugin = {"name": "project-extension", "source": "project"}
    state = _activation(enabled={"project-extension"}, safe_mode=True)
    assert web_server._dashboard_plugin_status(plugin, [], state) == "not enabled"


class TestProjectDashboardScopeInvalidation:
    """Cached project JavaScript must not outlive its cwd/env opt-in scope."""

    @pytest.mark.asyncio
    async def test_gate_disable_blocks_cached_asset_list_and_api(
        self,
        test_client,
        tmp_path,
        monkeypatch,
    ):
        project_root = tmp_path / "project"
        project_root.mkdir()
        monkeypatch.chdir(project_root)
        monkeypatch.setenv("HERMES_ENABLE_PROJECT_PLUGINS", "1")
        _make_project_plugin(project_root)
        activation = _activation(enabled={"project-extension"})

        with patch(
            "hermes_cli.config.load_plugin_activation_state",
            return_value=activation,
        ):
            cached = web_server._get_dashboard_plugins(force_rescan=True)
            stale_plugin = next(
                p for p in cached if p["name"] == "project-extension"
            )
            assert test_client.get(
                "/dashboard-plugins/project-extension/dist/index.js"
            ).status_code == 200
            listed = test_client.get("/api/dashboard/plugins")
            assert listed.status_code == 200
            assert "project-extension" in {
                item["name"] for item in listed.json()
            }

            # Keep the old directory in place: invalidation must be driven by
            # scope identity, not by the legacy "directory disappeared" check.
            monkeypatch.setenv("HERMES_ENABLE_PROJECT_PLUGINS", "0")
            assert test_client.get(
                "/dashboard-plugins/project-extension/dist/index.js"
            ).status_code == 404
            listed = test_client.get("/api/dashboard/plugins")
            assert listed.status_code == 200
            assert "project-extension" not in {
                item["name"] for item in listed.json()
            }
            assert stale_plugin["source"] == "project"
            request = _plugin_api_request("project-extension")
            _expected, call_next = _ok_call_next()
            with patch.object(
                web_server, "_get_dashboard_plugins", return_value=[stale_plugin]
            ):
                response = await web_server._plugin_api_runtime_gate(request, call_next)
            assert response.status_code == 404
            call_next.assert_not_called()

