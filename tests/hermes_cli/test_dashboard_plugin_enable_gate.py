"""Regression coverage for the ``plugins.enabled`` gate on dashboard
plugins whose manifest ``name`` differs from their install-directory key.

``hermes plugins enable <key>`` writes the *path-derived registry key*
(the plugin's directory name, e.g. ``hermes-mobile``) into
``plugins.enabled``, and the agent loader explicitly accepts both that
key and the manifest name ("Accept both the path-derived key and the
legacy bare name so existing configs keep working").  The two dashboard
gates added for #46435 (GHSA-mcfc-hp25-cjv7) compared only the
*dashboard* manifest ``name`` — which doubles as the mount prefix and
routinely differs from the directory key (a plugin installed at
``~/.hermes/plugins/hermes-mobile/`` with ``{"name": "mobile"}`` mounts
at ``/api/plugins/mobile/``).  Such a plugin loaded fine in the agent
while its backend API and dashboard assets were silently skipped (the
skip is logged at DEBUG), which presents as "plugin enabled but its
dashboard API 404s" after an upgrade.

These tests pin the fix: both gates accept *either* the directory key or
the manifest name, for enable and disable alike, without widening the
gate (a plugin enabled under neither identifier must still be skipped).
"""

from __future__ import annotations

import asyncio
import json
from contextlib import contextmanager
from unittest.mock import patch

import pytest

from hermes_cli import web_server


@pytest.fixture(autouse=True)
def _reset_plugin_cache():
    """Bust the per-process discovery cache before and after each test so
    the import-time production scan can't bleed in."""
    web_server._dashboard_plugins_cache = None
    yield
    web_server._dashboard_plugins_cache = None


API_SRC = (
    "from fastapi import APIRouter\n"
    "router = APIRouter()\n"
    "\n"
    "\n"
    '@router.get("/hello")\n'
    "def hello():\n"
    '    return {"ok": True}\n'
)


@pytest.fixture
def mobile_plugin(tmp_path, monkeypatch):
    """A user plugin whose directory key and dashboard name differ.

    Installed at ``<home>/plugins/hermes-mobile/`` (registry key
    ``hermes-mobile``) with ``dashboard/manifest.json`` declaring
    ``{"name": "mobile"}`` — the shape that regressed in the field.
    Bundled plugins are pointed at an empty directory so the only
    discoverable plugin is this one.
    """
    home = tmp_path / "home"
    (home / "plugins").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_ENABLE_PROJECT_PLUGINS", raising=False)
    empty_bundled = tmp_path / "no-bundled-plugins"
    empty_bundled.mkdir()
    monkeypatch.setattr(
        "hermes_cli.plugins.get_bundled_plugins_dir", lambda: empty_bundled
    )
    dash = home / "plugins" / "hermes-mobile" / "dashboard"
    dash.mkdir(parents=True)
    (dash / "manifest.json").write_text(
        json.dumps({
            "name": "mobile",
            "label": "Mobile",
            "entry": "dist/index.js",
            "api": "plugin_api.py",
        })
    )
    (dash / "plugin_api.py").write_text(API_SRC)
    return dash


@contextmanager
def gate_config(enabled, disabled=()):
    """Pin the ``plugins.enabled`` / ``plugins.disabled`` sets the gates read."""
    with (
        patch("hermes_cli.plugins_cmd._get_enabled_set", return_value=set(enabled)),
        patch("hermes_cli.plugins_cmd._get_disabled_set", return_value=set(disabled)),
    ):
        yield


def _mounted_prefixes():
    """Run the mount routine against a spy app; return mounted prefixes."""
    with patch.object(web_server.app, "include_router") as inc:
        web_server._mount_plugin_api_routes()
    return [c.kwargs.get("prefix") for c in inc.call_args_list]


class TestMountGateAcceptsDirectoryKey:
    def test_enabled_by_directory_key_mounts_api(self, mobile_plugin):
        """The field regression: enabled via ``hermes plugins enable
        hermes-mobile`` (directory key), API must mount under the
        manifest name."""
        with gate_config(enabled={"hermes-mobile"}):
            web_server._get_dashboard_plugins(force_rescan=True)
            prefixes = _mounted_prefixes()
        assert "/api/plugins/mobile" in prefixes

    def test_enabled_by_manifest_name_still_mounts_api(self, mobile_plugin):
        """Back-compat: configs that listed the manifest name keep working."""
        with gate_config(enabled={"mobile"}):
            web_server._get_dashboard_plugins(force_rescan=True)
            prefixes = _mounted_prefixes()
        assert "/api/plugins/mobile" in prefixes

    def test_disabled_by_directory_key_wins(self, mobile_plugin):
        """An explicit disable under either identifier must block the
        mount even when the plugin is also enabled."""
        with gate_config(
            enabled={"hermes-mobile", "mobile"}, disabled={"hermes-mobile"}
        ):
            web_server._get_dashboard_plugins(force_rescan=True)
            prefixes = _mounted_prefixes()
        assert prefixes == []

    def test_not_enabled_is_still_skipped(self, mobile_plugin):
        """#46435 invariant: a user plugin enabled under neither
        identifier must not have its backend imported or mounted."""
        with gate_config(enabled=set()):
            web_server._get_dashboard_plugins(force_rescan=True)
            prefixes = _mounted_prefixes()
        assert prefixes == []


class TestListingGateAcceptsDirectoryKey:
    def test_enabled_by_directory_key_served_to_frontend(self, mobile_plugin):
        """The /api/dashboard/plugins listing applies the same gate: a
        plugin enabled by its directory key must be served (its JS/CSS
        entry is what the frontend loads)."""
        with gate_config(enabled={"hermes-mobile"}):
            web_server._get_dashboard_plugins(force_rescan=True)
            served = asyncio.run(web_server.get_dashboard_plugins())
        assert "mobile" in {p["name"] for p in served}

    def test_internal_fields_not_leaked_to_frontend(self, mobile_plugin):
        """Whatever the gate records internally must stay internal —
        underscore-prefixed fields never reach the frontend."""
        with gate_config(enabled={"hermes-mobile"}):
            web_server._get_dashboard_plugins(force_rescan=True)
            served = asyncio.run(web_server.get_dashboard_plugins())
        assert served, "expected the enabled plugin to be served"
        assert all(not k.startswith("_") for p in served for k in p)

    def test_not_enabled_not_served(self, mobile_plugin):
        with gate_config(enabled=set()):
            web_server._get_dashboard_plugins(force_rescan=True)
            served = asyncio.run(web_server.get_dashboard_plugins())
        assert "mobile" not in {p["name"] for p in served}


class TestRequestTimeGatesAcceptDirectoryKey:
    """End-to-end through the live app: mounting is not enough — the
    ``_plugin_api_runtime_gate`` middleware re-checks the policy on every
    authenticated ``/api/plugins/<name>/…`` request, and
    ``/dashboard-plugins/<name>/…`` re-checks it per asset request.  Both
    must honor the directory key, or a plugin enabled the documented way
    mounts fine and then 404s at request time anyway.
    """

    @pytest.fixture
    def live_client(self, mobile_plugin, monkeypatch):
        """Authenticated TestClient against the real app with the
        mismatched-key plugin mounted (mirrors the route snapshot /
        reorder / restore dance of ``_install_example_plugin`` in
        ``test_web_server.py``)."""
        from starlette.testclient import TestClient

        import hermes_state
        from hermes_constants import get_hermes_home
        from hermes_cli.web_server import _SESSION_HEADER_NAME, _SESSION_TOKEN

        monkeypatch.setattr(
            hermes_state, "DEFAULT_DB_PATH", get_hermes_home() / "state.db"
        )
        app = web_server.app
        # Earlier suites in a full run can leak global app state that has
        # nothing to do with the plugin gates: an OAuth/password-login
        # suite leaves ``app.state.auth_required = True`` (engaging the
        # cookie auth gate, so the legacy session-token header 401s) and
        # a server-bind suite leaves ``app.state.bound_host`` set (the
        # DNS-rebinding host check then 400s TestClient's ``testserver``
        # host).  Neutralize both so these tests are order-independent;
        # monkeypatch restores the prior values on teardown.
        monkeypatch.setattr(app.state, "auth_required", False, raising=False)
        monkeypatch.setattr(app.state, "bound_host", None, raising=False)
        original_routes = list(app.router.routes)
        with gate_config(enabled={"hermes-mobile"}):
            web_server._get_dashboard_plugins(force_rescan=True)
            web_server._mount_plugin_api_routes()
        # Mid-flight mounts append after the SPA catch-all; move them to
        # the front so they win the match-order race (as the app does
        # when mounting at import time).
        new_routes = [r for r in app.router.routes if r not in original_routes]
        for route in new_routes:
            app.router.routes.remove(route)
        for offset, route in enumerate(new_routes):
            app.router.routes.insert(offset, route)
        client = TestClient(app)
        client.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN
        try:
            yield client
        finally:
            app.router.routes[:] = original_routes
            web_server._dashboard_plugins_cache = None

    def test_api_request_allowed_when_enabled_by_directory_key(self, live_client):
        """The field regression's second half: enabled as
        ``hermes plugins enable hermes-mobile``, an authenticated request
        to the mounted route must reach the handler."""
        with gate_config(enabled={"hermes-mobile"}):
            resp = live_client.get("/api/plugins/mobile/hello")
        assert resp.status_code == 200, resp.text

    def test_api_request_blocked_when_disabled_by_directory_key(self, live_client):
        """Runtime disable under either identifier wins immediately."""
        with gate_config(
            enabled={"hermes-mobile", "mobile"}, disabled={"hermes-mobile"}
        ):
            resp = live_client.get("/api/plugins/mobile/hello")
        assert resp.status_code == 404

    def test_api_request_blocked_when_not_enabled(self, live_client):
        """#46435 invariant at request time: no enable spelling → 404,
        even though the router is still mounted."""
        with gate_config(enabled=set()):
            resp = live_client.get("/api/plugins/mobile/hello")
        assert resp.status_code == 404

    def test_asset_served_when_enabled_by_directory_key(self, live_client):
        with gate_config(enabled={"hermes-mobile"}):
            resp = live_client.get("/dashboard-plugins/mobile/manifest.json")
        assert resp.status_code == 200, resp.text

    def test_asset_blocked_when_not_enabled(self, live_client):
        with gate_config(enabled=set()):
            resp = live_client.get("/dashboard-plugins/mobile/manifest.json")
        assert resp.status_code == 404
