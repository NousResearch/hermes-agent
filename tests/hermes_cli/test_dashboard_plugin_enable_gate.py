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


API_SRC = "from fastapi import APIRouter\\nrouter = APIRouter()\\n"


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
