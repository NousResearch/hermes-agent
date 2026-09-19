"""Test profile-scoped process home coverage for plugin enablement sets.

Regression for #112879: when a profile-scoped process (``HERMES_HOME=<root>/profiles/<name>``)
mounted a plugin's API routes because discovery saw it in the root, the runtime gate read
``plugins.enabled`` from the process home alone — whose config.yaml (a fresh profile's) had no
``plugins:`` section — and 404'd with "Plugin not found". The gate must mirror the homes
discovery scans so the two halves of one trust decision can't drift apart (#87197).
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.fixture
def test_client_profile(monkeypatch, tmp_path):
    """Isolated TestClient with a profile-scoped home."""
    try:
        from starlette.testclient import TestClient
    except ImportError:
        pytest.skip("fastapi/starlette not installed")

    root = tmp_path / "hermes-root"
    (root / "plugins").mkdir(parents=True)
    (root / "config.yaml").write_text("plugins:\n  enabled:\n  - root-plugin\n")
    profile_home = root / "profiles" / "test"
    profile_home.mkdir(parents=True)
    # Profile config has no plugins: section (the fresh-profile case).
    (profile_home / "config.yaml").write_text("# empty profile config\n")

    # Create the plugin BEFORE setting HERMES_HOME and importing web_server.
    _make_root_plugin(root)

    monkeypatch.setenv("HERMES_HOME", str(profile_home))

    # Reset the hermes-root memo so it sees the new tmp.
    import hermes_constants
    hermes_constants._default_hermes_root_memo = None

    from hermes_cli import web_server
    from hermes_cli.web_server import app, _SESSION_HEADER_NAME, _SESSION_TOKEN
    web_server._dashboard_plugins_cache = None

    client = TestClient(app)
    client.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN
    yield client, root
    web_server._dashboard_plugins_cache = None


def _make_root_plugin(root: Path, name="root-plugin"):
    """Create a minimal plugin in the hermes root."""
    dashboard_dir = root / "plugins" / name / "dashboard"
    dashboard_dir.mkdir(parents=True)
    (dashboard_dir / "manifest.json").write_text(json.dumps({
        "name": name,
        "label": name.title(),
        "api": "api.py"
    }))
    # api.py must live IN dashboard/ (web_server_dashboard resolves api relative to _dir).
    api_file = dashboard_dir / "api.py"
    api_file.write_text(
        "from fastapi import APIRouter\n"
        "router = APIRouter()\n"
        "@router.get('/health')\n"
        "def health():\n"
        "    return {'ok': True}\n"
    )
    return dashboard_dir


def test_profile_scoped_gate_mirrors_discovery(test_client_profile):
    """Profile-scoped process: root-enabled plugin mounts and serves 200."""
    client, root = test_client_profile

    # Discovery finds root-enabled plugin even when the profile config has no plugins: section.
    resp = client.get("/api/dashboard/plugins")
    assert resp.status_code == 200
    names = {p["name"] for p in resp.json()}
    assert "root-plugin" in names

    # The runtime gate must allow the mounted route (was 404 before the fix).
    resp = client.get("/api/plugins/root-plugin/health")
    assert resp.status_code == 200
    assert resp.json() == {"ok": True}


def test_profile_disabled_list_still_wins(test_client_profile, monkeypatch):
    """Profile-scoped process: an explicit plugins.disabled in profile config blocks the route."""
    client, root = test_client_profile

    # Add root-plugin to the profile's disabled set AFTER fixture setup.
    profile_home = root / "profiles" / "test"
    (profile_home / "config.yaml").write_text("plugins:\n  disabled:\n  - root-plugin\n")

    # Re-import so new config is read.
    import hermes_constants
    hermes_constants._default_hermes_root_memo = None
    from hermes_cli import web_server
    web_server._dashboard_plugins_cache = None

    # The gate must block it (deny-list stays authoritative).
    resp = client.get("/api/plugins/root-plugin/health")
    assert resp.status_code == 404
    assert "Plugin not found" in resp.text
