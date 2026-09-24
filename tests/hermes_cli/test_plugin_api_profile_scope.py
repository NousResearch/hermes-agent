"""Dashboard/desktop plugin backends must run under the served profile's secret scope.

A process that hosts more than one profile home — ``hermes serve`` / ``hermes dashboard``
pooling served profiles (``hermes_cli/web_server_profiles._config_profile_scope``), the
multiplexed messaging gateway, hosted rooms — flips ``agent.secret_scope`` to fail closed:
``get_secret`` RAISES instead of borrowing the launch profile's ``os.environ``.

Core routers bind that scope per request. Plugin backends (``plugin_api.py`` mounted by
``_mount_plugin_api_routes``) were mounted bare, so the first credential read inside a
plugin raised ``UnscopedSecretError`` and the panel rendered an error card instead of its
data — the user-visible "Honcho provider helpers unavailable ... could not read this
profile's HERMES_HONCHO_HOST" on the memory plugin, whose whole read path is
``get_secret``-based (``plugins/memory/honcho/client.py``).

These tests pin both halves of the contract: a plugin request resolves the profile's OWN
credentials (never a raise), and it never inherits the launch profile's value.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import yaml

import hermes_cli.web_server as web_server
import hermes_cli.web_server_dashboard as web_server_dashboard

_EXAMPLE_PLUGIN_FIXTURE = (
    Path(__file__).resolve().parent.parent / "fixtures" / "plugins" / "profile-scope-probe"
)


@pytest.fixture
def probe_plugin(_isolate_hermes_home):
    """Install the probe plugin into ``$HERMES_HOME/plugins`` and mount its routes."""
    from hermes_cli.config import get_process_hermes_home, load_config, save_config

    root = get_process_hermes_home()
    dst = root / "plugins" / "profile-scope-probe"
    shutil.copytree(_EXAMPLE_PLUGIN_FIXTURE, dst, dirs_exist_ok=True)

    cfg = load_config()
    plugins_cfg = cfg.setdefault("plugins", {})
    enabled = plugins_cfg.get("enabled")
    enabled = list(enabled) if isinstance(enabled, list) else []
    if "profile-scope-probe" not in enabled:
        enabled.append("profile-scope-probe")
    plugins_cfg["enabled"] = enabled
    save_config(cfg)

    original_routes = list(web_server.app.router.routes)
    web_server._dashboard_plugins_cache = None
    web_server._get_dashboard_plugins(force_rescan=True)
    web_server_dashboard._mount_plugin_api_routes()
    # Mid-flight mounts land after the SPA catch-all; hoist them so they match first.
    new_routes = [r for r in web_server.app.router.routes if r not in original_routes]
    for route in new_routes:
        web_server.app.router.routes.remove(route)
    for offset, route in enumerate(new_routes):
        web_server.app.router.routes.insert(offset, route)
    try:
        yield root
    finally:
        for route in list(web_server.app.router.routes):
            if route not in original_routes:
                web_server.app.router.routes.remove(route)
        shutil.rmtree(dst, ignore_errors=True)


@pytest.fixture
def multi_profile_host(probe_plugin, monkeypatch):
    """A process that hosts a second profile home: multiplex on, fail closed.

    Mirrors ``tui_gateway.launch_profile_policy.activate_multi_profile_hosting`` without
    its launch-env snapshot (which is process-global and would leak between tests).
    """
    import agent.secret_scope as secret_scope

    root = probe_plugin
    (root / ".env").write_text("PLUGIN_PROBE_SECRET=launch-value\n", encoding="utf-8")
    secondary = root / "profiles" / "systemos"
    secondary.mkdir(parents=True, exist_ok=True)
    (secondary / "config.yaml").write_text("{}\n", encoding="utf-8")
    (secondary / ".env").write_text("PLUGIN_PROBE_SECRET=systemos-value\n", encoding="utf-8")

    monkeypatch.setattr(secret_scope, "_MULTIPLEX_ACTIVE", True)
    return root


@pytest.fixture
def client(_isolate_hermes_home):
    from hermes_cli.web_server import _SESSION_HEADER_NAME, _SESSION_TOKEN

    try:
        from starlette.testclient import TestClient
    except ImportError:  # pragma: no cover - fastapi is a hard dependency in practice
        pytest.skip("fastapi/starlette not installed")

    test_client = TestClient(web_server.app, raise_server_exceptions=False)
    test_client.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN
    return test_client


def test_plugin_backend_reads_the_served_profiles_credentials(client, multi_profile_host):
    """The reported failure: a plugin read must not raise in a multi-profile process."""
    resp = client.get("/api/plugins/profile-scope-probe/probe?profile=systemos")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["secret"] == "systemos-value"
    assert body["multiplex_active"] is True
    assert body["scoped"] is True


def test_plugin_backend_serves_the_launch_profile_without_a_profile_param(
    client, multi_profile_host
):
    """The launch profile's own panel keeps working once the flip is on."""
    resp = client.get("/api/plugins/profile-scope-probe/probe")
    assert resp.status_code == 200, resp.text
    assert resp.json()["secret"] == "launch-value"


def test_plugin_backend_never_inherits_the_launch_profiles_secret(
    client, multi_profile_host
):
    """Isolation: a secondary profile's read resolves to ITS env, not the launch one's."""
    secondary_env = multi_profile_host / "profiles" / "systemos" / ".env"
    secondary_env.write_text("", encoding="utf-8")

    resp = client.get("/api/plugins/profile-scope-probe/probe?profile=systemos")
    assert resp.status_code == 200, resp.text
    assert resp.json()["secret"] == "unset"


def test_unknown_profile_is_a_client_error_not_a_500(client, multi_profile_host):
    """A bogus ``?profile=`` keeps the core routers' 400/404 contract."""
    resp = client.get("/api/plugins/profile-scope-probe/probe?profile=not-a-profile")
    assert resp.status_code in (400, 404), resp.text
    assert json.loads(resp.text)