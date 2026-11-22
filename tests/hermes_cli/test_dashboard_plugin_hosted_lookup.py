"""Under ``plugins.isolation: host`` a forwarded plugin request reuses the cached plugin list.

Each request to a hosted dashboard plugin looks the plugin up again to check it is still enabled.
That lookup reads the launch profile's cached discovery instead of scanning the plugin dirs again,
still checks ``plugins.enabled`` on every request, and rescans once when the cached list has no
hosted copy of the plugin (it changed since the last scan).

Driven through the real app on the consent-gate tests' multiplexed daemon stand-in, whose plugin
host is a recorder.
"""
from __future__ import annotations

import json

import pytest

from hermes_cli import web_server, web_server_dashboard
from tests.hermes_cli import test_plugin_api_consent_gate as consent_gate

daemon = consent_gate.daemon  # the multiplexed daemon stand-in fixture
_plugins = consent_gate._plugins
PING = "/api/plugins/alias-tab/ping"


def _install(home):
    """A user plugin in ``plugins/dash-dir`` whose dashboard calls itself ``alias-tab``."""
    dashboard = home / "plugins" / "dash-dir" / "dashboard"
    dashboard.mkdir(parents=True)
    (dashboard.parent / "plugin.yaml").write_text("name: dash-dir\nversion: 1.0.0\n")
    (dashboard / "manifest.json").write_text(json.dumps({"name": "alias-tab", "api": "plugin_api.py"}))
    (dashboard / "plugin_api.py").write_text(consent_gate._PLUGIN_API)
    (dashboard / "index.js").write_text("export {}\n")
    return dashboard.parent


@pytest.fixture
def scans(daemon, monkeypatch):
    """Counts plugin-dir scans by every caller of discovery."""
    counted, discover = [], web_server_dashboard._discover_dashboard_plugins

    def counting(*args, **kwargs):
        counted.append(1)
        return discover(*args, **kwargs)

    monkeypatch.setattr(web_server_dashboard, "_discover_dashboard_plugins", counting)
    monkeypatch.setattr(web_server, "_discover_dashboard_plugins", counting)
    return counted


def test_hosted_requests_do_not_rescan_the_plugin_dirs(daemon, scans):
    _install(daemon.launch)
    _plugins(daemon.launch, enabled=["dash-dir"], isolation="host")
    ping = lambda: daemon.client.get(PING, headers=daemon.owner)
    assert ping().json() == {"hosted": True}  # the first request mounts the plugin
    scans.clear()

    assert [ping().json() for _ in range(5)] == [{"hosted": True}] * 5
    assert scans == []
    assert daemon.hosted[-5:] == ["alias-tab"] * 5


def test_hosted_requests_still_check_consent_on_every_request(daemon, scans):
    _install(daemon.launch)
    _plugins(daemon.launch, enabled=["dash-dir"], isolation="host")
    assert daemon.client.get(PING, headers=daemon.owner).json() == {"hosted": True}
    forwarded = len(daemon.hosted)

    _plugins(daemon.launch, disabled=["dash-dir"], isolation="host")
    response = daemon.client.get(PING, headers=daemon.owner)

    assert response.status_code == 404
    assert len(daemon.hosted) == forwarded


def test_a_backend_added_since_the_last_scan_is_reached_with_one_scan(daemon, scans):
    """A cached entry with no hosted copy of the plugin (its backend was dropped at the last scan and is
    back now) costs one rescan, as the per-request scan found it before; the next request none."""
    plugin = _install(daemon.launch)
    _plugins(daemon.launch, enabled=["dash-dir"], isolation="host")
    assert daemon.client.get(PING, headers=daemon.owner).json() == {"hosted": True}
    manifest = plugin / "dashboard" / "manifest.json"
    manifest.write_text(json.dumps({"name": "alias-tab"}))
    assert daemon.client.get("/api/dashboard/plugins/rescan", headers=daemon.owner).status_code == 200
    assert daemon.client.get(PING, headers=daemon.owner).status_code == 404

    manifest.write_text(json.dumps({"name": "alias-tab", "api": "plugin_api.py"}))
    scans.clear()
    assert daemon.client.get(PING, headers=daemon.owner).json() == {"hosted": True}
    assert len(scans) == 1
    assert daemon.client.get(PING, headers=daemon.owner).json() == {"hosted": True}
    assert len(scans) == 1
