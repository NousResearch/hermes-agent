"""Regression pin for the #102592 review: serve startup stays fail-open on
plugin discovery failure.

Plugin discovery for the web/serve runtime is owned by the launcher
(``_dashboard_prepare_runtime()`` warns and continues) and by
``invoke_hook()``'s lazy path. ``start_server()`` deliberately does not
re-run it: ``PluginManager.discover_and_load()`` resets ``_discovered``
when loading raises, so an unguarded second scan in ``start_server()``
turns an enabled plugin's load failure into a fatal startup error — the
server never binds — instead of the launcher's warning.
"""

from __future__ import annotations

import hermes_cli.web_server as web_server
from tests.hermes_cli.test_dashboard_auth_gate import _stub_uvicorn_run


def test_serve_startup_survives_plugin_discovery_failure(monkeypatch):
    def _failing_discovery(*args, **kwargs):
        raise RuntimeError("plugin load failed")

    monkeypatch.setattr("hermes_cli.plugins.discover_plugins", _failing_discovery)
    _stub_uvicorn_run(monkeypatch)

    web_server.start_server(
        host="127.0.0.1", port=0, open_browser=False, headless=True
    )
