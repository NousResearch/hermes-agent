"""Plugin API dispatch binds the requested profile's secret scope (#120310).

``GET /api/plugins/<name>/...`` handlers that read credentials through
``agent.secret_scope.get_secret`` raised ``UnscopedSecretError`` under
multi-profile hosting: the per-plugin dispatch path installed no secret
scope, and the plugin's no-data contract swallowed the failure (blank
status-bar chips, nothing in any log).
"""
from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from hermes_cli import web_server


def _plugin_request(path="/api/plugins/demo/whoami", query=b""):
    from starlette.requests import Request

    scope = {
        "type": "http",
        "method": "GET",
        "path": path,
        "query_string": query,
        "headers": [],
        "state": {"token_authenticated": True},
    }
    return Request(scope)


def _gate_stubs(plugin):
    return (
        patch.object(web_server, "_get_dashboard_plugins", return_value=[plugin]),
        patch("hermes_cli.plugins_cmd._get_enabled_set", return_value=set()),
        patch("hermes_cli.plugins_cmd._get_disabled_set", return_value=set()),
    )


@pytest.mark.asyncio
async def test_plugin_route_reads_secret_under_multiplex(monkeypatch, tmp_path):
    """A credential read inside a plugin route resolves under multiplexing."""
    import agent.secret_scope as ss

    home = tmp_path / "home"
    home.mkdir()
    (home / ".env").write_text("PLUGIN_DEMO_KEY=sekrit\n")
    monkeypatch.setenv("HERMES_HOME", str(home))

    async def call_next(request):
        from agent.secret_scope import get_secret
        from starlette.responses import JSONResponse

        return JSONResponse({"key": get_secret("PLUGIN_DEMO_KEY")})

    request = _plugin_request()
    was_active = ss.is_multiplex_active()
    ss.set_multiplex_active(True)
    try:
        plugins_stub, enabled_stub, disabled_stub = _gate_stubs(
            {"name": "demo", "source": "bundled"}
        )
        with plugins_stub, enabled_stub, disabled_stub:
            response = await web_server._plugin_api_runtime_gate(request, call_next)
    finally:
        ss.set_multiplex_active(was_active)
    assert response.status_code == 200
    assert json.loads(response.body)["key"] == "sekrit"


@pytest.mark.asyncio
async def test_non_plugin_routes_pass_through_unscoped(monkeypatch, tmp_path):
    """The gate installs no scope for paths outside /api/plugins/."""
    import agent.secret_scope as ss

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))

    async def call_next(request):
        from starlette.responses import JSONResponse

        return JSONResponse({"scope": ss.current_secret_scope()})

    request = _plugin_request(path="/api/sessions", query=b"")
    was_active = ss.is_multiplex_active()
    ss.set_multiplex_active(True)
    try:
        plugins_stub, enabled_stub, disabled_stub = _gate_stubs(
            {"name": "demo", "source": "bundled"}
        )
        with plugins_stub, enabled_stub, disabled_stub:
            response = await web_server._plugin_api_runtime_gate(request, call_next)
    finally:
        ss.set_multiplex_active(was_active)
    assert response.status_code == 200
    assert json.loads(response.body)["scope"] is None


@pytest.mark.asyncio
async def test_plugin_route_bad_profile_renders_4xx_not_500(monkeypatch, tmp_path):
    """An unknown ?profile= on a plugin route renders the scope's 4xx, never a 500."""
    import agent.secret_scope as ss
    from unittest.mock import AsyncMock

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))

    call_next = AsyncMock()
    request = _plugin_request(query=b"profile=..%2Fevil")
    was_active = ss.is_multiplex_active()
    ss.set_multiplex_active(True)
    try:
        plugins_stub, enabled_stub, disabled_stub = _gate_stubs(
            {"name": "demo", "source": "bundled"}
        )
        with plugins_stub, enabled_stub, disabled_stub:
            response = await web_server._plugin_api_runtime_gate(request, call_next)
    finally:
        ss.set_multiplex_active(was_active)
    assert response.status_code == 400
    call_next.assert_not_called()
