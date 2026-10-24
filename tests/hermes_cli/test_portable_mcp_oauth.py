"""A portable package's OAuth MCP server, from declaration to sign-in.

Agent Plugins v1 leaves authorization to the client and gives ``mcp.json`` no field for it, so a
package declares it in the Hermes extension (``servers.<name>.auth: "oauth"``). Every sign-in surface
(``hermes mcp login``, the Desktop/TUI ``mcp.servers.oauth.start`` RPC, the dashboard route) then
accepts the plugin's server, and signing in writes only the profile's token store: a config.yaml copy
would replace the plugin's entry. These run real plugin discovery against a temp ``HERMES_HOME``.
"""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import pytest

import hermes_yaml as yaml
from hermes_cli.agent_plugins import MCP_SCHEMA_V1, PLUGIN_SCHEMA_V1, AgentPluginError, load_agent_plugin

URL = "https://mcp.example.com"


def _write_package(root: Path, declaration: dict | None, server: dict | None = None) -> None:
    root.mkdir(parents=True, exist_ok=True)
    servers = {"trade": declaration} if declaration is not None else {}
    (root / "plugin.json").write_text(json.dumps({
        "$schema": PLUGIN_SCHEMA_V1, "name": "trade-kit",
        "extensions": {"com.nousresearch.hermes": {"servers": servers}}}), encoding="utf-8")
    (root / "mcp.json").write_text(json.dumps({"$schema": MCP_SCHEMA_V1, "mcpServers": {
        "trade": server or {"type": "streamable-http", "url": URL}}}), encoding="utf-8")


def _read_config(home: Path) -> dict:
    path = home / "config.yaml"
    return (yaml.safe_load(path.read_text(encoding="utf-8")) or {}) if path.is_file() else {}


def _sign_in_succeeds(monkeypatch) -> list[dict]:
    """Stand in for the provider's browser round trip: the probe leaves a token on disk, as the SDK does."""
    from tools.mcp_oauth import HermesTokenStorage

    probed: list[dict] = []

    def probe(name, config, **_kwargs):
        probed.append(dict(config))
        path = HermesTokenStorage(name)._tokens_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"access_token": "t", "token_type": "Bearer"}), encoding="utf-8")
        return [("list_strategies", "List strategies")]

    monkeypatch.setattr("hermes_cli.mcp_config._probe_single_server", probe)
    return probed


@pytest.fixture
def plugin_home(tmp_path, monkeypatch):
    """A profile with the ``trade-kit`` package installed and enabled; ``declare`` rewrites its server entry."""
    from hermes_cli import plugins as plugins_mod

    home = tmp_path / ".hermes"
    bundled = tmp_path / "bundled"
    bundled.mkdir()
    monkeypatch.setenv("HOME", str(tmp_path / "os-home"))
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(plugins_mod, "get_bundled_plugins_dir", lambda: bundled)

    def declare(declaration: dict | None) -> Path:
        _write_package(home / "plugins" / "trade-kit", declaration)
        (home / "config.yaml").write_text(
            yaml.safe_dump({"plugins": {"enabled": ["trade-kit"]}}), encoding="utf-8")
        plugins_mod.discover_plugins(force=True)
        return home

    return declare


@pytest.mark.parametrize("declaration,expected", [({"auth": "oauth"}, "oauth"), (None, None)])
def test_package_declares_oauth_for_its_remote_server(tmp_path, declaration, expected):
    _write_package(tmp_path, declaration)
    package = load_agent_plugin(tmp_path, tmp_path / "data")
    server = package.mcp_servers["trade"]
    assert server.get("auth") == expected
    assert server["strict_redirect_headers"] is True  # the v1 redirect rule still holds
    assert "trade" not in package.server_declarations  # auth alone declares no application


@pytest.mark.parametrize("declaration,server,message", [
    ({"auth": "basic"}, None, "auth must be 'oauth'"),
    ({"auth": "oauth"}, {"type": "stdio", "command": "python"}, "only to a streamable-http server"),
])
def test_oauth_declaration_refuses_what_hermes_cannot_sign_in_to(tmp_path, declaration, server, message):
    _write_package(tmp_path, declaration, server)
    with pytest.raises(AgentPluginError, match=message):
        load_agent_plugin(tmp_path, tmp_path / "data")


def test_mcp_login_signs_in_to_a_plugin_server_without_copying_it_into_config(plugin_home, monkeypatch):
    from hermes_cli.mcp_config import cmd_mcp_login
    from tools.mcp_oauth import HermesTokenStorage

    home = plugin_home({"auth": "oauth"})
    probed = _sign_in_succeeds(monkeypatch)

    cmd_mcp_login(Namespace(name="trade", flow=None))

    assert [(c["url"], c["auth"]) for c in probed] == [(URL, "oauth")]
    assert HermesTokenStorage("trade").has_cached_tokens()
    assert "mcp_servers" not in _read_config(home)


def test_rpc_sign_in_keeps_a_plugin_server_out_of_config(plugin_home, monkeypatch):
    from tools.connectors import mcp_oauth
    from tui_gateway import mcp_oauth_sessions, server

    home = plugin_home({"auth": "oauth"})
    _sign_in_succeeds(monkeypatch)
    started: list[dict] = []

    def start_flow(hermes_home, name, cfg, **kwargs):
        started.append(kwargs)
        # The worker the real start_flow spawns, run inline with what the RPC handed it.
        mcp_oauth.probe_with_rollback(name, cfg, hermes_home, None, False, save_config=kwargs["save_config"])
        return {"session_id": "s", "auth_url": "https://idp.example/authorize", "flow": "pkce"}

    monkeypatch.setattr(mcp_oauth_sessions, "start_flow", start_flow)

    response = server._methods["mcp.servers.oauth.start"](1, {"name": "trade"})

    assert "error" not in response, response.get("error")
    assert started[0]["save_config"] is False
    assert "mcp_servers" not in _read_config(home)


def test_rpc_still_refuses_a_plugin_server_that_declares_no_oauth(plugin_home):
    from tui_gateway import server

    plugin_home(None)
    response = server._methods["mcp.servers.oauth.start"](1, {"name": "trade"})
    assert response["error"]["code"] == 4090


@pytest.mark.parametrize("own_entry", [False, True])
def test_removing_the_package_signs_its_server_out_unless_config_names_it(plugin_home, monkeypatch, own_entry):
    # Tokens are stored by server name: a later package declaring "trade" must not inherit this sign-in,
    # but the user's own config.yaml "trade" entry connects with the same token and keeps it.
    from hermes_cli.mcp_config import _oauth_tokens_present, _save_mcp_server, cmd_mcp_login
    from hermes_cli.plugins_cmd_remove import _remove_user_plugin

    home = plugin_home({"auth": "oauth"})
    _sign_in_succeeds(monkeypatch)
    cmd_mcp_login(Namespace(name="trade", flow=None))
    if own_entry:
        _save_mcp_server("trade", {"url": URL, "auth": "oauth"})
    assert _oauth_tokens_present("trade")

    _remove_user_plugin(home / "plugins", "trade-kit", home / "plugins" / "trade-kit")

    assert not (home / "plugins" / "trade-kit").exists()
    assert _oauth_tokens_present("trade") is own_entry


@pytest.mark.parametrize("declaration,status", [({"auth": "oauth"}, 200), (None, 409)])
def test_dashboard_auth_route_accepts_only_a_declared_plugin_oauth_server(plugin_home, monkeypatch, declaration, status):
    import asyncio

    from starlette.testclient import TestClient

    from hermes_cli import web_server_mcp
    from hermes_cli.web_server import _SESSION_HEADER_NAME, _SESSION_TOKEN, app

    plugin_home(declaration)
    monkeypatch.setattr(app.state, "auth_required", False, raising=False)
    workers: list[dict] = []

    def worker(flow, cfg, **kwargs):
        workers.append({"cfg": cfg, **kwargs})
        asyncio.run(flow.publish_authorization_url("https://idp.example/authorize?state=s1"))

    monkeypatch.setattr(web_server_mcp, "_run_dashboard_mcp_oauth", worker)
    web_server_mcp._mcp_oauth_flows.clear()
    client = TestClient(app)
    client.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN

    response = client.post("/api/mcp/servers/trade/auth")
    web_server_mcp._mcp_oauth_flows.clear()

    assert response.status_code == status
    if status == 200:
        assert workers[0]["save_config"] is False
        assert workers[0]["cfg"]["url"] == URL
