"""Adding or enabling a user-configured MCP server never needs the bundled MCP catalog module."""

import sys


def test_generic_mcp_paths_run_without_the_catalog_module(monkeypatch):
    from hermes_cli.config import load_config, save_config

    cfg = load_config()
    cfg["mcp_servers"] = {"mine": {"url": "https://mcp.example.test/mcp"}}
    save_config(cfg)
    monkeypatch.setitem(sys.modules, "hermes_cli.mcp_catalog", None)

    from tools.connectors.mcp import _configured_names
    from tui_gateway import server

    assert _configured_names() == ["mine"]
    resp = server._methods["mcp.servers.add"](1, {"name": "added", "config": {"url": "https://add.example.test/mcp"}})
    assert "error" not in resp, resp.get("error")
    assert load_config()["mcp_servers"]["added"]["url"] == "https://add.example.test/mcp"
