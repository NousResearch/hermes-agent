"""Adding or enabling a user-configured MCP server never needs the bundled MCP catalog module."""

import sys


def test_generic_mcp_paths_run_without_the_catalog_module(monkeypatch):
    from hermes_cli.config import load_config, save_config

    cfg = load_config()
    cfg["mcp_servers"] = {"mine": {"url": "https://mcp.example.test/mcp"}}
    save_config(cfg)
    monkeypatch.setitem(sys.modules, "hermes_cli.mcp_catalog", None)

    from tools.connectors.mcp import _configured_names
    from tui_gateway.mcp_rpc_helpers import record_mcp_add

    assert _configured_names() == ["mine"]
    record_mcp_add(None, {"url": "https://mcp.example.test/mcp"}, True)
