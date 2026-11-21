"""Regression tests for PUT /api/mcp/servers ownership.

A config-native MCP entry whose name is ALSO shipped by an enabled plugin is
config-owned (GET reports ``source: config`` and DELETE allows it). The bulk
save must use the same rule: it previously judged ownership against an empty
config, so any such duplicate made every whole-map save fail with
"Server '<name>' is provided by plugin '<name>' and cannot be modified".
"""
from contextlib import nullcontext
import asyncio

import pytest
from fastapi import HTTPException

from hermes_cli.web_models import MCPServersReplace
from hermes_cli.web_routers import mcp as mcp_router

# Names the installed plugins ship as portable servers.
PLUGIN_SERVERS = {"rea": "rea", "kinocut": "kinocut"}
REA = {"command": "npx", "args": ["-y", "rea-agents@6.4.0", "mcp"]}


def _run(coro):
    return asyncio.run(coro)


def _resolve(config):
    """Mirror server_configs_with_sources: a name in the supplied config is config-owned."""
    servers = dict(config)
    owners = {name: None for name in servers}
    for name, plugin in PLUGIN_SERVERS.items():
        if name not in servers:
            owners[name] = plugin
    return servers, owners


def _patch(monkeypatch, *, current, written):
    monkeypatch.setattr("hermes_cli.mcp_config._get_mcp_servers", lambda: current)
    monkeypatch.setattr("tui_gateway.mcp_rpc_helpers.server_configs_with_sources", _resolve)
    monkeypatch.setattr(
        "hermes_cli.mcp_config._replace_mcp_servers",
        lambda servers: (written.append(servers), (True, []))[1],
    )
    monkeypatch.setattr(mcp_router, "config_write_scope", lambda profile: nullcontext())


@pytest.mark.parametrize("enabled", [True, False])
def test_bulk_save_keeps_config_owned_duplicate_editable(monkeypatch, enabled):
    current = {"rea": {**REA, "enabled": enabled}}
    written = []
    _patch(monkeypatch, current=current, written=written)

    body = MCPServersReplace(servers={**current, "probe": {"url": "https://example.com/mcp"}})

    # On the old call (`server_configs_with_sources({})`) `rea` resolves as
    # plugin-owned and this raises 400; with the live config it saves.
    assert _run(mcp_router.replace_mcp_servers(body, None)) == {"ok": True}
    assert written and "rea" in written[-1]


def test_bulk_save_still_refuses_plugin_only_server(monkeypatch):
    written = []
    _patch(monkeypatch, current={}, written=written)

    body = MCPServersReplace(servers={"kinocut": {"command": "kino", "args": ["--mcp"]}})
    with pytest.raises(HTTPException) as excinfo:
        _run(mcp_router.replace_mcp_servers(body, None))
    assert excinfo.value.status_code == 400
    assert "is provided by plugin" in excinfo.value.detail
    assert not written
