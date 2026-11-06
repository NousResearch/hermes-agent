"""After the hosted boot froze the registries, an MCP reload that would change them is an expected,
named outcome (``restart_required``), never an exception the transport turns into -32603."""

import asyncio
from types import SimpleNamespace

import pytest

import registry_freeze


@pytest.fixture(autouse=True)
def thawed(monkeypatch):
    monkeypatch.setattr(registry_freeze, "_frozen", False)


def _connection(refreshed):
    runner = SimpleNamespace(config=SimpleNamespace(multiplex_profiles=False),
                             _mcp_reload_refresh_cached_agents=lambda multiplex, profile: refreshed.append(profile))
    return SimpleNamespace(actor=SimpleNamespace(capabilities={"session:control"}),
                           authority=SimpleNamespace(runner=runner, profile_id="fixture"))


def test_frozen_reload_answers_restart_required(monkeypatch):
    from gateway import session_ancillary

    def refused():
        registry_freeze.refuse_if_frozen("MCP registry")
    monkeypatch.setattr("tools.mcp_tool_discovery.reconcile_mcp_servers_with_config", refused)
    monkeypatch.setattr("tools.mcp_tool_discovery.mcp_servers_not_connected", list)
    monkeypatch.setattr("tools.mcp_tool_discovery.mcp_servers_ready", list)
    refreshed = []
    registry_freeze.freeze_registries()
    result = asyncio.run(session_ancillary.reload_mcp(_connection(refreshed), SimpleNamespace(session_id=None),
                                                      {"confirm": True}))
    assert result["status"] == "restart_required", result
    assert "restart required: registries are frozen" in result["message"], result
    assert refreshed == []  # nothing changed, so no cached agent is rebuilt
