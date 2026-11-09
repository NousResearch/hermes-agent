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


@pytest.mark.asyncio
async def test_frozen_reload_answers_restart_required_through_the_dispatcher(tmp_path, monkeypatch):
    """The canonical dispatcher validates every reply against the reload.mcp wire contract, so the
    frozen outcome must be a declared status: the reply dispatches clean and refreshes nothing."""
    from gateway.session_authority import LiveSession, SessionAuthority
    from gateway.session_controls import AuthorityConnection
    from hermes_state import SessionDB
    from hermes_state_runtime import begin_runtime_epoch

    def refused():
        registry_freeze.refuse_if_frozen("MCP registry")
    monkeypatch.setattr("tools.mcp_tool_discovery.reconcile_mcp_servers_with_config", refused)
    monkeypatch.setattr("tools.mcp_tool_discovery.mcp_servers_not_connected", list)
    monkeypatch.setattr("tools.mcp_tool_discovery.mcp_servers_ready", list)
    refreshed = []
    db = SessionDB(db_path=tmp_path / "state.db")
    viewer = None
    try:
        epoch = begin_runtime_epoch(db, instance_id="test")
        runner = SimpleNamespace(config=SimpleNamespace(multiplex_profiles=False),
                                 _mcp_reload_refresh_cached_agents=lambda multiplex, profile: refreshed.append(profile))
        authority = SessionAuthority(runner, profile_id="test", instance_id="test", db=db, epoch=epoch)
        authority.sessions["s"] = LiveSession(None, "route")
        viewer = AuthorityConnection(authority, object(), {"user_id": "human"})
        registry_freeze.freeze_registries()
        reply = await viewer.dispatch({"id": 1, "method": "reload.mcp",
                                       "params": {"session_id": "s", "confirm": True}})
        assert reply["result"]["status"] == "restart_required", reply
        assert "restart required: registries are frozen" in reply["result"]["message"], reply
        assert refreshed == []  # nothing changed, so no cached agent is rebuilt
    finally:
        if viewer is not None:
            await viewer.close()
        db.close()
