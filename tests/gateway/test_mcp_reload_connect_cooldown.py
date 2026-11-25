"""Explicit MCP reload retries a recovered server without waiting for backoff (#160034)."""

import asyncio
from types import SimpleNamespace

import pytest


@pytest.mark.asyncio
@pytest.mark.parametrize("surface", ["desktop", "chat"])
async def test_explicit_reload_retries_only_the_requesting_profiles_enabled_servers(monkeypatch, surface):
    from gateway import session_ancillary
    from tests.gateway.test_mcp_reload_refreshes_cached_agents import (
        _make_event,
        _make_runner_with_cached_agents,
    )
    from tools import mcp_tool as core
    from tools import mcp_tool_config as config
    from tools import mcp_tool_discovery as discovery

    monkeypatch.setattr(core, "_mcp_registry_scope", lambda: "requester")
    monkeypatch.setattr(config, "_load_mcp_config", lambda: {
        "recovered": {"url": "http://127.0.0.1:9/mcp"},
        "disabled": {"url": "http://127.0.0.1:9/mcp", "enabled": False},
    })
    for name in ("_servers", "_server_scope_keys", "_server_connect_errors",
                 "_server_connect_failures", "_server_connect_retry_after",
                 "_lazy_server_configs", "_server_tool_scopes"):
        monkeypatch.setattr(core, name, {})
    monkeypatch.setattr(core, "_server_connecting", set())
    monkeypatch.setattr(core, "_parallel_safe_servers", set())
    discovery._note_connect_failure("recovered", RuntimeError("offline"))
    discovery._note_connect_failure("disabled", RuntimeError("disabled"))
    other_key = ("other", "recovered")
    core._server_connect_failures[other_key] = 5
    core._server_connect_retry_after[other_key] = float("inf")
    attempts = []

    def discover():
        selected = discovery._select_new_servers(config._load_mcp_config())
        attempts.extend(selected)
        # A failed explicit attempt must re-arm automatic backoff.
        for name in selected:
            core._server_connecting.discard(("requester", name))
            discovery._note_connect_failure(name, RuntimeError("still offline"))
        return []

    monkeypatch.setattr(discovery, "discover_mcp_tools", discover)
    runner = _make_runner_with_cached_agents(num_agents=0)
    runner._mcp_reload_refresh_cached_agents = lambda *args: None
    if surface == "desktop":
        async def no_session(*args):
            return None

        monkeypatch.setattr(session_ancillary, "_commit_session_mcp", no_session)
        monkeypatch.setattr("gateway.session_authorities.served_profile_name", lambda _: "default")
        connection = SimpleNamespace(
            actor=SimpleNamespace(capabilities={"session:control"}),
            authority=SimpleNamespace(runner=runner, profile_id="requester"),
        )
        from gateway.session_authority import AuthorityRefusal
        with pytest.raises(AuthorityRefusal, match="mcp_servers_unavailable"):
            await session_ancillary.reload_mcp(connection, None, {"confirm": True})
    else:
        async def execute(fn):
            return await asyncio.to_thread(fn)

        runner._run_in_executor_with_context = execute
        monkeypatch.setattr("tools.mcp_tool_lifecycle.shutdown_mcp_servers", lambda **kwargs: None)
        monkeypatch.setattr("tools.mcp_tool_agent.reprobe_tool_availability", lambda: None)
        await runner._execute_mcp_reload(_make_event())

    assert attempts == ["recovered"]
    assert discovery._connect_cooldown_active("recovered")
    assert discovery._connect_cooldown_active("disabled")
    assert core._server_connect_failures[other_key] == 5
    assert core._server_connect_retry_after[other_key] == float("inf")
    assert discovery.reconcile_mcp_servers_with_config()["added"] == []
    assert attempts == ["recovered"], "automatic reconcile must keep honoring the new backoff"
