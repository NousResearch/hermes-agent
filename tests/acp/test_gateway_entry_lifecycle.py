"""The ACP transport owns neither MCP discovery nor a surviving client socket."""
import asyncio

import acp
import pytest

from acp_adapter import entry
from acp_adapter.gateway_server import GatewayACPAgent


@pytest.mark.parametrize("crash", [False, True])
def test_entry_closes_transport_without_discovering_mcp(monkeypatch, crash):
    calls = []
    agent = GatewayACPAgent()

    async def close():
        calls.append(("close", asyncio.get_running_loop()))

    async def serve(instance, **kwargs):
        assert instance is agent
        calls.append(("serve", asyncio.get_running_loop()))
        if crash:
            raise RuntimeError("owned transport failure")

    monkeypatch.setattr(entry, "_setup_logging", lambda: None)
    monkeypatch.setattr(entry, "_load_env", lambda: None)
    monkeypatch.setattr("acp_adapter.server.HermesACPAgent", lambda: agent)
    monkeypatch.setattr(agent, "aclose", close)
    monkeypatch.setattr(acp, "run_agent", serve)
    monkeypatch.setattr("hermes_cli.mcp_startup.start_background_mcp_discovery",
                        lambda **kw: calls.append(("mcp", None)))
    monkeypatch.delenv("HERMES_ACP_SKIP_CONFIGURED_MCP", raising=False)
    if crash:
        with pytest.raises(SystemExit):
            entry.main([])
    else:
        entry.main([])
    assert [name for name, loop in calls] == ["serve", "close"]
    assert calls[0][1] is calls[1][1]
