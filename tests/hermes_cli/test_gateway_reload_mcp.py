"""/reload-mcp on the shared gateway: the gateway reconciles its MCP servers and refreshes its
agents (``reload.mcp``); the CLI never touches its GatewayAgent stand-in's tools."""
import asyncio
from contextlib import asynccontextmanager

from cli import HermesCLI
from hermes_cli.gateway_connection import GatewayConnection


class _Client:
    def __init__(self):
        self.events = asyncio.Queue()
        self.calls = []

    async def rpc(self, method, *, _timeout=None, **params):
        self.calls.append((method, params))
        if method == "reload.mcp":
            return {"status": "reloaded", "message": "MCP servers reconciled (no server changes)"}
        return {"revision": 2, "execution_generation": 1, "messages": [], "info": {"tool_count": 20}}


def test_reload_mcp_runs_on_the_gateway(capsys):
    client = _Client()

    @asynccontextmanager
    async def connect():
        yield client
    conn = GatewayConnection(connect=connect)
    conn.start(5)
    cli = HermesCLI.__new__(HermesCLI)
    cli.session_id = "local-mcp"
    cli._command_running = False
    try:
        cli._reload_gateway_mcp(conn)
    finally:
        conn.close()
    assert client.calls[0] == ("reload.mcp", {"session_id": "local-mcp", "confirm": True})
    out = capsys.readouterr().out
    assert "20 tool(s) available" in out and "MCP reload failed" not in out, out
