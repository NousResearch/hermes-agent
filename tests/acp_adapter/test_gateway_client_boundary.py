"""The normal ACP constructor is a transport client, not an embedded owner."""
import pytest

from acp_adapter.server import HermesACPAgent


def test_default_acp_does_not_construct_an_embedded_session_manager(monkeypatch):
    import acp_adapter.server as server

    def forbidden(*args, **kwargs):
        raise AssertionError("ACP constructed an independent session owner")

    monkeypatch.setattr(server, "SessionManager", forbidden)
    agent = HermesACPAgent()
    assert not hasattr(agent, "session_manager")


@pytest.mark.asyncio
async def test_unadvertised_acp_creation_is_not_relabelled_as_cli(tmp_path):
    agent = HermesACPAgent()
    calls = []

    class RPC:
        async def rpc(self, method, **params):
            calls.append((method, params))
            return {"session_create": {"sources": ["cli", "tui", "gui"]}}

    agent._gateway = RPC()
    with pytest.raises(Exception, match="acp_policy_unavailable"):
        await agent.new_session(cwd=str(tmp_path), mcp_servers=[])
    assert calls == [("runtime.describe", {})]
