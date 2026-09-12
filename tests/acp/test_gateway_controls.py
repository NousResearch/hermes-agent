"""ACP controls retain generation fences and reject unavailable mutations."""
import pytest

from acp_adapter.server import HermesACPAgent
from hermes_cli.gateway_client import GatewayClientError
from hermes_state import SessionDB


@pytest.mark.asyncio
async def test_cancel_uses_observed_generation_and_mutations_fail_explicitly(tmp_path):
    agent = HermesACPAgent()
    calls = []

    class RPC:
        async def rpc(self, method, **params):
            calls.append((method, params))
            raise GatewayClientError("stale_generation")

    agent._gateway = RPC()
    agent._snapshots["shared"] = {"execution_generation": 4, "revision": 7}
    # No ACP prompt in flight: another surface's running turn is not ours to interrupt.
    await agent.cancel("shared")
    assert calls == []
    # Our own queued admission is cancelled by identity; a stale answer propagates verbatim
    # (the receipt lookup is the next call, which this authority also refuses).
    agent._admissions["shared"] = "adm-1"
    with pytest.raises(GatewayClientError, match="stale_generation"):
        await agent.cancel("shared")
    assert calls[0] == ("prompt.cancel", {"session_id": "shared", "admission_id": "adm-1"})
    assert calls[1][0] == "prompt.receipt"
    calls.clear()
    # Model changes are prepared canonical mutations: a stale authority answer propagates
    # verbatim instead of being retried through any alternate writer.
    with pytest.raises(GatewayClientError, match="stale_generation"):
        await agent.set_session_model(session_id="shared", model_id="new-model")
    methods = [method for method, _ in calls]
    assert methods and set(methods) <= {"session.resume", "session.mutate"}, methods
    # Edit-policy mode is still unsupported and must not reach the authority at all.
    before = len(calls)
    with pytest.raises(GatewayClientError, match="unavailable"):
        await agent.set_session_mode(session_id="shared", mode_id="dont_ask")
    assert len(calls) == before, "Unsupported operations must not reach an alternate writer"


@pytest.mark.asyncio
async def test_gateway_acp_catalog_does_not_boot_a_runtime(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    agent = HermesACPAgent()
    assert (await agent.list_sessions()).sessions == []
    assert not (tmp_path / "state.db").exists()
    with SessionDB(tmp_path / "state.db") as db:
        db.create_session(session_id="persisted-acp", source="acp", model="fixture", model_config={"cwd": str(tmp_path)})
        db.append_message("persisted-acp", "user", "picker history")
    assert [row.session_id for row in (await agent.list_sessions(cwd=str(tmp_path))).sessions] == ["persisted-acp"]
    assert agent._gateway is None
