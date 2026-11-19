"""Regression for #139165: ACP picker identity reaches canonical model mutations."""
import pytest

from acp_adapter.gateway_server import GatewayACPAgent


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model_id, expected",
    [
        ("deepseek:deepseek-flash", {"provider": "deepseek", "model": "deepseek-flash"}),
        ("openrouter:deepseek/example", {"provider": "openrouter", "model": "deepseek/example"}),
        ("anthropic/claude-example:beta", {"model": "anthropic/claude-example:beta"}),
        ("plain-model", {"model": "plain-model"}),
        ("", {"model": ""}),
        ("custom:fixture:local-model:tag", {"provider": "custom:fixture", "model": "local-model:tag"}),
    ],
)
async def test_picker_identity_reaches_prepared_mutation(model_id, expected, tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(
        'custom_providers:\n  - name: fixture\n    base_url: http://127.0.0.1:9/v1\n',
        encoding="utf-8",
    )
    agent = GatewayACPAgent()
    calls = []

    class RPC:
        async def rpc(self, method, **params):
            calls.append((method, params))
            if method == "session.resume":
                return {"execution_generation": 3, "revision": 7}
            assert method == "session.mutate"
            return {"status": "applied"}

    agent._gateway = RPC()
    agent._aliases["editor-tip"] = "logical-session"
    await agent.set_session_model(model_id=model_id, session_id="editor-tip")
    mutation = next(params for method, params in calls if method == "session.mutate")
    assert mutation["payload"] == expected
    assert mutation["session_id"] == "logical-session"
    assert mutation["expected_generation"] == 3
    assert mutation["expected_revision"] == 7
    assert agent._mutations.pending == {}


@pytest.mark.asyncio
@pytest.mark.parametrize("reason", ["connection_lost", "session_busy"])
async def test_prefixed_pick_preserves_mutation_failure_contract(reason):
    from hermes_cli.gateway_client import GatewayClientError

    agent = GatewayACPAgent()
    calls = []

    class RPC:
        async def rpc(self, method, **params):
            if method == "session.resume":
                return {"execution_generation": 3, "revision": 7}
            calls.append(params)
            if len(calls) == 1:
                raise GatewayClientError(reason)
            return {"status": "applied"}

    agent._gateway = RPC()
    with pytest.raises(GatewayClientError, match=reason):
        await agent.set_session_model(model_id="deepseek:example", session_id="session")
    assert "session" not in agent._snapshots
    await agent.set_session_model(model_id="deepseek:example", session_id="session")
    assert calls[0]["payload"] == calls[1]["payload"] == {"provider": "deepseek", "model": "example"}
    assert (calls[0]["request_id"] == calls[1]["request_id"]) == (reason == "connection_lost")
    assert agent._mutations.pending == {}
