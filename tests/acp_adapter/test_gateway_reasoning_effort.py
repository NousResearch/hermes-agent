"""The session's reasoning effort as an ACP config option on the gateway path (#136084, #18326).

``configOptions`` advertises a ``reasoning_effort`` select (category ``thought_level``) beside the edit mode
and the model picker, which a client that reads ``configOptions`` would otherwise lose (it ignores ``modes``).
A set commits the owner's ``reasoning`` mutation, the one Desktop's reasoning pill and ``/reasoning`` use.
"""
from types import SimpleNamespace
from unittest.mock import AsyncMock

import acp
import pytest
from acp.schema import ModelInfo, SessionModelState

from acp_adapter.gateway_server import GatewayACPAgent
from hermes_cli.gateway_client import GatewayClientError
from tests.acp_adapter.test_gateway_sessions import editor
from tests.fakes.fake_llm_provider import FakeLLMServer, Text

PICKER = SessionModelState(current_model_id="custom:a", available_models=[
    ModelInfo(model_id="custom:a", name="a"), ModelInfo(model_id="custom:b", name="b")])


def _agent(rpc):
    agent = GatewayACPAgent()
    agent._gateway, agent._conn = AsyncMock(), AsyncMock()
    agent._gateway.rpc.side_effect = rpc
    agent._snapshots["s"] = {"execution_generation": 7}
    return agent


def _options(options):
    return {o.id: (o.category, o.current_value, [row.value for row in o.options]) for o in options}


def _owner(sent, efforts):
    """A fake owner whose ``session.info`` reports the next effort in ``efforts`` after each mutation."""
    async def rpc(method, **params):
        if method == "session.resume":
            return {"revision": 3, "execution_generation": 7}
        if method == "session.info":
            return {"model": "a", "reasoning_effort": efforts[len(sent) - 1]}
        assert method == "session.mutate", method
        sent.append((params["operation"], params["payload"], params["expected_generation"]))
        return {"session_id": "s", "operation": params["operation"], "revision": 4}
    return rpc


@pytest.mark.asyncio
async def test_attach_advertises_mode_model_and_reasoning_effort(monkeypatch):
    monkeypatch.setattr("acp_adapter.gateway_session_meta.model_state", lambda info: PICKER)
    monkeypatch.setattr("acp_adapter.gateway_session_meta.provenance_meta", lambda *a: None)
    agent = _agent(None)
    agent._conn = None
    fields = await agent._session_fields("s", {"edit_approval_policy": "session", "reasoning_effort": "high"}, "s")
    options = _options(fields["config_options"])
    assert options["edit_approval_policy"][:2] == ("mode", "session")
    assert options["model"] == ("model", "custom:a", ["custom:a", "custom:b"])
    category, current, values = options["reasoning_effort"]
    assert (category, current) == ("thought_level", "high")
    assert values[:3] == ["default", "none", "minimal"] and {"low", "medium", "high", "xhigh", "max"} <= set(values)


@pytest.mark.asyncio
async def test_setting_reasoning_effort_commits_the_owners_reasoning_mutation():
    sent = []
    # The second answer is the profile's own level, which "default" hands the session back to.
    agent = _agent(_owner(sent, ["xhigh", "low"]))
    reply = await agent.set_config_option("reasoning_effort", "s", "xhigh")
    assert sent == [("reasoning", {"effort": "xhigh"}, 7)]
    assert _options(reply.config_options)["reasoning_effort"][1] == "xhigh"

    reply = await agent.set_config_option("reasoning_effort", "s", "default")
    assert sent[-1] == ("reasoning", {"effort": None}, 7)
    assert _options(reply.config_options)["reasoning_effort"][1] == "low"


@pytest.mark.asyncio
async def test_unknown_reasoning_level_is_refused_before_the_owner():
    sent = []
    agent = _agent(_owner(sent, []))
    with pytest.raises(GatewayClientError, match="invalid_params"):
        await agent.set_config_option("reasoning_effort", "s", "hgih")
    assert sent == []


@pytest.mark.asyncio
async def test_a_busy_session_refuses_the_level_by_name():
    async def rpc(method, **params):
        if method == "session.resume":
            return {"revision": 3, "execution_generation": 7}
        raise GatewayClientError("session_busy")

    agent = _agent(rpc)
    with pytest.raises(acp.RequestError):
        await agent.set_config_option("reasoning_effort", "s", "high")
    said = agent._conn.session_update.await_args.kwargs["update"].content.text
    assert said.startswith("Reasoning effort changes apply when the session is idle"), said


@pytest.mark.asyncio
async def test_another_surfaces_level_reaches_the_editor_once():
    agent = _agent(None)
    agent._efforts["s"] = "medium"
    for _ in range(2):
        await agent._project({"session_id": "s", "type": "session.info", "payload": {"reasoning_effort": "high"}})
    shown = [call.kwargs["update"] for call in agent._conn.session_update.await_args_list]
    assert [u.session_update for u in shown] == ["config_option_update"], shown
    assert _options(shown[0].config_options)["reasoning_effort"][1] == "high"


@pytest.mark.asyncio
async def test_the_model_option_is_a_model_pick(monkeypatch):
    monkeypatch.setattr("acp_adapter.gateway_session_meta.model_state", lambda info: PICKER)
    sent = []
    agent = _agent(_owner(sent, ["medium"]))
    reply = await agent.set_config_option("model", "s", "custom:b")
    assert sent == [("model", {"model": "b", "provider": "custom"}, 7)]
    assert _options(reply.config_options)["model"][0] == "model"


@pytest.fixture
def model_peer():
    with FakeLLMServer([Text("ok")]) as provider:
        yield SimpleNamespace(server_port=provider.port)


@pytest.mark.platforms("linux")
@pytest.mark.asyncio
@pytest.mark.parametrize("daemon", [{"agent": {"reasoning_effort": "low"}}], indirect=True)
async def test_editor_sets_reasoning_effort_and_it_persists_with_the_session(daemon, tmp_path):
    async with editor(daemon, tmp_path) as acp_client:
        await acp_client.rpc("initialize", protocolVersion=1, clientCapabilities={})
        created = (await acp_client.rpc("session/new", cwd=str(tmp_path), mcpServers=[]))["result"]
        sid = created["sessionId"]
        effort = {o["id"]: o for o in created["configOptions"]}["reasoning_effort"]
        # config.yaml's agent.reasoning_effort is the new session's level.
        assert (effort["category"], effort["currentValue"]) == ("thought_level", "low"), effort

        reply = await acp_client.rpc("session/set_config_option", sessionId=sid,
                                     configId="reasoning_effort", value="high")
        assert {o["id"]: o for o in reply["result"]["configOptions"]}["reasoning_effort"]["currentValue"] == "high"

        loaded = (await acp_client.rpc("session/load", sessionId=sid, cwd=str(tmp_path), mcpServers=[]))["result"]
        assert {o["id"]: o for o in loaded["configOptions"]}["reasoning_effort"]["currentValue"] == "high", loaded

        reply = await acp_client.rpc("session/set_config_option", sessionId=sid,
                                     configId="reasoning_effort", value="default")
        assert {o["id"]: o for o in reply["result"]["configOptions"]}["reasoning_effort"]["currentValue"] == "low"
