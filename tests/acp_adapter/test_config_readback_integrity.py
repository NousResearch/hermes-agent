"""Review regressions for complete, current ACP selector responses (#80575, #156312)."""
import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import acp
import pytest
from acp.agent.router import build_agent_router

from acp_adapter.gateway_server import GatewayACPAgent
from acp_adapter.gateway_session_meta import config_options, model_state
from acp_adapter.model_catalog import model_config_option
from acp_adapter.server import HermesACPAgent
from acp_adapter.session import SessionManager


def make_agent():
    manager = SessionManager(agent_factory=lambda: SimpleNamespace(model="same-model", provider="custom"))
    agent = HermesACPAgent(session_manager=manager)
    state = manager.create_session(cwd=".")
    return agent, state


@pytest.mark.asyncio
@pytest.mark.parametrize("value", ["not-an-offered-policy", "", True])
async def test_invalid_approval_choice_preserves_state_without_persisting(value):
    agent, state = make_agent()
    await agent.set_session_mode("dont_ask", state.session_id)
    with patch.object(agent.session_manager, "save_session", wraps=agent.session_manager.save_session) as save:
        with pytest.raises(acp.RequestError) as exc:
            await build_agent_router(agent)("session/set_config_option", {
                "sessionId": state.session_id, "configId": "edit_approval_policy",
                "value": value, **({"type": "boolean"} if isinstance(value, bool) else {}),
            }, False)
        assert exc.value.code == -32602
        assert state.mode == "dont_ask"
        save.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["session/new", "session/fork"])
async def test_initial_notices_follow_the_response_even_with_a_slow_catalog(monkeypatch, method):
    agent, state = make_agent()
    order = []
    notice = asyncio.Event()

    async def record(_session_id=None, update=None, **kwargs):
        order.append("notice")
        notice.set()

    agent.on_connect(SimpleNamespace(session_update=AsyncMock(side_effect=record)))
    original = agent._session_config_options
    building, release = threading.Event(), threading.Event()

    def slow_catalog(current):
        building.set()
        assert release.wait(10), "catalog was never released"
        return original(current)

    monkeypatch.setattr(agent, "_session_config_options", slow_catalog)
    params = {"cwd": ".", "mcpServers": []}
    if method == "session/fork":
        params["sessionId"] = state.session_id
    async def invoke():
        result = await build_agent_router(agent, use_unstable_protocol=True)(method, params, False)
        order.append("response")
        return result

    request = asyncio.create_task(invoke())
    try:
        assert await asyncio.to_thread(building.wait, 10)
        agent._conn.session_update.assert_not_awaited()
        release.set()
        await request
        await asyncio.wait_for(notice.wait(), 10)
        assert order[0] == "response"
    finally:
        release.set()
        await asyncio.gather(request, return_exceptions=True)


@pytest.mark.asyncio
async def test_approval_readback_cannot_replace_a_newer_model_pick(monkeypatch):
    from hermes_cli.model_switch import ModelSwitchResult

    agent, state = make_agent()
    original = agent._build_model_state
    building, release = threading.Event(), threading.Event()

    def slow_old_catalog(current):
        result = original(current)
        if current.model == "same-model":
            building.set()
            assert release.wait(10), "catalog was never released"
        return result

    monkeypatch.setattr(agent, "_build_model_state", slow_old_catalog)
    monkeypatch.setattr("hermes_cli.model_switch.switch_model", lambda **kw: ModelSwitchResult(
        success=True, new_model="new-model", target_provider="custom"))
    agent.session_manager._agent_factory = lambda: SimpleNamespace(model="new-model", provider="custom")
    older = asyncio.create_task(agent.set_config_option("edit_approval_policy", state.session_id, "session"))
    try:
        assert await asyncio.to_thread(building.wait, 10)
        newer = await agent.set_config_option("model", state.session_id, "custom:new-model")
        assert newer.config_options[0].current_value == "custom:new-model"
        release.set()
        reply = await older
        assert reply.config_options[0].current_value == "custom:new-model"
        assert reply.config_options[1].current_value == "session"
    finally:
        release.set()
        await asyncio.gather(older, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("identity", ["provider_route", "base_url"])
async def test_same_model_custom_server_switch_refreshes_the_picker(monkeypatch, identity):
    agent, state = make_agent()
    setattr(state.agent, identity, "old-server")
    connection = SimpleNamespace(session_update=AsyncMock())
    agent.on_connect(connection)

    def switch(_text, current):
        setattr(current.agent, identity, "new-server")
        return "Model switched"

    monkeypatch.setattr(agent, "_handle_slash_command", switch)
    await agent.prompt([acp.text_block("/model custom:new-server:same-model")], state.session_id)
    updates = [call.args[1] for call in connection.session_update.await_args_list]
    options = [update.config_options for update in updates if update.session_update == "config_option_update"]
    assert len(options) == 1
    expected = "new-server" if identity == "provider_route" else "custom"
    assert options[0][0].current_value == f"{expected}:same-model"


def test_both_adapters_preserve_the_canonical_model_selector(monkeypatch):
    agent, state = make_agent()
    monkeypatch.setattr("acp_adapter.server.build_model_state", lambda *args: None)
    monkeypatch.setattr("acp_adapter.model_catalog.build_model_state", lambda *args: None)
    in_process = agent._build_model_state(state)
    gateway = model_state({"model": state.model, "provider": "custom"})
    assert in_process is not None and gateway is not None
    assert in_process.model_dump() == gateway.model_dump()
    assert in_process.category == "model"
    assert in_process.description
    options = config_options(gateway, {}, "default")
    assert options[1] is gateway
    assert options[1].model_dump() == model_config_option(gateway.current_value, gateway.options).model_dump()


@pytest.mark.asyncio
@pytest.mark.parametrize("method,params", [
    ("session/close", {"sessionId": "s"}), ("session/delete", {"sessionId": "s"}),
    ("logout", {}), ("_unknown", {}),
])
async def test_gateway_unsupported_requests_return_method_not_found(method, params):
    router = build_agent_router(GatewayACPAgent(), use_unstable_protocol=True)
    with pytest.raises(acp.RequestError) as exc:
        await router(method, params, False)
    assert exc.value.code == -32601
    assert exc.value.data == {"method": method}


@pytest.mark.asyncio
async def test_gateway_extension_notifications_are_ignored():
    router = build_agent_router(GatewayACPAgent(), use_unstable_protocol=True)
    assert await router("_unknown", {}, True) is None
