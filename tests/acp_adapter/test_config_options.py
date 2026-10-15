"""SDK-v1 session selectors drive the same controls as text commands."""
from types import SimpleNamespace

import pytest
from acp.schema import ConfigOptionUpdate, TextContentBlock

from acp_adapter.server import HermesACPAgent
from acp_adapter.session import SessionManager
from tests.acp_adapter.test_acp_commands import CaptureConn


def values(response):
    options = response.config_options
    assert {o.id for o in options} == {"model", "reasoning", "fast"}
    for option in options:
        assert option.current_value in {row.value for row in option.options}
    return {o.id: o.current_value for o in options}


def runtime():
    return SimpleNamespace(model="gpt-6-sol", provider="openai-codex", api_mode="codex_responses",
                           base_url="https://chatgpt.com/backend-api/codex", reasoning_config=None,
                           service_tier=None, request_overrides={}, enabled_toolsets=[], disabled_toolsets=[])


@pytest.mark.asyncio
async def test_inprocess_sdk_options_text_and_independent_cold_fork(tmp_path, monkeypatch):
    from hermes_state import SessionDB
    with SessionDB(tmp_path / "state.db") as db:
        manager = SessionManager(agent_factory=runtime, db=db)
        agent = HermesACPAgent(session_manager=manager)
        conn = CaptureConn()
        agent.on_connect(conn)
        created = await agent.new_session(cwd=str(tmp_path))
        assert values(created)["fast"] == "normal"
        wire = created.model_dump(by_alias=True, exclude_none=True)
        assert all("id" in o and "configId" not in o for o in wire["configOptions"])
        sid = created.session_id
        state = manager.get_session(sid)
        state.history = [{"role": "user", "content": "keep transcript"}]
        state.agent._cached_system_prompt = "frozen prompt"
        state.agent.enabled_toolsets = ["file"]
        state.agent.request_overrides = {"extra_body": {"user": "keep"}}
        await agent.set_config_option("fast", sid, "fast")
        assert state.agent.request_overrides == {"extra_body": {"user": "keep"}, "service_tier": "priority"}
        await agent.prompt([TextContentBlock(type="text", text="/fast normal")], sid)
        assert state.agent.request_overrides == {"extra_body": {"user": "keep"}}
        assert values(await agent.set_config_option("reasoning", sid, "high"))["reasoning"] == "high"
        await agent.prompt([TextContentBlock(type="text", text="/reasoning low")], sid)
        assert state.agent.reasoning_config == {"enabled": True, "effort": "low"}
        await agent.set_config_option("reasoning", sid, "high")
        await agent.prompt([TextContentBlock(type="text", text="/fast cold")], sid)
        assert state.agent.service_tier == "cold"
        assert any(isinstance(update, ConfigOptionUpdate) for _, update in conn.updates)
        forked = await agent.fork_session(cwd=str(tmp_path), session_id=sid)
        assert values(forked)["fast"] == "cold"
        await agent.set_config_option("fast", forked.session_id, "normal")
        cold = HermesACPAgent(session_manager=SessionManager(agent_factory=runtime, db=db))
        resumed = await cold.resume_session(cwd=str(tmp_path), session_id=sid)
        assert values(resumed)["fast"] == "cold"
        assert values(resumed)["reasoning"] == "high"
        child = await cold.load_session(cwd=str(tmp_path), session_id=forked.session_id)
        assert values(child)["fast"] == "normal"
        assert manager.get_session(forked.session_id).agent.service_tier is None
        assert [(m["role"], m["content"]) for m in state.history] == [("user", "keep transcript")]
        assert state.agent._cached_system_prompt == "frozen prompt"
        assert state.agent.enabled_toolsets == ["file"]
        before = len(conn.updates)
        state.is_running = True
        with pytest.raises(Exception, match="busy"):
            await agent.set_config_option("reasoning", sid, "low")
        state.is_running = False
        from acp.exceptions import RequestError
        for key, value in [("opaque", "yes"), ("fast", "banana"), ("reasoning", "nonsense")]:
            with pytest.raises(RequestError):
                await agent.set_config_option(key, sid, value)
        assert len(conn.updates) == before
        assert "session-scoped" in agent._handle_slash_command("/fast fast --global", state)
        assert state.agent.service_tier == "cold"
        # Legacy model writes refresh all controls and retain explicit intent during reconstruction.
        from hermes_cli.model_switch import ModelSwitchResult
        monkeypatch.setattr("hermes_cli.model_switch.switch_model", lambda **kw: ModelSwitchResult(
            success=True, target_provider="custom", new_model="local-model"))
        empty = await agent.new_session(cwd=str(tmp_path))
        assert db.get_session(empty.session_id) is None
        # A newly paid profile default must not enable fast mode during a session model change.
        def changed_default(**kwargs):
            agent = runtime()
            agent.model, agent.provider = kwargs["model"], kwargs["requested_provider"]
            agent.service_tier = "priority"
            agent.request_overrides = {"service_tier": "priority"}
            return agent

        monkeypatch.setattr(manager, "_make_agent", changed_default)
        await agent.set_config_option("model", empty.session_id, "custom:local-model")
        assert db.get_session(empty.session_id)["model"] == "local-model"
        assert manager.get_session(empty.session_id).agent.service_tier is None
        assert "service_tier" not in manager.get_session(empty.session_id).agent.request_overrides
        await agent.set_session_model("custom:local-model", sid)
        assert state.agent.reasoning_config == {"enabled": True, "effort": "high"}
        assert state.agent.service_tier == "cold"
        assert state.agent._cached_system_prompt == "frozen prompt"
        assert isinstance(conn.updates[-1][1], ConfigOptionUpdate)
        fast = next(o for o in conn.updates[-1][1].config_options if o.id == "fast")
        assert {o.value for o in fast.options} == {"normal", "cold"}
        # Embedded reset restores configured defaults without changing the chosen model.
        monkeypatch.setattr("hermes_cli.config.load_config", lambda: {
            "agent": {"reasoning_effort": "low", "service_tier": "normal"}})
        await agent.prompt([TextContentBlock(type="text", text="/reset")], sid)
        assert state.agent.reasoning_config == {"enabled": True, "effort": "low"}
        assert state.agent.service_tier is None and state.control_values == {}
        assert state.model == "local-model" and state.history == []


@pytest.mark.asyncio
async def test_gateway_sdk_controls_use_owner_and_refusals_do_not_publish(tmp_path):
    from hermes_cli.gateway_client import GatewayClientError
    agent = HermesACPAgent()
    agent._home = tmp_path
    conn = CaptureConn()
    calls = []
    info = {"model": "gpt-6-sol", "provider": "openai-codex", "api_mode": "codex_responses",
            "reasoning_effort": "medium", "service_tier": "auto"}
    snapshot = {"session_id": "s", "revision": 1, "execution_generation": 0, "info": info}

    class Owner:
        refuse = False
        ambiguous = False

        async def rpc(self, method, **params):
            calls.append((method, params))
            if method == "runtime.describe":
                return {"session_create": {"sources": ["acp"]}, "capabilities": ["acp-editor-policy-v1"]}
            if method == "session.create":
                return {**snapshot, "info": dict(info)}
            if method == "session.info":
                return dict(info)
            if method == "session.resume":
                return {**snapshot, "info": dict(info)}
            if method == "session.mutate":
                if self.refuse:
                    raise GatewayClientError("session_busy")
                operation, payload = params["operation"], params["payload"]
                if operation == "reasoning":
                    info["reasoning_effort"] = payload["effort"]
                else:
                    info["service_tier"] = {"fast": "priority", "normal": None}.get(
                        payload["service_tier"], payload["service_tier"])
                if self.ambiguous:
                    self.ambiguous = False
                    raise GatewayClientError("transport_lost")
                return {"status": "applied"}
            raise AssertionError(method)

    owner = Owner()
    agent._gateway = owner
    created = await agent.new_session(cwd=str(tmp_path))
    assert values(created)["fast"] == "auto"
    agent._conn = conn
    assert values(await agent.set_config_option("reasoning", "s", "high"))["reasoning"] == "high"
    await agent.prompt([TextContentBlock(type="text", text="/fast cold")], "s")
    assert info["service_tier"] == "cold"
    from acp.agent.router import build_agent_router
    wire = await build_agent_router(agent)("session/set_config_option",
        {"sessionId": "s", "configId": "fast", "value": "normal"}, False)
    assert {option["id"] for option in wire["configOptions"]} == {"model", "reasoning", "fast"}
    await agent.set_config_option("fast", "s", "cold")
    mutations = [p for m, p in calls if m == "session.mutate"]
    assert [(p["operation"], p["payload"]) for p in mutations[:2]] == [
        ("reasoning", {"effort": "high"}), ("service_tier", {"service_tier": "cold"})]
    before = len(conn.updates)
    owner.refuse = True
    with pytest.raises(GatewayClientError, match="session_busy"):
        await agent.set_config_option("fast", "s", "normal")
    assert len(conn.updates) == before
    assert info["service_tier"] == "cold"
    owner.refuse = False
    owner.ambiguous = True
    with pytest.raises(GatewayClientError, match="transport_lost"):
        await agent.set_config_option("fast", "s", "normal")
    assert len(conn.updates) == before
    await agent.set_config_option("fast", "s", "normal")
    retries = [p for m, p in calls if m == "session.mutate"][-2:]
    assert retries[0] == retries[1]
    await agent.set_config_option("fast", "s", "cold")
    # Another owner surface changes the route: refresh availability without erasing current intent.
    await agent._project({"session_id": "s", "type": "session.info", "payload": {
        "model": "local-model", "provider": "custom", "base_url": "http://localhost:1234/v1"}})
    last = conn.updates[-1][1]
    assert isinstance(last, ConfigOptionUpdate)
    fast = next(option for option in last.config_options if option.id == "fast")
    assert fast.current_value == "cold"
    assert {option.value for option in fast.options} == {"normal", "cold"}
    assert agent._snapshots["s"]["info"]["model"] == "local-model"
    info.update(model="local-model", provider="custom", base_url="http://localhost:1234/v1")
    from acp.exceptions import RequestError
    before = len(conn.updates)
    for key, value in [("opaque", "yes"), ("fast", "fast"), ("reasoning", "nonsense")]:
        with pytest.raises(RequestError):
            await agent.set_config_option(key, "s", value)
    assert len(conn.updates) == before
    await agent.aclose()
