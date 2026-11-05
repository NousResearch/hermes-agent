"""SDK-v1 session selectors drive the same controls as text commands."""
from types import SimpleNamespace

import pytest
from acp.schema import ConfigOptionUpdate, TextContentBlock

from acp_adapter.server import HermesACPAgent
from acp_adapter.session import SessionManager
from tests.acp_adapter.test_acp_commands import CaptureConn


def values(response):
    options = response.config_options
    assert {o.id for o in options} == {"model", "reasoning_effort", "service_tier"}
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
        assert values(created)["service_tier"] == "normal"
        wire = created.model_dump(by_alias=True, exclude_none=True)
        assert all("id" in o and "configId" not in o for o in wire["configOptions"])
        sid = created.session_id
        state = manager.get_session(sid)
        state.history = [{"role": "user", "content": "keep transcript"}]
        state.agent._cached_system_prompt = "frozen prompt"
        state.agent.enabled_toolsets = ["file"]
        state.agent.request_overrides = {"extra_body": {"user": "keep"}}
        await agent.set_config_option("service_tier", sid, "fast")
        assert state.agent.request_overrides == {"extra_body": {"user": "keep"}, "service_tier": "priority"}
        await agent.prompt([TextContentBlock(type="text", text="/fast normal")], sid)
        assert state.agent.request_overrides == {"extra_body": {"user": "keep"}}
        assert values(await agent.set_config_option("reasoning_effort", sid, "high"))["reasoning_effort"] == "high"
        await agent.prompt([TextContentBlock(type="text", text="/reasoning low")], sid)
        assert state.agent.reasoning_config == {"enabled": True, "effort": "low"}
        await agent.set_config_option("reasoning_effort", sid, "high")
        await agent.prompt([TextContentBlock(type="text", text="/fast cold")], sid)
        assert state.agent.service_tier == "cold"
        assert any(isinstance(update, ConfigOptionUpdate) for _, update in conn.updates)
        forked = await agent.fork_session(cwd=str(tmp_path), session_id=sid)
        assert values(forked)["service_tier"] == "cold"
        await agent.set_config_option("service_tier", forked.session_id, "normal")
        cold = HermesACPAgent(session_manager=SessionManager(agent_factory=runtime, db=db))
        resumed = await cold.resume_session(cwd=str(tmp_path), session_id=sid)
        assert values(resumed)["service_tier"] == "cold"
        assert values(resumed)["reasoning_effort"] == "high"
        child = await cold.load_session(cwd=str(tmp_path), session_id=forked.session_id)
        assert values(child)["service_tier"] == "normal"
        assert manager.get_session(forked.session_id).agent.service_tier is None
        assert [(m["role"], m["content"]) for m in state.history] == [("user", "keep transcript")]
        assert state.agent._cached_system_prompt == "frozen prompt"
        assert state.agent.enabled_toolsets == ["file"]
        before = len(conn.updates)
        state.is_running = True
        with pytest.raises(Exception, match="busy"):
            await agent.set_config_option("reasoning_effort", sid, "low")
        state.is_running = False
        from acp.exceptions import RequestError
        for key, value in [("opaque", "yes"), ("service_tier", "banana"), ("reasoning_effort", "nonsense")]:
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
        fast = next(o for o in conn.updates[-1][1].config_options if o.id == "service_tier")
        assert {o.value for o in fast.options} == {"normal", "cold"}
        # Embedded reset restores configured defaults without changing the chosen model.
        monkeypatch.setattr("hermes_cli.config.load_config", lambda: {
            "agent": {"reasoning_effort": "low", "service_tier": "normal"}})
        await agent.prompt([TextContentBlock(type="text", text="/reset")], sid)
        assert state.agent.reasoning_config == {"enabled": True, "effort": "low"}
        assert state.agent.service_tier is None and state.control_values == {'model': 'custom:local-model'}
        assert state.model == "local-model" and state.history == []


@pytest.mark.asyncio
async def test_embedded_controls_reach_real_loopback_request_on_cold_restore(tmp_path, monkeypatch, model_peer):
    """Keep official route semantics, redirect only its HTTP transport to a loopback peer."""
    import httpx
    import json
    from acp import text_block
    from hermes_constants import get_hermes_home
    from hermes_state import SessionDB
    config = {'model': {'default': 'gpt-6.1-sol', 'provider': 'openai-api'},
              'platform_toolsets': {'acp': []}, 'agent': {'service_tier': 'normal'},
              'auxiliary': {'title_generation': {'enabled': False}}}
    (get_hermes_home() / 'config.yaml').write_text(json.dumps(config))
    monkeypatch.setattr('hermes_cli.runtime_provider.resolve_runtime_provider', lambda **kwargs: {
        'provider': 'openai-api', 'api_mode': 'chat_completions', 'api_key': 'loopback-only',
        'base_url': 'https://api.openai.com/v1'})
    send = httpx.Client.send

    def loopback(client, request, **kwargs):
        assert request.url.host == 'api.openai.com', request.url
        request.url = request.url.copy_with(scheme='http', host='127.0.0.1', port=model_peer.server_port)
        return send(client, request, **kwargs)

    monkeypatch.setattr(httpx.Client, 'send', loopback)
    with SessionDB(tmp_path / 'state.db') as db:
        manager = SessionManager(db=db)
        agent = HermesACPAgent(session_manager=manager)
        sid = (await agent.new_session(cwd=str(tmp_path))).session_id
        await agent.set_config_option('reasoning_effort', sid, 'high')
        await agent.set_config_option('service_tier', sid, 'fast')
        cold = HermesACPAgent(session_manager=SessionManager(db=db))
        conn = CaptureConn()
        cold.on_connect(conn)
        await cold.resume_session(cwd=str(tmp_path), session_id=sid)
        response = await cold.prompt([text_block('WS_SHARED embedded control request')], sid)
        assert response.stop_reason == 'end_turn'
        request = model_peer.requests[-1]
        # This fixture deliberately selects Chat Completions; official OpenAI's
        # reasoning transport is Responses. The wire guarantee exercised here is
        # the retained paid tier, while the gateway loopback case checks effort.
        assert cold.session_manager.get_session(sid).agent.reasoning_config['effort'] == 'high'
        assert request['service_tier'] == 'priority'
        assert any('LOCAL_ACK_WS_SHARED' in getattr(getattr(update, 'content', None), 'text', '')
                   for _, update in conn.updates)
