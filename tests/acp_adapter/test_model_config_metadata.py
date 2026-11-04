"""Declared model metadata drives native ACP choices without changing callable IDs."""

import copy
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from acp.exceptions import RequestError
from acp.schema import SessionConfigSelectGroup

from acp_adapter.gateway_server import GatewayACPAgent
from acp_adapter.config_options import config_options
from acp_adapter.gateway_session_meta import model_state
from acp_adapter.modes import DEFAULT_MODE
from acp_adapter.server import HermesACPAgent
from acp_adapter.session import SessionManager
from hermes_cli.gateway_client import GatewayClientError
from hermes_state import SessionDB


from tests.acp_adapter.model_metadata_fixtures import (
    CHOICE_A, CHOICE_B, CHOICE_OFF, MODEL_A, MODEL_B, MODEL_OFF, PROVIDER,
)


def _models(model=MODEL_A):
    return model_state({"model": model, "provider": PROVIDER})


def _options(model=MODEL_A, effort="low"):
    return config_options(_models(model), {"reasoning_effort": effort}, DEFAULT_MODE)


def test_native_groups_preserve_provider_identity_and_full_model_ids(declared_models):
    from hermes_cli.models import parse_model_input

    models = _models()
    option = next(option for option in _options() if option.category == "model")
    assert all(isinstance(group, SessionConfigSelectGroup) for group in option.options)
    grouped = {group.name: group for group in option.options}
    first = grouped["Relay one · upstream-a"]
    second = grouped["Relay two · upstream-a"]
    assert first.group != second.group
    assert [(choice.value, choice.name) for choice in first.options] == [(CHOICE_A, "Shared")]
    assert [(choice.value, choice.name) for choice in second.options] == [
        ("custom:relay-two:route-a/nested:variant/model", "Shared")]
    assert option.current_value == models.current_model_id == CHOICE_A
    assert parse_model_input(first.options[0].value, "") == (PROVIDER, MODEL_A)
    assert "First route" in next(model.description for model in models.available_models if model.model_id == CHOICE_A)


def test_explicit_efforts_are_per_model_and_empty_differs_from_missing(declared_models):
    first = next(option for option in _options(effort="high") if option.category == "thought_level")
    assert first.current_value == "high"
    assert [choice.value for choice in first.options] == ["low", "high"]
    second = next(option for option in _options(MODEL_B, "medium") if option.category == "thought_level")
    assert [choice.value for choice in second.options] == ["medium"]
    assert not any(option.category == "thought_level" for option in _options(MODEL_OFF))
    legacy = next(option for option in _options("legacy") if option.category == "thought_level")
    assert {"none", "low", "medium", "high"} <= {choice.value for choice in legacy.options}


@pytest.mark.parametrize("current,levels,expected", [
    ({"enabled": True, "effort": "high"}, ["none"], {"enabled": False}),
    ({"enabled": True, "effort": "vendor-tier"}, ["low", "high"], {"enabled": True, "effort": "low"}),
])
def test_an_explicit_effort_set_never_keeps_an_unlisted_runtime_value(current, levels, expected):
    from acp_adapter.config_options import constrain_reasoning
    assert constrain_reasoning(current, {"reasoning_efforts": levels}) == expected


def test_provider_and_group_colons_cannot_merge_different_routes(declared_models):
    _home, config, write = declared_models
    config["providers"]["relay"] = {
        "name": "Relay collision", "api": "https://collision.example.invalid/v1", "discover_models": False,
        "models": {MODEL_A: {"name": "Shared", "group": "one:upstream-a", "reasoning_efforts": ["low"]}}}
    write()
    option = next(option for option in _options() if option.category == "model")
    grouped = {group.name: group for group in option.options}
    first = grouped["Relay one · upstream-a"]
    collision = grouped["Relay collision · one:upstream-a"]
    assert first.group != collision.group
    assert [choice.value for choice in collision.options] == ["custom:relay:route-a/nested:variant/model"]


def test_named_groups_keep_other_inventory_providers_separate(declared_models, monkeypatch):
    monkeypatch.setattr("hermes_cli.inventory.build_models_payload", lambda *_args, **_kwargs: {"providers": [
        {"slug": "openrouter", "name": "First canonical", "models": ["native/model"]},
        {"slug": "anthropic", "name": "Second canonical", "models": ["native/model"]},
    ]})
    option = next(option for option in _options() if option.category == "model")
    grouped = {group.name: group for group in option.options}
    assert [choice.value for choice in grouped["First canonical"].options] == ["openrouter:native/model"]
    assert [choice.value for choice in grouped["Second canonical"].options] == ["anthropic:native/model"]


@pytest.mark.asyncio
@pytest.mark.parametrize("model,value", [(MODEL_A, "medium"), (MODEL_OFF, "high"), (MODEL_OFF, "default")])
async def test_gateway_rejects_an_effort_the_current_model_does_not_advertise(declared_models, model, value):
    info = {"model": model, "provider": PROVIDER, "reasoning_effort": "low"}

    class RPC:
        async def rpc(self, method, **params):
            if method == "session.info":
                return dict(info)
            if method == "session.resume":
                return {"revision": 1, "execution_generation": 1, "info": dict(info)}
            return {"status": "applied"}

    agent = GatewayACPAgent()
    agent._gateway = RPC()
    agent._snapshots["s"] = {"info": dict(info)}
    with pytest.raises(GatewayClientError, match="invalid_params"):
        await agent.set_config_option("reasoning_effort", "s", value)


def _agent(model=MODEL_A, provider=PROVIDER, session_id="fixture", **_kwargs):
    return SimpleNamespace(model=model, provider=provider, session_id=session_id,
                           requested_provider=_kwargs.get("requested_provider") or provider,
                           base_url=_kwargs.get("base_url") or "https://one.example.invalid/v1", api_key="fixture-only",
                           enabled_toolsets=[], disabled_toolsets=[], _pending_toolsets=None,
                           reasoning_config={"enabled": True, "effort": "low"},
                           _cached_system_prompt="frozen-system", context_compressor=None)


def _inprocess(home, monkeypatch):
    db = SessionDB(db_path=home / "state.db")
    manager = SessionManager(agent_factory=_agent, db=db)
    monkeypatch.setattr(manager, "_make_agent", lambda **kwargs: _agent(
        **{key: value for key, value in kwargs.items() if key != "requested_provider"},
        provider=kwargs.get("requested_provider") or PROVIDER))
    return HermesACPAgent(session_manager=manager), manager, db


@pytest.mark.asyncio
async def test_inprocess_choice_applies_to_runtime_and_survives_restore_and_fork(declared_models, monkeypatch):
    from agent.transports.chat_completions import ChatCompletionsTransport

    home, _config, _write = declared_models
    agent, manager, db = _inprocess(home, monkeypatch)
    try:
        created = await agent.new_session(cwd=str(home))
        state = manager.get_session(created.session_id)
        original = state.agent
        state.history[:] = [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "answer"}]
        manager.save_session(state.session_id)
        original._session_db, original._session_db_created = db, True
        history = copy.deepcopy(state.history)
        agent._conn = AsyncMock()
        response = await agent.set_config_option("reasoning_effort", state.session_id, "high")
        assert state.agent is original
        assert original.reasoning_config == {"enabled": True, "effort": "high"}
        assert original._cached_system_prompt == "frozen-system"
        assert state.history == history
        thought = next(option for option in response.config_options if option.category == "thought_level")
        assert thought.current_value == "high"
        request = ChatCompletionsTransport().build_kwargs(
            MODEL_A, state.history, supports_reasoning=True, reasoning_config=original.reasoning_config)
        assert request["extra_body"]["reasoning"] == {"enabled": True, "effort": "high"}
        assert any(call.kwargs["update"].session_update == "config_option_update"
                   for call in agent._conn.session_update.await_args_list)

        restored_manager = SessionManager(agent_factory=_agent, db=db)
        restored = restored_manager.get_session(state.session_id)
        assert restored.agent.reasoning_config == {"enabled": True, "effort": "high"}
        forked = restored_manager.fork_session(restored.session_id, cwd=str(home))
        assert forked.agent.reasoning_config == {"enabled": True, "effort": "high"}
        assert [(message["role"], message["content"]) for message in forked.history] == [
            ("user", "hello"), ("assistant", "answer")]
    finally:
        db.close()


@pytest.mark.asyncio
async def test_resolved_custom_runtime_keeps_its_configured_picker_identity(declared_models):
    home, _config, _write = declared_models
    db = SessionDB(db_path=home / "state.db")
    manager = SessionManager(agent_factory=lambda: _agent(provider="custom", requested_provider=PROVIDER), db=db)
    agent = HermesACPAgent(session_manager=manager)
    try:
        created = await agent.new_session(cwd=str(home))
        assert created.models.current_model_id == CHOICE_A
        options = {option.category: option for option in created.config_options}
        assert options["model"].current_value == CHOICE_A
        assert [choice.value for choice in options["thought_level"].options] == ["low", "high"]
    finally:
        db.close()


@pytest.mark.asyncio
async def test_fork_keeps_the_selected_endpoint_as_well_as_its_reasoning_pin(declared_models, monkeypatch):
    from hermes_cli.model_switch import ModelSwitchResult

    home, _config, _write = declared_models
    monkeypatch.setattr("hermes_cli.model_switch.switch_model", lambda **kwargs: ModelSwitchResult(
        success=True, new_model=kwargs["raw_input"], target_provider=kwargs["explicit_provider"]))
    agent, manager, db = _inprocess(home, monkeypatch)
    try:
        created = await agent.new_session(cwd=str(home))
        state = manager.get_session(created.session_id)
        await agent.set_config_option("reasoning_effort", state.session_id, "high")
        await agent.set_config_option("model", state.session_id, "custom:relay-two:route-a/nested:variant/model")
        forked = manager.fork_session(state.session_id, cwd=str(home))
        assert forked.agent.provider == "custom:relay-two"
        assert forked.agent.reasoning_config == {"enabled": True, "effort": "low"}
    finally:
        db.close()


@pytest.mark.asyncio
async def test_inprocess_model_choice_reconciles_the_pin_and_returns_all_options(declared_models, monkeypatch):
    from hermes_cli.model_switch import ModelSwitchResult

    home, _config, _write = declared_models
    monkeypatch.setattr("hermes_cli.model_switch.switch_model", lambda **kwargs: ModelSwitchResult(
        success=True, new_model=kwargs["raw_input"], target_provider=kwargs["explicit_provider"]))
    agent, manager, db = _inprocess(home, monkeypatch)
    try:
        created = await agent.new_session(cwd=str(home))
        state = manager.get_session(created.session_id)
        await agent.set_config_option("reasoning_effort", state.session_id, "high")
        reply = await agent.set_config_option("model", state.session_id, CHOICE_B)
        assert state.model == state.agent.model == MODEL_B
        assert state.agent.reasoning_config == {"enabled": True, "effort": "medium"}
        by_category = {option.category: option for option in reply.config_options}
        assert by_category["model"].current_value == CHOICE_B
        assert by_category["thought_level"].current_value == "medium"
        assert [choice.value for choice in by_category["thought_level"].options] == ["medium"]
        with pytest.raises(RequestError) as rejected:
            await agent.set_config_option("reasoning_effort", state.session_id, "high")
        assert rejected.value.code == -32602
        reply = await agent.set_config_option("model", state.session_id, CHOICE_OFF)
        assert state.model == MODEL_OFF
        assert state.agent.reasoning_config == {"enabled": False}
        assert not any(option.category == "thought_level" for option in reply.config_options)
    finally:
        db.close()
