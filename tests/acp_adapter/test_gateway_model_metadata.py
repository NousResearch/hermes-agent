"""ACP metadata controls reach the canonical owner's real durable session policy."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio

from acp_adapter.gateway_server import GatewayACPAgent
from hermes_cli.gateway_client import GatewayClientError, GatewayRPCError
from tests.acp_adapter.model_metadata_fixtures import (
    CHOICE_B, CHOICE_OFF, MODEL_A, PROVIDER,
)


@pytest_asyncio.fixture
async def owned_editor(declared_models, monkeypatch):
    from gateway import run
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from gateway.session_authority import SessionAuthority
    from gateway.session_controls import AuthorityConnection
    from gateway.session_local import create_local_session
    import hermes_state_runtime as runtime

    home, config, _write = declared_models
    monkeypatch.setattr(run, "_load_gateway_config", lambda *_args: config)
    store = SessionStore(home / "sessions", GatewayConfig())
    db = store._db
    runner = SimpleNamespace(
        session_store=store, _session_db=db, adapters={}, _draining=False,
        _evict_cached_agent=lambda _route: None, _cached_agent_for=lambda _route: None,
        _set_session_reasoning_override=lambda _route, _value: None, _handle_message=None,
        _resolve_session_agent_runtime=lambda **_kwargs: (MODEL_A, {
            "provider": "custom", "base_url": "https://one.example.invalid/v1", "api_key": "fixture-only"}))
    runner._adapter_for_source = lambda source: runner.adapters.get(source.platform)
    authority = SessionAuthority(runner, profile_id=str(home), instance_id="owner", db=db,
                                 epoch=runtime.begin_runtime_epoch(db, instance_id="owner"))
    connection = AuthorityConnection(authority, object(), {"user_id": "human"})
    ref = create_local_session(authority, connection.actor, {
        "request_id": "editor", "source": "acp", "cwd": str(home),
        "model": MODEL_A, "provider": PROVIDER, "toolsets": [], "reasoning": "low"})

    class Bridge:
        async def rpc(self, method, **params):
            response = await connection.dispatch({"id": method, "method": method, "params": params})
            if "error" in response:
                raise GatewayRPCError(response["error"]["data"]["reason"])
            return response["result"]

    agent = GatewayACPAgent()
    agent._gateway = Bridge()
    agent._conn = AsyncMock()
    await agent._adopt(ref.session_id, await agent._gateway.rpc("session.resume", session_id=ref.session_id))
    try:
        yield agent, authority, ref, db
    finally:
        await connection.close()
        store.close_all_db_handles()


@pytest.mark.asyncio
async def test_gateway_effort_pick_changes_the_durable_policy_without_changing_the_prompt(owned_editor):
    from gateway.session_policy import policy_for_source
    from hermes_state_local import local_receipt

    agent, authority, ref, db = owned_editor
    before = db.get_session(ref.session_id)["system_prompt"]
    reply = await agent.set_config_option("reasoning_effort", ref.session_id, "high")
    policy = policy_for_source(authority.runner, authority.sessions[ref.session_id].source)
    assert policy.reasoning_config == {"enabled": True, "effort": "high"}
    saved = json.loads(local_receipt(db, ref.session_id)["policy"]["config_json"])
    assert saved["agent"]["reasoning_effort"] == "high"
    assert db.get_session(ref.session_id)["system_prompt"] == before
    thought = next(option for option in reply.config_options if option.category == "thought_level")
    assert thought.current_value == "high"
    assert [choice.value for choice in thought.options] == ["low", "high"]


@pytest.mark.asyncio
async def test_gateway_model_pick_updates_the_effort_in_the_same_session_policy(owned_editor):
    from gateway.session_policy import policy_for_source

    agent, authority, ref, _db = owned_editor
    await agent.set_config_option("reasoning_effort", ref.session_id, "high")
    reply = await agent.set_config_option("model", ref.session_id, CHOICE_B)
    policy = policy_for_source(authority.runner, authority.sessions[ref.session_id].source)
    assert policy.reasoning_config == {"enabled": True, "effort": "medium"}
    by_category = {option.category: option for option in reply.config_options}
    assert by_category["model"].current_value == CHOICE_B
    assert by_category["thought_level"].current_value == "medium"
    reply = await agent.set_config_option("model", ref.session_id, CHOICE_OFF)
    policy = policy_for_source(authority.runner, authority.sessions[ref.session_id].source)
    assert policy.reasoning_config == {"enabled": False}
    assert not any(option.category == "thought_level" for option in reply.config_options)


@pytest.mark.asyncio
async def test_a_model_change_during_effort_validation_cannot_enable_reasoning_on_the_new_model(owned_editor):
    from gateway.session_policy import policy_for_source
    from tests.acp_adapter.model_metadata_fixtures import MODEL_OFF

    agent, authority, ref, _db = owned_editor
    original = agent._gateway.rpc
    raced = False

    async def rpc(method, **params):
        nonlocal raced
        if method == "session.resume" and not raced:
            raced = True
            snapshot = await original(method, **params)
            await original("session.mutate", session_id=ref.session_id, request_id="other-model",
                           expected_revision=snapshot["revision"],
                           expected_generation=snapshot["execution_generation"], operation="model",
                           payload={"provider": PROVIDER, "model": MODEL_OFF, "reasoning": "none"})
        return await original(method, **params)

    agent._gateway.rpc = rpc
    with pytest.raises(GatewayClientError, match="revision_conflict"):
        await agent.set_config_option("reasoning_effort", ref.session_id, "high")
    policy = policy_for_source(authority.runner, authority.sessions[ref.session_id].source)
    assert policy.reasoning_config == {"enabled": False}
