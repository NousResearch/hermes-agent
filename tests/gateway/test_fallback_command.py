"""Durable manual selection through native ingress, routing storage and constructors."""
import asyncio
import copy
import json
import logging
import sqlite3
import threading
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock, AsyncMock, patch

import pytest
import yaml

import gateway.run as gateway_run
from gateway.run import GatewayRunner, _gateway_config_home
from gateway.platforms.event import MessageEvent
from hermes_cli.commands import resolve_command
from tests.gateway.test_compression_exhaustion_reset_policy import (
    env, native_env, native_message, reload_entry,
)

KEY = "manual_fallback_index"


@pytest.fixture
def manual(native_env, monkeypatch):
    e = native_env
    e.chain = [dict(provider="custom", model="gpt-5.5", api_key="fixture-manual-key",
                    base_url="https://fixture.example/v1", api_mode="chat_completions",
                    service_tier_override="normal"),
               dict(provider="openrouter", model="next-model")]
    (e.home / "config.yaml").write_text(yaml.safe_dump({"fallback_providers": e.chain}))
    e.runner._refresh_fallback_model = Mock(side_effect=AssertionError("manual path refreshed automatic fallback"))
    return e


def test_registration():
    command = resolve_command("fallback")
    assert command is not None
    assert command.gateway_only and command.busy_policy == "reject"
    assert command.args_hint == "[on|off|status]"


@pytest.mark.asyncio
async def test_durable_on_off_is_index_only_and_preserves_model_pause(manual):
    e = manual
    e.store.set_model_override(e.key, {"model": "saved", "provider": "openai"})
    e.store.set_session_metadata(e.key, "compression_exhausted", True)
    before = e.entry.updated_at
    assert "Manual fallback: ON" in str(await native_message(e, "/fallback ON"))
    row = reload_entry(e)
    assert row.metadata == {"compression_exhausted": True, KEY: 0}
    assert row.updated_at == before and row.model_override["model"] == "saved"
    model, rt = e.runner._resolve_session_agent_runtime(source=e.source)
    assert model == "gpt-5.5" and rt[KEY] == 0
    assert rt["fallback_model"] == e.chain[1:]
    assert rt["fallback_service_tier_override"] == "normal"
    other = e.store.get_or_create_session(replace(e.source, chat_id="other"))
    assert KEY not in other.metadata
    (e.home / "config.yaml").write_text("broken: [")
    for _ in range(2):
        assert "Manual fallback: OFF" in str(await native_message(e, "/fallback off"))
    row = reload_entry(e)
    assert row.metadata == {"compression_exhausted": True}
    assert row.model_override["model"] == "saved"
    assert "fixture-manual-key" not in str(row.to_dict())


@pytest.mark.asyncio
@pytest.mark.parametrize("text", ["/fallback invalid", "/fallback on 1", "/fallback 0", "/fallback on global"])
async def test_grammar_is_mutation_free(manual, text):
    assert "Usage:" in str(await native_message(manual, text))
    assert KEY not in reload_entry(manual).metadata


@pytest.mark.asyncio
async def test_status_is_read_only_and_force_redacted(manual, monkeypatch):
    e = manual
    e.store.set_session_metadata(e.key, KEY, 0)
    agent = SimpleNamespace(_fallback_activated=True, model="https://u:password@host/?api_key=secret", provider="openrouter")
    monkeypatch.setattr(e.runner, "_cached_agent_for", lambda *a, **k: agent)
    reply = str(await native_message(e, "/fallback status"))
    assert "Manual fallback: ON" in reply and "Automatic fallback: active" in reply
    assert "password" not in reply and "api_key=secret" not in reply
    assert reload_entry(e).metadata == {KEY: 0}


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["on", "off", "new"])
async def test_failed_primary_retains_route_and_cache(manual, monkeypatch, action):
    e = manual
    if action != "on":
        e.store.set_session_metadata(e.key, KEY, 0)
    before = copy.deepcopy(e.entry.to_dict())
    evict = Mock()
    monkeypatch.setattr(e.runner, "_evict_cached_agent", evict)
    monkeypatch.setattr(e.db, "replace_gateway_routing_entries", Mock(side_effect=OSError("secret-failure")))
    reply = str(await native_message(e, "/new" if action == "new" else f"/fallback {action}"))
    assert "secret-failure" not in reply
    assert e.store.lookup_by_session_key(e.key).to_dict() == before
    evict.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["on", "off"])
async def test_cancelled_storage_retains_native_lease_through_publication(manual, monkeypatch, action):
    e = manual
    if action == "off":
        e.store.set_session_metadata(e.key, KEY, 0)
    entered, release = threading.Event(), threading.Event()
    original = e.db.replace_gateway_routing_entries
    def blocked(*a, **kw):
        entered.set()
        assert release.wait(10)
        return original(*a, **kw)
    monkeypatch.setattr(e.db, "replace_gateway_routing_entries", blocked)
    evict = Mock()
    monkeypatch.setattr(e.runner, "_evict_cached_agent", evict)
    task = asyncio.create_task(native_message(e, f"/fallback {action}"))
    lease = None
    try:
        assert await asyncio.to_thread(entered.wait, 3)
        state = e.runner._peek_session_state(e.key)
        owner, lease = state.turn.agent, state.turn.lease
        assert lease is not None and not lease.released
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        for incoming in ("message", "/new", "/stop", "/fallback off"):
            assert "retry" in str(await asyncio.wait_for(native_message(e, incoming), 1)).lower()
            assert state.turn.agent is owner
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
    assert task.cancelled() and lease.released
    evict.assert_called_once_with(e.key)
    assert (KEY in reload_entry(e).metadata) is (action == "on")


@pytest.fixture
def reset_resources(manual, monkeypatch, request):
    """Real /new, primary DB, full cleanup helper, executor and cache; recorded resource edges."""
    e = manual
    e.manual_selected = getattr(request, "param", True)
    if e.manual_selected:
        e.store.set_session_metadata(e.key, KEY, 0)
        e.store.set_session_metadata(e.key, "compression_exhausted", True)
    e.steps = []
    e.primary_entered, e.primary_release = threading.Event(), threading.Event()
    e.cleanup_entered, e.cleanup_release = threading.Event(), threading.Event()
    e.primary_release.set()
    e.cleanup_release.set()
    e.fail_primary = False
    resource = e.tmp / "old-tool-resource"
    resource.write_text("owned by the old conversation")
    e.resource = resource

    def close():
        e.steps.append("full-cleanup-start")
        e.cleanup_entered.set()
        assert e.cleanup_release.wait(10)
        resource.unlink()
        e.steps.append("full-cleanup")

    e.agent = SimpleNamespace(close=close, release_clients=Mock())
    e.runner._agent_cache = {e.key: (e.agent, "signature")}
    e.runner._agent_cache_lock = threading.Lock()
    # The shared fixture stubs this helper: restore it so omitting full cleanup fails here.
    monkeypatch.delattr(e.runner, "_cleanup_old_agent_for_reset")
    primary = e.db.replace_gateway_routing_entries
    def persist(*args, **kwargs):
        e.steps.append("primary-start")
        e.primary_entered.set()
        assert e.primary_release.wait(10)
        if e.fail_primary:
            e.steps.append("primary-failed")
            raise OSError("private-primary-failure")
        result = primary(*args, **kwargs)
        e.steps.append("primary")
        return result
    monkeypatch.setattr(e.db, "replace_gateway_routing_entries", persist)
    evict = e.runner._evict_cached_agent
    def evict_recorded(key):
        e.steps.append("evict")
        evict(key)
    monkeypatch.setattr(e.runner, "_evict_cached_agent", evict_recorded)
    # Run only the soft-release thread's leaf inline; full cleanup still uses the native executor.
    monkeypatch.setattr(e.runner, "_spawn_release_thread", lambda target, args, *a, **kw: target(*args))
    e.interrupt = Mock(side_effect=lambda **kw: e.steps.append("delegations"))
    monkeypatch.setattr("tools.async_delegation.interrupt_for_session", e.interrupt)
    monkeypatch.setattr("gateway.slash_commands_session._reset_process_scoped_tool_state",
                        lambda: e.steps.append("tool-state"))
    e.runner._session_state(e.key).conversation.last_resolved_model = "old-model"
    yield e
    e.primary_release.set()
    e.cleanup_release.set()


@pytest.mark.asyncio
@pytest.mark.parametrize("reset_resources", [False, True], indirect=True, ids=["native", "manual"])
async def test_public_new_full_teardown_order(reset_resources):
    e = reset_resources
    old_id = e.entry.session_id
    reply = str(await native_message(e, "/new"))
    teardown = ["full-cleanup-start", "full-cleanup"]
    expected = (["primary-start", "primary", *teardown, "delegations", "tool-state", "evict"]
                if e.manual_selected else
                [*teardown, "evict", "delegations", "tool-state", "primary-start", "primary"])
    assert e.steps == expected
    assert "could not be persisted" not in reply
    assert not e.resource.exists() and e.runner._cached_agent_for(e.key) is None
    e.agent.release_clients.assert_called_once_with()
    e.interrupt.assert_called_once_with(session_key=e.key, reason="session_reset", parent_session_id=old_id)
    row = reload_entry(e)
    assert row is not None
    assert row.session_id != old_id and KEY not in row.metadata and not row.compression_paused
    assert not e.runner._session_state(e.key).conversation.last_resolved_model
    assert not e.runner._is_session_running(e.key)


@pytest.mark.asyncio
async def test_public_new_failed_primary_preserves_all_resources(reset_resources):
    e = reset_resources
    before = copy.deepcopy(e.entry.to_dict())
    history = e.db.get_messages(e.entry.session_id)
    e.fail_primary = True
    reply = str(await native_message(e, "/new"))
    assert e.steps == ["primary-start", "primary-failed"]
    assert "could not be persisted" in reply and "private-primary-failure" not in reply
    assert e.store.lookup_by_session_key(e.key).to_dict() == before
    row = reload_entry(e)
    assert row is not None and row.to_dict() == before
    assert e.db.get_messages(e.entry.session_id) == history
    assert e.resource.exists() and e.runner._cached_agent_for(e.key) is e.agent
    assert e.runner._session_state(e.key).conversation.last_resolved_model == "old-model"
    e.agent.release_clients.assert_not_called()
    assert not e.runner._is_session_running(e.key)


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary,fail", [("primary", False), ("cleanup", False), ("primary", True)])
async def test_public_new_repeated_cancel_settles_full_teardown(reset_resources, boundary, fail):
    e = reset_resources
    before = copy.deepcopy(e.entry.to_dict())
    e.fail_primary = fail
    if boundary == "primary":
        e.primary_release.clear()
    e.cleanup_release.clear()
    entered = e.primary_entered if boundary == "primary" else e.cleanup_entered
    task = asyncio.create_task(native_message(e, "/new"))
    lease = None
    try:
        assert await asyncio.to_thread(entered.wait, 3), f"{boundary} was not reached"
        state = e.runner._peek_session_state(e.key)
        owner, lease = state.turn.agent, state.turn.lease
        assert owner is not None and lease is not None and not lease.released
        for _ in range(2):
            task.cancel()
            await asyncio.sleep(0)
        assert not task.done() and not lease.released
        if boundary == "primary" and not fail:
            e.primary_release.set()
            assert await asyncio.to_thread(e.cleanup_entered.wait, 3), "full cleanup was skipped"
            for _ in range(2):
                task.cancel()
                await asyncio.sleep(0)
        assert not task.done() and state.turn.agent is owner and state.turn.lease is lease
        assert e.resource.exists() and e.runner._cached_agent_for(e.key) is e.agent
        assert "evict" not in e.steps and "delegations" not in e.steps and "tool-state" not in e.steps
        for incoming in ("ordinary message", "/new", "/stop", "/fallback off"):
            assert "retry" in str(await asyncio.wait_for(native_message(e, incoming), 2)).lower()
            assert state.turn.agent is owner and state.turn.lease is lease and not lease.released
        e.runner._hmwa_prepare_turn.assert_not_awaited()
        e.runner._run_agent.assert_not_awaited()
    finally:
        e.primary_release.set()
        e.cleanup_release.set()
        await asyncio.gather(task, return_exceptions=True)
    assert task.cancelled() and lease.released and not e.runner._is_session_running(e.key)
    if fail:
        assert e.steps == ["primary-start", "primary-failed"]
        row = reload_entry(e)
        assert row is not None and row.to_dict() == before
        assert e.resource.exists() and e.runner._cached_agent_for(e.key) is e.agent
        e.agent.release_clients.assert_not_called()
    else:
        assert e.steps == ["primary-start", "primary", "full-cleanup-start", "full-cleanup",
                           "delegations", "tool-state", "evict"]
        row = reload_entry(e)
        assert row is not None
        assert row.session_id != before["session_id"] and KEY not in row.metadata
        assert not e.resource.exists() and e.runner._cached_agent_for(e.key) is None
        e.agent.release_clients.assert_called_once_with()


@pytest.mark.asyncio
async def test_validation_cancellation_never_starts_storage(manual, monkeypatch):
    e = manual
    entered, release = threading.Event(), threading.Event()
    resolver = getattr(gateway_run, "_resolve_fallback_entry_agent_kwargs", None)
    assert resolver is not None
    def blocked(*a, **kw):
        entered.set()
        assert release.wait(10)
        return resolver(*a, **kw)
    monkeypatch.setattr(gateway_run, "_resolve_fallback_entry_agent_kwargs", blocked)
    task = asyncio.create_task(native_message(e, "/fallback on"))
    try:
        assert await asyncio.to_thread(entered.wait, 3)
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        assert not task.done()
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
    assert task.cancelled() and KEY not in reload_entry(e).metadata
    assert not e.runner._is_session_running(e.key)


@pytest.mark.asyncio
async def test_acl_and_busy_race_reject_before_validation(manual, monkeypatch):
    e = manual
    e.runner._check_slash_access = Mock(return_value="denied")
    assert await native_message(e, "/fallback on") == "denied"
    assert KEY not in e.entry.metadata
    e.runner._check_slash_access = Mock(return_value=None)
    original = e.runner.async_session_store.lookup_by_session_key
    async def race(*a, **kw):
        value = await original(*a, **kw)
        e.runner._session_state(e.key).turn.agent = Mock()
        return value
    monkeypatch.setattr(e.runner.async_session_store, "lookup_by_session_key", race)
    assert "running" in str(await native_message(e, "/fallback on")).lower()
    assert KEY not in e.entry.metadata


@pytest.mark.parametrize("index", [None, True, -1, 2, "0"])
def test_invalid_marker_never_uses_cached_or_model_route(manual, index):
    e = manual
    e.store.set_session_metadata(e.key, KEY, index)
    e.runner._session_state(e.key).conversation.last_resolved_model = "cached"
    with pytest.raises(RuntimeError, match="/fallback off"):
        e.runner._resolve_session_agent_runtime(source=e.source)


@pytest.mark.parametrize("contents", ["broken: [", "[]", "fallback_providers: []"])
def test_config_uncertainty_never_uses_warm_substitute(manual, contents):
    e = manual
    e.store.set_session_metadata(e.key, KEY, 0)
    (e.home / "config.yaml").write_text(contents)
    with pytest.raises(RuntimeError, match="/fallback off"):
        e.runner._resolve_session_agent_runtime(source=e.source, user_config={"fallback_providers": e.chain})


@pytest.mark.parametrize("changes", [
    {"api_key": "${UNSET_MANUAL_KEY}"}, {"api_mode": "unknown"},
    {"api_mode": "bedrock_converse"}, {"api_mode": "codex_app_server"},
    {"base_url": "https://u:p@host/v1"}, {"base_url": "https://host/v1?key=secret"},
    {"base_url": "https://host:bad/v1"}, {"base_url": "${UNSET_ENDPOINT}"},
    {"api_key": "no-key-required"}, {"model": "${UNSET_MANUAL_MODEL}"},
])
def test_manual_resolver_rejects_invalid_route_without_secrets(manual, changes):
    entry = dict(manual.chain[0], **changes)
    resolver = getattr(gateway_run, "_resolve_fallback_entry_agent_kwargs", None)
    assert resolver is not None
    with pytest.raises((RuntimeError, ValueError)):
        resolver(entry, config={})


def test_ephemeral_projection_and_cache_identity(manual):
    e = manual
    e.store.set_session_metadata(e.key, KEY, 0)
    model, runtime = e.runner._resolve_session_agent_runtime(source=e.source)
    e.runner._service_tier = "priority"
    route = e.runner._resolve_turn_agent_config("hi", model, runtime)
    assert route["runtime"][KEY] == 0
    assert route["runtime"]["fallback_model"] == e.chain[1:]
    ctor = gateway_run._runtime_kwargs_for_agent_constructor(route["runtime"])
    assert KEY not in ctor and "has_header_auth" not in ctor
    assert ctor["fallback_service_tier_override"] == "normal"
    sig = lambda rt: e.runner._agent_config_signature(model, rt, [], "")
    assert sig(runtime) != sig({k:v for k,v in runtime.items() if k != KEY})
    assert sig(runtime) != sig(dict(runtime, fallback_service_tier_override=None))



@pytest.mark.asyncio
async def test_main_reuse_and_background_consume_one_manual_tail(manual, monkeypatch):
    from run_agent import AIAgent
    from gateway.run_turn_runner import TurnRunner
    e = manual
    e.store.set_session_metadata(e.key, KEY, 0)
    e.runner._service_tier = "priority"
    e.runner._prefill_messages = []
    e.runner._reasoning_config = None
    e.runner._provider_routing = {}
    model, rt = e.runner._resolve_session_agent_runtime(source=e.source)
    route = e.runner._resolve_turn_agent_config("hi", model, rt)
    agents = []
    class RecordingAgent(AIAgent):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            agents.append(self)
        def run_conversation(self, **kw):
            wire = self._build_api_kwargs([{"role": "user", "content": "hi"}])
            assert "service_tier" not in wire
            return {"final_response": "checked"}
    ctx = SimpleNamespace(source=e.source, session_key=e.key, session_id=e.entry.session_id,
                          AIAgent=RecordingAgent, user_config={}, enabled_toolsets=[], disabled_toolsets=None)
    turn = TurnRunner(e.runner, ctx)
    monkeypatch.setattr(turn, "_skip_context_files", lambda _: True)
    try:
        with patch("agent.process_bootstrap.OpenAI"), patch("model_tools.get_tool_definitions", return_value=[]), patch("model_tools.check_toolset_requirements", return_value={}):
            first = turn._build_fresh_agent(route, "telegram", "", 4, None, {}, True)
            assert first._fallback_chain == e.chain[1:]
            monkeypatch.setattr(turn, "_cached_sid_is_dead", lambda *a: (e.entry.session_id, False))
            monkeypatch.setattr(turn, "_current_message_count", lambda: 0)
            monkeypatch.setattr(turn, "_lookup_cached_agent", lambda *a: SimpleNamespace(agent=first, reused=True, evicted=None))
            model2, rt2 = e.runner._resolve_session_agent_runtime(source=e.source)
            rt2["fallback_model"] = []
            route2 = e.runner._resolve_turn_agent_config("hi", model2, rt2)
            reused, yes = turn._resolve_turn_agent(route2, "telegram", "", 4, None, {})
            assert yes and reused is first and first._fallback_chain == []
            adapter = SimpleNamespace(send=AsyncMock(), extract_media=lambda x: ([], x),
                                      extract_images=lambda x: ([], x))
            monkeypatch.setattr(e.runner, "_adapter_for_source", lambda source: adapter)
            monkeypatch.setattr(e.runner, "_resolve_turn_toolsets", lambda *a: ([], None))
            monkeypatch.setattr("run_agent.AIAgent", RecordingAgent)
            await e.runner._run_background_task("hi", e.source, "manual-background")
            assert len(agents) == 2
            assert agents[-1]._fallback_chain == e.chain[1:]
            assert "checked" in str(adapter.send.call_args)
        e.runner._refresh_fallback_model.assert_not_called()
    finally:
        for agent in agents:
            agent.shutdown_memory_provider()
            agent.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("stale_runtime", [False, True])
@pytest.mark.parametrize("fallback_active", [False, True])
async def test_manual_endpoint_diagnostics_omit_credentials(
    manual, monkeypatch, caplog, stale_runtime, fallback_active,
):
    from gateway.run_turn_runner import TurnRunner
    from gateway.turn_context import TurnContext
    from run_agent import AIAgent

    e = manual
    route_secret = "violet-orchard-door"
    old_secret = "previous-orchard-door"
    endpoint = f"https://proxy.example/{route_secret}/v1"
    monkeypatch.setenv("ROUTE_TOKEN", route_secret)
    e.chain[0]["base_url"] = "https://proxy.example/${ROUTE_TOKEN}/v1"
    config_path = e.home / "config.yaml"
    config_path.write_text(yaml.safe_dump({"fallback_providers": e.chain}))
    config_before = config_path.read_bytes()
    caplog.set_level(logging.DEBUG, logger="gateway.run")
    reply = str(await native_message(e, "/fallback on"))
    assert "Manual fallback: ON" in reply
    selection_before = copy.deepcopy(reload_entry(e).to_dict())
    assert selection_before["metadata"] == {KEY: 0}
    e.runner._provider_routing = {}
    e.runner._prefill_messages = []
    monkeypatch.setattr(e.runner, "_resolve_session_reasoning_config", lambda **kw: None)
    monkeypatch.setattr(e.runner, "_resolve_session_service_tier", lambda **kw: None)
    ctx = TurnContext(source=e.source, session_key=e.key, session_id=e.entry.session_id,
                      message="hi", user_config={}, enabled_toolsets=[], AIAgent=AIAgent)
    turn = TurnRunner(e.runner, ctx)
    projected = []
    agents = []

    def construct(route, platform_key, prompt, max_iterations, reasoning, routing):
        projected.append((route, copy.deepcopy(route)))
        agent = turn._build_fresh_agent(
            route, platform_key, prompt, max_iterations, reasoning, routing, True,
        )
        agents.append(agent)
        ctx.agent_holder[0] = agent
        return agent, False

    def complete(agent, *args):
        # Model I/O boundary: preserve the native constructor's resolved route.
        assert agent.base_url == endpoint and agent.api_key == e.chain[0]["api_key"]
        agent._fallback_activated = fallback_active
        config = {"retain": {"nested": True}}
        if stale_runtime:
            config["gateway_runtime"] = {
                "provider": agent.provider, "api_mode": agent.api_mode,
                "fallback_active": fallback_active,
                "base_url": f"https://old.example/{old_secret}/v1",
            }
        e.db.update_session_meta(agent.session_id, json.dumps(config), model=agent.model)
        return {"final_response": "checked", "messages": [], "completed": True}

    monkeypatch.setattr(turn, "_combined_ephemeral_prompt", lambda: "")
    monkeypatch.setattr(turn, "_setup_stream_consumer", lambda *a: (None, None, None, False))
    monkeypatch.setattr(turn, "_resolve_turn_agent", construct)
    monkeypatch.setattr(turn, "_wire_turn_agent_callbacks", lambda *a: None)
    monkeypatch.setattr(turn, "_load_turn_history", lambda *a: ([], None, []))
    monkeypatch.setattr(turn, "_prepare_turn_message", lambda *a: (None, None))
    monkeypatch.setattr(turn, "_run_conversation_with_approval", complete)
    monkeypatch.setattr(turn, "_append_auto_media_tags", lambda response, *a: response)
    try:
        with patch("agent.process_bootstrap.OpenAI"), patch("model_tools.get_tool_definitions", return_value=[]), patch("model_tools.check_toolset_requirements", return_value={}):
            result = turn.run_sync()
        assert result["final_response"] == "checked"
        agent = agents[0]
        with sqlite3.connect(e.tmp / "state.db") as db:
            model, raw_config = db.execute(
                "SELECT model, model_config FROM sessions WHERE id = ?", (agent.session_id,),
            ).fetchone()
        saved = json.loads(raw_config)
        assert saved == {"retain": {"nested": True}, "gateway_runtime": {
            "provider": agent.provider, "api_mode": agent.api_mode,
            "fallback_active": fallback_active,
        }}
        assert model == result["model"] == agent.model
        assert agent.base_url == endpoint and agent.api_key == e.chain[0]["api_key"]
        assert projected[0][0] == projected[0][1]
        assert projected[0][0]["runtime"]["base_url"] == endpoint
        assert projected[0][0]["runtime"][KEY] == 0
        assert reload_entry(e).to_dict() == selection_before
        assert config_path.read_bytes() == config_before
        diagnostic = str(await native_message(e, "/fallback status"))
        # Native agent endpoint logging is a separate inherited risk. Keep all captured
        # records intact and scope this diagnostic assertion to gateway-owned records.
        gateway_records = [record for record in caplog.records
                           if record.name == "gateway" or record.name.startswith("gateway.")]
        raw_logs = repr([(record.msg, record.args) for record in gateway_records])
        formatted_logs = "\n".join(caplog.handler.format(record) for record in gateway_records)
        for secret in (route_secret, old_secret, e.chain[0]["api_key"]):
            assert secret not in raw_config + repr(result) + reply + diagnostic + formatted_logs + raw_logs
        e.runner._refresh_fallback_model.assert_not_called()
    finally:
        for agent in agents:
            agent.shutdown_memory_provider()
            agent.close()


@pytest.mark.parametrize("fallback_active", [False, True])
def test_ordinary_endpoint_diagnostics_preserve_native_sync(manual, fallback_active):
    e = manual
    agent = SimpleNamespace(model="ordinary-model", provider="custom",
                            base_url="https://ordinary.example/v1", api_mode="chat_completions",
                            _fallback_activated=fallback_active)
    e.db.update_session_meta(e.entry.session_id, json.dumps({"retain": True}))
    before = vars(agent).copy()
    # The default helper call remains the ordinary, nonmanual contract.
    e.runner._sync_session_model_from_agent(e.entry.session_id, agent)
    with sqlite3.connect(e.tmp / "state.db") as db:
        model, raw_config = db.execute(
            "SELECT model, model_config FROM sessions WHERE id = ?", (e.entry.session_id,),
        ).fetchone()
    assert model == agent.model
    assert json.loads(raw_config) == {"retain": True, "gateway_runtime": {
        "provider": agent.provider, "base_url": agent.base_url, "api_mode": agent.api_mode,
        "fallback_active": fallback_active,
    }}
    assert vars(agent) == before and KEY not in reload_entry(e).metadata


def test_entry_metadata_reaches_request_and_max_tokens(manual):
    from run_agent import AIAgent
    entry = dict(manual.chain[0], max_output_tokens=2048, capabilities={"vision": False},
                 request_overrides={"service_tier": "priority", "extra_body": {"retain": True}})
    model, rt = gateway_run._resolve_fallback_entry_agent_kwargs(entry, config={})
    assert rt["max_tokens"] == 2048 and rt["capabilities"] == {"vision": False}
    manual.runner._service_tier = "priority"
    route = manual.runner._resolve_turn_agent_config("hi", model, rt)
    with patch("agent.process_bootstrap.OpenAI"), patch("model_tools.get_tool_definitions", return_value=[]), patch("model_tools.check_toolset_requirements", return_value={}):
        agent = AIAgent(model=model, **gateway_run._runtime_kwargs_for_agent_constructor(route["runtime"]),
                        request_overrides=route["request_overrides"], quiet_mode=True,
                        skip_context_files=True, skip_memory=True)
    try:
        wire = agent._build_api_kwargs([{"role": "user", "content": "hi"}])
        assert "service_tier" not in wire and wire["extra_body"]["retain"]
        assert agent.max_tokens == 2048
    finally:
        agent.shutdown_memory_provider()
        agent.close()


@pytest.mark.asyncio
async def test_source_scope_expands_env_and_scoped_miss_never_uses_ambient(manual, monkeypatch):
    from agent import secret_scope
    from hermes_constants import get_hermes_home
    from gateway.run import _profile_runtime_scope
    e = manual
    (e.home / ".env").write_text("MANUAL_ENDPOINT=https://scoped.example/v1\nMANUAL_KEY=scoped-key\n")
    config = {"fallback_providers": [dict(provider="custom", model="gpt-5.5",
                base_url="${MANUAL_ENDPOINT}", key_env="MANUAL_KEY")]}
    (e.home / "config.yaml").write_text(yaml.safe_dump(config))
    monkeypatch.setattr(gateway_run, "_gateway_config_home", get_hermes_home)
    monkeypatch.setenv("MANUAL_ENDPOINT", "https://ambient.invalid/v1")
    monkeypatch.setenv("MANUAL_KEY", "ambient-secret")
    monkeypatch.setattr(secret_scope, "_MULTIPLEX_ACTIVE", True)
    with _profile_runtime_scope(e.home):
        assert "Manual fallback: ON" in str(await native_message(e, "/fallback on"))
        _, rt = e.runner._resolve_session_agent_runtime(source=e.source)
        assert rt["base_url"] == "https://scoped.example/v1" and rt["api_key"] == "scoped-key"
    (e.home / ".env").write_text("MANUAL_ENDPOINT=https://scoped.example/v1\n")
    with _profile_runtime_scope(e.home):
        with pytest.raises(RuntimeError, match="/fallback off"):
            e.runner._resolve_session_agent_runtime(source=e.source)


def _fallback_scope_state():
    from agent.secret_scope import current_secret_scope
    from hermes_constants import get_hermes_home
    from tools.terminal_scope import get_terminal_scope
    return get_hermes_home(), current_secret_scope(), get_terminal_scope()


@pytest.fixture
def fallback_profiles(native_env, monkeypatch):
    """Real source resolution, effective YAML and native profile/secret scopes."""
    from pathlib import Path
    from agent import secret_scope
    e = native_env
    monkeypatch.setattr(Path, "home", lambda: e.tmp)
    monkeypatch.setenv("HERMES_HOME", str(e.home))
    # Undo the inherited fixture's fixed-home stand-in; use the production reader.
    monkeypatch.setattr(gateway_run, "_gateway_config_home", _gateway_config_home)
    monkeypatch.setattr(gateway_run, "_hermes_home", e.home)
    monkeypatch.setattr(secret_scope, "_MULTIPLEX_ACTIVE", True)
    e.runner.config.multiplex_profiles = True
    del e.runner._normalize_source_for_session_key
    e.runner._recover_telegram_topic_thread_id.return_value = "recovered-topic"
    e.cases, e.provider_calls = {}, []
    for name, home in (("default", e.home), ("secondary", e.home / "profiles" / "secondary")):
        home.mkdir(parents=True, exist_ok=True)
        (home / ".env").write_text(
            f"PROFILE_FALLBACK_URL=https://{name}.example/v1\nPROFILE_FALLBACK_KEY={name}-fixture-key\n")
        (home / "config.yaml").write_text(yaml.safe_dump({"fallback_providers": [dict(
            provider="custom", model=f"{name}-model", base_url="${PROFILE_FALLBACK_URL}",
            key_env="PROFILE_FALLBACK_KEY")]}))
        source = replace(e.source, profile=name)
        normalized = e.runner._normalize_source_for_session_key(source)
        assert e.runner._resolve_profile_home_for_source(normalized) == home
        entry = e.store.get_or_create_session(normalized)
        e.cases[name] = SimpleNamespace(runner=e.runner, store=e.store, db=e.db, tmp=e.tmp,
                                       source=source, home=home, entry=entry, key=entry.session_key)

    def provider(**kwargs):
        e.provider_calls.append((_fallback_scope_state(), kwargs, threading.get_ident()))
        return dict(provider=kwargs["requested"], api_mode="chat_completions",
                    base_url=kwargs["explicit_base_url"], api_key=kwargs["explicit_api_key"])

    # External provider boundary only: config, key_env lookup and validation remain native.
    e.provider = provider
    monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider", provider)
    return e


@pytest.mark.asyncio
@pytest.mark.parametrize("secondary_state", ["valid", "empty", "missing_key"])
async def test_activation_uses_source_profile_and_restores_caller(fallback_profiles, monkeypatch, secondary_state):
    import contextlib
    e = fallback_profiles
    default, secondary = e.cases["default"], e.cases["secondary"]
    valid = secondary_state == "valid"
    if valid:
        (default.home / "config.yaml").write_text("{}")
    elif secondary_state == "empty":
        (secondary.home / "config.yaml").write_text("{}")
    else:
        (secondary.home / ".env").write_text("PROFILE_FALLBACK_URL=https://secondary.example/v1\n")
    writes = Mock(wraps=e.db.replace_gateway_routing_entries)
    monkeypatch.setattr(e.db, "replace_gateway_routing_entries", writes)
    before = _fallback_scope_state()
    # Missing secondary credentials must not borrow a valid caller/default secret scope.
    async with (contextlib.nullcontext() if valid else gateway_run._async_profile_runtime_scope(default.home)):
        caller = _fallback_scope_state()
        reply = str(await native_message(secondary, "/fallback on"))
        assert _fallback_scope_state() == caller
        assert ("Manual fallback: ON" in reply) is valid
        assert (KEY in reload_entry(secondary).metadata) is valid
        assert bool(writes.call_count) is valid
        if valid:
            scope, kwargs, worker_thread = e.provider_calls[-1]
            assert scope[0] == secondary.home
            assert kwargs["target_model"] == "secondary-model"
            assert kwargs["explicit_base_url"] == "https://secondary.example/v1"
            assert kwargs["explicit_api_key"] == scope[1]["PROFILE_FALLBACK_KEY"] == "secondary-fixture-key"
            assert kwargs["config"]["fallback_providers"][0]["model"] == "secondary-model"
            assert worker_thread != threading.get_ident()
        else:
            assert e.provider_calls == []
        # The following default command must see only its own configuration.
        reply = str(await native_message(default, "/fallback on"))
        assert ("Manual fallback: ON" in reply) is not valid
        assert (KEY in reload_entry(default).metadata) is not valid
        assert _fallback_scope_state() == caller
    assert _fallback_scope_state() == before


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ["secrets", "validation"])
@pytest.mark.parametrize("fail", [False, True])
async def test_profile_validation_repeated_cancel_settles_before_release(fallback_profiles, monkeypatch, boundary, fail):
    from pathlib import Path
    e = fallback_profiles
    default, secondary = e.cases["default"], e.cases["secondary"]
    entered, release = threading.Event(), threading.Event()
    observed = []
    read_bytes = Path.read_bytes
    writes = Mock(wraps=e.db.replace_gateway_routing_entries)
    monkeypatch.setattr(e.db, "replace_gateway_routing_entries", writes)

    def park():
        observed.append((_fallback_scope_state(), threading.get_ident()))
        entered.set()
        assert release.wait(10)
        if fail:
            raise RuntimeError("fixture-profile-validation-failure")

    def read(path):
        if boundary == "secrets" and path == secondary.home / ".env" and not entered.is_set():
            park()
        return read_bytes(path)

    def provider(**kwargs):
        if boundary == "validation" and kwargs["target_model"] == "secondary-model":
            park()
        return e.provider(**kwargs)

    monkeypatch.setattr(Path, "read_bytes", read)
    monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider", provider)
    before = _fallback_scope_state()
    async with gateway_run._async_profile_runtime_scope(default.home):
        caller = _fallback_scope_state()
        task = asyncio.create_task(native_message(secondary, "/fallback on"))
        lease = None
        try:
            assert await asyncio.to_thread(entered.wait, 3)
            assert observed[0][0][0] == secondary.home
            assert observed[0][1] != threading.get_ident()
            if boundary == "validation":
                assert observed[0][0][1]["PROFILE_FALLBACK_KEY"] == "secondary-fixture-key"
            lease = e.runner._peek_session_state(secondary.key).turn.lease
            assert lease is not None and not lease.released
            for _ in range(3):
                task.cancel()
                await asyncio.sleep(0)
                assert not task.done() and not lease.released
                assert _fallback_scope_state() == caller
            writes.assert_not_called()
            assert "session recovery is still settling" in str(await native_message(secondary, "/fallback off")).lower()
            # Another profile remains independent while secondary validation is parked.
            assert "Manual fallback: ON" in str(await native_message(default, "/fallback on"))
            assert e.provider_calls[-1][0][0] == default.home
            assert e.provider_calls[-1][1]["explicit_api_key"] == "default-fixture-key"
            writes.reset_mock()
        finally:
            release.set()
            await asyncio.gather(task, return_exceptions=True)
        assert task.cancelled() and lease.released
        assert _fallback_scope_state() == caller
        writes.assert_not_called()
        assert KEY not in reload_entry(secondary).metadata
        assert not e.runner._is_session_running(secondary.key)
    assert _fallback_scope_state() == before


@pytest.mark.asyncio
@pytest.mark.parametrize("multiplex", [False, True])
async def test_profile_status_off_and_ordinary_activation_controls(fallback_profiles, monkeypatch, multiplex):
    import builtins
    from pathlib import Path
    from agent import secret_scope
    e = fallback_profiles
    default, secondary = e.cases["default"], e.cases["secondary"]
    if not multiplex:
        e.runner.config.multiplex_profiles = False
        monkeypatch.setattr(secret_scope, "_MULTIPLEX_ACTIVE", False)
        (default.home / "config.yaml").write_text(yaml.safe_dump({"fallback_providers": [dict(
            provider="custom", model="ordinary-model", base_url="https://ordinary.example/v1",
            api_key="ordinary-fixture-key")]}))
        caller = _fallback_scope_state()
        assert "Manual fallback: ON" in str(await native_message(default, "/fallback on"))
        assert e.provider_calls[-1][0] == caller
        assert _fallback_scope_state() == caller
    case = secondary if multiplex else default
    assert e.store.set_session_metadata(case.key, KEY, 0)
    for home in (default.home, secondary.home):
        (home / "config.yaml").write_text("broken: [")
    reads = []
    builtin_open, path_open = builtins.open, Path.open
    guarded = {home / name for home in (default.home, secondary.home) for name in (".env", "config.yaml")}

    def opened(path, *args, **kwargs):
        if not isinstance(path, int) and Path(path) in guarded:
            reads.append(Path(path))
            raise OSError("fixture-profile-files-unavailable")
        return builtin_open(path, *args, **kwargs)

    def path_opened(path, *args, **kwargs):
        if path in guarded:
            reads.append(path)
            raise OSError("fixture-profile-files-unavailable")
        return path_open(path, *args, **kwargs)

    caller = _fallback_scope_state()
    # Measure the handler itself, excluding dispatcher/reopen configuration reads.
    with monkeypatch.context() as io_guard:
        io_guard.setattr(builtins, "open", opened)
        io_guard.setattr(Path, "open", path_opened)
        for command in ("/fallback", "/fallback status", "/fallback off", "/fallback off"):
            reply = str(await e.runner._handle_fallback_command(MessageEvent(text=command, source=case.source)))
            assert "Manual fallback:" in reply and "could not" not in reply
        assert reads == [] and _fallback_scope_state() == caller
    assert KEY not in reload_entry(case).metadata


@pytest.mark.parametrize("pooled", [False, True])
def test_named_custom_header_and_pool_precedence(manual, monkeypatch, pooled):
    from hermes_cli import runtime_provider as rp
    from agent.credential_pool import CredentialPool, PooledCredential
    config = {"providers": {"fixture": {"base_url": "https://fixture.example/v1",
                "api_key": "config-key", "extra_headers": {"Authorization": "Bearer header-fixture"},
                "capabilities": {"vision": False}, "extra_body": {"retain": True}}}}
    pool = CredentialPool("custom:fixture", [])
    if pooled:
        # Selection itself is native; only credential discovery is replaced.
        pool._entries = [PooledCredential(id="fixture", provider="custom:fixture", auth_type="api_key",
                                           source="manual", label="fixture", access_token="pool-key", priority=0)]
    monkeypatch.setattr(rp, "load_pool", lambda name: pool)
    monkeypatch.setattr(rp, "custom_provider_pool_key_candidates", lambda *a: ["custom:fixture"])
    entry = dict(provider="custom:fixture", model="target", api_key="explicit-entry-key")
    model, runtime = gateway_run._resolve_fallback_entry_agent_kwargs(entry, config=config)
    assert model == "target" and runtime["requested_provider"] == "custom:fixture"
    assert runtime["api_key"] == ("pool-key" if pooled else "explicit-entry-key")
    assert runtime["has_header_auth"] is True and "extra_headers" not in runtime
    assert runtime["capabilities"] == {"vision": False}
    assert runtime["request_overrides"]["extra_body"]["retain"] is True


def test_callable_auth_is_not_materialized_or_serialized(manual, monkeypatch):
    from hermes_cli import runtime_provider as rp
    token = Mock(return_value="must-not-be-minted")
    # Auth backend boundary; the complete native resolver and projection still run.
    monkeypatch.setattr(rp, "load_pool", lambda name: SimpleNamespace(has_credentials=lambda: False))
    monkeypatch.setattr("hermes_cli.runtime_provider_backends._azure_entra_credentials", lambda config: token)
    model, runtime = gateway_run._resolve_fallback_entry_agent_kwargs(
        {"provider": "azure-foundry", "model": "gpt-4o"}, config={"model": {
            "provider": "azure-foundry", "base_url": "https://fixture.services.ai.azure.com",
            "auth_mode": "entra_id"}})
    assert runtime["api_key"] is token
    token.assert_not_called()


def test_unresolved_header_auth_is_rejected(manual):
    with pytest.raises((RuntimeError, ValueError)):
        gateway_run._resolve_fallback_entry_agent_kwargs(
            {"provider": "custom:header", "model": "target"}, config={"providers": {
                "header": {"base_url": "https://header.example/v1",
                           "extra_headers": {"Authorization": "${MISSING_HEADER}"}}}})


def test_header_only_and_native_local_noauth_remain_available(manual):
    model, runtime = gateway_run._resolve_fallback_entry_agent_kwargs(
        {"provider": "custom:header", "model": "target"}, config={"providers": {
            "header": {"base_url": "https://header.example/v1",
                       "extra_headers": {"Authorization": "Bearer fixture"}}}})
    assert runtime["has_header_auth"] is True
    assert "extra_headers" not in runtime
    _, local = gateway_run._resolve_fallback_entry_agent_kwargs(
        {"provider": "custom", "model": "target", "base_url": "http://127.0.0.1:1234/v1"}, config={})
    assert local["api_key"] == "no-key-required"


def _arm_pending_one_shot(e):
    """Native /model --once state, including the earliest standing override snapshot."""
    state = e.runner._session_state(e.key)
    state.conversation.model_override = {"model": "standing-model", "provider": "custom"}
    e.runner._claim_one_turn_restore(e.key)
    state.conversation.model_override = {"model": "temporary-model", "provider": "custom"}
    return copy.copy(state.conversation), state.persistent.run_generation


@pytest.mark.asyncio
async def test_public_manual_new_pending_one_shot_success(reset_resources):
    e = reset_resources
    _, generation = _arm_pending_one_shot(e)
    await test_public_new_full_teardown_order(e)
    state = e.runner._session_state(e.key)
    assert state.conversation.one_turn_restore is None
    assert state.conversation.model_override is None
    assert state.persistent.run_generation > generation
    assert not e.runner._is_session_run_current(e.key, generation)


@pytest.mark.asyncio
async def test_public_manual_new_pending_one_shot_failed_primary(reset_resources):
    e = reset_resources
    conversation, generation = _arm_pending_one_shot(e)
    await test_public_new_failed_primary_preserves_all_resources(e)
    state = e.runner._session_state(e.key)
    assert state.conversation == conversation
    assert state.persistent.run_generation > generation
    assert not e.runner._is_session_run_current(e.key, generation)


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary,fail", [("primary", False), ("cleanup", False), ("primary", True)])
async def test_public_manual_new_pending_one_shot_repeated_cancel(reset_resources, boundary, fail):
    e = reset_resources
    conversation, generation = _arm_pending_one_shot(e)
    await test_public_new_repeated_cancel_settles_full_teardown(e, boundary, fail)
    state = e.runner._session_state(e.key)
    if fail:
        assert state.conversation == conversation
    else:
        assert state.conversation.one_turn_restore is None
        assert state.conversation.model_override is None
    assert state.persistent.run_generation > generation
    assert not e.runner._is_session_run_current(e.key, generation)


@pytest.mark.parametrize("selected", [True, False], ids=["manual", "ordinary"])
@pytest.mark.parametrize("before,after", [(4096, 2048), (4096, None), (None, 2048), (2048, 2048)])
def test_limit_only_edit_rebuilds_manual_cache_constructor_and_request(manual, selected, before, after):
    from gateway.run_turn_runner import TurnRunner
    from gateway.turn_context import TurnContext
    from run_agent import AIAgent

    e = manual
    if selected:
        e.store.set_session_metadata(e.key, KEY, 0)
    else:
        e.runner._refresh_fallback_model = Mock(return_value=e.chain[1:])
    e.runner._agent_cache = {}
    e.runner._agent_cache_lock = threading.Lock()
    e.runner._prefill_messages = []
    e.runner._service_tier = None
    agents, constructor_limits = [], []

    class RecordingAgent(AIAgent):
        def __init__(self, *args, **kwargs):
            constructor_limits.append(kwargs.get("max_tokens"))
            super().__init__(*args, **kwargs)
            agents.append(self)

    ctx = TurnContext(source=e.source, session_key=e.key, session_id=e.entry.session_id,
                      AIAgent=RecordingAgent, enabled_toolsets=[], user_config={
                          "gateway": {"platforms": {"telegram": {"skip_context_files": True}}}})
    turn = TurnRunner(e.runner, ctx)

    def route_for(limit):
        e.chain[0].pop("max_output_tokens", None)
        if limit is not None:
            e.chain[0]["max_output_tokens"] = limit
        (e.home / "config.yaml").write_text(yaml.safe_dump({"fallback_providers": e.chain}))
        if selected:
            model, runtime = e.runner._resolve_session_agent_runtime(source=e.source)
        else:
            # Identical provider projection without a manual session selection: native identity control.
            model, runtime = gateway_run._resolve_fallback_entry_agent_kwargs(e.chain[0], config={})
        return e.runner._resolve_turn_agent_config("hi", model, runtime)

    def check_budget(agent, expected):
        assert agent.max_tokens == expected
        assert agent.context_compressor.max_tokens == expected
        wire = agent._build_api_kwargs([{"role": "user", "content": "hi"}])
        limits = {k: v for k, v in wire.items() if k in {"max_tokens", "max_completion_tokens"}}
        assert list(limits.values()) == ([] if expected is None else [expected])

    try:
        with patch("agent.process_bootstrap.OpenAI"), patch("model_tools.get_tool_definitions", return_value=[]), patch("model_tools.check_toolset_requirements", return_value={}):
            # Complete native startup discovery before measuring a limit-only cache transition.
            from hermes_cli.plugins import discover_plugins
            discover_plugins()
            first_route = route_for(before)
            first, reused = turn._resolve_turn_agent(first_route, "telegram", "", 4, None, {})
            assert not reused and constructor_limits == [before]
            check_budget(first, before)
            warm, reused = turn._resolve_turn_agent(first_route, "telegram", "", 4, None, {})
            assert reused and warm is first
            next_route = route_for(after)
            assert {k: v for k, v in first_route["runtime"].items() if k != "max_tokens"} == {
                k: v for k, v in next_route["runtime"].items() if k != "max_tokens"}
            current, reused = turn._resolve_turn_agent(next_route, "telegram", "", 4, None, {})
            rebuild = selected and before != after
            assert reused is (not rebuild)
            assert (current is first) is (not rebuild)
            assert constructor_limits == ([before, after] if rebuild else [before])
            check_budget(current, after if rebuild else before)
            warm, reused = turn._resolve_turn_agent(next_route, "telegram", "", 4, None, {})
            assert reused and warm is current
    finally:
        for agent in agents:
            agent.shutdown_memory_provider()
            agent.close()
