"""A failed turn-start restore keeps the live fallback usable until a successful retry."""

import json

import httpx
from openai import OpenAI
import pytest

from agent import agent_runtime_helpers as runtime
from agent import chat_completion_helpers as completion
from agent.credential_pool import CredentialPool, PooledCredential
from run_agent import AIAgent


@pytest.fixture
def fallback_runtime(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(
        "model:\n  context_length: 200000\nagent:\n  reasoning_effort: medium\n", encoding="utf-8",
    )
    requests, clients, all_clients = [], [], []

    def respond(request):
        body = json.loads(request.content)
        requests.append((request.url.host, body))
        return httpx.Response(200, json={
            "id": "reply", "object": "chat.completion", "created": 1, "model": body["model"],
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "usable"},
                         "finish_reason": "stop"}],
        })

    def sdk_client(*, _runtime=True, **kwargs):
        unused_http_client = kwargs.pop("http_client", None)
        if unused_http_client is not None:
            unused_http_client.close()
        client = OpenAI(**kwargs, http_client=httpx.Client(transport=httpx.MockTransport(respond)))
        all_clients.append(client)
        if _runtime:
            clients.append(client)
        return client

    monkeypatch.setattr("agent.process_bootstrap.OpenAI", sdk_client)
    monkeypatch.setattr("agent.auxiliary_client.OpenAI", lambda **kwargs: sdk_client(_runtime=False, **kwargs))
    monkeypatch.setattr("agent.context_compressor.get_model_context_length", lambda *_args, **_kwargs: 200000)
    monkeypatch.setattr("agent.model_metadata.get_model_context_length", lambda *_args, **_kwargs: 200000)
    agent = AIAgent(
        model="primary-model", provider="custom", api_key="test-primary", base_url="http://primary.example.test/v1",
        enabled_toolsets=[], quiet_mode=True, skip_context_files=True, skip_memory=True, save_trajectories=False,
        request_overrides={"extra_body": {"policy": "primary"}},
        fallback_model={"provider": "custom", "model": "fallback-model", "api_key": "test-fallback",
                        "base_url": "http://fallback.example.test/v1"},
    )
    assert agent._try_activate_fallback()
    agent.request_overrides = {"extra_body": {"policy": "fallback"}}
    agent.reasoning_config = {"effort": "medium"}
    agent._primary_runtime["reasoning_config"] = {"effort": "low"}
    agent._cached_system_prompt = "Model: fallback-model\nProvider: custom"
    agent._transport_cache = {"fallback-wire": object()}
    agent._consecutive_stale_streams = 3
    agent._restore_wait_logged = True
    agent._compression_feasibility_checked = True
    agent._last_feasibility_notice = agent._compression_warning = "fallback budget"
    agent.context_compressor.last_prompt_tokens = 5000
    agent.context_compressor._aux_context_ceiling = 100000
    notices, retired = [], []
    agent._emit_diagnostic_status = notices.append
    retire = agent._retire_shared_openai_client

    def retire_recorded(client, **kwargs):
        retired.append(client)
        retire(client, **kwargs)

    monkeypatch.setattr(agent, "_retire_shared_openai_client", retire_recorded)
    try:
        yield agent, requests, clients, notices, retired
    finally:
        for client in all_clients:
            client.close()


def _fail_once(agent, monkeypatch, stage):
    armed = [True]

    def fail():
        if armed[0]:
            armed[0] = False
            raise RuntimeError("injected restore failure")

    if stage == "client":
        create = agent._create_openai_client

        def create_primary(*args, **kwargs):
            fail()
            return create(*args, **kwargs)

        monkeypatch.setattr(agent, "_create_openai_client", create_primary)
    elif stage == "bedrock":
        agent._primary_runtime.update(provider="bedrock", base_url="bedrock://us-west-2", api_mode="anthropic_messages")
        monkeypatch.setattr("agent.anthropic_adapter.build_anthropic_bedrock_client", lambda *_args: fail())
    elif stage == "engine":
        update = agent.context_compressor.update_model

        def update_primary(*args, **kwargs):
            update(*args, **kwargs)
            agent.context_compressor.restore_only_attribute = "temporary"
            fail()

        monkeypatch.setattr(agent.context_compressor, "update_model", update_primary)
    elif stage == "pool":
        pool = CredentialPool(provider="custom", entries=[PooledCredential.from_dict("custom", {
            "id": "primary-entry", "auth_type": "api_key", "access_token": "test-primary",
            "base_url": "http://primary.example.test/v1",
        })])
        agent._credential_pool = pool
        select = pool.select

        def select_primary(**kwargs):
            fail()
            return select(**kwargs)

        monkeypatch.setattr(pool, "select", select_primary)
    else:
        rewrite = completion.rewrite_prompt_model_identity

        def rewrite_primary(*args, **kwargs):
            rewrite(*args, **kwargs)
            fail()

        monkeypatch.setattr(completion, "rewrite_prompt_model_identity", rewrite_primary)


def _route_state(agent):
    names = ("model", "provider", "requested_provider", "base_url", "api_mode", "api_key", "client",
             "request_overrides", "runtime_capabilities", "_client_kwargs", "reasoning_config",
             "_use_prompt_caching", "_use_native_cache_layout", "_reasoning_echo_flag", "_credential_pool",
             "_credential_pool_entry_id", "_cached_system_prompt", "_fallback_activated", "_fallback_index",
             "_rate_limit_backoff_count", "_provider_fallback_active", "_provider_fallback_route",
             "_consecutive_stale_streams", "_compression_feasibility_checked", "_last_feasibility_notice",
             "_compression_warning", "_restore_wait_logged", "_bedrock_region", "_bedrock_guardrail_config")
    return {name: (hasattr(agent, name), getattr(agent, name, None)) for name in names}


@pytest.mark.parametrize("stage", ["client", "bedrock", "engine", "pool", "prompt"])
def test_failed_primary_restore_keeps_the_fallback_route_and_real_sdk_request_usable(
    fallback_runtime, monkeypatch, stage,
):
    agent, requests, clients, notices, retired = fallback_runtime
    _fail_once(agent, monkeypatch, stage)
    before = _route_state(agent)
    engine = agent.context_compressor
    engine_state = dict(vars(engine))
    cache, entries = agent._transport_cache, dict(agent._transport_cache)
    old_client, first_new_client = agent.client, len(clients)

    assert not runtime.restore_primary_runtime(agent)

    assert _route_state(agent) == before
    assert agent.context_compressor is engine and vars(engine) == engine_state
    assert agent._transport_cache is cache and cache == entries
    assert old_client not in retired and not old_client.is_closed()
    assert all(client in retired for client in clients[first_new_client:])
    assert not any("Primary model restored" in notice for notice in notices)
    assert agent.client.chat.completions.create(
        model=agent.model, messages=[{"role": "user", "content": "probe"}], **agent.request_overrides,
    ).choices[0].message.content == "usable"
    assert requests[-1][0] == "fallback.example.test"
    assert requests[-1][1]["model"] == before["model"][1]
    assert requests[-1][1]["policy"] == "fallback"


@pytest.mark.parametrize("stage", ["engine", "prompt"])
def test_failed_restore_can_retry_and_commit_the_primary_with_one_success_notice(
    fallback_runtime, monkeypatch, stage,
):
    agent, requests, clients, notices, retired = fallback_runtime
    _fail_once(agent, monkeypatch, stage)
    fallback_client = agent.client

    assert not runtime.restore_primary_runtime(agent)
    assert agent.client is fallback_client and not fallback_client.is_closed()
    assert not any("Primary model restored" in notice for notice in notices)
    assert runtime.restore_primary_runtime(agent)

    assert (agent.model, agent.context_compressor.model) == ("primary-model", "primary-model")
    assert agent.reasoning_config == agent._primary_runtime["reasoning_config"]
    assert not agent._fallback_activated and agent._fallback_index == 0
    assert sum("Primary model restored" in notice for notice in notices) == 1
    assert agent.client not in retired and not agent.client.is_closed()
    assert agent.client.chat.completions.create(
        model=agent.model, messages=[{"role": "user", "content": "probe"}], **agent.request_overrides,
    ).choices[0].message.content == "usable"
    assert requests[-1][0] == "primary.example.test" and requests[-1][1]["policy"] == "primary"



@pytest.mark.parametrize("stage", ["engine", "prompt"])
def test_failed_restore_preserves_durable_guards_until_successful_retry(
    fallback_runtime, monkeypatch, tmp_path, stage,
):
    import sqlite3

    from agent.context_compressor import PROACTIVE_PRUNE_REARM_MODEL_CONFIG_KEY
    from hermes_state import SessionDB

    agent, requests, clients, notices, retired = fallback_runtime
    state_path = tmp_path / "restore-state.db"
    session_id = "restore-durable-guards"
    db = SessionDB(db_path=state_path)
    db.create_session(session_id, source="test")
    db.record_compression_failure_cooldown(session_id, 9999999999.0, "fallback overloaded")
    db.set_compression_fallback_streak(session_id, 3)
    db.set_compression_overload_streak(session_id, 4)
    db.set_compression_ineffective_count(session_id, 2)
    db.patch_session_model_config(session_id, {
        PROACTIVE_PRUNE_REARM_MODEL_CONFIG_KEY: 12000, "keep_other_setting": "unchanged",
    })
    agent.context_compressor.bind_session_state(db, session_id)

    def persisted():
        with sqlite3.connect(state_path) as reader:
            reader.row_factory = sqlite3.Row
            return dict(reader.execute(
                "SELECT compression_failure_cooldown_until, compression_failure_error, "
                "compression_fallback_streak, compression_overload_streak, "
                "compression_ineffective_count, model_config FROM sessions WHERE id = ?",
                (session_id,),
            ).fetchone())

    try:
        before = persisted()
        _fail_once(agent, monkeypatch, stage)
        assert not runtime.restore_primary_runtime(agent)
        assert persisted() == before
        assert agent._fallback_activated and agent.model == "fallback-model"
        assert agent.client.chat.completions.create(
            model=agent.model, messages=[{"role": "user", "content": "probe"}],
            **agent.request_overrides,
        ).choices[0].message.content == "usable"
        assert requests[-1][0] == "fallback.example.test"

        assert runtime.restore_primary_runtime(agent)
        restored = persisted()
        assert restored["compression_failure_cooldown_until"] is None
        assert restored["compression_failure_error"] is None
        assert restored["compression_fallback_streak"] == 0
        assert restored["compression_overload_streak"] == 0
        assert restored["compression_ineffective_count"] == 0
        config = json.loads(restored["model_config"])
        assert PROACTIVE_PRUNE_REARM_MODEL_CONFIG_KEY not in config
        assert config["keep_other_setting"] == "unchanged"
    finally:
        agent.context_compressor.bind_session_state()
        db.close()



def test_restore_write_deferral_keeps_other_instances_and_workers_independent(
    tmp_path, monkeypatch, caplog,
):
    from concurrent.futures import ThreadPoolExecutor
    from contextvars import copy_context
    import logging

    from agent.context_compressor import ContextCompressor
    from agent.context_compressor_state import defer_compressor_state_writes
    from hermes_state import SessionDB

    db = SessionDB(db_path=tmp_path / "restore-context.db")
    db.create_session("first", source="test")
    db.create_session("second", source="test")
    first = ContextCompressor(model="first", config_context_length=200000, quiet_mode=True)
    second = ContextCompressor(model="second", config_context_length=200000, quiet_mode=True)
    first.bind_session_state(db, "first")
    second.bind_session_state(db, "second")
    try:
        with pytest.raises(RuntimeError, match="abandon restore"):
            with defer_compressor_state_writes(first):
                assert first._durable_write("set_compression_fallback_streak", "restore reset", 0)
                with ThreadPoolExecutor(max_workers=1) as pool:
                    assert pool.submit(
                        copy_context().run, first._durable_write,
                        "set_compression_fallback_streak", "worker verdict", 7,
                    ).result(timeout=5)
                assert second._durable_write("set_compression_fallback_streak", "other verdict", 8)
                raise RuntimeError("abandon restore")
        assert db.get_compression_fallback_streak("first") == 7
        assert db.get_compression_fallback_streak("second") == 8

        def rejected_write(*_args):
            raise RuntimeError("storage blocked")

        monkeypatch.setattr(db, "set_compression_fallback_streak", rejected_write)
        with caplog.at_level(logging.DEBUG, logger="agent.context_compressor"):
            with defer_compressor_state_writes(first):
                assert first._durable_write("set_compression_fallback_streak", "restore reset", 0)
        assert db.get_compression_fallback_streak("first") == 7
        assert "restore reset persist failed (non-sqlite): storage blocked" in caplog.text
    finally:
        db.close()
