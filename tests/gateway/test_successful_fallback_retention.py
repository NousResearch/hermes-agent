"""Successful gateway turns retain recovery state through real cache/profile boundaries."""
from collections import OrderedDict
from pathlib import Path
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agent.agent_runtime_helpers import restore_primary_runtime
from gateway.run import GatewayRunner
from gateway.run_turn_runner import TurnRunner
from hermes_constants import set_hermes_home_override, reset_hermes_home_override
from hermes_state import SessionDB


@pytest.mark.parametrize("deadline", [200.0, 0.0])
def test_successful_fallback_keeps_agent_until_core_restores(monkeypatch, tmp_path, deadline):
    import agent.agent_runtime_helpers as runtime

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    clock = [100.0]
    monkeypatch.setattr(runtime.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr("agent.credential_pool.load_pool", lambda *_a, **_kw: None)
    runner = object.__new__(GatewayRunner)
    runner._agent_cache = OrderedDict()
    runner._agent_cache_lock = threading.Lock()
    runner._spawn_release_thread = MagicMock()  # resource teardown only, NOT cache eviction
    agents = []
    for profile in ("alpha", "beta"):
        home = tmp_path / profile
        home.mkdir()
        primary = f"primary-{profile}"
        (home / "config.yaml").write_text(f"model:\n  default: {primary}\n  provider: openrouter\n")
        key = f"agent:{profile}:matrix:room:shared-room"
        agent = SimpleNamespace(
            model="fallback", provider="openrouter", _fallback_activated=True,
            _rate_limited_until=deadline, _fallback_index=1,
            _primary_runtime={
                "model": primary, "provider": "openrouter", "base_url": "",
                "api_mode": "chat_completions", "api_key": "test", "client_kwargs": {},
                "use_prompt_caching": False, "compressor_model": primary,
                "compressor_context_length": 100000, "compressor_base_url": "",
                "compressor_api_key": "test", "compressor_provider": "openrouter",
            },
            context_compressor=MagicMock(), _transport_cache={},
            _create_openai_client=MagicMock(), _ensure_lmstudio_runtime_loaded=MagicMock(),
            _cached_system_prompt="stable prefix", _provider_fallback_active=False,
        )
        signature = runner._agent_config_signature(primary, {"provider": "openrouter"}, [], "stable prefix")
        runner._agent_cache[key] = (agent, signature, 0, "sid")
        runner._session_state(key).conversation.ephemeral_pin = "stable prefix"
        agents.append((profile, home, primary, key, agent, signature))

    for profile, home, primary, key, agent, signature in agents:
        token = set_hermes_home_override(str(home))
        db = SessionDB(db_path=home / "state.db")
        try:
            db.create_session("sid", source="matrix", model=primary,
                              model_config={"gateway_runtime": {"provider": "openrouter"}})
            runner._session_db = SimpleNamespace(_db=db)
            ctx = SimpleNamespace(session_key=key, session_id="sid", source=None, _interrupt_depth=0,
                                  agent_holder=[agent], result_holder=[{"failed": False}])
            runner._sync_session_model_from_agent("sid", agent)
            runner._run_agent_evict_on_fallback(ctx)
            assert runner._agent_cache[key][0] is agent
            assert runner._session_state(key).conversation.ephemeral_pin == "stable prefix"
            assert db.get_session("sid")["model"] == primary
            turn = TurnRunner(runner, ctx)
            found = turn._lookup_cached_agent(signature, runner._agent_cache_lock,
                                              runner._agent_cache, 123, "sid", False, 0)
            assert found.reused and found.agent is agent
            assert found.agent._cached_system_prompt == "stable prefix"
            assert found.agent._rate_limited_until == deadline
            clock[0] = 100.0
            if deadline:
                assert restore_primary_runtime(found.agent) is False
                assert agent.model == "fallback"
                clock[0] = 201.0
            assert restore_primary_runtime(found.agent) is True
            assert agent.model == primary
            assert agent._fallback_activated is False
            runner._sync_session_model_from_agent("sid", agent)
            assert db.get_session("sid")["model"] == primary
            # Config changes and cross-process history invalidation still win.
            changed = runner._agent_config_signature("manual-model", {"provider": "openrouter"}, [], "stable prefix")
            assert not turn._lookup_cached_agent(changed, runner._agent_cache_lock,
                                                 runner._agent_cache, 123, "sid", False, 0).reused
            stale = turn._lookup_cached_agent(signature, runner._agent_cache_lock,
                                              runner._agent_cache, 123, "sid", False, 1)
            assert not stale.reused and stale.evicted is agent
            assert key not in runner._agent_cache
        finally:
            db.close()
            reset_hermes_home_override(token)
    runner._spawn_release_thread.assert_not_called()


@pytest.mark.parametrize("failed,intentional,active,evicted", [
    (False, False, False, True),
    (True, False, False, False),
    (False, True, False, False),
])
def test_unmanaged_drift_and_intentional_switch_policy(monkeypatch, failed, intentional, active, evicted):
    monkeypatch.setattr("gateway.run._resolve_gateway_model", lambda: "primary")
    runner = object.__new__(GatewayRunner)
    runner._is_intentional_model_switch = lambda *_a: intentional
    runner._evict_cached_agent = MagicMock()
    agent = SimpleNamespace(model="other", provider="openrouter", _fallback_activated=active)
    runner._run_agent_evict_on_fallback(SimpleNamespace(
        session_key="room", source=None, agent_holder=[agent], result_holder=[{"failed": failed}],
    ))
    assert runner._evict_cached_agent.called is evicted
