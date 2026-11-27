"""Cached gateway conversations must see credentials added between turns."""

import json
import threading
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from agent.agent_runtime_helpers import recover_with_credential_pool
from agent.credential_pool import CredentialPool
from gateway.config import Platform
from gateway.run import GatewayRunner, _profile_runtime_scope, _resolve_runtime_agent_kwargs_for_provider
from gateway.run_turn_runner import TurnRunner


@pytest.fixture
def profile_home(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    home = tmp_path / "profile"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


def _write_pool(home, tokens):
    home.mkdir(exist_ok=True)
    (home / "auth.json").write_text(json.dumps({
        "version": 1,
        "credential_pool": {"xai-oauth": [
            {"id": f"account-{i}", "label": f"account-{i}", "auth_type": "oauth",
             "priority": i, "source": "manual", "access_token": token,
             "expires_at": 4102444800, "base_url": "https://api.x.ai/v1"}
            for i, token in enumerate(tokens)
        ]},
    }), encoding="utf-8")


def _runtime():
    return _resolve_runtime_agent_kwargs_for_provider("xai-oauth", target_model="grok-fixture")


def _cache_turn(agent, initial_runtime, current_runtime):
    runner = object.__new__(GatewayRunner)
    runner._agent_cache_lock = threading.Lock()
    runner._session_db = None
    runner._refresh_fallback_model = Mock(return_value=None)
    runner._apply_fallback_chain_to_agent = Mock()
    signature = runner._agent_config_signature(
        "grok-fixture", initial_runtime, [], "",
        cache_keys=runner._extract_cache_busting_config({}),
        user_id=None, user_id_alt=None, skip_context_files=False,
    )
    runner._agent_cache = {"session": (agent, signature, None, "sid")}
    context = SimpleNamespace(user_config={}, source=SimpleNamespace(platform=Platform.TELEGRAM), enabled_toolsets=[],
                              session_key="session", session_id="sid", _interrupt_depth=0)
    return TurnRunner(runner, context)._resolve_turn_agent(
        {"model": "grok-fixture", "runtime": current_runtime}, "telegram", "", 10, None, None,
    )


def _agent(runtime):
    from run_agent import AIAgent

    return AIAgent(
        model="grok-fixture", provider=runtime["provider"], api_key=runtime["api_key"],
        base_url=runtime["base_url"], api_mode=runtime["api_mode"],
        credential_pool=runtime["credential_pool"], quiet_mode=True,
        skip_context_files=True, skip_memory=True, enabled_toolsets=[],
    )


def test_cache_hit_can_rotate_to_account_added_after_previous_turn(profile_home):
    _write_pool(profile_home, ["primary-token"])
    with _profile_runtime_scope(profile_home):
        initial = _runtime()
        agent = _agent(initial)
        client = agent.client
        agent._cached_system_prompt = "frozen prompt"
        try:
            _write_pool(profile_home, ["primary-token", "new-backup-token"])
            current = _runtime()
            reused, hit = _cache_turn(agent, initial, current)
            assert hit and reused is agent
            assert agent.client is client
            assert agent._cached_system_prompt == "frozen prompt"

            recovered, _ = recover_with_credential_pool(
                agent, status_code=429, has_retried_429=True,
            )
            assert recovered, "cached pool must include the newly configured backup"
            assert agent.client.auth_headers["Authorization"] == "Bearer new-backup-token"
            assert agent._credential_pool_entry_id == "account-1"
        finally:
            agent.client.close()
            client.close()


def test_cache_hit_does_not_rotate_to_account_removed_between_turns(profile_home):
    _write_pool(profile_home, ["primary-token", "removed-token"])
    with _profile_runtime_scope(profile_home):
        initial = _runtime()
        agent = _agent(initial)
        try:
            _write_pool(profile_home, ["primary-token"])
            current = _runtime()
            reused, hit = _cache_turn(agent, initial, current)
            assert hit and reused is agent
            recovered, _ = recover_with_credential_pool(agent, status_code=429, has_retried_429=True)
            assert not recovered
            assert agent.client.auth_headers["Authorization"] == "Bearer primary-token"
        finally:
            agent.client.close()


def test_cached_pools_follow_the_owning_profile_in_a_b_a_order(profile_home):
    from agent.secret_scope import set_multiplex_active

    homes = [profile_home / "a", profile_home / "b"]
    agents, initial = [], []
    set_multiplex_active(True)
    try:
        for home in homes:
            _write_pool(home, ["shared-primary-token"])
            with _profile_runtime_scope(home):
                initial.append(_runtime())
                agents.append(_agent(initial[-1]))
        clients = [agent.client for agent in agents]
        for idx, backup in [(0, "a-backup"), (1, "b-backup"), (0, "a-new-backup")]:
            _write_pool(homes[idx], ["shared-primary-token", backup])
            with _profile_runtime_scope(homes[idx]):
                reused, hit = _cache_turn(agents[idx], initial[idx], _runtime())
                assert hit and reused is agents[idx]
                assert reused.client is clients[idx]
                assert [entry.runtime_api_key for entry in reused._credential_pool.entries()] == [
                    "shared-primary-token", backup,
                ]
        for idx, expected in enumerate(["a-new-backup", "b-backup"]):
            with _profile_runtime_scope(homes[idx]):
                recovered, _ = recover_with_credential_pool(agents[idx], status_code=429, has_retried_429=True)
                assert recovered
                assert agents[idx].client.auth_headers["Authorization"] == f"Bearer {expected}"
    finally:
        set_multiplex_active(False)
        for agent in agents:
            agent.client.close()


def test_pool_refresh_preserves_an_already_rotated_live_credential(profile_home):
    _write_pool(profile_home, ["primary-token", "backup-token"])
    with _profile_runtime_scope(profile_home):
        initial = _runtime()
        agent = _agent(initial)
        try:
            assert agent._swap_credential(initial["credential_pool"].entries()[1])
            client = agent.client
            reused, hit = _cache_turn(agent, initial, _runtime())
            assert hit and reused is agent
            assert agent.client is client
            assert agent.api_key == "backup-token"
            assert agent._credential_pool_entry_id == "account-1"
        finally:
            agent.client.close()


@pytest.mark.parametrize("mismatch", [
    "agent_provider", "agent_endpoint", "pool_provider", "entry_endpoint", "missing_key", "fallback",
])
def test_pool_refresh_preserves_foreign_or_fallback_runtime(profile_home, mismatch):
    _write_pool(profile_home, ["primary-token"])
    with _profile_runtime_scope(profile_home):
        initial, current = _runtime(), _runtime()
        agent = _agent(initial)
        original_pool, original_id, client = agent._credential_pool, agent._credential_pool_entry_id, agent.client
        try:
            mutations = {
                "agent_provider": lambda: setattr(agent, "provider", "openai-codex"),
                "agent_endpoint": lambda: setattr(agent, "base_url", "https://custom.example/v1"),
                "pool_provider": lambda: current.update(credential_pool=CredentialPool(
                    "openai-codex", current["credential_pool"].entries())),
                "entry_endpoint": lambda: current.update(credential_pool=CredentialPool("xai-oauth", [
                    replace(entry, base_url="https://custom.example/v1")
                    for entry in current["credential_pool"].entries()
                ])),
                "missing_key": lambda: setattr(agent, "api_key", "explicit-session-key"),
                "fallback": lambda: setattr(agent, "_fallback_activated", True),
            }
            mutations[mismatch]()
            reused, hit = _cache_turn(agent, initial, current)
            assert hit and reused is agent
            assert agent._credential_pool is original_pool
            assert agent._credential_pool_entry_id == original_id
            assert agent.client is client
        finally:
            agent.client.close()
