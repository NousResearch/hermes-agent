"""Preserve subscription account rotation across cached turns (PR #43277)."""
import json
import time
from pathlib import Path

import pytest

from run_agent import AIAgent


@pytest.fixture
def pool_home(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "missing-codex"))
    monkeypatch.setattr("hermes_cli.auth_codex._probe_codex_pool_entry_quota_restored", lambda *args: False)
    (tmp_path / "config.yaml").write_text(
        "model:\n  provider: openai-codex\n  default: gpt-6-luna\n"
        "credential_pool_strategies:\n  openai-codex: fill_first\n")
    (tmp_path / "auth.json").write_text(json.dumps({
        "version": 1, "providers": {}, "credential_pool": {"openai-codex": [
            {"id": f"cred-{i}", "source": f"manual:test-{i}", "label": f"account-{i}",
             "auth_type": "api_key", "access_token": f"key-{i}", "priority": i,
             "base_url": "https://chatgpt.com/backend-api/codex"}
            for i in range(2)
        ]},
    }))
    return tmp_path


def _agent(pool):
    agent = AIAgent.__new__(AIAgent)
    agent.__dict__.update(
        provider="openai-codex", model="gpt-6-luna", api_key="key-0",
        _base_url="https://chatgpt.com/backend-api/codex", _credential_pool=pool,
        _credential_pool_entry_id="cred-0", _fallback_activated=False, _is_entitlement_failure=lambda *args: False,
        _primary_runtime={"api_key": "key-0", "client_kwargs": {"api_key": "key-0"}},
    )
    def swap(entry):
        agent.api_key = entry.runtime_api_key
        agent._credential_pool_entry_id = entry.id
        return True
    agent._swap_credential = swap
    return agent


@pytest.mark.parametrize("status", ["exhausted", "dead"])
def test_new_turn_replaces_current_key_exhausted_by_another_process(pool_home, status):
    from agent.credential_pool import load_pool
    from agent.agent_runtime_helpers import restore_primary_runtime

    pool = load_pool("openai-codex")
    pool.select()
    agent = _agent(pool)
    auth_path = pool_home / "auth.json"
    data = json.loads(auth_path.read_text())
    data["credential_pool"]["openai-codex"][0].update(
        last_status=status, last_status_at=time.time(), last_error_code=429,
        last_error_reset_at=time.time() + 3600)
    auth_path.write_text(json.dumps(data))
    restore_primary_runtime(agent)
    assert agent.api_key == "key-1"
    assert agent._primary_runtime["api_key"] == "key-1"


def test_persistent_401_rotates_after_same_entry_refresh_limit(pool_home, monkeypatch):
    from agent.credential_pool import load_pool
    from agent.agent_runtime_helpers import recover_with_credential_pool

    pool = load_pool("openai-codex")
    pool.select()
    agent = _agent(pool)
    # Network refresh succeeds but the inference endpoint continues rejecting this account.
    monkeypatch.setattr(pool, "try_refresh_matching", lambda **kwargs: pool.current())
    for _ in range(3):
        recovered, _ = recover_with_credential_pool(agent, status_code=401, has_retried_429=False)
    assert recovered
    assert agent.api_key == "key-1"
    assert next(e for e in pool.entries() if e.id == "cred-0").last_status == "exhausted"


def test_overload_retries_then_rotates_with_short_cooldown(pool_home):
    from agent.credential_pool import load_pool, _exhausted_until
    from agent.agent_runtime_helpers import recover_with_credential_pool
    from agent.error_classifier import FailoverReason

    pool = load_pool("openai-codex")
    pool.select()
    agent = _agent(pool)
    args = {"status_code": 503, "classified_reason": FailoverReason.overloaded}
    assert recover_with_credential_pool(agent, has_retried_429=False, **args) == (False, True)
    assert recover_with_credential_pool(agent, has_retried_429=True, **args) == (True, False)
    assert agent.api_key == "key-1"
    failed = next(e for e in pool.entries() if e.id == "cred-0")
    assert _exhausted_until(failed) - failed.last_status_at == 60


def test_direct_codex_agent_attaches_only_its_matching_pool_on_recovery(pool_home, monkeypatch):
    from agent.agent_runtime_helpers import recover_with_credential_pool

    auth_path = pool_home / "auth.json"
    data = json.loads(auth_path.read_text())
    for entry in data["credential_pool"]["openai-codex"]:
        entry["auth_type"] = "oauth"
    auth_path.write_text(json.dumps(data))
    monkeypatch.setattr("agent.account_usage._get_json", lambda *args, **kwargs: {})
    agent = _agent(None)
    agent._credential_pool_entry_id = None
    recovered, _ = recover_with_credential_pool(agent, status_code=429, has_retried_429=True)
    assert recovered
    assert agent.api_key == "key-1"
    assert agent._credential_pool.provider == "openai-codex"


def test_direct_agent_does_not_attach_credential_from_another_endpoint(pool_home):
    from agent.agent_runtime_helpers import recover_with_credential_pool

    auth_path = pool_home / "auth.json"
    data = json.loads(auth_path.read_text())
    data["credential_pool"]["openai-codex"][0]["base_url"] = "https://other.example/v1"
    auth_path.write_text(json.dumps(data))
    agent = _agent(None)
    assert recover_with_credential_pool(agent, status_code=429, has_retried_429=True) == (False, True)
    assert agent._credential_pool is None
