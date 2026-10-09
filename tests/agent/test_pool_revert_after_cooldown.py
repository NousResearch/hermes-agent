# Copyright 2025 Nous Research (Licensed under the Apache License, Version 2.0)
"""A live session rotated off a quota-benched credential moves back once the bench lifts.

New sessions already do this (``load_pool().select()`` prefers the priority-0 entry again once
its 429/402 cooldown has elapsed); the per-turn ``restore_primary_runtime`` hook must do the same
for the session that took the rotation, or a long-lived gateway agent bills the fallback for
its whole life. Credential-only: it must not touch the model/base_url/compressor restore path.
"""

import logging
import threading
import time
from dataclasses import replace

import pytest

import agent.credential_pool as cp
import agent.agent_runtime_helpers as runtime_helpers
from agent.agent_runtime_helpers import recover_with_credential_pool, restore_primary_runtime
from agent.client_lifecycle import ClientLifecycleMixin
from agent.credential_pool import EXHAUSTED_TTL_429_SECONDS, CredentialPool, PooledCredential

_BASE = "https://api.anthropic.com"


def _entry(entry_id, label, *, priority, auth_type, token):
    raw = {
        "id": entry_id, "label": label, "auth_type": auth_type, "priority": priority,
        "access_token": token, "base_url": _BASE, "source": "manual",
    }
    if auth_type == "oauth":
        raw["refresh_token"] = f"rt-{entry_id}"
        raw["expires_at_ms"] = int((time.time() + 30 * 86400) * 1000)
    return PooledCredential.from_dict("anthropic", raw)


class _LiveAgent:
    """Long-lived session stand-in: real pool + real recovery/restore helpers, no client build."""

    _fallback_activated = False
    _fallback_index = 0
    _primary_runtime = {"provider": "anthropic", "model": "claude-opus-5", "base_url": _BASE}
    provider = "anthropic"
    model = "claude-opus-5"
    base_url = _BASE

    def __init__(self, pool):
        self._credential_pool = pool
        first = pool.select()
        self._credential_pool_entry_id = first.id
        self.api_key = first.runtime_api_key

    def _swap_credential(self, entry):
        self.api_key = entry.runtime_api_key
        self._credential_pool_entry_id = entry.id
        return True

    def _prepare_credential_swap(self, entry):
        from hermes_cli.anon_auth import route_can_serve_model

        runtime_base = entry.runtime_base_url or entry.base_url or self.base_url
        if not route_can_serve_model(self.provider, runtime_base, self.model):
            return None
        agent = self

        class _DeferredSwap:
            def commit(self, final_entry):
                return agent._swap_credential(final_entry)

            def abort(self):
                return None

        return _DeferredSwap()

    def _is_entitlement_failure(self, error_context, status_code):
        return False


class _Resource:
    def __init__(self, name):
        self.name = name
        self.close_calls = 0
        self.retire_calls = 0

    def close(self):
        self.close_calls += 1


class _LifecycleAgent(ClientLifecycleMixin):
    """Client-lifecycle owner with fixture resources and no network-capable clients."""

    _fallback_activated = False
    _fallback_index = 0
    _primary_runtime = {"provider": "anthropic", "model": "claude-opus-5", "base_url": _BASE}
    provider = "anthropic"
    model = "claude-opus-5"
    base_url = _BASE

    def __init__(self, pool, *, api_mode="chat_completions"):
        fallback = next(entry for entry in pool.entries() if entry.id == "fall0000")
        self._credential_pool = pool
        self._credential_pool_entry_id = fallback.id
        self._credential_pool_revert_id = "pref0000"
        self.api_mode = api_mode
        self.api_key = fallback.runtime_api_key
        self._client_kwargs = {"api_key": self.api_key, "base_url": self.base_url}
        self._client_lock = threading.RLock()
        self._transport_cache = {}
        self.client = _Resource("original-openai")
        self._anthropic_client = _Resource("original-anthropic")
        self._anthropic_api_key = self.api_key
        self._anthropic_base_url = self.base_url
        self._is_anthropic_oauth = False
        self.created_resources = []
        self.events = []

    def _create_openai_client(self, client_kwargs, *, reason, shared):
        resource = _Resource(f"candidate-openai-{len(self.created_resources)}")
        self.created_resources.append(resource)
        self.events.append("build")
        return resource

    def _build_direct_anthropic_client(self, token, base_url):
        resource = _Resource(f"candidate-anthropic-{len(self.created_resources)}")
        self.created_resources.append(resource)
        self.events.append("build")
        return resource

    def _anthropic_oauth_flag(self, token):
        return token.startswith("sk-ant-oat")

    def _close_openai_client(self, client, *, reason, shared):
        client.close()
        self.events.append("candidate_close")

    def _retire_shared_openai_client(self, client, *, reason):
        assert self._credential_pool_revert_id == "pref0000"
        client.retire_calls += 1
        self.events.append("swap_commit")


def _expired_reclaim_pool(tmp_path, monkeypatch, *, provider="anthropic", oauth=True):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    auth_type = "oauth" if oauth else "api_key"
    preferred = _entry(
        "pref0000", "preferred", priority=0, auth_type=auth_type,
        token="sk-ant-oat01-PREF" if oauth else "fixture-pref",
    )
    fallback = _entry(
        "fall0000", "fallback", priority=1, auth_type="api_key", token="fixture-fall",
    )
    if provider != "anthropic":
        preferred = replace(preferred, provider=provider, base_url="https://example.invalid/v1")
        fallback = replace(fallback, provider=provider, base_url="https://example.invalid/v1")
    preferred = replace(
        preferred,
        last_status=cp.STATUS_EXHAUSTED,
        last_status_at=time.time() - EXHAUSTED_TTL_429_SECONDS - 120,
        last_error_code=429,
        last_error_reason="rate_limit",
        last_error_message="fixture cooldown",
        last_error_reset_at=time.time() - 60,
    )
    pool = CredentialPool(provider, [preferred, fallback])
    pool._persist()
    return pool


def _expire_cooldowns(monkeypatch, real=None, windows=1):
    real = real or time.time
    monkeypatch.setattr(cp.time, "time", lambda: real() + windows * (EXHAUSTED_TTL_429_SECONDS + 120))


def test_live_session_reverts_to_quota_benched_credential_once_cooldown_lifts(monkeypatch):
    real_time = time.time
    pool = CredentialPool(provider="anthropic", entries=[
        _entry("pref0000", "subscription-oauth", priority=0, auth_type="oauth", token="sk-ant-oat01-PREF"),
        _entry("fall0000", "paid-api-key", priority=1, auth_type="api_key", token="sk-ant-api03-FALL"),
    ])
    agent = _LiveAgent(pool)
    assert agent.api_key == "sk-ant-oat01-PREF"

    # 429 twice = retry once, then rotate (the real rate-limit ladder).
    recover_with_credential_pool(agent, status_code=429, has_retried_429=False, error_context={"message": "Error"})
    recovered, _ = recover_with_credential_pool(agent, status_code=429, has_retried_429=True, error_context={"message": "Error"})
    assert recovered and agent.api_key == "sk-ant-api03-FALL"

    # Control: while the bench is still active the session stays on the fallback.
    assert restore_primary_runtime(agent) is False
    assert agent.api_key == "sk-ant-api03-FALL"

    _expire_cooldowns(monkeypatch)
    assert restore_primary_runtime(agent) is False  # credential-only: no primary-runtime restore ran
    assert agent.api_key == "sk-ant-oat01-PREF"
    assert agent._credential_pool_entry_id == "pref0000"
    assert agent._fallback_activated is False and agent.model == "claude-opus-5"
    assert agent._credential_pool_revert_id is None
    # The pool agrees the preferred entry is healthy again (cooldown cleared, not merely elapsed).
    assert next(e for e in pool.entries() if e.id == "pref0000").last_status != cp.STATUS_EXHAUSTED

    # Control (two-session interleaving): a session that started on the FALLBACK because another
    # session benched the preferred entry, then rotates UP to the preferred entry once its window
    # reopened, must not be pulled back DOWN when the fallback's own cooldown lifts.
    pool2 = CredentialPool(provider="anthropic", entries=[
        _entry("pref0000", "subscription-oauth", priority=0, auth_type="oauth", token="sk-ant-oat01-PREF"),
        _entry("fall0000", "paid-api-key", priority=1, auth_type="api_key", token="sk-ant-api03-FALL"),
    ])
    monkeypatch.setattr(cp.time, "time", real_time)
    pool2.mark_exhausted_and_rotate(  # the OTHER session benches the preferred entry
        status_code=429, credential_id="pref0000", failure_reason="rate_limit", error_context={"message": "Error"},
    )
    late = _LiveAgent(pool2)
    assert late.api_key == "sk-ant-api03-FALL"
    _expire_cooldowns(monkeypatch, real_time, 1)  # pref's window reopened; fall benched from now
    recover_with_credential_pool(late, status_code=429, has_retried_429=False, error_context={"message": "Error"})
    recovered, _ = recover_with_credential_pool(late, status_code=429, has_retried_429=True, error_context={"message": "Error"})
    assert recovered and late.api_key == "sk-ant-oat01-PREF"
    assert getattr(late, "_credential_pool_revert_id", None) is None  # rotated UP: nothing to revert to
    _expire_cooldowns(monkeypatch, real_time, 2)  # fall's cooldown lifts too
    assert restore_primary_runtime(late) is False
    assert late.api_key == "sk-ant-oat01-PREF" and pool2.select().id == "pref0000"


def test_auth_bench_does_not_arm_a_revert(monkeypatch):
    """A 401 bench is not a quota window: the session keeps the credential it rotated to."""
    pool = CredentialPool(provider="anthropic", entries=[
        _entry("pref0000", "primary-key", priority=0, auth_type="api_key", token="sk-ant-api03-PREF"),
        _entry("fall0000", "backup-key", priority=1, auth_type="api_key", token="sk-ant-api03-FALL"),
    ])
    agent = _LiveAgent(pool)
    recovered, _ = recover_with_credential_pool(agent, status_code=401, has_retried_429=False, error_context={"message": "invalid"})
    assert recovered and agent.api_key == "sk-ant-api03-FALL"

    _expire_cooldowns(monkeypatch)
    assert restore_primary_runtime(agent) is False
    assert agent.api_key == "sk-ant-api03-FALL"
    assert getattr(agent, "_credential_pool_revert_id", None) is None


def test_reclaim_route_veto_preserves_cooldown_and_revert_intent(
    tmp_path, monkeypatch, caplog,
):
    """A route veto is not a successful reclaim and must remain retryable."""
    pool = _expired_reclaim_pool(tmp_path, monkeypatch)
    fallback = next(entry for entry in pool.entries() if entry.id == "fall0000")
    pool._current_id = fallback.id
    agent = _LiveAgent.__new__(_LiveAgent)
    agent._credential_pool = pool
    agent._credential_pool_entry_id = fallback.id
    agent._credential_pool_revert_id = "pref0000"
    agent.api_key = fallback.runtime_api_key
    active_key = agent.api_key

    attempts = []

    def _veto(*_args):
        attempts.append("pref0000")
        return False

    monkeypatch.setattr("hermes_cli.anon_auth.route_can_serve_model", _veto)
    caplog.set_level(logging.INFO, logger="agent.agent_runtime_helpers")

    assert restore_primary_runtime(agent) is False
    assert agent.api_key == active_key
    assert agent._credential_pool_entry_id == "fall0000"
    assert agent._credential_pool_revert_id == "pref0000"
    in_memory = next(entry for entry in pool.entries() if entry.id == "pref0000")
    assert in_memory.last_status == cp.STATUS_EXHAUSTED
    from hermes_cli.auth import read_credential_pool

    persisted = next(row for row in read_credential_pool("anthropic") if row["id"] == "pref0000")
    assert persisted["last_status"] == cp.STATUS_EXHAUSTED
    assert "available again — reverted pool rotation" not in caplog.text

    assert restore_primary_runtime(agent) is False
    assert attempts == ["pref0000", "pref0000"]


def test_reclaim_swap_build_failure_aborts_only_replacement_resources(tmp_path, monkeypatch):
    pool = _expired_reclaim_pool(tmp_path, monkeypatch)
    agent = _LifecycleAgent(pool, api_mode="anthropic_messages")
    original = agent._anthropic_client
    original_identity = (
        agent.api_key,
        agent.base_url,
        agent._credential_pool_entry_id,
        agent._anthropic_api_key,
        agent._anthropic_base_url,
    )

    def _fail_after_build(_token, _base_url):
        raise RuntimeError("fixture replacement configuration failed")

    agent._derive_anthropic_oauth_flag = _fail_after_build

    assert restore_primary_runtime(agent) is False
    assert agent._anthropic_client is original
    assert original.close_calls == 0
    assert len(agent.created_resources) == 1
    assert agent.created_resources[0].close_calls == 1
    assert (
        agent.api_key,
        agent.base_url,
        agent._credential_pool_entry_id,
        agent._anthropic_api_key,
        agent._anthropic_base_url,
    ) == original_identity
    assert agent._credential_pool_revert_id == "pref0000"
    assert next(entry for entry in pool.entries() if entry.id == "pref0000").last_status == cp.STATUS_EXHAUSTED


def test_reclaim_swap_derives_anthropic_oauth_from_candidate_route(tmp_path, monkeypatch):
    """A route change must not classify a third-party endpoint using the old native base URL."""
    pool = _expired_reclaim_pool(tmp_path, monkeypatch, provider="glm")
    preferred = next(entry for entry in pool.entries() if entry.id == "pref0000")
    third_party_base = "https://llmbox.bytedance.net"
    pool._replace_entry(preferred, replace(preferred, base_url=third_party_base))
    pool._persist()
    agent = _LifecycleAgent(pool, api_mode="anthropic_messages")
    agent.provider = "glm"
    agent._anthropic_base_url = _BASE
    agent._is_anthropic_oauth = True
    monkeypatch.setattr("hermes_cli.anon_auth.route_can_serve_model", lambda *_args: True)

    assert restore_primary_runtime(agent) is False

    assert agent._anthropic_base_url == third_party_base
    assert agent._is_anthropic_oauth is False


def test_reclaim_abort_does_not_overwrite_newer_cooldown(tmp_path, monkeypatch):
    pool = _expired_reclaim_pool(tmp_path, monkeypatch, oauth=False)
    ticket = pool.prepare_reclaim("pref0000", model="claude-opus-5")
    assert ticket is not None
    ticket.commit()

    committed = next(entry for entry in pool.entries() if entry.id == "pref0000")
    newer_at = time.time() + 30
    newer = replace(
        committed,
        last_status=cp.STATUS_EXHAUSTED,
        last_status_at=newer_at,
        last_error_code=429,
        last_error_reason="newer_rate_limit",
        last_error_message="concurrent cooldown",
        last_error_reset_at=newer_at + 3600,
    )
    pool._replace_entry(committed, newer)
    pool._persist()

    ticket.abort()

    in_memory = next(entry for entry in pool.entries() if entry.id == "pref0000")
    assert in_memory.last_status == cp.STATUS_EXHAUSTED
    assert in_memory.last_status_at == newer_at
    from hermes_cli.auth import read_credential_pool

    persisted = next(row for row in read_credential_pool("anthropic") if row["id"] == "pref0000")
    assert persisted["last_status"] == cp.STATUS_EXHAUSTED
    assert cp._parse_absolute_timestamp(persisted["last_status_at"]) == pytest.approx(newer_at)


def test_reclaim_abort_preserves_single_use_refresh_result(tmp_path, monkeypatch):
    pool = _expired_reclaim_pool(
        tmp_path, monkeypatch, provider="openai-codex", oauth=True,
    )
    monkeypatch.setattr(pool, "_entry_needs_refresh", lambda _entry: True)

    def _single_use_refresh(entry, *, force):
        assert force is False
        refreshed = replace(
            entry,
            access_token="single-use-access-new",
            refresh_token="single-use-refresh-new",
            expires_at_ms=2**53,
            **cp._MARK_OK,
        )
        pool._replace_entry(entry, refreshed)
        pool._persist(status_cleared_ids=[entry.id])
        return refreshed

    monkeypatch.setattr(pool, "_refresh_entry", _single_use_refresh)
    ticket = pool.prepare_reclaim("pref0000", model="gpt-5.4")
    assert ticket is not None

    refreshed = ticket.commit()
    assert refreshed.access_token == "single-use-access-new"
    assert refreshed.last_status == cp.STATUS_OK
    ticket.abort()

    in_memory = next(entry for entry in pool.entries() if entry.id == "pref0000")
    assert in_memory.access_token == "single-use-access-new"
    assert in_memory.refresh_token == "single-use-refresh-new"
    assert in_memory.last_status == cp.STATUS_EXHAUSTED
    assert in_memory.last_error_reason == "rate_limit"
    from hermes_cli.auth import read_credential_pool

    persisted = next(row for row in read_credential_pool("openai-codex") if row["id"] == "pref0000")
    assert persisted["access_token"] == "single-use-access-new"
    assert persisted["refresh_token"] == "single-use-refresh-new"
    assert persisted["last_status"] == cp.STATUS_EXHAUSTED


def test_reclaim_commit_clears_once_then_logs_once(tmp_path, monkeypatch, caplog):
    pool = _expired_reclaim_pool(tmp_path, monkeypatch, oauth=False)
    assert hasattr(pool, "prepare_reclaim")
    agent = _LifecycleAgent(pool)
    original = agent.client
    events = agent.events
    persist_calls = 0
    original_persist = pool._persist

    def _tracked_persist(**kwargs):
        nonlocal persist_calls
        assert agent._credential_pool_revert_id == "pref0000"
        persist_calls += 1
        events.append("pool_commit")
        return original_persist(**kwargs)

    monkeypatch.setattr(pool, "_persist", _tracked_persist)
    original_info = runtime_helpers.logger.info

    def _tracked_info(message, *args, **kwargs):
        if "available again — reverted pool rotation" in str(message):
            events.append("success_log")
            assert agent._credential_pool_revert_id is None
        return original_info(message, *args, **kwargs)

    monkeypatch.setattr(runtime_helpers.logger, "info", _tracked_info)
    caplog.set_level(logging.INFO, logger="agent.agent_runtime_helpers")

    assert restore_primary_runtime(agent) is False

    assert persist_calls == 1
    assert events == ["build", "pool_commit", "swap_commit", "success_log"]
    assert agent._credential_pool_revert_id is None
    assert agent._credential_pool_entry_id == "pref0000"
    assert agent.client is agent.created_resources[0]
    assert original.retire_calls == 1
    assert agent.created_resources[0].close_calls == 0
    assert caplog.text.count("available again — reverted pool rotation") == 1
