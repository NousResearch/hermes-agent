# Copyright 2025 Nous Research (Licensed under the Apache License, Version 2.0)
"""A live session rotated off a quota-benched credential moves back once the bench lifts.

New sessions already do this (``load_pool().select()`` prefers the priority-0 entry again once
its 429/402 cooldown has elapsed); the per-turn ``restore_primary_runtime`` hook must do the same
for the session that took the rotation, or a long-lived gateway agent bills the fallback for
its whole life. Credential-only: it must not touch the model/base_url/compressor restore path.
"""

import json
import logging
import threading
import time
from contextlib import contextmanager
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


def test_reclaim_persist_failure_keeps_live_and_durable_cooldown(tmp_path, monkeypatch):
    """A durable write failure must not publish the cleared row in memory."""
    pool = _expired_reclaim_pool(tmp_path, monkeypatch, oauth=False)
    ticket = pool.prepare_reclaim("pref0000", model="claude-opus-5")
    assert ticket is not None
    import hermes_cli.auth as auth_mod

    monkeypatch.setattr(
        auth_mod,
        "_save_auth_store",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("fixture persist failed")),
    )

    with pytest.raises(OSError, match="fixture persist failed"):
        ticket.commit()
    ticket.abort()

    live = next(entry for entry in pool.entries() if entry.id == "pref0000")
    assert live.last_status == cp.STATUS_EXHAUSTED
    persisted = next(row for row in auth_mod.read_credential_pool("anthropic") if row["id"] == "pref0000")
    assert persisted["last_status"] == cp.STATUS_EXHAUSTED
    assert pool.prepare_reclaim("pref0000", model="claude-opus-5") is not None


@pytest.mark.parametrize("api_mode", ["chat_completions", "anthropic_messages"])
def test_credential_swap_refuses_concurrent_client_replacement(
    tmp_path, monkeypatch, api_mode,
):
    """A prepared ticket owns its original client, never a newer concurrent replacement."""
    pool = _expired_reclaim_pool(tmp_path, monkeypatch)
    preferred = next(entry for entry in pool.entries() if entry.id == "pref0000")
    agent = _LifecycleAgent(pool, api_mode=api_mode)
    original = agent._anthropic_client if api_mode == "anthropic_messages" else agent.client
    ticket = agent._prepare_credential_swap(preferred)
    assert ticket is not None
    candidate = agent.created_resources[-1]
    concurrent = _Resource(f"concurrent-{api_mode}")
    if api_mode == "anthropic_messages":
        agent._anthropic_client = concurrent
    else:
        agent.client = concurrent

    assert ticket.commit(preferred) is False
    ticket.abort()

    if api_mode == "anthropic_messages":
        assert agent._anthropic_client is concurrent
        assert concurrent.close_calls == 0
        assert original.close_calls == 0
    else:
        assert agent.client is concurrent
        assert concurrent.retire_calls == 0
        assert original.retire_calls == 0
    assert candidate.close_calls == 1
    assert agent._credential_pool_revert_id == "pref0000"


def _run_blocked_reclaim_refresh(pool, monkeypatch):
    """Start a deterministic in-flight Codex refresh and return its controls/results."""
    import hermes_cli.auth as auth_mod

    started = threading.Event()
    release = threading.Event()
    outcome = {"entry": None, "error": None}
    monkeypatch.setattr(pool, "_entry_needs_refresh", lambda _entry: True)

    def _refresh(_access_token, _refresh_token):
        started.set()
        assert release.wait(5), "fixture refresh was never released"
        return {
            "access_token": "refreshed-access",
            "refresh_token": "refreshed-token",
            "last_refresh": "2099-01-01T00:00:00Z",
        }

    monkeypatch.setattr(auth_mod, "refresh_codex_oauth_pure", _refresh)
    ticket = pool.prepare_reclaim("pref0000", model="gpt-5.4")
    assert ticket is not None

    def _commit():
        try:
            outcome["entry"] = ticket.commit()
        except Exception as exc:  # expected by the fixed stale-ticket path
            outcome["error"] = exc

    thread = threading.Thread(target=_commit)
    thread.start()
    assert started.wait(5), "fixture refresh never started"
    return ticket, thread, release, outcome


def test_reclaim_refresh_does_not_overwrite_newer_in_memory_cooldown(tmp_path, monkeypatch):
    pool = _expired_reclaim_pool(
        tmp_path, monkeypatch, provider="openai-codex", oauth=True,
    )
    ticket, thread, release, outcome = _run_blocked_reclaim_refresh(pool, monkeypatch)
    live = next(entry for entry in pool.entries() if entry.id == "pref0000")
    newer_at = time.time() + 60
    newer = replace(
        live,
        last_status=cp.STATUS_EXHAUSTED,
        last_status_at=newer_at,
        last_error_reason="newer_concurrent_cooldown",
        last_error_reset_at=newer_at + 3600,
    )
    pool._replace_entry(live, newer)
    release.set()
    thread.join(5)
    assert not thread.is_alive()

    assert outcome["entry"] is None
    assert isinstance(outcome["error"], RuntimeError)
    ticket.abort()
    surviving = next(entry for entry in pool.entries() if entry.id == "pref0000")
    assert surviving.last_status == cp.STATUS_EXHAUSTED
    assert surviving.last_error_reason == "newer_concurrent_cooldown"


def test_reclaim_refresh_detects_equal_value_aba(tmp_path, monkeypatch):
    pool = _expired_reclaim_pool(
        tmp_path, monkeypatch, provider="openai-codex", oauth=True,
    )
    ticket, thread, release, outcome = _run_blocked_reclaim_refresh(pool, monkeypatch)
    staged = next(entry for entry in pool.entries() if entry.id == "pref0000")
    changed = replace(staged, label="temporary-aba-value")
    pool._replace_entry(staged, changed)
    pool._replace_entry(changed, staged)
    release.set()
    thread.join(5)
    assert not thread.is_alive()

    assert outcome["entry"] is None
    assert isinstance(outcome["error"], RuntimeError)
    ticket.abort()
    surviving = next(entry for entry in pool.entries() if entry.id == "pref0000")
    assert surviving.last_status == cp.STATUS_EXHAUSTED
    assert surviving.label == "preferred"


def test_reclaim_refresh_detects_public_admin_move_aba(tmp_path, monkeypatch):
    """A move-away/back through the supported admin API invalidates a prepared reclaim."""
    import agent.credential_pool_reclaim as reclaim_mod

    pool = _expired_reclaim_pool(
        tmp_path, monkeypatch, provider="openai-codex", oauth=True,
    )
    before = [entry.to_dict() for entry in pool.entries()]
    ticket = pool.prepare_reclaim("pref0000", model="gpt-5.4")
    assert ticket is not None

    assert pool.move_entry("pref0000", 1) is not None
    assert pool.move_entry("pref0000", 0) is not None
    assert [entry.to_dict() for entry in pool.entries()] == before
    assert reclaim_mod._durable_row(
        pool, "pref0000", ticket._store_path,
    ) == ticket._durable_basis

    with pytest.raises(RuntimeError, match="stale credential reclaim ticket"):
        ticket.commit()
    ticket.abort()
    surviving = next(entry for entry in pool.entries() if entry.id == "pref0000")
    assert surviving.last_status == cp.STATUS_EXHAUSTED


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


def test_borrowed_root_reclaim_serializes_durable_cas_and_write(tmp_path, monkeypatch):
    """The root owner lock keeps compare+write atomic; a peer root update wins afterward."""
    fake_home = tmp_path / "fake-home"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))
    import hermes_constants
    import hermes_cli.auth as auth_mod
    import agent.credential_pool_reclaim as reclaim_mod

    root_pool = _expired_reclaim_pool(
        tmp_path, monkeypatch, provider="xai-oauth", oauth=True,
    )
    root = tmp_path / "hermes"
    hermes_constants._default_hermes_root_memo = None  # type: ignore[attr-defined]
    from hermes_cli.profiles import create_profile

    profile = create_profile("worker")
    monkeypatch.setenv("HERMES_HOME", str(profile))
    hermes_constants._default_hermes_root_memo = None  # type: ignore[attr-defined]
    auth_mod._global_auth_store_cache = None
    pool = cp.load_pool("xai-oauth")
    assert pool._borrowed_root_ids == {entry.id for entry in root_pool.entries()}
    monkeypatch.setattr(pool, "_entry_needs_refresh", lambda _entry: False)
    ticket = pool.prepare_reclaim("pref0000", model="grok-4")
    assert ticket is not None
    lock_targets = []
    original_pool_lock = cp._auth_store_lock

    def _observed_pool_lock(*args, **kwargs):
        lock_targets.append(kwargs.get("target_path"))
        return original_pool_lock(*args, **kwargs)

    monkeypatch.setattr(cp, "_auth_store_lock", _observed_pool_lock)

    compare_seen = threading.Event()
    peer_done = threading.Event()
    original_durable_row = reclaim_mod._durable_row
    reads = 0

    def _observed_durable_row(*args, **kwargs):
        nonlocal reads
        row = original_durable_row(*args, **kwargs)
        reads += 1
        if reads == 1:
            compare_seen.set()
            peer_done.wait(0.25)
        return row

    monkeypatch.setattr(reclaim_mod, "_durable_row", _observed_durable_row)
    root_auth = root / "auth.json"
    newer_at = time.time() + 120

    def _peer_write():
        assert compare_seen.wait(5)
        with auth_mod._auth_store_lock(target_path=root_auth):
            store = auth_mod._load_auth_store(root_auth)
            row = next(item for item in store["credential_pool"]["xai-oauth"] if item["id"] == "pref0000")
            row.update({
                "access_token": "peer-root-access",
                "refresh_token": "peer-root-refresh",
                "last_status": cp.STATUS_EXHAUSTED,
                "last_status_at": newer_at,
                "last_error_reason": "peer_root_cooldown",
                "last_error_reset_at": newer_at + 3600,
            })
            auth_mod._save_auth_store(store, target_path=root_auth)
        peer_done.set()

    peer = threading.Thread(target=_peer_write)
    peer.start()
    ticket.commit()
    peer.join(5)
    assert not peer.is_alive()
    assert lock_targets[0] == root_auth

    root_store = json.loads(root_auth.read_text(encoding="utf-8"))
    persisted = next(row for row in root_store["credential_pool"]["xai-oauth"] if row["id"] == "pref0000")
    assert persisted["access_token"] == "peer-root-access"
    assert persisted["refresh_token"] == "peer-root-refresh"
    assert persisted["last_status"] == cp.STATUS_EXHAUSTED
    assert persisted["last_error_reason"] == "peer_root_cooldown"


def test_reclaim_peer_token_adoption_persists_pool_once(tmp_path, monkeypatch):
    """Adopting a peer-rotated singleton and clearing cooldown is one pool-store write."""
    pool = _expired_reclaim_pool(
        tmp_path, monkeypatch, provider="openai-codex", oauth=True,
    )
    preferred = next(entry for entry in pool.entries() if entry.id == "pref0000")
    seeded = replace(
        preferred,
        source="device_code",
        access_token="stale-access",
        refresh_token="stale-refresh",
    )
    pool._replace_entry(preferred, seeded)
    pool._persist()
    import hermes_cli.auth as auth_mod

    auth_file = tmp_path / "hermes" / "auth.json"
    store = json.loads(auth_file.read_text(encoding="utf-8"))
    store.setdefault("providers", {})["openai-codex"] = {
        "tokens": {
            "access_token": "peer-access",
            "refresh_token": "peer-refresh",
        },
        "last_refresh": "2099-01-01T00:00:00Z",
    }
    auth_file.write_text(json.dumps(store), encoding="utf-8")
    monkeypatch.setattr(cp, "_codex_entry_tracks_singleton", lambda *_args: True)
    monkeypatch.setattr(
        cp, "_codex_access_token_is_expiring",
        lambda token, _skew=0: token == "stale-access",
    )
    writes = 0
    original_save = auth_mod._save_auth_store

    def _tracked_save(*args, **kwargs):
        nonlocal writes
        writes += 1
        return original_save(*args, **kwargs)

    monkeypatch.setattr(auth_mod, "_save_auth_store", _tracked_save)
    ticket = pool.prepare_reclaim("pref0000", model="grok-4")
    assert ticket is not None

    committed = ticket.commit()

    assert committed.access_token == "peer-access"
    assert committed.refresh_token == "peer-refresh"
    assert committed.last_status == cp.STATUS_OK
    assert writes == 1


def test_reclaim_terminal_manual_refresh_publishes_dead(tmp_path, monkeypatch):
    """A terminal manual grant verdict must reach the real live and durable owner row."""
    from agent import anthropic_credentials as ac
    import hermes_cli.auth as auth_mod

    pool = _expired_reclaim_pool(tmp_path, monkeypatch, provider="anthropic", oauth=True)
    monkeypatch.setattr(pool, "_entry_needs_refresh", lambda _entry: True)
    terminal = ac.AnthropicOAuthError(
        400, "invalid_grant", "fixture refresh token revoked", what="refresh",
    )

    def _terminal_refresh(*_args, **_kwargs):
        raise terminal

    monkeypatch.setattr(ac, "refresh_anthropic_oauth_pure", _terminal_refresh)
    ticket = pool.prepare_reclaim("pref0000", model="claude-opus-5")
    assert ticket is not None

    with pytest.raises(RuntimeError):
        ticket.commit()

    live = next(entry for entry in pool.entries() if entry.id == "pref0000")
    assert live.last_status == cp.STATUS_DEAD
    assert live.last_error_reason == "invalid_grant"
    durable = next(
        row for row in auth_mod.read_credential_pool("anthropic")
        if row["id"] == "pref0000"
    )
    assert durable["last_status"] == cp.STATUS_DEAD
    assert durable["last_error_reason"] == "invalid_grant"


def test_reclaim_terminal_singleton_refresh_quarantines_real_owner(tmp_path, monkeypatch):
    """A terminal singleton grant is removed from the real pool and its owning store."""
    import hermes_cli.auth as auth_mod

    pool = _expired_reclaim_pool(
        tmp_path, monkeypatch, provider="openai-codex", oauth=True,
    )
    preferred = next(entry for entry in pool.entries() if entry.id == "pref0000")
    singleton = replace(
        preferred,
        source="device_code",
        access_token="terminal-access",
        refresh_token="terminal-refresh",
    )
    pool._replace_entry(preferred, singleton)
    pool._persist()
    store = auth_mod._load_auth_store()
    store.setdefault("providers", {})["openai-codex"] = {
        "tokens": {
            "access_token": "terminal-access",
            "refresh_token": "terminal-refresh",
        },
    }
    auth_mod._save_auth_store(store)
    monkeypatch.setattr(pool, "_entry_needs_refresh", lambda _entry: True)

    def _terminal_refresh(*_args, **_kwargs):
        raise RuntimeError("invalid_grant fixture")

    monkeypatch.setattr(auth_mod, "refresh_codex_oauth_pure", _terminal_refresh)
    monkeypatch.setattr(
        auth_mod, "_is_terminal_codex_oauth_refresh_error", lambda _exc: True,
    )
    ticket = pool.prepare_reclaim("pref0000", model="gpt-5.4")
    assert ticket is not None

    with pytest.raises(RuntimeError):
        ticket.commit()

    assert [entry.id for entry in pool.entries()] == ["fall0000"]
    assert [row["id"] for row in auth_mod.read_credential_pool("openai-codex")] == [
        "fall0000",
    ]


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

    def _single_use_refresh(entry):
        return replace(
            entry,
            access_token="single-use-access-new",
            refresh_token="single-use-refresh-new",
            expires_at_ms=2**53,
            **cp._MARK_OK,
        )

    monkeypatch.setattr(pool, "_refresh_reclaim_candidate", _single_use_refresh)
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
    import agent.credential_pool_reclaim as reclaim_mod
    original_persist = reclaim_mod._write_durable_row

    def _tracked_persist(*args, **kwargs):
        nonlocal persist_calls
        assert agent._credential_pool_revert_id == "pref0000"
        persist_calls += 1
        events.append("pool_commit")
        return original_persist(*args, **kwargs)

    monkeypatch.setattr(reclaim_mod, "_write_durable_row", _tracked_persist)
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


def test_reclaim_and_normal_persist_share_pool_then_auth_lock_order(tmp_path, monkeypatch):
    """Normal persistence and reclaim both finish with the normal write as CAS winner."""
    import agent.credential_pool_reclaim as reclaim_mod
    import hermes_cli.auth as auth_mod

    pool = _expired_reclaim_pool(tmp_path, monkeypatch, oauth=False)
    ticket = pool.prepare_reclaim("pref0000", model="claude-opus-5")
    assert ticket is not None

    refresh_ready = threading.Event()
    allow_refresh = threading.Event()
    normal_holds_pool = threading.Event()
    reclaim_waiting_for_pool = threading.Event()
    base_pool_lock = pool._lock

    class _ObservedRLock:
        def acquire(self, *args, **kwargs):
            if (
                threading.current_thread().name == "reclaim-commit"
                and normal_holds_pool.is_set()
            ):
                reclaim_waiting_for_pool.set()
            return base_pool_lock.acquire(*args, **kwargs)

        def release(self):
            return base_pool_lock.release()

        def __enter__(self):
            self.acquire()
            return self

        def __exit__(self, *_exc):
            self.release()

    pool._lock = _ObservedRLock()
    monkeypatch.setattr(pool, "_entry_needs_refresh", lambda _entry: True)

    def _paused_refresh(entry):
        refresh_ready.set()
        assert allow_refresh.wait(5), "normal persist never took the pool lock"
        return entry

    monkeypatch.setattr(pool, "_refresh_reclaim_candidate", _paused_refresh)

    auth_lock = threading.RLock()

    @contextmanager
    def _bounded_auth_lock(*_args, **_kwargs):
        if not auth_lock.acquire(timeout=0.25):
            raise TimeoutError("lock-order inversion")
        try:
            yield
        finally:
            auth_lock.release()

    monkeypatch.setattr(cp, "_auth_store_lock", _bounded_auth_lock)
    monkeypatch.setattr(auth_mod, "_auth_store_lock", _bounded_auth_lock)
    outcomes = {"persist_error": None, "reclaim_error": None}

    def _reclaim():
        try:
            ticket.commit()
        except Exception as exc:
            outcomes["reclaim_error"] = exc

    def _normal_persist():
        assert refresh_ready.wait(5), "reclaim never reached its refresh boundary"
        try:
            with pool._lock:
                normal_holds_pool.set()
                allow_refresh.set()
                assert reclaim_waiting_for_pool.wait(5), "reclaim never attempted its transaction"
                current = next(entry for entry in pool._entries if entry.id == "pref0000")
                newer_at = time.time() + 60
                winner = replace(
                    current,
                    last_status=cp.STATUS_EXHAUSTED,
                    last_status_at=newer_at,
                    last_error_reason="normal_persist_winner",
                    last_error_reset_at=newer_at + 3600,
                )
                pool._replace_entry(current, winner)
                pool._persist()
        except Exception as exc:
            outcomes["persist_error"] = exc

    reclaim_thread = threading.Thread(target=_reclaim, name="reclaim-commit")
    persist_thread = threading.Thread(target=_normal_persist, name="normal-persist")
    reclaim_thread.start()
    persist_thread.start()
    reclaim_thread.join(5)
    persist_thread.join(5)

    assert not reclaim_thread.is_alive()
    assert not persist_thread.is_alive()
    assert outcomes["persist_error"] is None
    assert isinstance(outcomes["reclaim_error"], RuntimeError)
    assert "stale credential reclaim ticket" in str(outcomes["reclaim_error"])
    live = next(entry for entry in pool.entries() if entry.id == "pref0000")
    durable = reclaim_mod._durable_row(pool, "pref0000", ticket._store_path)
    assert live.last_error_reason == "normal_persist_winner"
    assert durable["last_error_reason"] == "normal_persist_winner"


def test_borrowed_anthropic_reclaim_locks_owner_store_before_source(
    tmp_path, monkeypatch,
):
    """Every Anthropic path uses pool -> active -> owner -> source."""
    import hermes_cli.auth as auth_mod

    pool = _expired_reclaim_pool(tmp_path, monkeypatch)
    current = next(entry for entry in pool._entries if entry.id == "pref0000")
    source_entry = replace(current, source="claude_code")
    pool._replace_entry(current, source_entry)
    pool._borrowed_root_ids = {source_entry.id}
    ticket = pool.prepare_reclaim(source_entry.id, model="claude-opus-5")
    assert ticket is not None

    events = []
    base_pool_lock = pool._lock

    class _RecordedPoolLock:
        def __enter__(self):
            base_pool_lock.acquire()
            events.append("pool+")
            return self

        def __exit__(self, *_exc):
            events.append("pool-")
            base_pool_lock.release()

    @contextmanager
    def _recorded_auth_lock(*_args, target_path=None, **_kwargs):
        label = "owner" if target_path is not None else "active"
        events.append(f"{label}+")
        try:
            yield
        finally:
            events.append(f"{label}-")

    @contextmanager
    def _recorded_source_lock():
        events.append("source+")
        try:
            yield
        finally:
            events.append("source-")

    pool._lock = _RecordedPoolLock()
    monkeypatch.setattr(auth_mod, "_auth_store_lock", _recorded_auth_lock)
    monkeypatch.setattr(auth_mod, "_same_path", lambda _left, _right: False)
    monkeypatch.setattr(pool, "_claude_code_credentials_lock", _recorded_source_lock)

    with pool._borrowed_reclaim_transaction(ticket, source_entry):
        events.append("body")

    assert events == [
        "pool+", "active+", "owner+", "source+", "body",
        "source-", "owner-", "active-", "pool-",
    ]


def test_borrowed_reclaim_and_deferred_refresh_have_one_pool_first_winner(
    tmp_path, monkeypatch,
):
    """A normal deferred refresh cannot hold auth while reclaim holds the pool."""
    import hermes_cli.auth as auth_mod

    pool = _expired_reclaim_pool(
        tmp_path, monkeypatch, provider="openai-codex", oauth=True,
    )
    current = next(entry for entry in pool._entries if entry.id == "pref0000")
    expired = replace(
        current,
        source="device_code",
        access_token="deferred-access-0",
        refresh_token="deferred-refresh-0",
        expires_at_ms=1,
    )
    pool._replace_entry(current, expired)
    pool._borrowed_root_ids = {expired.id}
    pool._persist()
    ticket = pool.prepare_reclaim(expired.id, model="gpt-5.4")
    assert ticket is not None
    monkeypatch.setattr(pool, "_entry_needs_refresh", lambda _entry: True)

    boundary = threading.Event()
    normal_has_auth = threading.Event()
    base_pool_lock = pool._lock
    reclaim_pool_attempts = 0

    class _BoundedObservedRLock:
        def acquire(self, *_args, **_kwargs):
            nonlocal reclaim_pool_attempts
            name = threading.current_thread().name
            if name == "borrowed-reclaim":
                if base_pool_lock.acquire(blocking=False):
                    reclaim_pool_attempts += 1
                    if reclaim_pool_attempts >= 2:
                        boundary.set()
                    return True
                boundary.set()
                if not base_pool_lock.acquire(timeout=1):
                    raise TimeoutError("reclaim waited for the live pool")
                reclaim_pool_attempts += 1
                return True
            if name == "deferred-refresh":
                if not base_pool_lock.acquire(timeout=0.2):
                    raise TimeoutError("deferred refresh waited for the live pool")
                return True
            return base_pool_lock.acquire()

        def release(self):
            return base_pool_lock.release()

        def __enter__(self):
            self.acquire()
            return self

        def __exit__(self, *_exc):
            self.release()

    pool._lock = _BoundedObservedRLock()
    auth_lock = threading.RLock()
    auth_depth = threading.local()

    @contextmanager
    def _bounded_auth_lock(*_args, **_kwargs):
        if not auth_lock.acquire(timeout=1):
            raise TimeoutError("auth lock inversion")
        depth = getattr(auth_depth, "value", 0)
        auth_depth.value = depth + 1
        try:
            if threading.current_thread().name == "deferred-refresh" and depth == 0:
                normal_has_auth.set()
                assert boundary.wait(2), "reclaim never reached the competing pool boundary"
            yield
        finally:
            auth_depth.value -= 1
            auth_lock.release()

    monkeypatch.setattr(cp, "_auth_store_lock", _bounded_auth_lock)
    monkeypatch.setattr(auth_mod, "_auth_store_lock", _bounded_auth_lock)
    posts = []

    def _rotate(_access_token, refresh_token):
        posts.append(refresh_token)
        return {
            "access_token": "deferred-access-1",
            "refresh_token": "deferred-refresh-1",
            "last_refresh": "2099-01-01T00:00:01Z",
        }

    monkeypatch.setattr(auth_mod, "refresh_codex_oauth_pure", _rotate)
    results = {"refresh": None, "reclaim": None}
    errors = {"refresh": None, "reclaim": None}

    def _normal_refresh():
        try:
            results["refresh"] = pool._refresh_entry(expired, force=False)
        except Exception as exc:
            errors["refresh"] = exc

    def _reclaim():
        try:
            results["reclaim"] = ticket.commit()
        except Exception as exc:
            errors["reclaim"] = exc

    normal = threading.Thread(target=_normal_refresh, name="deferred-refresh")
    reclaim = threading.Thread(target=_reclaim, name="borrowed-reclaim")
    normal.start()
    assert normal_has_auth.wait(2), "normal refresh never acquired auth"
    reclaim.start()
    normal.join(5)
    reclaim.join(5)

    assert not normal.is_alive() and not reclaim.is_alive()
    assert errors["refresh"] is None
    assert results["refresh"] is not None
    assert results["refresh"].access_token == "deferred-access-1"
    assert isinstance(errors["reclaim"], RuntimeError)
    assert "stale credential reclaim ticket" in str(errors["reclaim"])
    assert posts == ["deferred-refresh-0"]
    winner = next(entry for entry in pool.entries() if entry.id == expired.id)
    assert (winner.access_token, winner.refresh_token) == (
        "deferred-access-1", "deferred-refresh-1",
    )


def _shared_claude_code_reclaim_fleet(tmp_path, monkeypatch, *, profile_names):
    import hermes_constants
    import hermes_cli.auth as auth_mod
    from agent import anthropic_credentials as anthropic_mod
    from agent.credential_persistence import fingerprint_secret_value
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    fake_home = tmp_path / "fake-home"
    fake_home.mkdir()
    claude_dir = fake_home / "claude"
    claude_dir.mkdir()
    root = tmp_path / "hermes"
    root.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(claude_dir))
    monkeypatch.setenv("HERMES_HOME", str(root))
    for name in ("ANTHROPIC_TOKEN", "ANTHROPIC_API_KEY", "CLAUDE_CODE_OAUTH_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    hermes_constants._default_hermes_root_memo = None  # type: ignore[attr-defined]
    monkeypatch.setattr(
        auth_mod, "is_provider_explicitly_configured", lambda provider: provider == "anthropic",
    )
    monkeypatch.setattr(
        anthropic_mod, "_read_claude_code_credentials_from_keychain", lambda: None,
    )
    monkeypatch.setattr(
        anthropic_mod, "_mirror_claude_code_credentials_to_keychain", lambda *_a, **_k: None,
    )

    access_token = "claude-access-0"
    refresh_token = "claude-refresh-0"
    expired_ms = int((time.time() - 3600) * 1000)
    source_path = claude_dir / ".credentials.json"
    source_path.write_text(json.dumps({
        "claudeAiOauth": {
            "accessToken": access_token,
            "refreshToken": refresh_token,
            "expiresAt": expired_ms,
            "scopes": ["user:inference"],
        },
    }), encoding="utf-8")
    exhausted_at = time.time() - EXHAUSTED_TTL_429_SECONDS - 120
    (root / "auth.json").write_text(json.dumps({
        "version": 1,
        "providers": {},
        "credential_pool": {
            "anthropic": [{
                "id": "shared-claude",
                "label": "shared Claude Code grant",
                "auth_type": "oauth",
                "priority": 0,
                "source": "claude_code",
                "expires_at_ms": expired_ms,
                "secret_fingerprint": fingerprint_secret_value(access_token),
                "last_status": cp.STATUS_EXHAUSTED,
                "last_status_at": exhausted_at,
                "last_error_code": 429,
                "last_error_reason": "rate_limit",
                "last_error_reset_at": exhausted_at + 30,
            }],
        },
    }), encoding="utf-8")
    profiles = [root / "profiles" / name for name in profile_names]
    for profile in profiles:
        profile.mkdir(parents=True)
        (profile / "auth.json").write_text(
            json.dumps({"version": 1, "providers": {}}), encoding="utf-8",
        )

    def under(home, callback):
        token = set_hermes_home_override(home)
        try:
            auth_mod._global_auth_store_cache = None
            return callback()
        finally:
            reset_hermes_home_override(token)

    return {
        "root": root,
        "profiles": profiles,
        "source_path": source_path,
        "under": under,
    }


def test_root_anthropic_refresh_and_borrowed_reclaim_do_not_cycle(
    tmp_path, monkeypatch,
):
    """Root refresh and profile reclaim agree on owner-before-source ordering."""
    import hermes_cli.auth as auth_mod
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    fleet = _shared_claude_code_reclaim_fleet(
        tmp_path, monkeypatch, profile_names=("borrower",),
    )
    root_pool = fleet["under"](fleet["root"], lambda: cp.load_pool("anthropic"))
    profile = fleet["profiles"][0]
    borrowed_pool = fleet["under"](profile, lambda: cp.load_pool("anthropic"))
    assert borrowed_pool._borrowed_root_ids == {"shared-claude"}
    ticket = fleet["under"](
        profile,
        lambda: borrowed_pool.prepare_reclaim("shared-claude", model="claude-opus-5"),
    )
    assert ticket is not None
    root_entry = next(entry for entry in root_pool.entries() if entry.id == "shared-claude")

    root_path = (fleet["root"] / "auth.json").resolve()
    source_path = fleet["source_path"].resolve()
    locks = {root_path: threading.RLock(), source_path: threading.RLock()}
    locks_guard = threading.Lock()
    depths = threading.local()
    root_has_owner = threading.Event()
    competing_boundary = threading.Event()

    @contextmanager
    def _bounded_path_lock(*_args, target_path=None, **_kwargs):
        path = (target_path or auth_mod._auth_file_path()).resolve()
        with locks_guard:
            lock = locks.setdefault(path, threading.RLock())
        local_depths = getattr(depths, "values", {})
        depth = local_depths.get(path, 0)
        name = threading.current_thread().name
        if name == "profile-reclaim" and depth == 0 and path == root_path:
            competing_boundary.set()
        if not lock.acquire(timeout=0.75):
            raise TimeoutError(f"lock cycle at {path.name}")
        local_depths[path] = depth + 1
        depths.values = local_depths
        try:
            if name == "profile-reclaim" and depth == 0 and path == source_path:
                competing_boundary.set()
            if name == "root-refresh" and depth == 0 and path == root_path:
                root_has_owner.set()
                assert competing_boundary.wait(2), "profile reclaim never reached root/source"
            yield
        finally:
            local_depths[path] -= 1
            lock.release()

    monkeypatch.setattr(cp, "_auth_store_lock", _bounded_path_lock)
    monkeypatch.setattr(auth_mod, "_auth_store_lock", _bounded_path_lock)
    posts = []

    def _rotate(_refresh_token, *, use_json=False):
        posts.append((_refresh_token, use_json))
        return {
            "access_token": "claude-access-1",
            "refresh_token": "claude-refresh-1",
            "expires_at_ms": int((time.time() + 3600) * 1000),
        }

    monkeypatch.setattr(
        "agent.anthropic_credentials.refresh_anthropic_oauth_pure", _rotate,
    )
    results = {"refresh": None, "reclaim": None}
    errors = {"refresh": None, "reclaim": None}

    def _root_refresh():
        token = set_hermes_home_override(fleet["root"])
        try:
            results["refresh"] = root_pool._refresh_entry(root_entry, force=False)
        except Exception as exc:
            errors["refresh"] = exc
        finally:
            reset_hermes_home_override(token)

    def _profile_reclaim():
        token = set_hermes_home_override(profile)
        try:
            results["reclaim"] = ticket.commit()
        except Exception as exc:
            errors["reclaim"] = exc
        finally:
            reset_hermes_home_override(token)

    root_thread = threading.Thread(target=_root_refresh, name="root-refresh")
    reclaim_thread = threading.Thread(target=_profile_reclaim, name="profile-reclaim")
    root_thread.start()
    assert root_has_owner.wait(2), "root refresh never acquired the owner store"
    reclaim_thread.start()
    root_thread.join(5)
    reclaim_thread.join(5)

    assert not root_thread.is_alive() and not reclaim_thread.is_alive()
    assert errors["refresh"] is None
    assert results["refresh"] is not None
    assert results["refresh"].access_token == "claude-access-1"
    assert isinstance(errors["reclaim"], RuntimeError)
    assert "stale credential reclaim ticket" in str(errors["reclaim"])
    assert posts == [("claude-refresh-0", False)]
    source = json.loads(fleet["source_path"].read_text(encoding="utf-8"))["claudeAiOauth"]
    assert (source["accessToken"], source["refreshToken"]) == (
        "claude-access-1", "claude-refresh-1",
    )


def test_two_profile_claude_code_reclaim_posts_once_and_both_adopt(
    tmp_path, monkeypatch,
):
    """A tokenless metadata-row loser hydrates from the rotated Claude source."""
    from agent.credential_persistence import fingerprint_secret_value
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    fleet = _shared_claude_code_reclaim_fleet(
        tmp_path, monkeypatch, profile_names=("alpha", "beta"),
    )
    pools = [
        fleet["under"](profile, lambda: cp.load_pool("anthropic"))
        for profile in fleet["profiles"]
    ]
    assert all(pool._borrowed_root_ids == {"shared-claude"} for pool in pools)
    tickets = [
        fleet["under"](
            profile,
            lambda pool=pool: pool.prepare_reclaim(
                "shared-claude", model="claude-opus-5",
            ),
        )
        for profile, pool in zip(fleet["profiles"], pools)
    ]
    assert all(ticket is not None for ticket in tickets)

    posts = []
    post_guard = threading.Lock()
    second_post = threading.Event()

    def _single_use_refresh(refresh_token, *, use_json=False):
        with post_guard:
            sequence = len(posts) + 1
            posts.append((refresh_token, use_json))
        if sequence == 1:
            second_post.wait(0.5)
        else:
            second_post.set()
        return {
            "access_token": f"claude-access-{sequence}",
            "refresh_token": f"claude-refresh-{sequence}",
            "expires_at_ms": int((time.time() + 3600) * 1000),
        }

    monkeypatch.setattr(
        "agent.anthropic_credentials.refresh_anthropic_oauth_pure",
        _single_use_refresh,
    )
    start = threading.Barrier(3)
    results = [None, None]
    errors = [None, None]

    def _commit(index):
        token = set_hermes_home_override(fleet["profiles"][index])
        try:
            start.wait()
            results[index] = tickets[index].commit()
        except Exception as exc:
            errors[index] = exc
        finally:
            reset_hermes_home_override(token)

    threads = [
        threading.Thread(target=_commit, args=(index,), name=f"claude-borrower-{index}")
        for index in range(2)
    ]
    for thread in threads:
        thread.start()
    start.wait()
    for thread in threads:
        thread.join(5)

    assert all(not thread.is_alive() for thread in threads)
    assert errors == [None, None]
    assert posts == [("claude-refresh-0", False)]
    assert {
        (result.access_token, result.refresh_token) for result in results
    } == {("claude-access-1", "claude-refresh-1")}
    root_store = json.loads((fleet["root"] / "auth.json").read_text(encoding="utf-8"))
    row = root_store["credential_pool"]["anthropic"][0]
    assert "access_token" not in row and "refresh_token" not in row
    assert row["secret_fingerprint"] == fingerprint_secret_value("claude-access-1")
    assert row["_credential_reclaim_revision"] == 1
    assert row["last_status"] == cp.STATUS_OK
    source = json.loads(fleet["source_path"].read_text(encoding="utf-8"))["claudeAiOauth"]
    assert (source["accessToken"], source["refreshToken"]) == (
        "claude-access-1", "claude-refresh-1",
    )
    assert all(
        "anthropic" not in json.loads(
            (profile / "auth.json").read_text(encoding="utf-8"),
        ).get("credential_pool", {})
        for profile in fleet["profiles"]
    )


def test_borrowed_root_single_use_reclaim_refreshes_once_and_loser_adopts(
    tmp_path, monkeypatch,
):
    """Two profiles sharing one root grant consume one token and converge on its rotation."""
    import hermes_constants
    import hermes_cli.auth as auth_mod
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    fake_home = tmp_path / "fake-home"
    fake_home.mkdir()
    root = tmp_path / "hermes"
    root.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setenv("HERMES_HOME", str(root))
    hermes_constants._default_hermes_root_memo = None  # type: ignore[attr-defined]
    expired_at = time.time() - EXHAUSTED_TTL_429_SECONDS - 120
    root_store = {
        "version": 1,
        "providers": {
            "openai-codex": {
                "tokens": {
                    "access_token": "root-access-0",
                    "refresh_token": "root-refresh-0",
                },
            },
        },
        "credential_pool": {
            "openai-codex": [
                {
                    "id": "shared-codex",
                    "label": "shared root grant",
                    "auth_type": "oauth",
                    "priority": 0,
                    "source": "device_code",
                    "access_token": "root-access-0",
                    "refresh_token": "root-refresh-0",
                    "last_status": cp.STATUS_EXHAUSTED,
                    "last_status_at": expired_at,
                    "last_error_code": 429,
                    "last_error_reason": "rate_limit",
                    "last_error_reset_at": expired_at + 30,
                },
            ],
        },
    }
    (root / "auth.json").write_text(json.dumps(root_store), encoding="utf-8")
    profiles = [root / "profiles" / name for name in ("alpha", "beta")]
    for profile in profiles:
        profile.mkdir(parents=True)
        (profile / "auth.json").write_text(
            json.dumps({"version": 1, "providers": {}}), encoding="utf-8",
        )

    def _under_profile(profile, callback):
        token = set_hermes_home_override(profile)
        try:
            auth_mod._global_auth_store_cache = None
            return callback()
        finally:
            reset_hermes_home_override(token)

    pools = [_under_profile(profile, lambda: cp.load_pool("openai-codex")) for profile in profiles]
    assert all(pool._borrowed_root_ids == {"shared-codex"} for pool in pools)
    tickets = [
        _under_profile(
            profile,
            lambda pool=pool: pool.prepare_reclaim("shared-codex", model="gpt-5.4"),
        )
        for profile, pool in zip(profiles, pools)
    ]
    assert all(ticket is not None for ticket in tickets)

    monkeypatch.setattr(
        cp,
        "_codex_access_token_is_expiring",
        lambda access_token, _skew: access_token == "root-access-0",
    )
    posts = []
    rotations = []
    post_lock = threading.Lock()
    second_post = threading.Event()

    def _single_use_refresh(_access_token, refresh_token):
        with post_lock:
            sequence = len(posts) + 1
            posts.append(refresh_token)
        if sequence == 1:
            second_post.wait(0.5)
        else:
            second_post.set()
        rotated = {
            "access_token": f"root-access-{sequence}",
            "refresh_token": f"root-refresh-{sequence}",
            "last_refresh": f"2099-01-01T00:00:0{sequence}Z",
        }
        with post_lock:
            rotations.append((rotated["access_token"], rotated["refresh_token"]))
        return rotated

    monkeypatch.setattr(auth_mod, "refresh_codex_oauth_pure", _single_use_refresh)
    start = threading.Barrier(3)
    results = [None, None]
    errors = [None, None]

    def _commit(index):
        token = set_hermes_home_override(profiles[index])
        try:
            start.wait()
            results[index] = tickets[index].commit()
        except Exception as exc:
            errors[index] = exc
        finally:
            reset_hermes_home_override(token)

    threads = [
        threading.Thread(target=_commit, args=(index,), name=f"borrower-{index}")
        for index in range(2)
    ]
    for thread in threads:
        thread.start()
    start.wait()
    for thread in threads:
        thread.join(5)

    assert all(not thread.is_alive() for thread in threads)
    assert errors == [None, None]
    assert posts == ["root-refresh-0"]
    assert rotations == [("root-access-1", "root-refresh-1")]
    assert {
        (result.access_token, result.refresh_token) for result in results
    } == {("root-access-1", "root-refresh-1")}
    persisted = json.loads((root / "auth.json").read_text(encoding="utf-8"))
    provider_tokens = persisted["providers"]["openai-codex"]["tokens"]
    pool_row = persisted["credential_pool"]["openai-codex"][0]
    assert (provider_tokens["access_token"], provider_tokens["refresh_token"]) == (
        "root-access-1", "root-refresh-1",
    )
    assert (pool_row["access_token"], pool_row["refresh_token"]) == (
        "root-access-1", "root-refresh-1",
    )
    assert all(
        "openai-codex"
        not in json.loads((profile / "auth.json").read_text(encoding="utf-8")).get(
            "credential_pool", {}
        )
        for profile in profiles
    )
