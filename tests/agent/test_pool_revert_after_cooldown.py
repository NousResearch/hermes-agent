# Copyright 2025 Nous Research (Licensed under the Apache License, Version 2.0)
"""A live session rotated off a quota-benched credential moves back once the bench lifts.

New sessions already do this (``load_pool().select()`` prefers the priority-0 entry again once
its 429/402 cooldown has elapsed); the per-turn ``restore_primary_runtime`` hook must do the same
for the session that took the rotation, or a long-lived gateway agent bills the fallback for
its whole life. Credential-only: it must not touch the model/base_url/compressor restore path.
"""

import time

import agent.credential_pool as cp
from agent.agent_runtime_helpers import recover_with_credential_pool, restore_primary_runtime
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

    def _is_entitlement_failure(self, error_context, status_code):
        return False


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


class _SessionOnEntry(_LiveAgent):
    """A session already bound to a given entry (earlier ones were taken by other sessions).

    Bypasses ``_LiveAgent``'s opening ``select()`` because under ``round_robin`` that call itself
    renumbers the pool, which is the behaviour under test.
    """

    def __init__(self, pool, entry_id):
        self._credential_pool = pool
        entry = next(e for e in pool.entries() if e.id == entry_id)
        pool._current_id = entry.id
        self._credential_pool_entry_id = entry.id
        self.api_key = entry.runtime_api_key


def _round_robin_pool(monkeypatch):
    monkeypatch.setattr(cp, "get_pool_strategy", lambda provider: cp.STRATEGY_ROUND_ROBIN)
    pool = CredentialPool(provider="anthropic", entries=[
        _entry("pref0000", "primary-key", priority=0, auth_type="api_key", token="sk-ant-api03-PREF"),
        _entry("mid00000", "second-key", priority=1, auth_type="api_key", token="sk-ant-api03-MID"),
        _entry("fall0000", "backup-key", priority=2, auth_type="api_key", token="sk-ant-api03-FALL"),
    ])
    monkeypatch.setattr(pool, "_persist", lambda *a, **k: None)
    return pool


def test_round_robin_priorities_are_a_cursor_not_a_preference(monkeypatch):
    """Pins the premise: ``select()`` renumbers every entry, so ``priority`` cannot rank them."""
    pool = _round_robin_pool(monkeypatch)
    assert [(e.id, e.priority) for e in pool.entries()] == [("pref0000", 0), ("mid00000", 1), ("fall0000", 2)]
    pool.select()
    assert [(e.id, e.priority) for e in pool.entries()] != [("pref0000", 0), ("mid00000", 1), ("fall0000", 2)]


def test_round_robin_rotation_never_arms_a_revert(monkeypatch):
    """A round-robin pool has no preferred entry, so nothing may pin the session back to one.

    The rank guard reads ``priority`` after ``mark_exhausted_and_rotate``. Under ``round_robin``
    that read happens *after* select() renumbered the pool, so a session that rotated UP off the
    lowest-numbered entry was seen as having rotated DOWN: the revert armed and
    ``restore_primary_runtime`` later dragged the session back onto the credential it had just
    been rate-limited off.
    """
    real_time = time.time
    pool = _round_robin_pool(monkeypatch)
    agent = _SessionOnEntry(pool, "fall0000")

    recover_with_credential_pool(agent, status_code=429, has_retried_429=False, error_context={"message": "Error"})
    recovered, _ = recover_with_credential_pool(agent, status_code=429, has_retried_429=True, error_context={"message": "Error"})

    assert recovered and agent._credential_pool_entry_id != "fall0000"
    assert getattr(agent, "_credential_pool_revert_id", None) is None
    rotated_to = agent._credential_pool_entry_id
    _expire_cooldowns(monkeypatch, real_time, 2)
    assert restore_primary_runtime(agent) is False
    assert agent._credential_pool_entry_id == rotated_to  # not dragged back to the benched entry


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
