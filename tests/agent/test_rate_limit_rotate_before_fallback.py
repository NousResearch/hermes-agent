# Copyright 2025 Nous Research (Licensed under the Apache License, Version 2.0)
"""A rate-limit 429 must give the credential pool one rotation before fallback_providers.

Regression tests for #120216: with a multi-credential pool (``hermes auth list`` shows
three Anthropic OAuth entries) a generic 429 switched straight to the configured
fallback provider ~70 ms after the error — no ``credential pool:`` log line, and the
priority-1 entry's ``request_count`` stayed 0. ``recover_after_classification`` defers
the first 429 to a same-credential retry, and the eager-fallback hinge trusts a
snapshot availability probe; when that probe says "no" the turn jumped providers while
the pool could still rotate to a healthy sibling.

``_rotate_rate_limited_credential_before_fallback`` is the last-chance rotation at the
fallback boundary: the pool's own rotation hook resolves the failed entry and only
reports failure when every entry really is exhausted or cooling down.
"""

import time
from types import SimpleNamespace
from typing import Any, Dict
from unittest.mock import MagicMock

import agent.conversation_loop as conversation_loop
import agent.turn_recovery as turn_recovery
from agent.error_classifier import FailoverReason
from agent.credential_pool import CredentialPool, PooledCredential

_BASE = "https://api.anthropic.com"
_MODEL = "claude-opus-5"


def _entry(entry_id, label, *, priority, token, model_cooldowns=None):
    raw = {
        "id": entry_id, "label": label, "auth_type": "oauth", "priority": priority,
        "access_token": token, "refresh_token": f"rt-{entry_id}", "base_url": _BASE,
        "source": "manual", "expires_at_ms": int((time.time() + 30 * 86400) * 1000),
    }
    entry = PooledCredential.from_dict("anthropic", raw)
    if model_cooldowns:
        entry.model_cooldowns = dict(model_cooldowns)
    return entry


class _LiveAgent:
    """Minimal agent stand-in: real pool + real recovery helpers, no client build."""

    _fallback_activated = False
    _fallback_index = 0
    _fallback_chain: Any = None
    _try_activate_fallback: Any = None
    _buffer_diagnostic_status: Any = None
    _primary_runtime = {"provider": "anthropic", "model": _MODEL, "base_url": _BASE}
    provider = "anthropic"
    model = _MODEL
    base_url = _BASE

    def __init__(self, pool, on_entry_id="pref0000"):
        self._credential_pool = pool
        self._credential_pool_entry_id = on_entry_id
        self.api_key = next(e.runtime_api_key for e in pool.entries() if e.id == on_entry_id)

    def _swap_credential(self, entry):
        self.api_key = entry.runtime_api_key
        self._credential_pool_entry_id = entry.id
        return True

    def _is_entitlement_failure(self, error_context, status_code):
        return False


def _pool_with_two(model_cooldowns):
    """Two healthy Anthropic OAuth entries; per-entry (credential, model) cooldowns."""
    return CredentialPool(provider="anthropic", entries=[
        _entry("pref0000", "subscription-oauth", priority=0, token="tok-pref",
               model_cooldowns=model_cooldowns.get("pref0000")),
        _entry("fall0000", "paid-api-key", priority=1, token="tok-fall",
               model_cooldowns=model_cooldowns.get("fall0000")),
    ])


_RL_VERDICT = SimpleNamespace(
    reason=FailoverReason.rate_limit, should_rotate_credential=True,
)


def test_rotates_to_sibling_when_snapshot_probe_is_blind_to_model_scope():
    """A cooldown recorded for ANOTHER model must not hide an entry from rotation.

    ``model_cooldown_until(entry, model=None)`` conservatively blocks an entry with any
    active model cooldown, so the eager-fallback hinge's ``has_available()`` snapshot
    said "nothing usable" while both entries were healthy for the failing model — the
    reported jump to fallback_providers with the priority-1 entry untouched.
    """
    other_model_bench = time.time() + 3600
    pool = _pool_with_two({
        "pref0000": {"other-model": other_model_bench},
        "fall0000": {"another-model": other_model_bench},
    })
    agent = _LiveAgent(pool)
    assert pool.has_available() is False  # the snapshot the hinge consults says "no"

    rotated = turn_recovery._rotate_rate_limited_credential_before_fallback(
        agent, _RL_VERDICT, 429, {"message": "Error code: 429 - rate_limit_error"},
    )

    assert rotated is True
    assert agent._credential_pool_entry_id == "fall0000"
    assert agent.api_key == "tok-fall"


def test_fallback_when_every_entry_is_cooling_for_the_failing_model():
    """The rotation hook is authoritative: nothing reachable -> return False, take the chain."""
    bench = time.time() + 3600
    pool = _pool_with_two({
        "pref0000": {_MODEL: bench},
        "fall0000": {_MODEL: bench},
    })
    agent = _LiveAgent(pool)

    rotated = turn_recovery._rotate_rate_limited_credential_before_fallback(
        agent, _RL_VERDICT, 429, {"message": "Error code: 429 - rate_limit_error"},
    )

    assert rotated is False
    assert agent._credential_pool_entry_id == "pref0000"  # no swap happened


def test_single_entry_pool_takes_the_fallback_chain():
    pool = CredentialPool(provider="anthropic", entries=[
        _entry("pref0000", "subscription-oauth", priority=0, token="tok-pref"),
    ])
    agent = _LiveAgent(pool)

    assert turn_recovery._rotate_rate_limited_credential_before_fallback(
        agent, _RL_VERDICT, 429, {"message": "Error code: 429"},
    ) is False


def test_non_rotating_verdicts_and_upstream_429s_never_touch_the_pool():
    pool = _pool_with_two({})
    agent = _LiveAgent(pool)

    overloaded = SimpleNamespace(reason=FailoverReason.overloaded, should_rotate_credential=False)
    assert turn_recovery._rotate_rate_limited_credential_before_fallback(
        agent, overloaded, 429, {"message": "overloaded"},
    ) is False

    upstream = SimpleNamespace(reason=FailoverReason.upstream_rate_limit, should_rotate_credential=False)
    assert turn_recovery._rotate_rate_limited_credential_before_fallback(
        agent, upstream, 429, {"message": "429 from upstream"},
    ) is False

    unrotatable = SimpleNamespace(reason=FailoverReason.rate_limit, should_rotate_credential=False)
    assert turn_recovery._rotate_rate_limited_credential_before_fallback(
        agent, unrotatable, 429, {"message": "rate limit"},
    ) is False


# ---- wire-level: the fallback boundary consults the rotation before switching providers ----


def _route_args(agent, monkeypatch, *, activate_result=True) -> Dict[str, Any]:
    agent._fallback_index = 0
    agent._fallback_chain = [object()]
    agent._credential_pool = MagicMock()
    agent._credential_pool.has_available.return_value = False  # hinge says "no"
    agent._credential_pool.entries.return_value = [object(), object()]
    agent._try_activate_fallback = MagicMock(return_value=activate_result)
    agent._buffer_diagnostic_status = MagicMock()
    monkeypatch.setattr(conversation_loop, "_arm_fallback_restart",
                        lambda agent, api_messages, active_system_prompt, _retry: active_system_prompt)
    return dict(
        agent=agent, api_error=Exception("Error code: 429"), classified=_RL_VERDICT,
        _retry=SimpleNamespace(), error_msg="Error code: 429", error_context={"message": "rate limit"},
        recovered_with_pool=False, base_url=_BASE, model=_MODEL,
        messages=[], api_messages=[], system_message=None, active_system_prompt=None,
        conversation_history=[], retry_count=1, max_retries=3, compression_attempts=0,
        max_compression_attempts=2, api_call_count=1, effective_task_id=None,
    )


def test_route_retries_on_rotation_instead_of_falling_back(monkeypatch):
    agent = _LiveAgent(_pool_with_two({}))
    calls = []
    monkeypatch.setattr(
        turn_recovery, "_rotate_rate_limited_credential_before_fallback",
        lambda *a, **k: calls.append(a) or True,
    )
    kwargs = _route_args(agent, monkeypatch)

    verdict = turn_recovery.route_classified_error(**kwargs)

    assert len(calls) == 1
    assert verdict.action == "continue"
    agent._try_activate_fallback.assert_not_called()


def test_route_activates_fallback_when_rotation_yields_nothing(monkeypatch):
    agent = _LiveAgent(_pool_with_two({}))
    monkeypatch.setattr(
        turn_recovery, "_rotate_rate_limited_credential_before_fallback", lambda *a, **k: False,
    )
    kwargs = _route_args(agent, monkeypatch)

    verdict = turn_recovery.route_classified_error(**kwargs)

    agent._try_activate_fallback.assert_called_once()
    assert verdict.action == "break"
