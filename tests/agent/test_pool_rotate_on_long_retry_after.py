"""A 429 whose Retry-After outlasts a live wait rotates the pool on the FIRST hit.

Anthropic answers a spent subscription window with 429 + ``Retry-After`` of several minutes. The
retry-once-then-rotate ladder sat that window out (capped at 600s) on the dead credential while a
second pooled account stood idle; a short Retry-After still gets its one retry first.
"""
from __future__ import annotations

import time
from types import SimpleNamespace

from agent.agent_runtime_helpers import recover_with_credential_pool
from agent.credential_pool import CredentialPool, PooledCredential
from agent.error_classifier import classify_api_error
from agent.turn_recovery import recover_after_classification
from agent.turn_retry_state import TurnRetryState

_BASE = "https://api.anthropic.com"
_MSG = "This request would exceed your account's rate limit. Please try again later."


def _entry(entry_id, priority):
    return PooledCredential.from_dict("anthropic", {
        "id": entry_id, "label": entry_id, "auth_type": "oauth", "priority": priority,
        "access_token": f"tok-{entry_id}", "refresh_token": f"rt-{entry_id}", "base_url": _BASE,
        "source": "manual", "expires_at_ms": int((time.time() + 30 * 86400) * 1000),
    })


class _RateLimited(Exception):
    def __init__(self, retry_after, message=_MSG):
        super().__init__(f"HTTP 429: {message}")
        self.status_code = 429
        self.message = message
        self.body = {"type": "error", "error": {"type": "rate_limit_error", "message": message}}
        headers = {"retry-after": str(retry_after)} if retry_after is not None else {}
        self.response = SimpleNamespace(headers=headers)


class _Agent:
    log_prefix = ""
    provider = "anthropic"
    api_mode = "anthropic_messages"
    base_url = _BASE
    model = "claude-opus-5-5"
    _fallback_activated = False

    def __init__(self, pool):
        self._credential_pool = pool
        first = pool.select()
        self._credential_pool_entry_id = first.id
        self.api_key = first.runtime_api_key

    def _recover_with_credential_pool(self, **kwargs):
        return recover_with_credential_pool(self, **kwargs)

    def _swap_credential(self, entry):
        self.api_key = entry.runtime_api_key
        self._credential_pool_entry_id = entry.id
        return True

    def _is_entitlement_failure(self, error_context, status_code):
        return False

    def __getattr__(self, name):
        return lambda *args, **kwargs: None


def _hit(agent, retry_after, message=_MSG):
    err = _RateLimited(retry_after, message)
    classified = classify_api_error(err, provider=agent.provider, model=agent.model)
    return recover_after_classification(
        agent, err, classified, TurnRetryState(), status_code=429,
        error_context={"message": message}, messages=[], api_messages=[],
    )


def _pool():
    return CredentialPool(provider="anthropic", entries=[_entry("acct1", 0), _entry("acct2", 1)])


def test_long_retry_after_rotates_immediately():
    agent = _Agent(_pool())
    retry_now, rotated = _hit(agent, 524)
    assert retry_now and rotated
    assert agent._credential_pool_entry_id == "acct2"


def test_short_retry_after_still_retries_same_credential_first():
    agent = _Agent(_pool())
    _, rotated = _hit(agent, 5)
    assert not rotated
    assert agent._credential_pool_entry_id == "acct1"


def test_message_only_long_reset_rotates_on_first_429():
    """Rotate on a long text-only reset; short waits and explicit headers still win."""
    agent = _Agent(_pool())
    retry_now, rotated = _hit(agent, None, "Rate limit reached. Resets in 3 minutes.")
    assert retry_now and rotated
    assert agent._credential_pool_entry_id == "acct2"

    short_agent = _Agent(_pool())
    _, rotated = _hit(short_agent, None, "Rate limit reached. Retry after 5 seconds.")
    assert not rotated
    assert short_agent._credential_pool_entry_id == "acct1"

    header_agent = _Agent(_pool())
    _, rotated = _hit(header_agent, 5, "Rate limit reached. Resets in 3 minutes.")
    assert not rotated
    assert header_agent._credential_pool_entry_id == "acct1"


def test_top_level_retry_after_beside_error_rotates_on_first_429():
    """Use a top-level body cooldown even when a nested error object is present."""
    agent = _Agent(_pool())
    err = _RateLimited(168)
    err.response.headers = {}
    err.body = {
        "retry_after": 168,
        "error": {"type": "rate_limit_error", "message": _MSG},
    }
    classified = classify_api_error(err, provider=agent.provider, model=agent.model)
    retry_now, rotated = recover_after_classification(
        agent, err, classified, TurnRetryState(), status_code=429,
        error_context={"message": _MSG}, messages=[], api_messages=[],
    )
    assert retry_now and rotated
    assert agent._credential_pool_entry_id == "acct2"
