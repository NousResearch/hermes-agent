"""Credential pool must rotate to the next healthy key on 5xx (#121841).

A key-scoped 5xx on an aggregator (one key 503s while another is healthy) must
rotate within the pool before the caller jumps to cross-provider fallback.
429/402 rotation already works; only the overloaded/server_error family was
missing from recover_with_credential_pool's reason dispatch.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from agent.agent_runtime_helpers import recover_with_credential_pool
from agent.credential_pool import STATUS_EXHAUSTED, CredentialPool, PooledCredential
from agent.error_classifier import FailoverReason

_BASE_URL = "https://aggregator.example/v1"


def _entry(i: int) -> PooledCredential:
    return PooledCredential(
        provider="agg", id=f"p{i + 1}", label=f"key-{i + 1}", auth_type="api_key",
        priority=i, source="manual", access_token=f"tok-{i + 1}-1234567890",
        base_url=_BASE_URL,
    )


class _Agent:
    log_prefix = ""
    quiet_mode = True
    api_mode = "chat_completions"
    provider = "agg"
    model = "m"
    base_url = _BASE_URL
    _credential_pool_revert_id = None

    def __init__(self, pool: CredentialPool) -> None:
        self._credential_pool = pool
        self.api_key = pool.select().access_token
        self.swapped_to: list = []

    def _swap_credential(self, entry):
        self.swapped_to.append(entry.id)
        self.api_key = entry.access_token
        return True

    def __getattr__(self, name):
        return lambda *args, **kwargs: None


def _recover(agent, status_code, reason):
    return recover_with_credential_pool(
        agent, status_code=status_code, has_retried_429=False,
        classified_reason=reason, error_context=None,
    )


def test_503_overloaded_rotates_to_next_healthy_key():
    pool = CredentialPool("agg", [_entry(0), _entry(1)])
    agent = _Agent(pool)

    recovered, retried = _recover(agent, 503, FailoverReason.overloaded)

    assert (recovered, retried) == (True, False)
    assert agent.swapped_to == ["p2"] and agent.api_key == "tok-2-1234567890"
    benched = next(e for e in pool.entries() if e.id == "p1")
    assert benched.last_status == STATUS_EXHAUSTED
    assert benched.last_error_code == 503
    assert benched.extra.get("failure_reason") == "overloaded"


def test_500_server_error_rotates_to_next_healthy_key():
    pool = CredentialPool("agg", [_entry(0), _entry(1)])
    agent = _Agent(pool)

    recovered, retried = _recover(agent, 500, FailoverReason.server_error)

    assert (recovered, retried) == (True, False)
    assert agent.swapped_to == ["p2"] and agent.api_key == "tok-2-1234567890"
    benched = next(e for e in pool.entries() if e.id == "p1")
    assert benched.last_status == STATUS_EXHAUSTED
    assert benched.last_error_code == 500
    assert benched.extra.get("failure_reason") == "server_error"
    # 5xx rotation must not arm the per-turn revert hook (gated to 429/402).
    assert agent._credential_pool_revert_id is None


def test_exhausted_pool_returns_false_on_5xx():
    pool = CredentialPool("agg", [_entry(0)])
    agent = _Agent(pool)
    pool.mark_exhausted_and_rotate(
        status_code=503, credential_id="p1", failure_reason="overloaded",
    )
    agent.swapped_to.clear()

    recovered, retried = _recover(agent, 503, FailoverReason.overloaded)

    assert (recovered, retried) == (False, False)
    assert agent.swapped_to == []
