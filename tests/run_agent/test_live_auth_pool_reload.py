"""Long-lived agents must observe credentials added by another process."""

from __future__ import annotations

from dataclasses import dataclass

from agent.agent_runtime_helpers import recover_with_credential_pool
from agent.credential_pool import (
    AUTH_TYPE_API_KEY,
    STATUS_DEAD,
    CredentialPool,
    PooledCredential,
)
from agent.error_classifier import FailoverReason


@dataclass(frozen=True)
class _Entry:
    id: str
    runtime_api_key: str

    @property
    def access_token(self) -> str:
        return self.runtime_api_key


class _Pool:
    def __init__(self, entries, *, provider="openai-codex"):
        self.provider = provider
        self._entries = list(entries)
        self.refresh_calls = 0
        self.mark_calls = []

    def entries(self):
        return list(self._entries)

    def has_available_alternative(self, api_key_hint):
        return any(entry.runtime_api_key != api_key_hint for entry in self._entries)

    def mark_exhausted_and_rotate(
        self,
        *,
        status_code,
        error_context=None,
        api_key_hint=None,
    ):
        self.mark_calls.append((status_code, error_context, api_key_hint))
        if api_key_hint is None:
            return None
        return next(
            (
                entry
                for entry in self._entries
                if entry.runtime_api_key != api_key_hint
            ),
            None,
        )

    def select(self):
        return self._entries[0] if self._entries else None

    def try_refresh_current(self):
        self.refresh_calls += 1
        return None


class _Agent:
    provider = "openai-codex"
    api_mode = "codex_responses"
    base_url = "https://chatgpt.com/backend-api/codex"

    def __init__(self, pool, api_key="stale-token"):
        self._credential_pool = pool
        self.api_key = api_key
        self.swapped_to = None

    def _is_entitlement_failure(self, _error_context, _status_code):
        return False

    def _swap_credential(self, entry):
        self.swapped_to = entry
        self.api_key = entry.runtime_api_key


def _pooled(entry_id, token, *, priority, last_status=None):
    return PooledCredential(
        provider="openai-codex",
        id=entry_id,
        label=entry_id,
        auth_type=AUTH_TYPE_API_KEY,
        priority=priority,
        source="manual",
        access_token=token,
        last_status=last_status,
    )


def test_pool_alternative_check_uses_exact_identity_and_skips_dead_entries():
    pool = CredentialPool(
        "openai-codex",
        [
            _pooled("current", "current-token", priority=0),
            _pooled(
                "dead-alternative",
                "dead-token",
                priority=1,
                last_status=STATUS_DEAD,
            ),
            _pooled("live-alternative", "live-token", priority=2),
        ],
    )

    assert pool.has_available_alternative("current-token") is True
    assert pool.has_available_alternative("live-token") is True

    dead_only = CredentialPool(
        "openai-codex",
        [
            _pooled("current", "current-token", priority=0),
            _pooled("dead", "dead-token", priority=1, last_status=STATUS_DEAD),
        ],
    )
    assert dead_only.has_available_alternative("current-token") is False


def test_exact_api_key_hint_never_marks_a_different_current_entry():
    pool = CredentialPool(
        "openai-codex",
        [
            _pooled("current", "current-token", priority=0),
            _pooled("alternative", "alternative-token", priority=1),
        ],
    )

    assert (
        pool.mark_exhausted_and_rotate(
            status_code=401,
            error_context={"reason": "invalid_token"},
            api_key_hint="missing-token",
        )
        is None
    )
    assert all(entry.last_status is None for entry in pool.entries())


def test_existing_agent_adopts_credential_added_to_auth_store(monkeypatch):
    stale = _Entry("old", "stale-token")
    fresh = _Entry("new", "fresh-token")
    cached_pool = _Pool([stale])
    disk_pool = _Pool([stale, fresh])
    agent = _Agent(cached_pool)

    monkeypatch.setattr("agent.credential_pool.load_pool", lambda provider: disk_pool)

    recovered, retried = recover_with_credential_pool(
        agent,
        status_code=401,
        has_retried_429=False,
        classified_reason=FailoverReason.auth,
        error_context={"reason": "invalid_token"},
    )

    assert recovered is True
    assert retried is False
    assert agent._credential_pool is disk_pool
    assert agent.swapped_to is fresh
    assert cached_pool.refresh_calls == 0
    assert disk_pool.mark_calls == [
        (401, {"reason": "invalid_token"}, "stale-token")
    ]


def test_unchanged_disk_pool_keeps_existing_refresh_path(monkeypatch):
    stale = _Entry("old", "stale-token")
    cached_pool = _Pool([stale])
    same_disk_pool = _Pool([stale])
    agent = _Agent(cached_pool)

    monkeypatch.setattr(
        "agent.credential_pool.load_pool",
        lambda provider: same_disk_pool,
    )

    recovered, retried = recover_with_credential_pool(
        agent,
        status_code=401,
        has_retried_429=False,
        classified_reason=FailoverReason.auth,
        error_context={"reason": "invalid_token"},
    )

    assert recovered is False
    assert retried is False
    assert agent._credential_pool is cached_pool
    assert cached_pool.refresh_calls == 1
    assert same_disk_pool.mark_calls == []


def test_removed_failed_credential_adopts_disk_authoritative_replacement(monkeypatch):
    stale = _Entry("old", "stale-token")
    replacement = _Entry("replacement", "fresh-token")
    cached_pool = _Pool([stale])
    disk_pool = _Pool([replacement])
    agent = _Agent(cached_pool)

    monkeypatch.setattr("agent.credential_pool.load_pool", lambda provider: disk_pool)

    recovered, _ = recover_with_credential_pool(
        agent,
        status_code=401,
        has_retried_429=False,
        classified_reason=FailoverReason.auth,
        error_context={"reason": "invalid_token"},
    )

    assert recovered is True
    assert agent._credential_pool is disk_pool
    assert agent.swapped_to is replacement
    assert disk_pool.mark_calls == []


def test_pool_reload_rotation_requires_exact_http_auth_status(monkeypatch):
    stale = _Entry("old", "stale-token")
    fresh = _Entry("new", "fresh-token")
    cached_pool = _Pool([stale])
    disk_pool = _Pool([stale, fresh])
    agent = _Agent(cached_pool)

    monkeypatch.setattr("agent.credential_pool.load_pool", lambda provider: disk_pool)

    recovered, _ = recover_with_credential_pool(
        agent,
        status_code=None,
        has_retried_429=False,
        classified_reason=FailoverReason.auth,
        error_context={"reason": "auth-shaped transport error"},
    )

    assert recovered is False
    assert agent._credential_pool is cached_pool
    assert agent.swapped_to is None
    assert disk_pool.mark_calls == []
