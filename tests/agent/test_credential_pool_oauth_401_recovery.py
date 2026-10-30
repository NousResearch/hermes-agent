"""Anthropic OAuth 401 recovery for pooled sessions, including children that share one pool object.

delegate_task children dispatched in one coordinator turn share that turn's ``CredentialPool``, and a
child runs one long turn. Three defects turned an ordinary token expiry into dead sessions:

* On a 401, ``try_refresh_matching(credential_id=...)`` force-POSTed a refresh even when a sibling had
  already rotated the entry, so the failed key was no longer the entry's key. Every POST revokes the
  access token the siblings now hold, and the siblings ping-pong "revoked" 401s.
* The refresh cap (``_MAX_AUTH_REFRESH_ATTEMPTS``) resets only at turn start, so a long child turn ended at
  its third 401 even when the refreshed keys had authenticated in between (the account answered 429s).
* After the cap, recovery returned False without rotating, although another row had room.
"""

from __future__ import annotations

import time

import pytest

from agent.agent_runtime_helpers import recover_with_credential_pool
from agent.credential_pool import AUTH_TYPE_OAUTH, STATUS_EXHAUSTED, CredentialPool, PooledCredential

_API = "https://api.anthropic.com"
_EIGHT_HOURS_MS = 8 * 3600 * 1000
_RATE_LIMIT = {"message": "This request would exceed your account's rate limit. Please try again later."}


@pytest.fixture(autouse=True)
def _fake_pool_store(monkeypatch):
    """Back the pool's auth.json reads/writes with a dict (the shared on-disk store)."""
    store: dict[str, list] = {}

    def _write(provider, entries, *, removed_ids=None, status_cleared_ids=None, token_bases=None):
        store[provider] = list(entries)

    def _read(provider=None):
        return dict(store) if provider is None else list(store.get(provider, []))

    monkeypatch.setattr("agent.credential_pool.write_credential_pool", _write)
    monkeypatch.setattr("agent.credential_pool.read_credential_pool", _read)
    monkeypatch.setattr("agent.anthropic_credentials.read_claude_code_credentials", lambda: None)
    return store


class _TokenEndpoint:
    """Anthropic's OAuth token endpoint: a refresh rotates the pair, and the API then accepts only the
    newest access token of that login (the previous one is revoked, whoever holds it)."""

    def __init__(self):
        self.posts: list[str] = []
        self.live: dict[str, str] = {}

    def refresh(self, refresh_token, *, use_json=False):
        self.posts.append(refresh_token)
        login, rotation = refresh_token.rsplit("-rt-", 1)
        rotation = int(rotation) + 1
        self.live[login] = f"{login}-at-{rotation}"
        return {
            "access_token": self.live[login],
            "refresh_token": f"{login}-rt-{rotation}",
            "expires_at_ms": int(time.time() * 1000) + _EIGHT_HOURS_MS,
        }

    def accepts(self, access_token):
        return access_token in self.live.values()


@pytest.fixture
def endpoint(monkeypatch):
    server = _TokenEndpoint()
    monkeypatch.setattr("agent.anthropic_credentials.refresh_anthropic_oauth_pure", server.refresh)
    return server


def _row(eid, priority=0, *, access=None, refresh=None, expires_in_ms=_EIGHT_HOURS_MS):
    return PooledCredential(
        provider="anthropic", id=eid, label=eid, auth_type=AUTH_TYPE_OAUTH, priority=priority,
        source="manual:hermes_pkce", access_token=access or f"{eid}-at-0", refresh_token=refresh or f"{eid}-rt-0",
        expires_at_ms=int(time.time() * 1000) + expires_in_ms, base_url=_API,
    )


class _Child:
    """A delegated session leasing one row of a (possibly shared) pool; real pool, real recovery helper."""

    provider = "anthropic"
    model = "claude-opus-4-6"
    base_url = _API
    _fallback_activated = False

    def __init__(self, pool, entry_id):
        self._credential_pool = pool
        self._credential_pool_entry_id = entry_id
        self.api_key = next(e for e in pool.entries() if e.id == entry_id).runtime_api_key

    def _swap_credential(self, entry):
        self.api_key = entry.runtime_api_key
        self._credential_pool_entry_id = entry.id
        return True

    def _is_entitlement_failure(self, error_context, status_code):
        return False


def _on_401(agent):
    return recover_with_credential_pool(
        agent, status_code=401, has_retried_429=False,
        error_context={"message": "OAuth access token has expired"},
    )


def _status(pool, eid):
    return next(e for e in pool.entries() if e.id == eid).last_status


class TestRotatedEntryIsNotRefreshedAgain:
    def test_stale_hint_returns_the_rotated_entry_without_a_refresh_post(self, endpoint):
        pool = CredentialPool("anthropic", [_row("claude-3", access="claude-3-at-1", refresh="claude-3-rt-1")])

        got = pool.try_refresh_matching(api_key_hint="claude-3-at-0", credential_id="claude-3")

        assert endpoint.posts == []
        assert (got.id, got.access_token) == ("claude-3", "claude-3-at-1")

    def test_the_entrys_own_failed_key_is_still_refreshed(self, endpoint):
        pool = CredentialPool("anthropic", [_row("claude-3")])

        got = pool.try_refresh_matching(api_key_hint="claude-3-at-0", credential_id="claude-3")

        assert endpoint.posts == ["claude-3-rt-0"]
        assert got.access_token == "claude-3-at-1"

    def test_a_rotated_entry_that_is_itself_expiring_is_refreshed(self, endpoint):
        pool = CredentialPool(
            "anthropic", [_row("claude-3", access="claude-3-at-1", refresh="claude-3-rt-1", expires_in_ms=30_000)],
        )

        got = pool.try_refresh_matching(api_key_hint="claude-3-at-0", credential_id="claude-3")

        assert endpoint.posts == ["claude-3-rt-1"]
        assert got.access_token == "claude-3-at-2"

    def test_siblings_sharing_a_pool_do_not_revoke_each_other(self, endpoint):
        pool = CredentialPool("anthropic", [_row("claude-3")])
        first, second = _Child(pool, "claude-3"), _Child(pool, "claude-3")

        assert _on_401(first) == (True, False)
        # The sibling still sends the pre-rotation token: it adopts the rotation instead of POSTing again.
        assert _on_401(second) == (True, False)

        assert endpoint.posts == ["claude-3-rt-0"]
        assert first.api_key == second.api_key == "claude-3-at-1"
        assert endpoint.accepts(first.api_key)


class TestRefreshCap:
    def test_after_the_cap_the_session_rotates_to_another_healthy_row(self, endpoint):
        pool = CredentialPool("anthropic", [_row("claude-3", 0), _row("claude-4", 1)])
        child = _Child(pool, "claude-3")
        assert _on_401(child)[0] is True
        assert _on_401(child)[0] is True

        recovered, _ = _on_401(child)  # auth still failing after two refreshes

        assert recovered is True
        assert (child._credential_pool_entry_id, child.api_key) == ("claude-4", "claude-4-at-0")
        assert _status(pool, "claude-3") == STATUS_EXHAUSTED
        # No refresh past the cap: it would only revoke the token the row's other holders have.
        assert endpoint.posts == ["claude-3-rt-0", "claude-3-rt-1"]

    def test_a_429_restarts_the_cap(self, endpoint):
        pool = CredentialPool("anthropic", [_row("claude-3", 0), _row("claude-4", 1)])
        child = _Child(pool, "claude-3")
        assert _on_401(child)[0] is True
        assert _on_401(child)[0] is True
        # The refreshed key authenticated: the account answers 429, and the session waits on the same row.
        assert recover_with_credential_pool(
            child, status_code=429, has_retried_429=False, error_context=_RATE_LIMIT,
        ) == (False, True)

        recovered, _ = _on_401(child)  # hours later, the access token expires again

        assert recovered is True
        assert (child._credential_pool_entry_id, child.api_key) == ("claude-3", "claude-3-at-3")
        assert endpoint.posts == ["claude-3-rt-0", "claude-3-rt-1", "claude-3-rt-2"]
        assert _status(pool, "claude-3") != STATUS_EXHAUSTED
