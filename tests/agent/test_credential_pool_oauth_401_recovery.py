from __future__ import annotations

import itertools
import time

import pytest

from agent.agent_runtime_helpers import recover_with_credential_pool
from agent.credential_pool import AUTH_TYPE_OAUTH, CredentialPool, PooledCredential

_API = "https://api.anthropic.com"
_EIGHT_HOURS_MS = 8 * 3600 * 1000


@pytest.fixture(autouse=True)
def _no_claude_code_credentials(monkeypatch):
    monkeypatch.setattr("agent.anthropic_credentials.read_claude_code_credentials", lambda: None)


class _TokenEndpoint:
    def __init__(self):
        self.posts: list[str] = []
        self.live: dict[str, str] = {}
        self.ttl_ms = _EIGHT_HOURS_MS

    def refresh(self, refresh_token, *, use_json=False):
        self.posts.append(refresh_token)
        login, rotation = refresh_token.rsplit("-rt-", 1)
        rotation = int(rotation) + 1
        self.live[login] = f"{login}-at-{rotation}"
        return {
            "access_token": self.live[login],
            "refresh_token": f"{login}-rt-{rotation}",
            "expires_at_ms": int(time.time() * 1000) + self.ttl_ms,
        }

    def accepts(self, access_token):
        return access_token in self.live.values()


@pytest.fixture
def endpoint(monkeypatch):
    server = _TokenEndpoint()
    monkeypatch.setattr("agent.anthropic_credentials.refresh_anthropic_oauth_pure", server.refresh)
    return server


def _row(eid, *, access=None, refresh=None, expires_in_ms=_EIGHT_HOURS_MS):
    return PooledCredential(
        provider="anthropic", id=eid, label=eid, auth_type=AUTH_TYPE_OAUTH, priority=0,
        source="manual:hermes_pkce", access_token=f"{eid}-at-0" if access is None else access,
        refresh_token=refresh or f"{eid}-rt-0", expires_at_ms=int(time.time() * 1000) + expires_in_ms,
        base_url=_API,
    )


class _DelegatedChild:
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


def _reload(fake_pool_store):
    persisted = next(row for row in fake_pool_store["anthropic"] if row["id"] == "claude-3")
    return CredentialPool("anthropic", [PooledCredential.from_dict("anthropic", persisted)])


class TestStale401AfterRotation:
    def test_stale_hint_adopts_the_rotated_token_without_a_post(self, endpoint, fake_pool_store):
        pool = CredentialPool("anthropic", [_row("claude-3", access="claude-3-at-1", refresh="claude-3-rt-1")])

        got = pool.try_refresh_matching(api_key_hint="claude-3-at-0", credential_id="claude-3")

        assert endpoint.posts == []
        assert (got.id, got.access_token) == ("claude-3", "claude-3-at-1")

    def test_the_failed_current_token_is_refreshed(self, endpoint, fake_pool_store):
        pool = CredentialPool("anthropic", [_row("claude-3")])

        got = pool.try_refresh_matching(api_key_hint="claude-3-at-0", credential_id="claude-3")

        assert endpoint.posts == ["claude-3-rt-0"]
        assert got.access_token == "claude-3-at-1"

    def test_a_rotated_token_that_is_expiring_is_refreshed(self, endpoint, fake_pool_store):
        pool = CredentialPool(
            "anthropic", [_row("claude-3", access="claude-3-at-1", refresh="claude-3-rt-1", expires_in_ms=30_000)],
        )

        got = pool.try_refresh_matching(api_key_hint="claude-3-at-0", credential_id="claude-3")

        assert endpoint.posts == ["claude-3-rt-1"]
        assert got.access_token == "claude-3-at-2"

    def test_a_row_without_a_bearer_is_refreshed_not_adopted(self, endpoint, fake_pool_store):
        pool = CredentialPool("anthropic", [_row("claude-3", access="", refresh="claude-3-rt-1")])

        got = pool.try_refresh_matching(api_key_hint="claude-3-at-0", credential_id="claude-3")

        assert endpoint.posts == ["claude-3-rt-1"]
        assert got.access_token == "claude-3-at-2"

    def test_siblings_sharing_a_pool_do_not_revoke_each_other(self, endpoint, fake_pool_store):
        pool = CredentialPool("anthropic", [_row("claude-3")])
        first, second = _DelegatedChild(pool, "claude-3"), _DelegatedChild(pool, "claude-3")

        assert _on_401(first) == (True, False)
        assert _on_401(second) == (True, False)

        assert endpoint.posts == ["claude-3-rt-0"]
        assert first.api_key == second.api_key == "claude-3-at-1"
        assert endpoint.accepts(first.api_key)

    def test_a_reloaded_pool_adopts_the_token_rotated_under_its_inflight_request(self, endpoint, fake_pool_store):
        first = _DelegatedChild(CredentialPool("anthropic", [_row("claude-3")]), "claude-3")
        stale_key = first.api_key
        assert _on_401(first) == (True, False)
        second = _DelegatedChild(_reload(fake_pool_store), "claude-3")
        second.api_key = stale_key

        assert _on_401(second) == (True, False)

        assert second.api_key == first.api_key == "claude-3-at-1"
        assert endpoint.accepts(second.api_key)
        assert endpoint.posts == ["claude-3-rt-0"]


@pytest.mark.parametrize(
    "topology,sent,current_expiring",
    list(itertools.product(("shared_pool", "reloaded_pool"), ("stale_gen0", "current_gen1", "no_hint"), (False, True))),
)
def test_a_second_401_refreshes_only_when_its_own_token_is_due(
    endpoint, fake_pool_store, topology, sent, current_expiring,
):
    if current_expiring:
        endpoint.ttl_ms = 30_000
    first_pool = CredentialPool("anthropic", [_row("claude-3")])
    assert _on_401(_DelegatedChild(first_pool, "claude-3")) == (True, False)
    assert endpoint.posts == ["claude-3-rt-0"]
    pool = first_pool if topology == "shared_pool" else _reload(fake_pool_store)

    if sent == "no_hint":
        got = pool.try_refresh_matching(credential_id="claude-3")
    else:
        second = _DelegatedChild(pool, "claude-3")
        second.api_key = "claude-3-at-0" if sent == "stale_gen0" else "claude-3-at-1"
        assert _on_401(second) == (True, False)
        got = next(e for e in pool.entries() if e.id == "claude-3")
        assert second.api_key == got.access_token

    must_refresh = sent != "stale_gen0" or current_expiring
    assert endpoint.posts == (["claude-3-rt-0", "claude-3-rt-1"] if must_refresh else ["claude-3-rt-0"])
    assert got.access_token == ("claude-3-at-2" if must_refresh else "claude-3-at-1")
    assert endpoint.accepts(got.access_token)
