"""Concurrent Nous 401 recovery must not stampede the shared OAuth grant.

Sep 2 2026 incident: ~120 subagent processes shared one Nous OAuth pool entry
whose access token hit its hourly expiry. Every process got a 401, every
process force-refreshed, and each rotation invalidated the token a sibling had
just adopted — 81 refreshes and ~540 401s in eight minutes. Processes that
lost the auth-store flock race had their only entry benched ("matched no nous
entry ... pool size 0") and surfaced the 401 to the user as "out of funds".

Two invariants pinned here:

1. ``resolve_nous_runtime_credentials(force_refresh=True, stale_access_token=X)``
   does NOT POST a refresh when the store already holds a usable token that
   is not X — a peer already rotated; adopt it.
2. A lock-timeout during a pool-level Nous refresh leaves the entry
   untouched instead of marking it exhausted.
"""

import json
import logging

import hermes_cli.auth as auth_mod
from agent.credential_pool import CredentialPool, PooledCredential

from tests.hermes_cli.test_auth_nous_provider import _invoke_jwt, _setup_nous_auth


def test_forced_refresh_adopts_peer_rotation_instead_of_reposting(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes"
    peer_token = _invoke_jwt(seconds=3600)
    failed_token = _invoke_jwt(seconds=3000)  # what THIS process still holds
    _setup_nous_auth(
        hermes_home,
        access_token=peer_token,
        refresh_token="rt-after-peer-rotation",
        scope=auth_mod.DEFAULT_NOUS_SCOPE,
        expires_at=auth_mod.datetime.fromtimestamp(
            auth_mod.time.time() + 3600, tz=auth_mod.timezone.utc
        ).isoformat(),
        expires_in=3600,
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    posts = []

    def _fake_refresh_access_token(*, client, portal_base_url, client_id, refresh_token):
        posts.append(refresh_token)
        return {
            "access_token": _invoke_jwt(seconds=7200),
            "refresh_token": "rt-should-not-happen",
            "expires_in": 7200,
            "token_type": "Bearer",
            "scope": auth_mod.DEFAULT_NOUS_SCOPE,
        }

    monkeypatch.setattr(auth_mod, "_refresh_access_token", _fake_refresh_access_token)

    creds = auth_mod.resolve_nous_runtime_credentials(
        force_refresh=True, stale_access_token=failed_token
    )

    assert posts == [], "peer already rotated — must not consume the refresh token again"
    assert creds["api_key"] == peer_token

    # Same call WITHOUT the hint keeps the pre-existing force semantics.
    auth_mod.resolve_nous_runtime_credentials(force_refresh=True)
    assert posts == ["rt-after-peer-rotation"]


def test_lock_timeout_during_nous_refresh_does_not_bench_entry(monkeypatch, caplog):
    entry = PooledCredential(
        id="267aed",
        provider="nous",
        auth_type="oauth",
        access_token=_invoke_jwt(seconds=3600),
        refresh_token="rt",
        label="test@nous",
        source="device_code",
        priority=0,
    )
    pool = CredentialPool.__new__(CredentialPool)
    pool._lock = __import__("threading").RLock()
    pool._entries = [entry]
    pool._active_leases = {}
    pool._current_id = None
    pool._max_concurrent = 2
    pool._unmatched_rotation_streak = 0
    pool.provider = "nous"

    monkeypatch.setattr(pool, "_sync_nous_entry_from_auth_store", lambda e: e)
    monkeypatch.setattr(pool, "_persist", lambda *a, **k: None)

    def _busy(*a, **k):
        raise TimeoutError("Timed out waiting for auth store lock")

    monkeypatch.setattr(auth_mod, "resolve_nous_runtime_credentials", _busy)

    result = pool._refresh_entry_impl(entry, force=True)

    assert result is entry
    assert pool._entries[0].last_status is None, "lock contention is not a credential failure"


def test_agent_401_refresh_passes_failed_bearer_as_stale_hint(monkeypatch):
    """Every 401-recovery caller must hand the auth store the bearer that
    failed — without it ``_already_rotated_by_peer`` can never fire and each
    subagent rotates the shared grant again (the "no crash, N refreshes"
    variant of the Sep 2 stampede).
    """
    from run_agent import AIAgent

    agent = AIAgent.__new__(AIAgent)
    agent.provider = "nous"
    agent.api_mode = "chat_completions"
    agent.api_key = "jwt-that-just-401d"
    agent.base_url = "https://inference-api.nousresearch.com/v1"
    agent._client_kwargs = {}
    monkeypatch.setattr(agent, "_replace_primary_openai_client", lambda **k: True)

    seen = {}

    def _fake_resolve(**kwargs):
        seen.update(kwargs)
        return {"api_key": "fresh", "base_url": agent.base_url}

    monkeypatch.setattr(auth_mod, "resolve_nous_runtime_credentials", _fake_resolve)

    assert agent._try_refresh_nous_client_credentials(force=True) is True
    assert seen["force_refresh"] is True
    assert seen["stale_access_token"] == "jwt-that-just-401d"
