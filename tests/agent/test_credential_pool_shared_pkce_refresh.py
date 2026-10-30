"""Cross-profile shared Hermes PKCE refresh and recovery behavior."""

import json
import threading

import pytest

from agent.credential_pool import AUTH_TYPE_OAUTH, CredentialPool, PooledCredential


@pytest.fixture
def shared_homes(tmp_path, monkeypatch):
    default_root = tmp_path / "default"
    profile_root = default_root / "profiles" / "profile-b"
    default_root.mkdir()
    profile_root.mkdir(parents=True)
    for root in (default_root, profile_root):
        (root / "auth.json").write_text(json.dumps({"providers": {}}), encoding="utf-8")

    monkeypatch.setattr("agent.anthropic_adapter.get_default_hermes_root", lambda: default_root)
    monkeypatch.setattr("agent.credential_pool.get_default_hermes_root", lambda: default_root)
    monkeypatch.setenv("HERMES_HOME", str(default_root))
    return default_root, profile_root


def _entry(label="team", access="old-access", refresh="old-refresh"):
    return PooledCredential(
        provider="anthropic",
        id=f"id-{label}",
        label=label,
        auth_type=AUTH_TYPE_OAUTH,
        priority=0,
        source="hermes_pkce_shared",
        access_token=access,
        refresh_token=refresh,
    )


def _set_home(monkeypatch, root):
    monkeypatch.setenv("HERMES_HOME", str(root))


def test_profile_a_refresh_visible_to_profile_b_select(shared_homes, monkeypatch):
    default_root, profile_root = shared_homes
    from agent.anthropic_adapter import (
        read_shared_hermes_oauth_credentials,
        write_shared_hermes_oauth_credentials,
    )

    _set_home(monkeypatch, default_root)
    write_shared_hermes_oauth_credentials(
        {"accessToken": "old-access", "refreshToken": "old-refresh"}, label="team"
    )
    pool_a = CredentialPool("anthropic", [_entry()])

    refresh_calls = []

    def refresh(refresh_token, **_kwargs):
        refresh_calls.append(refresh_token)
        return {
            "access_token": "new-access",
            "refresh_token": "new-refresh",
            "expires_at_ms": 4_000_000_000_000,
        }

    monkeypatch.setattr("agent.anthropic_adapter.refresh_anthropic_oauth_pure", refresh)
    refreshed = pool_a._refresh_entry(pool_a.entries()[0], force=True)

    assert refreshed is not None
    published = read_shared_hermes_oauth_credentials("team")
    assert published is not None
    assert published["accessToken"] == "new-access"

    _set_home(monkeypatch, profile_root)
    pool_b = CredentialPool("anthropic", [_entry()])
    selected = pool_b.select()

    assert selected is not None
    assert selected.access_token == "new-access"
    assert selected.refresh_token == "new-refresh"
    assert refresh_calls == ["old-refresh"]


def test_revoked_refresh_token_syncs_from_shared_then_retries_once(shared_homes, monkeypatch):
    default_root, profile_root = shared_homes
    from agent.anthropic_adapter import write_shared_hermes_oauth_credentials

    _set_home(monkeypatch, default_root)
    write_shared_hermes_oauth_credentials(
        {"accessToken": "winner-access", "refreshToken": "winner-refresh"}, label="team"
    )
    _set_home(monkeypatch, profile_root)
    pool_b = CredentialPool("anthropic", [_entry()])
    calls = []

    def refresh(refresh_token, **_kwargs):
        calls.append(refresh_token)
        if len(calls) == 1:
            raise RuntimeError("404 invalid_grant")
        return {
            "access_token": "final-access",
            "refresh_token": "final-refresh",
            "expires_at_ms": 456,
        }

    monkeypatch.setattr("agent.anthropic_adapter.refresh_anthropic_oauth_pure", refresh)
    result = pool_b._refresh_entry_impl(pool_b.entries()[0], force=True)

    assert result is not None
    assert result.access_token == "final-access"
    assert result.refresh_token == "final-refresh"
    assert calls == ["old-refresh", "winner-refresh"]
    assert pool_b.entries()[0].access_token == "final-access"
    assert pool_b.entries()[0].refresh_token == "final-refresh"


def test_concurrent_refresh_loser_recovers_via_sync_without_rerefresh(shared_homes, monkeypatch):
    default_root, profile_root = shared_homes
    from agent.anthropic_adapter import write_shared_hermes_oauth_credentials

    _set_home(monkeypatch, default_root)
    write_shared_hermes_oauth_credentials(
        {"accessToken": "old-access", "refreshToken": "old-refresh"}, label="team"
    )
    pool_a = CredentialPool("anthropic", [_entry()])
    _set_home(monkeypatch, profile_root)
    pool_b = CredentialPool("anthropic", [_entry()])

    first_call_started = threading.Event()
    release_first_call = threading.Event()
    calls = []
    calls_lock = threading.Lock()

    def refresh(refresh_token, **_kwargs):
        with calls_lock:
            calls.append(refresh_token)
            call_number = len(calls)
        if call_number == 1:
            first_call_started.set()
            assert release_first_call.wait(timeout=5)
            return {
                "access_token": "winner-access",
                "refresh_token": "winner-refresh",
                "expires_at_ms": 789,
            }
        raise AssertionError("loser must sync the winner's grant instead of refreshing")

    monkeypatch.setattr("agent.anthropic_adapter.refresh_anthropic_oauth_pure", refresh)
    results = {}

    def run(name, pool):
        results[name] = pool._refresh_entry(pool.entries()[0], force=True)

    winner = threading.Thread(target=run, args=("winner", pool_a))
    loser = threading.Thread(target=run, args=("loser", pool_b))
    winner.start()
    loser_started = False
    try:
        assert first_call_started.wait(timeout=5)
        loser.start()
        loser_started = True
        release_first_call.set()
    finally:
        release_first_call.set()
        winner.join(timeout=5)
        if loser_started:
            loser.join(timeout=5)

    assert not winner.is_alive()
    assert not loser.is_alive()
    assert len(calls) == 1
    assert calls == ["old-refresh"]
    assert results["winner"].access_token == "winner-access"
    assert results["loser"].access_token == "winner-access"
    assert pool_b.entries()[0].refresh_token == "winner-refresh"
