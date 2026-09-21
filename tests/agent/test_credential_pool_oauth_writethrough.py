"""Credential-pool OAuth refresh persistence.

Single-use refresh tokens (Codex, xAI, Anthropic PKCE) must be POSTed under the cross-process
auth lock and the rotated pair committed back to the store/singleton that seeds the pool, or the
next ``load_pool()`` re-seeds the consumed pair over the rotated one. Real on-disk stores under
``tmp_path``; only the token endpoint is faked.
"""

import json
import time

from agent import credential_pool as CP
from agent.credential_pool import (
    AUTH_TYPE_OAUTH,
    CredentialPool,
    PooledCredential,
    load_pool,
)
from hermes_cli import auth as A
import hermes_cli.auth_codex as auth_codex


def _write_store(path, store):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(store), encoding="utf-8")


def _entry(provider: str, *, id: str, access_token: str, refresh_token: str):
    return PooledCredential(
        provider=provider,
        id=id,
        label="cred",
        auth_type=AUTH_TYPE_OAUTH,
        priority=0,
        source="device_code",
        access_token=access_token,
        refresh_token=refresh_token,
    )


def test_codex_pool_refresh_holds_auth_store_lock_across_post(monkeypatch, tmp_path):
    """The Codex OAuth pool refresh must POST under the cross-process auth lock.

    Codex refresh tokens are single-use. If two Hermes processes both read the
    same on-disk token and both POST it, the loser gets ``refresh_token_reused``.
    Serializing the sync -> refresh POST -> write-back sequence through the
    shared ``_auth_store_lock`` closes that window: a second process blocks on
    the flock and, once inside, adopts the rotated token instead of re-POSTing.

    This asserts the invariant directly — that ``refresh_codex_oauth_pure`` is
    only ever called while the auth-store lock is held — rather than snapshotting
    any token value.
    """
    provider = "openai-codex"
    profile_path = tmp_path / "auth.json"
    monkeypatch.setattr(A, "_auth_file_path", lambda: profile_path)

    lock_held: dict = {"during_post": None}
    real_lock = A._auth_store_lock

    depth = {"n": 0}

    import contextlib

    @contextlib.contextmanager
    def tracking_lock(*args, **kwargs):
        depth["n"] += 1
        try:
            with real_lock(*args, **kwargs):
                yield
        finally:
            depth["n"] -= 1

    monkeypatch.setattr(A, "_auth_store_lock", tracking_lock)
    # credential_pool imported _auth_store_lock by name; patch that binding too.
    monkeypatch.setattr(CP, "_auth_store_lock", tracking_lock)

    def fake_refresh(access_token, refresh_token, **kwargs):
        # The POST to the token endpoint must happen with the lock held.
        lock_held["during_post"] = depth["n"] > 0
        return {
            "access_token": "rotated-access",
            "refresh_token": "rotated-refresh",
            "last_refresh": "2020-01-02T00:00:00Z",
        }

    monkeypatch.setattr(A, "refresh_codex_oauth_pure", fake_refresh)
    monkeypatch.setattr(auth_codex, "refresh_codex_oauth_pure", fake_refresh)

    entry = _entry(
        provider,
        id="codex-1",
        access_token="stale-access",
        refresh_token="stale-refresh",
    )
    pool = CredentialPool(provider, [entry])

    refreshed = pool._refresh_entry(entry, force=True)

    assert refreshed is not None
    assert refreshed.access_token == "rotated-access"
    assert refreshed.refresh_token == "rotated-refresh"
    # The invariant: the single-use token POST ran inside the auth-store lock.
    assert lock_held["during_post"] is True


def test_hermes_pkce_refresh_writes_back_to_singleton(tmp_path, monkeypatch):
    """A successful hermes_pkce refresh must update
    ~/.hermes/.anthropic_oauth.json, or ``_seed_from_singletons()`` on the
    next ``load_pool()`` re-seeds the pre-refresh (already-consumed,
    single-use) token pair over the freshly rotated one.
    """
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setattr("hermes_cli.auth.is_provider_explicitly_configured", lambda pid: True)

    oauth_file = hermes_home / ".anthropic_oauth.json"
    oauth_file.write_text(
        json.dumps({"accessToken": "sk-ant-oat-rt0", "refreshToken": "rt0", "expiresAt": 0}),
        encoding="utf-8",
    )
    _write_store(hermes_home / "auth.json", {"version": 1, "providers": {}})

    monkeypatch.setattr(
        "agent.anthropic_credentials.refresh_anthropic_oauth_pure",
        lambda refresh_token, use_json=False: {
            "access_token": "sk-ant-oat-rt1",
            "refresh_token": "rt1",
            "expires_at_ms": int(time.time() * 1000) + 3_600_000,
        },
    )
    monkeypatch.setattr("agent.anthropic_credentials.read_claude_code_credentials", lambda: None)

    entry = PooledCredential(
        provider="anthropic",
        id="pool-entry",
        label="cred",
        auth_type=AUTH_TYPE_OAUTH,
        priority=0,
        source="hermes_pkce",
        access_token="sk-ant-oat-rt0",
        refresh_token="rt0",
    )
    pool = CredentialPool("anthropic", [entry])
    updated = pool._refresh_entry(entry, force=True)
    assert updated is not None
    assert updated.refresh_token == "rt1"

    on_disk = json.loads(oauth_file.read_text(encoding="utf-8"))
    assert on_disk["refreshToken"] == "rt1", (
        "successful hermes_pkce refresh must write back to "
        "~/.hermes/.anthropic_oauth.json, or _seed_from_singletons() will "
        "revert the pool entry to the pre-refresh (spent) token on next load"
    )

    reloaded = load_pool("anthropic")
    reloaded_entries = [e for e in reloaded.entries() if e.source.endswith("hermes_pkce")]
    assert reloaded_entries, "hermes_pkce entry should still be present after reload"
    assert reloaded_entries[0].refresh_token == "rt1", (
        "regression: fresh load_pool() re-seeded the pre-refresh refresh "
        "token from the stale singleton file, reverting a successful "
        "rotation and orphaning the already-consumed rt0"
    )


def test_manual_hermes_pkce_refresh_does_not_create_duplicate_singleton(
    tmp_path, monkeypatch
):
    """A pool-owned manual:hermes_pkce entry must not create a second source."""
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setattr("hermes_cli.auth.is_provider_explicitly_configured", lambda pid: True)
    monkeypatch.setattr("agent.anthropic_credentials.read_claude_code_credentials", lambda: None)
    monkeypatch.setattr(
        "agent.anthropic_credentials.refresh_anthropic_oauth_pure",
        lambda refresh_token, use_json=False: {
            "access_token": "manual-at-1",
            "refresh_token": "manual-rt-1",
            "expires_at_ms": int(time.time() * 1000) + 3_600_000,
        },
    )
    _write_store(hermes_home / "auth.json", {"version": 1, "providers": {}})

    entry = PooledCredential(
        provider="anthropic",
        id="manual-entry",
        label="cred",
        auth_type=AUTH_TYPE_OAUTH,
        priority=0,
        source="manual:hermes_pkce",
        access_token="manual-at-0",
        refresh_token="manual-rt-0",
        expires_at_ms=0,
    )
    pool = CredentialPool("anthropic", [entry])
    refreshed = pool._refresh_entry(entry, force=True)

    assert refreshed is not None
    assert refreshed.refresh_token == "manual-rt-1"
    oauth_file = hermes_home / ".anthropic_oauth.json"
    assert not oauth_file.exists(), (
        "manual:hermes_pkce is already pool-owned; refreshing it must not "
        "create a second hermes_pkce singleton source"
    )

    reloaded = load_pool("anthropic")
    matching = [e for e in reloaded.entries() if e.id == "manual-entry"]
    assert len(matching) == 1
    assert matching[0].source == "manual:hermes_pkce"
    assert matching[0].refresh_token == "manual-rt-1"

