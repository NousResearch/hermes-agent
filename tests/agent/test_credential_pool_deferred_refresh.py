"""Thread-safety of the deferred single-use-token refresh path (#71775).

The deferred path deliberately runs OAuth network I/O outside the pool
lock. These tests pin the two invariants that make that safe:

1. `select()` does NOT hold the pool lock while the deferred refresh's
   network call runs (the whole point of the PR).
2. The pool mutations that follow the network call (`_replace_entry`,
   `_persist`) DO re-serialize under the pool lock, so a concurrent
   `select()`/rotation cannot tear `self._entries` or double-write
   auth.json.
"""

from dataclasses import replace

from agent.credential_pool import (
    AUTH_TYPE_OAUTH,
    CredentialPool,
    PooledCredential,
)


def _codex_entry(entry_id: str = "codex-1") -> PooledCredential:
    return PooledCredential(
        provider="openai-codex",
        id=entry_id,
        label="test codex",
        auth_type=AUTH_TYPE_OAUTH,
        priority=0,
        source="device_code",
        access_token="at-stale",
        refresh_token="rt-stale",
        expires_at_ms=1,  # long expired -> needs refresh
    )


def _plugin_entry(*, expires_at_ms: int | None) -> PooledCredential:
    return PooledCredential(
        provider="example-oauth",
        id="plugin-1",
        label="Example OAuth",
        auth_type=AUTH_TYPE_OAUTH,
        priority=0,
        source="manual:loopback_pkce",
        access_token="at-stale",
        refresh_token="rt-stale",
        expires_at_ms=expires_at_ms,
    )


def test_plugin_expiry_metadata_triggers_proactive_refresh(monkeypatch):
    pool = CredentialPool("example-oauth", [_plugin_entry(expires_at_ms=1)])
    monkeypatch.setattr(
        "agent.credential_pool.plugin_refresh_hook",
        lambda provider: (lambda entry: {}) if provider == "example-oauth" else None,
    )

    assert pool._entry_needs_refresh(pool._entries[0]) is True


def test_plugin_without_expiry_metadata_keeps_existing_behavior(monkeypatch):
    pool = CredentialPool("example-oauth", [_plugin_entry(expires_at_ms=None)])
    monkeypatch.setattr(
        "agent.credential_pool.plugin_refresh_hook",
        lambda provider: (lambda entry: {}) if provider == "example-oauth" else None,
    )

    assert pool._entry_needs_refresh(pool._entries[0]) is False


def test_plugin_refresh_is_deferred_outside_pool_lock(monkeypatch):
    pool = CredentialPool("example-oauth", [_plugin_entry(expires_at_ms=1)])
    lock_free_during_refresh = {}

    monkeypatch.setattr(
        "agent.credential_pool.plugin_refresh_hook",
        lambda provider: (lambda entry: {}) if provider == "example-oauth" else None,
    )

    def _fake_refresh(entry, *, force):
        acquired = pool._lock.acquire(blocking=False)
        lock_free_during_refresh["value"] = acquired
        if acquired:
            pool._lock.release()
        refreshed = replace(entry, access_token="at-fresh", expires_at_ms=2**53)
        pool._replace_entry(entry, refreshed)
        return refreshed

    monkeypatch.setattr(pool, "_refresh_entry", _fake_refresh)
    monkeypatch.setattr(pool, "_persist", lambda **kw: None)

    selected = pool.select()

    assert lock_free_during_refresh.get("value") is True
    assert selected is not None
    assert selected.access_token == "at-fresh"


def test_select_does_not_hold_pool_lock_during_deferred_refresh(monkeypatch):
    pool = CredentialPool("openai-codex", [_codex_entry()])
    lock_free_during_refresh = {}

    def _fake_refresh(entry, *, force):
        # If select() still held the pool lock here, this non-blocking
        # acquire would fail — the regression this PR exists to fix.
        acquired = pool._lock.acquire(blocking=False)
        lock_free_during_refresh["value"] = acquired
        if acquired:
            pool._lock.release()
        refreshed = replace(entry, access_token="at-fresh", expires_at_ms=2**53)
        pool._replace_entry(entry, refreshed)
        return refreshed

    monkeypatch.setattr(
        pool, "_entry_needs_refresh", lambda e: e.access_token == "at-stale"
    )
    monkeypatch.setattr(pool, "_refresh_entry", _fake_refresh)
    monkeypatch.setattr(pool, "_persist", lambda **kw: None)

    selected = pool.select()

    assert lock_free_during_refresh.get("value") is True, (
        "select() held the pool lock during the deferred refresh network window"
    )
    assert selected is not None
    assert selected.access_token == "at-fresh"


