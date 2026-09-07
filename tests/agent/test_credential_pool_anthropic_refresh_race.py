"""Regression tests for cross-process races refreshing Anthropic OAuth tokens.

``CredentialPool._refresh_entry`` explicitly documents (see the comment
above the ``if self.provider in ("openai-codex", "xai-oauth", "anthropic"):`` branch in
``agent/credential_pool.py``) that single-use OAuth refresh tokens require
the whole sync -> POST -> write-back sequence to be serialized across
Hermes *processes* via the cross-process ``_auth_store_lock`` flock,
otherwise "two processes can both adopt the same on-disk token, both POST
it, and the loser gets ``refresh_token_reused``".

Anthropic's OAuth refresh tokens have the identical single-use property --
``agent.anthropic_credentials._refresh_oauth_token`` says so explicitly:
"Claude Code's OAuth refresh tokens are single-use: a successful refresh
rotates the pair and invalidates the old refresh token." Before the PR, ``"anthropic"`` was absent from the
``("openai-codex", "xai-oauth")`` tuple that gets the cross-process flock,
and the *only* on-failure recovery path
(``CredentialPool._sync_anthropic_entry_from_credentials_file``) was
hard-scoped to ``entry.source == "claude_code"``. Entries sourced from
Hermes's own PKCE login (``manual:hermes_pkce`` / ``hermes_pkce``) got no
recovery at all and were marked exhausted on a lost race, even though a
fresh, valid token pair existed on disk (written by the winner). The old
dashboard source ``manual:dashboard_pkce`` is now retired with the removed
dashboard flow.

These tests reproduce that race deterministically with a fake OAuth server
that enforces single-use refresh tokens, run concurrent pool instances against
it, and assert the current protection. The process-level, cross-profile
Claude Code witness lives in ``test_anthropic_oauth_stress.py``.
"""

from __future__ import annotations

import threading
import time
from dataclasses import replace as dc_replace

import pytest

from agent.credential_pool import (
    AUTH_TYPE_OAUTH,
    STATUS_EXHAUSTED,
    CredentialPool,
    PooledCredential,
)


def _entry(*, id: str, access_token: str, refresh_token: str, source: str) -> PooledCredential:
    return PooledCredential(
        provider="anthropic",
        id=id,
        label="anthropic oauth",
        auth_type=AUTH_TYPE_OAUTH,
        priority=0,
        source=source,
        access_token=access_token,
        refresh_token=refresh_token,
        expires_at_ms=0,  # already expired -> force path always refreshes
    )


@pytest.fixture(autouse=True)
def _fake_pool_store(monkeypatch):
    """Back write/read_credential_pool with a shared in-memory dict.

    This stands in for ``~/.hermes/auth.json`` so the two "process-local"
    CredentialPool instances used by the race tests can actually see each
    other's persisted writes (exactly what the real cross-process recovery
    path depends on), without touching the real filesystem.
    """
    store: Dict[str, list] = {}

    def _write(provider, entries, *, removed_ids=None):
        store[provider] = list(entries)

    def _read(provider=None):
        if provider is None:
            return dict(store)
        return list(store.get(provider, []))

    monkeypatch.setattr("agent.credential_pool.write_credential_pool", _write)
    monkeypatch.setattr("agent.credential_pool.read_credential_pool", _read)
    return store


class _SingleUseTokenServer:
    """Minimal fake of Anthropic's OAuth token endpoint.

    Enforces the real single-use-refresh-token contract: the first caller
    to present a given refresh_token gets a fresh pair; every subsequent
    caller presenting that same (now-spent) refresh_token gets an
    ``invalid_grant`` error, mirroring Anthropic's actual behavior.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._spent: set[str] = set()
        self._rotation = 0
        self.calls: list[str] = []
        # Small artificial delay to widen the race window deterministically.
        self.delay_seconds = 0.05

    def refresh(self, refresh_token: str, *, use_json: bool = False):
        self.calls.append(refresh_token)
        time.sleep(self.delay_seconds)
        with self._lock:
            if refresh_token in self._spent:
                raise ValueError("invalid_grant: refresh token already used")
            self._spent.add(refresh_token)
            self._rotation += 1
            rotation = self._rotation
        return {
            "access_token": f"sk-ant-oat-rotated-{rotation}",
            "refresh_token": f"sk-ant-ort-rotated-{rotation}",
            "expires_at_ms": int(time.time() * 1000) + 3_600_000,
        }


def test_anthropic_refresh_is_protected_by_cross_process_lock(monkeypatch):
    """Structural check: anthropic refresh acquires ``_auth_store_lock``.

    Regression guard for the gap where "anthropic" was missing from the
    ``("openai-codex", "xai-oauth")`` tuple despite Anthropic OAuth refresh
    tokens being single-use too -- see the module docstring.
    """
    lock_calls: list[str] = []

    class _RecordingLock:
        def __init__(self, *a, **kw):
            lock_calls.append("anthropic")

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr("agent.credential_pool._auth_store_lock", _RecordingLock)
    monkeypatch.setattr(
        "agent.anthropic_credentials.refresh_anthropic_oauth_pure",
        lambda refresh_token, use_json=False: {
            "access_token": "sk-ant-oat-new",
            "refresh_token": "sk-ant-ort-new",
            "expires_at_ms": int(time.time() * 1000) + 3_600_000,
        },
    )
    monkeypatch.setattr(
        "agent.anthropic_credentials.read_claude_code_credentials", lambda: None
    )

    pool = CredentialPool(
        "anthropic",
        [_entry(id="a1", access_token="stale-at", refresh_token="stale-rt", source="manual:hermes_pkce")],
    )
    entry = pool.entries()[0]
    pool._refresh_entry(entry, force=True)

    assert lock_calls, (
        "regression: CredentialPool._refresh_entry() for provider='anthropic' "
        "did not acquire the cross-process _auth_store_lock, even though "
        "Anthropic OAuth refresh tokens are single-use (same property "
        "openai-codex/xai-oauth are explicitly locked for)."
    )


def test_concurrent_hermes_pkce_refresh_loses_credential_despite_valid_token_on_disk(monkeypatch):
    """Two 'Hermes processes' race to refresh the same stale hermes_pkce
    refresh token. The winner gets a valid new pair. The loser -- despite a
    fresh, valid credential now existing -- has no recovery path and is
    marked exhausted, because ``_sync_anthropic_entry_from_credentials_file``
    only helps ``entry.source == "claude_code"``.
    """
    server = _SingleUseTokenServer()
    monkeypatch.setattr(
        "agent.anthropic_credentials.refresh_anthropic_oauth_pure",
        lambda refresh_token, use_json=False: server.refresh(refresh_token, use_json=use_json),
    )
    # No Claude Code credential file in play for this scenario.
    monkeypatch.setattr(
        "agent.anthropic_credentials.read_claude_code_credentials", lambda: None
    )

    shared_stale_entry = _entry(
        id="pool-entry",
        access_token="stale-at",
        refresh_token="stale-rt",
        source="manual:hermes_pkce",
    )

    # Simulate two independent OS processes, each with its own in-memory
    # pool constructed from the SAME on-disk stale entry.
    pool_process_a = CredentialPool("anthropic", [dc_replace(shared_stale_entry)])
    pool_process_b = CredentialPool("anthropic", [dc_replace(shared_stale_entry)])

    results: dict[str, object] = {}

    def _run(pool, key):
        entry = pool.entries()[0]
        results[key] = pool._refresh_entry(entry, force=True)

    t_a = threading.Thread(target=_run, args=(pool_process_a, "a"))
    t_b = threading.Thread(target=_run, args=(pool_process_b, "b"))
    t_a.start()
    t_b.start()
    t_a.join(timeout=5)
    t_b.join(timeout=5)

    # With the cross-process lock in place, the second thread to acquire it
    # re-syncs from the pool store first and may adopt the winner's
    # already-persisted token before ever calling the token endpoint --
    # so only ONE call is expected now (the race is avoided entirely,
    # which is strictly better than the old behavior of two racing calls).
    # Both calls (if two do happen) must only ever spend the one stale
    # token that was actually on disk -- never anything else.
    assert 1 <= len(server.calls) <= 2, f"unexpected call count: {server.calls!r}"
    assert set(server.calls) == {"stale-rt"}, f"unexpected tokens posted: {server.calls!r}"

    winner_result = results["a"] or results["b"]
    loser_key = "b" if results["a"] else "a"
    loser_result = results[loser_key]
    loser_pool = pool_process_b if loser_key == "b" else pool_process_a

    assert winner_result is not None and winner_result.access_token.startswith(
        "sk-ant-oat-rotated-"
    ), "the winning process should obtain a fresh, valid token pair"

    # This is the bug: the loser should be able to recover (a valid token
    # now exists -- the winner's), not be left exhausted.
    assert loser_result is not None, (
        "regression: the losing process's CredentialPool._refresh_entry() "
        "returned None (unrecoverable) after losing the single-use-token "
        "race, even though a fresh valid Anthropic credential exists on "
        "disk. hermes_pkce-sourced entries must adopt the winner's token "
        "via _sync_anthropic_entry_from_pool_store(), same as claude_code."
    )
    loser_entry_after = loser_pool.entries()[0]
    assert loser_entry_after.last_status != STATUS_EXHAUSTED, (
        "regression: the losing process marked its Anthropic hermes_pkce "
        "credential STATUS_EXHAUSTED after a lost refresh race, even "
        "though the account is not actually exhausted -- a sibling Hermes "
        "process holds a perfectly valid rotated token. This causes "
        "spurious 're-authenticate with Anthropic' failures under "
        "ordinary concurrent Hermes usage (fleet workers, cron jobs, "
        "multiple CLI sessions)."
    )


def test_concurrent_claude_code_refresh_recovers_via_credentials_file(monkeypatch):
    """Contrast case: entry.source == 'claude_code' DOES recover from a lost
    race, because ``_sync_anthropic_entry_from_credentials_file`` re-reads
    ``~/.claude/.credentials.json`` on failure. This asymmetry is itself
    evidence that hermes_pkce/dashboard-sourced credentials were simply
    never given the same treatment, not that recovery is impossible.
    """
    server = _SingleUseTokenServer()
    monkeypatch.setattr(
        "agent.anthropic_credentials.refresh_anthropic_oauth_pure",
        lambda refresh_token, use_json=False: server.refresh(refresh_token, use_json=use_json),
    )

    winner_creds: dict = {}
    write_lock = threading.Lock()

    def _fake_read_claude_code_credentials():
        with write_lock:
            return dict(winner_creds) if winner_creds else None

    def _fake_write_claude_code_credentials(access_token, refresh_token, expires_at_ms, **_kw):
        with write_lock:
            winner_creds.update(
                accessToken=access_token,
                refreshToken=refresh_token,
                expiresAt=expires_at_ms,
            )

    monkeypatch.setattr(
        "agent.anthropic_credentials.read_claude_code_credentials",
        _fake_read_claude_code_credentials,
    )
    monkeypatch.setattr(
        "agent.anthropic_credentials._write_claude_code_credentials",
        _fake_write_claude_code_credentials,
    )

    shared_stale_entry = _entry(
        id="pool-entry",
        access_token="stale-at",
        refresh_token="stale-rt",
        source="claude_code",
    )
    pool_process_a = CredentialPool("anthropic", [dc_replace(shared_stale_entry)])
    pool_process_b = CredentialPool("anthropic", [dc_replace(shared_stale_entry)])

    results: dict[str, object] = {}

    def _run(pool, key):
        entry = pool.entries()[0]
        results[key] = pool._refresh_entry(entry, force=True)

    t_a = threading.Thread(target=_run, args=(pool_process_a, "a"))
    t_b = threading.Thread(target=_run, args=(pool_process_b, "b"))
    t_a.start()
    t_b.start()
    t_a.join(timeout=5)
    t_b.join(timeout=5)

    # Both processes should end up with a usable credential: the winner via
    # its own refresh, the loser via the credentials-file sync-and-retry.
    assert results["a"] is not None, "claude_code process A should recover"
    assert results["b"] is not None, "claude_code process B should recover"
