"""Deferred refresh leaves unrelated credential selection available (#71775)."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import replace
from threading import Event

import pytest

from agent import credential_pool, credential_pool_deferred_refresh
from agent.credential_pool import (
    AUTH_TYPE_OAUTH,
    STRATEGY_ROUND_ROBIN,
    CredentialPool,
    PooledCredential,
)
from hermes_cli import auth

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

@pytest.mark.parametrize("blocked_at", ["file-lock", "provider"])
def test_select_can_use_another_entry_during_deferred_refresh(tmp_path, monkeypatch, blocked_at):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    stale = replace(_codex_entry(), source="manual:device_code", priority=1)
    healthy = replace(stale, id="healthy", priority=0, access_token="healthy-access", refresh_token="healthy-refresh")
    pool = CredentialPool(stale.provider, [healthy, stale])
    pool._persist()
    monkeypatch.setattr(pool, "_entry_needs_refresh", lambda e: e.access_token == stale.access_token)
    started, release = Event(), Event()
    real_store_lock = credential_pool._auth_store_lock

    def pause():
        started.set()
        assert release.wait(10)

    @contextmanager
    def store_lock(*args, **kwargs):
        if blocked_at == "file-lock" and not started.is_set():
            pause()
        with real_store_lock(*args, **kwargs):
            yield

    def refresh(*_args):
        if blocked_at == "provider":
            pause()
        return {"access_token": "at-fresh", "refresh_token": "rt-fresh"}

    monkeypatch.setattr(credential_pool, "_auth_store_lock", store_lock)
    monkeypatch.setattr(auth, "refresh_codex_oauth_pure", refresh)

    with ThreadPoolExecutor(max_workers=2) as executor:
        pending = executor.submit(pool._refresh_pending_entries, [stale])
        try:
            assert started.wait(5)
            selected = executor.submit(pool.select).result(timeout=3)
            assert selected.id == healthy.id
        finally:
            release.set()
        pending.result(timeout=5)

    assert [entry.access_token for entry in pool.entries()] == [healthy.access_token, "at-fresh"]
    assert [row["id"] for row in auth.read_credential_pool(stale.provider)] == [healthy.id, stale.id]


@pytest.mark.parametrize("new_login", [False, True], ids=["metadata", "new-generation"])
def test_round_robin_retains_deferred_refresh_across_row_rebuild(tmp_path, monkeypatch, new_login):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    stale = replace(_codex_entry(), source="manual:device_code", priority=1)
    first = replace(stale, id="first", priority=0, access_token="first-access", refresh_token="first-refresh")
    second = replace(stale, id="second", priority=2, access_token="second-access", refresh_token="second-refresh")
    pool = CredentialPool(stale.provider, [first, stale, second])
    pool._strategy = STRATEGY_ROUND_ROBIN
    pool._persist()
    monkeypatch.setattr(pool, "_entry_needs_refresh", lambda row: row.access_token == stale.access_token)
    refresh_calls = []

    def refresh(access_token, refresh_token):
        refresh_calls.append((access_token, refresh_token))
        return {"access_token": "at-fresh", "refresh_token": "rt-fresh"}

    publish = credential_pool_deferred_refresh._publish_deferred_entry

    def update_before_publish(live_pool, original, staged):
        current, *_ = live_pool.entries()
        updates = {"request_count": 7}
        if new_login:
            updates.update(access_token="new-login-access", refresh_token="new-login-refresh")
        live_pool._replace_entry(current, replace(current, **updates))
        publish(live_pool, original, staged)

    monkeypatch.setattr(auth, "refresh_codex_oauth_pure", refresh)
    monkeypatch.setattr(credential_pool_deferred_refresh, "_publish_deferred_entry", update_before_publish)

    assert pool.select().id == first.id

    assert refresh_calls == [(stale.access_token, stale.refresh_token)]
    rows = pool.entries()
    assert [(row.id, row.priority, row.request_count) for row in rows] == [
        (stale.id, 0, 7), (second.id, 1, 0), (first.id, 2, 1),
    ]
    assert rows[0].refresh_token == ("new-login-refresh" if new_login else "rt-fresh")
