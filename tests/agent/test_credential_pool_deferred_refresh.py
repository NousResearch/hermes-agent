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

import json
import threading
import time
from dataclasses import replace

import pytest

from agent import anthropic_adapter
from agent import credential_pool as CP
from agent.credential_pool import (
    AUTH_TYPE_OAUTH,
    CredentialPool,
    PooledCredential,
    STATUS_EXHAUSTED,
    STRATEGY_FILL_FIRST,
    STRATEGY_LEAST_USED,
    STRATEGY_ROUND_ROBIN,
)
from hermes_cli import auth as auth_mod


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


def test_deferred_mutations_serialize_against_concurrent_rotation(monkeypatch):
    """_replace_entry/_persist from the deferred path must contend on the
    pool lock: with the lock held by another thread, the deferred mutation
    must block rather than mutate concurrently."""
    pool = CredentialPool("openai-codex", [_codex_entry()])
    monkeypatch.setattr(pool, "_persist", lambda **kw: None)

    entry = pool._entries[0]
    refreshed = replace(entry, access_token="at-fresh")

    mutated = threading.Event()

    def _deferred_mutation():
        pool._replace_entry(entry, refreshed)  # self-locking
        mutated.set()

    with pool._lock:
        t = threading.Thread(target=_deferred_mutation)
        t.start()
        # While we hold the lock, the deferred mutation must NOT complete.
        assert not mutated.wait(timeout=0.3), (
            "_replace_entry mutated the pool while another thread held the lock"
        )
    t.join(timeout=5)
    assert mutated.is_set()
    assert pool._entries[0].access_token == "at-fresh"


def test_external_state_lock_is_never_acquired_under_pool_lock():
    pool = CredentialPool("anthropic", [_anthropic_entry("entry")])

    with pool._lock:
        with pytest.raises(RuntimeError, match="pool lock is held"):
            pool._sync_external_status_entries()
        with pytest.raises(RuntimeError, match="pool lock is held"):
            pool._refresh_entry(pool._entries[0], force=False)


def _anthropic_entry(
    entry_id: str,
    *,
    priority: int = 0,
    access_token: str | None = None,
    refresh_token: str | None = None,
    expires_at_ms: int = 1,
    request_count: int = 0,
    source: str = "manual:oauth",
    last_status: str | None = None,
    last_status_at: float | None = None,
) -> PooledCredential:
    return PooledCredential(
        provider="anthropic",
        id=entry_id,
        label=entry_id,
        auth_type=AUTH_TYPE_OAUTH,
        priority=priority,
        source=source,
        access_token=access_token or f"{entry_id}-access",
        refresh_token=refresh_token or f"{entry_id}-refresh",
        expires_at_ms=expires_at_ms,
        request_count=request_count,
        last_status=last_status,
        last_status_at=last_status_at,
        last_error_code=429 if last_status == STATUS_EXHAUSTED else None,
    )


def _pool_with_store(tmp_path, monkeypatch, entries, *, strategy=STRATEGY_FILL_FIRST):
    auth_path = tmp_path / "auth.json"
    monkeypatch.setattr(auth_mod, "_auth_file_path", lambda: auth_path)
    monkeypatch.setattr(auth_mod, "_global_auth_file_path", lambda: None)
    monkeypatch.setattr(
        auth_mod,
        "is_provider_explicitly_configured",
        lambda _provider: True,
    )
    monkeypatch.setattr(CP, "load_env", lambda: {})
    monkeypatch.setattr(CP, "_get_secret", lambda *_args: "")
    auth_path.write_text(
        json.dumps(
            {
                "version": 1,
                "credential_pool": {
                    "anthropic": [entry.to_dict() for entry in entries],
                },
            }
        )
    )
    pool = CredentialPool("anthropic", entries)
    pool._strategy = strategy
    return auth_path, pool


def _disk_entries(auth_path):
    payloads = json.loads(auth_path.read_text())["credential_pool"]["anthropic"]
    return [PooledCredential.from_dict("anthropic", payload) for payload in payloads]


def _reloaded_claude_entry():
    return next(
        entry
        for entry in CP.load_pool("anthropic").entries()
        if entry.source == "claude_code"
    )


def _install_claude_source(monkeypatch, initial):
    state = {"credentials": dict(initial), "writes": []}
    source_lock = threading.RLock()

    def read_credentials():
        return {
            **state["credentials"],
            "source": "claude_code_credentials_file",
        }

    def write_credentials(access_token, refresh_token, expires_at_ms):
        write = (access_token, refresh_token, expires_at_ms)
        state["writes"].append(write)
        state["credentials"] = {
            "accessToken": access_token,
            "refreshToken": refresh_token,
            "expiresAt": expires_at_ms,
        }
        return True

    def write_locked(
        access_token,
        refresh_token,
        expires_at_ms,
        *,
        expected_refresh_token,
        allow_missing=False,
    ):
        del allow_missing
        current = state["credentials"]
        if current.get("refreshToken") != expected_refresh_token:
            raise auth_mod.SourceCredentialLineageChanged("anthropic")
        return write_credentials(access_token, refresh_token, expires_at_ms)

    def refresh_source(observed):
        with source_lock:
            current = read_credentials()
            if current.get("refreshToken") != observed.get("refreshToken"):
                return current
            refreshed = anthropic_adapter.refresh_anthropic_oauth_pure(
                observed["refreshToken"],
                use_json=False,
            )
            try:
                anthropic_adapter._write_claude_code_credentials_locked(
                    refreshed["access_token"],
                    refreshed["refresh_token"],
                    refreshed["expires_at_ms"],
                    expected_refresh_token=observed["refreshToken"],
                    allow_missing=False,
                )
            except auth_mod.SourceCredentialLineageChanged:
                return read_credentials()
            return read_credentials()

    monkeypatch.setattr(
        anthropic_adapter,
        "read_claude_code_credentials",
        read_credentials,
    )
    monkeypatch.setattr(
        anthropic_adapter,
        "_read_claude_code_credentials_from_file",
        read_credentials,
    )
    monkeypatch.setattr(
        anthropic_adapter,
        "_write_claude_code_credentials",
        write_credentials,
    )
    monkeypatch.setattr(
        anthropic_adapter,
        "_write_claude_code_credentials_locked",
        write_locked,
    )
    monkeypatch.setattr(
        anthropic_adapter,
        "_refresh_claude_code_source_credentials",
        refresh_source,
    )
    return state


def _tokens(entry):
    return entry.access_token, entry.refresh_token


def test_public_status_sync_wins_before_deferred_refresh_starts(tmp_path, monkeypatch):
    old = _anthropic_entry(
        "claude-entry",
        source="claude_code",
        access_token="old-access",
        refresh_token="old-refresh",
        last_status=STATUS_EXHAUSTED,
        last_status_at=0.0,
    )
    auth_path, pool = _pool_with_store(tmp_path, monkeypatch, [old])
    source = _install_claude_source(
        monkeypatch,
        {
            "accessToken": "old-access",
            "refreshToken": "old-refresh",
            "expiresAt": 1,
        },
    )
    posts = []

    def refresh_credentials(refresh_token, *, use_json):
        posts.append((refresh_token, use_json))
        return {
            "access_token": "old-rotated-access",
            "refresh_token": "old-rotated-refresh",
            "expires_at_ms": 9_999_999_999_999,
        }

    monkeypatch.setattr(
        anthropic_adapter,
        "refresh_anthropic_oauth_pure",
        refresh_credentials,
    )
    candidate_captured = threading.Event()
    allow_refresh = threading.Event()
    real_refresh = pool._refresh_entry

    def paused_refresh(entry, *, force):
        candidate_captured.set()
        assert allow_refresh.wait(timeout=5)
        return real_refresh(entry, force=force)

    monkeypatch.setattr(pool, "_refresh_entry", paused_refresh)
    results = []
    errors = []

    def select_worker():
        try:
            results.append(pool.select())
        except BaseException as exc:
            errors.append(exc)

    selector = threading.Thread(target=select_worker)
    selector.start()
    assert candidate_captured.wait(timeout=5)

    current = pool.entries()[0]
    pool._replace_entry(
        current,
        replace(
            current,
            last_status=STATUS_EXHAUSTED,
            last_status_at=time.time(),
            last_error_code=429,
        ),
        mark_dirty=False,
    )
    source["credentials"] = {
        "accessToken": "concurrent-new-access",
        "refreshToken": "concurrent-new-refresh",
        "expiresAt": 9_999_999_999_999,
    }
    assert pool.has_available() is True
    allow_refresh.set()
    selector.join(timeout=5)

    assert not selector.is_alive()
    assert errors == []
    assert posts == []
    assert source["writes"] == []
    assert source["credentials"] == {
        "accessToken": "concurrent-new-access",
        "refreshToken": "concurrent-new-refresh",
        "expiresAt": 9_999_999_999_999,
    }
    assert len(results) == 1 and results[0] is not None
    assert _tokens(results[0]) == (
        "concurrent-new-access",
        "concurrent-new-refresh",
    )
    assert _tokens(pool.entries()[0]) == _tokens(results[0])
    assert _disk_entries(auth_path)[0].source == "claude_code"
    assert _tokens(_reloaded_claude_entry()) == _tokens(results[0])


def test_inflight_refresh_cannot_commit_over_newer_chain(tmp_path, monkeypatch):
    old = _anthropic_entry(
        "claude-entry",
        source="claude_code",
        access_token="old-access",
        refresh_token="old-refresh",
    )
    auth_path, pool = _pool_with_store(tmp_path, monkeypatch, [old])
    source = _install_claude_source(
        monkeypatch,
        {
            "accessToken": "old-access",
            "refreshToken": "old-refresh",
            "expiresAt": 1,
        },
    )
    post_entered = threading.Event()
    release_post = threading.Event()
    posts = []

    def refresh_credentials(refresh_token, *, use_json):
        posts.append((refresh_token, use_json))
        post_entered.set()
        assert release_post.wait(timeout=5)
        return {
            "access_token": "old-rotated-access",
            "refresh_token": "old-rotated-refresh",
            "expires_at_ms": 9_999_999_999_999,
        }

    monkeypatch.setattr(
        anthropic_adapter,
        "refresh_anthropic_oauth_pure",
        refresh_credentials,
    )
    results = []
    errors = []

    def select_worker():
        try:
            results.append(pool.select())
        except BaseException as exc:
            errors.append(exc)

    selector = threading.Thread(target=select_worker)
    selector.start()
    assert post_entered.wait(timeout=5)

    current = pool.entries()[0]
    newer = replace(
        current,
        access_token="concurrent-new-access",
        refresh_token="concurrent-new-refresh",
        expires_at_ms=9_999_999_999_999,
    )
    assert pool._replace_entry(current, newer) == newer
    pool._persist_pending_changes()
    source["credentials"] = {
        "accessToken": "concurrent-new-access",
        "refreshToken": "concurrent-new-refresh",
        "expiresAt": 9_999_999_999_999,
    }
    release_post.set()
    selector.join(timeout=5)

    assert not selector.is_alive()
    assert errors == []
    assert posts == [("old-refresh", False)]
    assert source["writes"] == []
    assert len(results) == 1 and results[0] is not None
    assert _tokens(results[0]) == _tokens(newer)
    assert _tokens(pool.entries()[0]) == _tokens(newer)
    assert _disk_entries(auth_path)[0].source == "claude_code"
    assert _tokens(_reloaded_claude_entry()) == _tokens(newer)
    assert source["credentials"]["refreshToken"] == "concurrent-new-refresh"


def test_removed_inflight_candidate_is_not_written_or_resurrected(tmp_path, monkeypatch):
    old = _anthropic_entry(
        "claude-entry",
        source="claude_code",
        access_token="old-access",
        refresh_token="old-refresh",
    )
    auth_path, pool = _pool_with_store(tmp_path, monkeypatch, [old])
    source = _install_claude_source(
        monkeypatch,
        {
            "accessToken": "old-access",
            "refreshToken": "old-refresh",
            "expiresAt": 1,
        },
    )
    post_entered = threading.Event()
    release_post = threading.Event()

    def refresh_credentials(_refresh_token, *, use_json):
        assert use_json is False
        post_entered.set()
        assert release_post.wait(timeout=5)
        return {
            "access_token": "removed-rotated-access",
            "refresh_token": "removed-rotated-refresh",
            "expires_at_ms": 9_999_999_999_999,
        }

    monkeypatch.setattr(
        anthropic_adapter,
        "refresh_anthropic_oauth_pure",
        refresh_credentials,
    )
    results = []
    selector = threading.Thread(target=lambda: results.append(pool.select()))
    selector.start()
    assert post_entered.wait(timeout=5)
    assert pool.remove_index(1) is not None
    release_post.set()
    selector.join(timeout=5)

    assert not selector.is_alive()
    assert results == [None]
    assert source["writes"] == [
        (
            "removed-rotated-access",
            "removed-rotated-refresh",
            9_999_999_999_999,
        )
    ]
    assert source["credentials"]["refreshToken"] == "removed-rotated-refresh"
    assert pool.entries() == []
    assert _disk_entries(auth_path) == []


def test_concurrent_selectors_refresh_rotating_chain_once(tmp_path, monkeypatch):
    old = _anthropic_entry(
        "claude-entry",
        source="claude_code",
        access_token="old-access",
        refresh_token="old-refresh",
    )
    auth_path, pool = _pool_with_store(tmp_path, monkeypatch, [old])
    source = _install_claude_source(
        monkeypatch,
        {
            "accessToken": "old-access",
            "refreshToken": "old-refresh",
            "expiresAt": 1,
        },
    )
    first_post_entered = threading.Event()
    second_post_entered = threading.Event()
    release_post = threading.Event()
    calls = []
    calls_lock = threading.Lock()

    def refresh_credentials(refresh_token, *, use_json):
        with calls_lock:
            calls.append((refresh_token, use_json))
            if len(calls) == 1:
                first_post_entered.set()
            else:
                second_post_entered.set()
        assert release_post.wait(timeout=5)
        return {
            "access_token": "rotated-access",
            "refresh_token": "rotated-refresh",
            "expires_at_ms": 9_999_999_999_999,
        }

    monkeypatch.setattr(
        anthropic_adapter,
        "refresh_anthropic_oauth_pure",
        refresh_credentials,
    )
    start = threading.Barrier(3)
    results = []
    errors = []

    def select_worker():
        try:
            start.wait(timeout=5)
            results.append(pool.select())
        except BaseException as exc:
            errors.append(exc)

    selectors = [threading.Thread(target=select_worker) for _ in range(2)]
    for selector in selectors:
        selector.start()
    start.wait(timeout=5)
    assert first_post_entered.wait(timeout=5)
    second_post_entered.wait(timeout=0.3)
    release_post.set()
    for selector in selectors:
        selector.join(timeout=5)

    assert all(not selector.is_alive() for selector in selectors)
    assert errors == []
    assert calls == [("old-refresh", False)]
    assert source["writes"] == [
        ("rotated-access", "rotated-refresh", 9_999_999_999_999)
    ]
    assert len(results) == 2 and all(result is not None for result in results)
    assert {_tokens(result) for result in results} == {
        ("rotated-access", "rotated-refresh")
    }
    assert _tokens(pool.entries()[0]) == ("rotated-access", "rotated-refresh")
    assert _disk_entries(auth_path)[0].source == "claude_code"
    assert _tokens(_reloaded_claude_entry()) == (
        "rotated-access",
        "rotated-refresh",
    )


@pytest.mark.parametrize("mutation", ["remove", "replace"])
def test_post_cas_lineage_mutation_cannot_write_or_return_stale_refresh(
    tmp_path,
    monkeypatch,
    mutation,
):
    old = _anthropic_entry(
        "claude-entry",
        source="claude_code",
        access_token="old-access",
        refresh_token="old-refresh",
    )
    _auth_path, pool = _pool_with_store(tmp_path, monkeypatch, [old])
    source = _install_claude_source(
        monkeypatch,
        {
            "accessToken": "old-access",
            "refreshToken": "old-refresh",
            "expiresAt": 1,
        },
    )
    monkeypatch.setattr(
        anthropic_adapter,
        "refresh_anthropic_oauth_pure",
        lambda *_args, **_kwargs: {
            "access_token": "obsolete-access",
            "refresh_token": "obsolete-refresh",
            "expires_at_ms": 9_999_999_999_999,
        },
    )

    cas_complete = threading.Event()
    allow_write_boundary = threading.Event()
    real_replace = pool._replace_entry

    def pause_after_refresh_cas(previous, updated, **kwargs):
        result = real_replace(previous, updated, **kwargs)
        if (
            threading.current_thread().name == "refresh-selector"
            and result is not None
            and updated.access_token == "obsolete-access"
        ):
            cas_complete.set()
            assert allow_write_boundary.wait(timeout=5)
        return result

    pool._replace_entry = pause_after_refresh_cas
    results = []
    errors = []

    def select_worker():
        try:
            results.append(pool.select())
        except BaseException as exc:
            errors.append(exc)

    selector = threading.Thread(target=select_worker, name="refresh-selector")
    selector.start()
    assert cas_complete.wait(timeout=5)

    if mutation == "remove":
        assert pool.remove_index(1) is not None
        expected = None
    else:
        current = pool.entries()[0]
        winner = replace(
            current,
            access_token="winner-access",
            refresh_token="winner-refresh",
            expires_at_ms=9_999_999_999_999,
        )
        source["credentials"] = {
            "accessToken": "winner-access",
            "refreshToken": "winner-refresh",
            "expiresAt": 9_999_999_999_999,
        }
        assert real_replace(current, winner) == winner
        expected = winner

    allow_write_boundary.set()
    selector.join(timeout=5)

    assert not selector.is_alive()
    assert errors == []
    assert source["writes"] == [
        ("obsolete-access", "obsolete-refresh", 9_999_999_999_999)
    ]
    if expected is None:
        assert results == [None]
        assert pool.entries() == []
        assert source["credentials"]["refreshToken"] == "obsolete-refresh"
    else:
        assert len(results) == 1 and results[0] is not None
        assert _tokens(results[0]) == _tokens(expected)
        assert _tokens(pool.entries()[0]) == _tokens(expected)
        assert source["credentials"]["refreshToken"] == "winner-refresh"


def test_source_lineage_change_after_commit_read_wins_before_write(
    tmp_path,
    monkeypatch,
):
    old = _anthropic_entry(
        "claude-entry",
        source="claude_code",
        access_token="old-access",
        refresh_token="old-refresh",
    )
    _auth_path, pool = _pool_with_store(tmp_path, monkeypatch, [old])
    source = _install_claude_source(
        monkeypatch,
        {
            "accessToken": "old-access",
            "refreshToken": "old-refresh",
            "expiresAt": 1,
        },
    )
    monkeypatch.setattr(
        anthropic_adapter,
        "refresh_anthropic_oauth_pure",
        lambda *_args, **_kwargs: {
            "access_token": "obsolete-access",
            "refresh_token": "obsolete-refresh",
            "expires_at_ms": 9_999_999_999_999,
        },
    )

    commit_read = threading.Event()
    allow_commit = threading.Event()
    real_write_locked = anthropic_adapter._write_claude_code_credentials_locked

    def pause_at_commit_boundary(*args, **kwargs):
        commit_read.set()
        assert allow_commit.wait(timeout=5)
        return real_write_locked(*args, **kwargs)

    monkeypatch.setattr(
        anthropic_adapter,
        "_write_claude_code_credentials_locked",
        pause_at_commit_boundary,
    )
    results = []
    selector = threading.Thread(target=lambda: results.append(pool.select()))
    selector.start()
    assert commit_read.wait(timeout=5)

    source["credentials"] = {
        "accessToken": "winner-access",
        "refreshToken": "winner-refresh",
        "expiresAt": 9_999_999_999_999,
    }
    allow_commit.set()
    selector.join(timeout=5)

    assert not selector.is_alive()
    assert source["writes"] == []
    assert source["credentials"]["refreshToken"] == "winner-refresh"
    assert len(results) == 1 and results[0] is not None
    assert _tokens(results[0]) == ("winner-access", "winner-refresh")
    assert _tokens(pool.entries()[0]) == ("winner-access", "winner-refresh")


@pytest.mark.parametrize(
    ("refresh_succeeds", "expected_id"),
    [(True, "preferred-expiring"), (False, "healthy-fallback")],
)
def test_fill_first_waits_for_pending_refresh(
    tmp_path,
    monkeypatch,
    refresh_succeeds,
    expected_id,
):
    preferred = _anthropic_entry("preferred-expiring", priority=0)
    healthy = _anthropic_entry(
        "healthy-fallback",
        priority=1,
        expires_at_ms=9_999_999_999_999,
    )
    auth_path, pool = _pool_with_store(
        tmp_path,
        monkeypatch,
        [preferred, healthy],
        strategy=STRATEGY_FILL_FIRST,
    )

    def refresh_credentials(_refresh_token, *, use_json):
        assert use_json is False
        if not refresh_succeeds:
            raise RuntimeError("refresh failed")
        return {
            "access_token": "preferred-refreshed-access",
            "refresh_token": "preferred-refreshed-refresh",
            "expires_at_ms": 9_999_999_999_999,
        }

    monkeypatch.setattr(
        anthropic_adapter,
        "refresh_anthropic_oauth_pure",
        refresh_credentials,
    )
    selected = pool.select()

    assert selected is not None and selected.id == expected_id
    memory = pool.entries()
    disk = _disk_entries(auth_path)
    assert [(entry.id, entry.priority) for entry in disk] == [
        (entry.id, entry.priority) for entry in memory
    ]


def test_round_robin_refreshes_before_one_rotation(tmp_path, monkeypatch):
    entries = [
        _anthropic_entry(
            "healthy-0",
            priority=0,
            expires_at_ms=9_999_999_999_999,
        ),
        _anthropic_entry("expiring-1", priority=1),
        _anthropic_entry(
            "healthy-2",
            priority=2,
            expires_at_ms=9_999_999_999_999,
        ),
    ]
    auth_path, pool = _pool_with_store(
        tmp_path,
        monkeypatch,
        entries,
        strategy=STRATEGY_ROUND_ROBIN,
    )
    monkeypatch.setattr(
        anthropic_adapter,
        "refresh_anthropic_oauth_pure",
        lambda *_args, **_kwargs: {
            "access_token": "expiring-refreshed-access",
            "refresh_token": "expiring-refreshed-refresh",
            "expires_at_ms": 9_999_999_999_999,
        },
    )

    selected = pool.select()

    assert selected is not None and selected.id == "healthy-0"
    memory = pool.entries()
    assert [(entry.id, entry.priority) for entry in memory] == [
        ("expiring-1", 0),
        ("healthy-2", 1),
        ("healthy-0", 2),
    ]
    assert [(entry.id, entry.priority) for entry in _disk_entries(auth_path)] == [
        (entry.id, entry.priority) for entry in memory
    ]


def test_least_used_refreshes_before_incrementing_final_selection(tmp_path, monkeypatch):
    entries = [
        _anthropic_entry("expiring", priority=0, request_count=0),
        _anthropic_entry(
            "healthy-low",
            priority=1,
            request_count=3,
            expires_at_ms=9_999_999_999_999,
        ),
        _anthropic_entry(
            "healthy-high",
            priority=2,
            request_count=7,
            expires_at_ms=9_999_999_999_999,
        ),
    ]
    auth_path, pool = _pool_with_store(
        tmp_path,
        monkeypatch,
        entries,
        strategy=STRATEGY_LEAST_USED,
    )
    monkeypatch.setattr(
        anthropic_adapter,
        "refresh_anthropic_oauth_pure",
        lambda *_args, **_kwargs: {
            "access_token": "expiring-refreshed-access",
            "refresh_token": "expiring-refreshed-refresh",
            "expires_at_ms": 9_999_999_999_999,
        },
    )

    selected = pool.select()

    assert selected is not None and selected.id == "expiring"
    counts = {entry.id: entry.request_count for entry in pool.entries()}
    assert counts == {"expiring": 1, "healthy-low": 3, "healthy-high": 7}
    assert {
        entry.id: entry.request_count for entry in _disk_entries(auth_path)
    } == counts


def test_refresh_commit_preserves_concurrent_routing_fields(tmp_path, monkeypatch):
    entry = _anthropic_entry("expiring", priority=0, request_count=2)
    auth_path, pool = _pool_with_store(tmp_path, monkeypatch, [entry])
    post_entered = threading.Event()
    release_post = threading.Event()

    def refresh_credentials(_refresh_token, *, use_json):
        assert use_json is False
        post_entered.set()
        assert release_post.wait(timeout=5)
        return {
            "access_token": "refreshed-access",
            "refresh_token": "refreshed-refresh",
            "expires_at_ms": 9_999_999_999_999,
        }

    monkeypatch.setattr(
        anthropic_adapter,
        "refresh_anthropic_oauth_pure",
        refresh_credentials,
    )
    results = []
    refresher = threading.Thread(
        target=lambda: results.append(pool._refresh_entry(entry, force=False))
    )
    refresher.start()
    assert post_entered.wait(timeout=5)
    current = pool.entries()[0]
    routed = replace(current, priority=4, request_count=9)
    assert pool._replace_entry(current, routed) == routed
    pool._persist_pending_changes()
    release_post.set()
    refresher.join(timeout=5)

    assert not refresher.is_alive()
    assert len(results) == 1 and results[0] is not None
    assert _tokens(results[0]) == ("refreshed-access", "refreshed-refresh")
    assert (results[0].priority, results[0].request_count) == (4, 9)
    memory = pool.entries()[0]
    disk = _disk_entries(auth_path)[0]
    assert (memory.priority, memory.request_count) == (4, 9)
    assert (disk.priority, disk.request_count) == (4, 9)
