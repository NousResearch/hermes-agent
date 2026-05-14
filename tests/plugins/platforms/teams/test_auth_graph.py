"""Tests for the MSAL-backed Graph token provider."""
from __future__ import annotations

import asyncio
import sys
import time
from unittest.mock import MagicMock

import pytest

from plugins.platforms.teams.auth_graph import AuthError, GraphTokenProvider


def _make_fake_msal_app(token: str = "tok", expires_in: int = 3600) -> MagicMock:
    """Return a MagicMock standing in for msal.ConfidentialClientApplication."""
    fake = MagicMock()
    fake.acquire_token_for_client.return_value = {
        "access_token": token,
        "expires_in": expires_in,
    }
    return fake


def _make_provider_with_fake(fake_app: MagicMock) -> GraphTokenProvider:
    p = GraphTokenProvider(
        client_id="client-id",
        tenant_id="tenant-id",
        client_secret="secret",
    )
    # Patch the lazy builder to return our fake — never touches MSAL or network.
    p._build_msal_app = lambda: fake_app  # type: ignore[method-assign]
    return p


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_token_caches_per_scope():
    fake = _make_fake_msal_app(token="tok", expires_in=3600)
    p = _make_provider_with_fake(fake)

    t1 = await p.get_token("https://graph.microsoft.com/.default")
    t2 = await p.get_token("https://graph.microsoft.com/.default")

    assert t1 == "tok"
    assert t2 == "tok"
    assert fake.acquire_token_for_client.call_count == 1


@pytest.mark.asyncio
async def test_get_token_separates_caches_per_scope():
    fake = _make_fake_msal_app(token="tok", expires_in=3600)
    p = _make_provider_with_fake(fake)

    await p.get_token("scope-a")
    await p.get_token("scope-b")

    assert fake.acquire_token_for_client.call_count == 2
    # Verify the actual scope arg was passed in each call.
    call_scopes = [
        call.args[0] if call.args else call.kwargs.get("scopes")
        for call in fake.acquire_token_for_client.call_args_list
    ]
    assert call_scopes == [["scope-a"], ["scope-b"]]


# ---------------------------------------------------------------------------
# Refresh near expiry
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_token_refreshes_when_near_expiry():
    fake = _make_fake_msal_app(token="tok", expires_in=120)
    p = _make_provider_with_fake(fake)

    # First call caches with ~120s lifetime.
    await p.get_token("scope-x")
    assert fake.acquire_token_for_client.call_count == 1

    # Second call with a refresh window LARGER than the remaining lifetime
    # should treat the cached token as near-expiry and refresh.
    await p.get_token("scope-x", refresh_if_within_seconds=200)
    assert fake.acquire_token_for_client.call_count == 2


@pytest.mark.asyncio
async def test_get_token_refreshes_when_clock_advances(monkeypatch):
    """If wall-clock advances past expires_at - leeway, refresh."""
    fake = _make_fake_msal_app(token="tok", expires_in=120)
    p = _make_provider_with_fake(fake)

    real_time = time.time
    fake_now = [real_time()]

    def fake_time() -> float:
        return fake_now[0]

    # Patch the time function used inside auth_graph.
    monkeypatch.setattr(
        "plugins.platforms.teams.auth_graph.time.time",
        fake_time,
    )

    await p.get_token("scope-x")
    assert fake.acquire_token_for_client.call_count == 1

    # Jump forward beyond the cached token's effective lifetime.
    fake_now[0] += 200

    await p.get_token("scope-x")
    assert fake.acquire_token_for_client.call_count == 2


# ---------------------------------------------------------------------------
# Error path
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_token_raises_on_msal_error():
    fake = MagicMock()
    fake.acquire_token_for_client.return_value = {
        "error": "invalid_client",
        "error_description": "bad",
    }
    p = _make_provider_with_fake(fake)

    with pytest.raises(AuthError) as exc_info:
        await p.get_token("scope-x")

    msg = str(exc_info.value)
    assert "invalid_client" in msg
    assert "bad" in msg


@pytest.mark.asyncio
async def test_get_token_raises_on_msal_error_without_description():
    fake = MagicMock()
    fake.acquire_token_for_client.return_value = {"error": "invalid_client"}
    p = _make_provider_with_fake(fake)

    with pytest.raises(AuthError) as exc_info:
        await p.get_token("scope-x")

    assert "invalid_client" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_token_concurrent_calls_only_acquire_once():
    """Two concurrent get_token calls for the same scope must collapse into
    a single MSAL acquisition thanks to the per-scope async lock."""
    import threading

    loop = asyncio.get_running_loop()
    call_event = asyncio.Event()
    release_event = threading.Event()  # cross-thread signal — safe from worker

    call_count = 0

    def _acquire(scopes):
        nonlocal call_count
        call_count += 1
        # Signal (from worker thread) that we're inside MSAL.
        loop.call_soon_threadsafe(call_event.set)
        # Block until the test releases us, forcing the second coroutine
        # to contend on the per-scope lock.
        release_event.wait(timeout=5.0)
        return {"access_token": "tok", "expires_in": 3600}

    fake = MagicMock()
    fake.acquire_token_for_client.side_effect = _acquire
    p = _make_provider_with_fake(fake)

    async def call() -> str:
        return await p.get_token("scope-shared")

    task1 = asyncio.create_task(call())
    # Wait until task1 has reached the executor (and thus is holding the lock).
    await call_event.wait()
    task2 = asyncio.create_task(call())
    # Give task2 a moment to reach the lock.
    await asyncio.sleep(0.05)
    # Now release the MSAL call so both tasks can complete.
    release_event.set()

    results = await asyncio.gather(task1, task2)

    assert results == ["tok", "tok"]
    assert call_count == 1
    assert fake.acquire_token_for_client.call_count == 1


@pytest.mark.asyncio
async def test_get_token_concurrent_unseen_scope_locks_correctly():
    """Two concurrent get_token() calls on a scope NEVER seen before must
    still collapse onto a single MSAL acquisition.

    This pins the lock-dict invariant: the lock for a fresh scope must be
    created atomically (setdefault), not via a get-check-set sequence that
    could race even if today's CPython asyncio scheduling masks it.
    """
    import threading

    loop = asyncio.get_running_loop()
    call_event = asyncio.Event()
    release_event = threading.Event()

    call_count = 0

    def _acquire(scopes):
        nonlocal call_count
        call_count += 1
        loop.call_soon_threadsafe(call_event.set)
        release_event.wait(timeout=5.0)
        return {"access_token": "tok", "expires_in": 3600}

    fake = MagicMock()
    fake.acquire_token_for_client.side_effect = _acquire
    p = _make_provider_with_fake(fake)

    # Sanity: scope has never been seen — no lock entry yet.
    assert "fresh-scope" not in p._locks

    async def call() -> str:
        return await p.get_token("fresh-scope")

    # Kick off two tasks before the lock dict has any entry for this scope.
    task1 = asyncio.create_task(call())
    task2 = asyncio.create_task(call())

    # Wait until at least one task is inside MSAL.
    await call_event.wait()
    # Give the other task time to attempt the lock.
    await asyncio.sleep(0.05)
    release_event.set()

    results = await asyncio.gather(task1, task2)

    assert results == ["tok", "tok"]
    assert call_count == 1
    assert fake.acquire_token_for_client.call_count == 1
    # Both coroutines must have ended up sharing the same Lock instance.
    assert len(p._locks) == 1
    assert "fresh-scope" in p._locks


@pytest.mark.asyncio
async def test_get_token_different_scopes_do_not_block():
    """Per-scope locks must not serialize calls across different scopes.

    With scope A's MSAL call blocked, a concurrent call for scope B must
    complete promptly — proving locks are partitioned per scope, not global.
    """
    import threading

    loop = asyncio.get_running_loop()
    a_in_msal = asyncio.Event()
    a_release = threading.Event()
    b_done = asyncio.Event()

    def _acquire(scopes):
        scope = scopes[0]
        if scope == "scope-A":
            loop.call_soon_threadsafe(a_in_msal.set)
            a_release.wait(timeout=5.0)
            return {"access_token": "tok-a", "expires_in": 3600}
        # scope-B: return immediately.
        return {"access_token": "tok-b", "expires_in": 3600}

    fake = MagicMock()
    fake.acquire_token_for_client.side_effect = _acquire
    p = _make_provider_with_fake(fake)

    async def call_a() -> str:
        return await p.get_token("scope-A")

    async def call_b() -> str:
        try:
            return await p.get_token("scope-B")
        finally:
            b_done.set()

    task_a = asyncio.create_task(call_a())
    # Wait until A is inside MSAL and holding A's lock.
    await a_in_msal.wait()
    # Now start B — it must NOT be blocked by A.
    task_b = asyncio.create_task(call_b())

    # B must complete BEFORE we release A.
    await asyncio.wait_for(b_done.wait(), timeout=2.0)
    assert task_b.done(), "scope-B was blocked by scope-A's lock"
    assert not task_a.done(), "scope-A should still be waiting on MSAL"

    # Now let A finish.
    a_release.set()

    results = await asyncio.gather(task_a, task_b)
    assert results == ["tok-a", "tok-b"]
    assert fake.acquire_token_for_client.call_count == 2


# ---------------------------------------------------------------------------
# Lazy import of msal
# ---------------------------------------------------------------------------

def test_msal_is_lazy_imported():
    """Importing the auth_graph module must NOT pull in msal at import time.

    msal is only needed when _build_msal_app() actually runs, so the
    plugin can be loaded even if msal isn't installed (lazy_deps will
    install it on demand).
    """
    # Drop msal & auth_graph from sys.modules so we can observe a fresh import.
    sys.modules.pop("msal", None)
    sys.modules.pop("plugins.platforms.teams.auth_graph", None)
    # Importing should not pull msal in.
    import importlib
    importlib.import_module("plugins.platforms.teams.auth_graph")
    assert "msal" not in sys.modules


# ---------------------------------------------------------------------------
# expires_in default
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_token_handles_missing_expires_in():
    fake = MagicMock()
    fake.acquire_token_for_client.return_value = {"access_token": "tok"}
    p = _make_provider_with_fake(fake)

    token = await p.get_token("scope-x")
    assert token == "tok"
