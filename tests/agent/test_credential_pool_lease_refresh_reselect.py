"""acquire_lease must re-select after a deferred single-use-token refresh.

Post-merge gate-sweep finding on the #71775 salvage (deferred refresh moved
OUTSIDE the pool lock). ``select()`` re-selects once the refreshed entries
are back in rotation (credential_pool.py, select() -> "if pending_refresh:
re-select"); ``acquire_lease()`` did not, so a pool whose only entries all
needed a refresh returned None even though the refresh had just succeeded —
the caller saw "no credentials available" and failed a request that should
have gone through.

These tests stub ``_available_entries`` / ``_refresh_pending_entries`` at the
same seam the production deferred-refresh contract uses: _available_entries
returns ``(available, pending_refresh)`` and entries pending a refresh are
NOT in ``available`` until the refresh has run.
"""

import threading

from agent.credential_pool import CredentialPool, PooledCredential


def _entry(entry_id: str) -> PooledCredential:
    return PooledCredential(
        id=entry_id,
        provider="anthropic",
        auth_type="oauth",
        access_token="tok",
        label=entry_id,
        source="oauth",
        priority=0,
    )


def _bare_pool(entries):
    """Minimal pool shell — avoids disk/keyring I/O in __init__."""
    pool = CredentialPool.__new__(CredentialPool)
    pool._lock = threading.RLock()
    pool._entries = list(entries)
    pool._active_leases = {}
    pool._current_id = None
    pool._max_concurrent = 2
    pool._unmatched_rotation_streak = 0
    pool.provider = "anthropic"
    return pool


def _wire_deferred_refresh(pool, *, refresh_succeeds: bool = True):
    """Model the deferred-refresh contract with an explicit state flag."""
    state = {"needs_refresh": True, "refresh_calls": 0}

    def fake_refresh(pending):
        state["refresh_calls"] += 1
        if refresh_succeeds:
            state["needs_refresh"] = False

    def fake_available(clear_expired=False, refresh=False):
        if state["needs_refresh"]:
            # Pending a refresh -> not yet available.
            pending = [(e.id, "tok") for e in pool._entries] if refresh else []
            return [], pending
        return list(pool._entries), []

    pool._refresh_pending_entries = fake_refresh
    pool._available_entries = fake_available
    return state


def test_acquire_lease_reselects_after_deferred_refresh():
    """The only entry needs a refresh; once refreshed it is available, so a
    lease MUST be granted rather than reporting no credentials."""
    pool = _bare_pool([_entry("e1")])
    state = _wire_deferred_refresh(pool)

    lease = pool.acquire_lease()

    assert state["refresh_calls"] == 1, "the deferred refresh should run once"
    assert state["needs_refresh"] is False, "entry is available post-refresh"
    assert lease == "e1", (
        "acquire_lease returned None despite a successfully refreshed, "
        "available entry — the caller would fail an answerable request"
    )
    assert pool._active_leases.get("e1") == 1, "the lease must be recorded"


def test_acquire_lease_without_pending_refresh_does_not_double_select():
    """No pending refresh -> exactly one selection pass (no wasted work)."""
    pool = _bare_pool([_entry("e1")])
    state = _wire_deferred_refresh(pool)
    state["needs_refresh"] = False  # already healthy

    passes = {"n": 0}
    original = pool._acquire_lease_under_lock

    def counting(credential_id):
        passes["n"] += 1
        return original(credential_id)

    pool._acquire_lease_under_lock = counting

    lease = pool.acquire_lease()

    assert lease == "e1"
    assert passes["n"] == 1, "healthy pool must not trigger the retry path"
    assert state["refresh_calls"] == 0


def test_acquire_lease_still_none_when_refresh_does_not_help():
    """If the refresh leaves nothing available, None is still the answer —
    the retry must not loop or invent a credential."""
    pool = _bare_pool([_entry("e1")])
    state = _wire_deferred_refresh(pool, refresh_succeeds=False)

    assert pool.acquire_lease() is None
    assert state["refresh_calls"] == 1, "retry must not refresh repeatedly"
    assert pool._active_leases == {}
