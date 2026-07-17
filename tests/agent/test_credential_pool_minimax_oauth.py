"""Regression tests for MiniMax OAuth credential-pool refresh parity.

Companion to ``test_credential_pool_oauth_writethrough.py``.  These tests
bring ``minimax-oauth`` to the same cross-profile OAuth safety already
present for Nous / OpenAI Codex / xAI:

- Source-aware, source-locked refresh: the token-endpoint POST runs inside
  the shared cross-process ``_auth_store_lock`` so two profiles sharing the
  same MiniMax OAuth grant cannot both POST the same single-use refresh
  token (issue #48415, the MiniMax analog).
- Root write-through: when a profile pool refreshes a grant it resolved
  from the global-root fallback, the rotated chain is written back to root.
- Profile-local shadow: a profile that genuinely owns its own
  ``providers.minimax-oauth`` block is NOT clobbered by root, and its
  refresh does not promote into root.
- Terminal quarantine: an ``invalid_grant`` / ``refresh_token_reused``
  wipes the dead tokens from profile and root and marks the pool entry
  ``STATUS_DEAD``.
- Transient failure: a non-terminal refresh error (``relogin_required``
  ``False``) retains credentials — no quarantine.

All tests drive the real ``CredentialPool`` against real on-disk auth
stores (profile + root under ``tmp_path``) with the network POST stubbed.
No live credentials are read, printed, or exchanged.
"""

from __future__ import annotations

import contextlib
import json
import threading
import time
from datetime import datetime, timedelta, timezone

import pytest

from agent import credential_pool as CP
from agent.credential_pool import (
    AUTH_TYPE_OAUTH,
    CredentialPool,
    PooledCredential,
    STATUS_DEAD,
)
from hermes_cli import auth as A
from hermes_cli.auth import AuthError


# ---------------------------------------------------------------------------
# Store helpers
# ---------------------------------------------------------------------------

def _write_store(path, store):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(store), encoding="utf-8")


def _read_store(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _minimax_state(
    *,
    access_token="old-access",
    refresh_token="old-refresh",
    region="global",
    expires_in_future_seconds=3600,
):
    """Return a complete MiniMax OAuth singleton state dict."""
    now = datetime.now(timezone.utc)
    expires_at = now + timedelta(seconds=expires_in_future_seconds)
    portal = "https://api.minimax.io"
    inference = "https://api.minimax.io/anthropic"
    if region == "cn":
        portal = "https://api.minimaxi.com"
        inference = "https://api.minimaxi.com/anthropic"
    return {
        "provider": "minimax-oauth",
        "region": region,
        "portal_base_url": portal,
        "inference_base_url": inference,
        "client_id": "test-minimax-client-id",
        "scope": "group_id profile model.completion",
        "token_type": "Bearer",
        "access_token": access_token,
        "refresh_token": refresh_token,
        "obtained_at": now.isoformat(),
        "expires_at": expires_at.isoformat(),
        "expires_in": expires_in_future_seconds,
    }


def _entry(
    *,
    id="e1",
    access_token="old-access",
    refresh_token="old-refresh",
    source="oauth",
    expires_at=None,
):
    """Build a MiniMax OAuth pool entry seeded from the singleton."""
    if expires_at is None:
        expires_at = (datetime.now(timezone.utc) - timedelta(seconds=10)).isoformat()
    return PooledCredential(
        provider="minimax-oauth",
        id=id,
        label="minimax-oauth",
        auth_type=AUTH_TYPE_OAUTH,
        priority=0,
        source=source,
        access_token=access_token,
        refresh_token=refresh_token,
        expires_at=expires_at,
        base_url="https://api.minimax.io/anthropic",
    )


@pytest.fixture
def profile_and_root(tmp_path, monkeypatch):
    """Wire a profile auth store + a distinct global-root auth store on disk."""
    profile_path = tmp_path / "profiles" / "work" / "auth.json"
    root_path = tmp_path / "root" / "auth.json"
    monkeypatch.setattr(A, "_auth_file_path", lambda: profile_path)
    monkeypatch.setattr(A, "_global_auth_file_path", lambda: root_path)
    # Keep the pytest seat-belt in _write_through_provider_state_to_global_root
    # from tripping by pointing HOME away from the tmp root.
    monkeypatch.setenv("HOME", str(tmp_path / "not-the-root"))
    return profile_path, root_path


def _patch_refresh_pure(monkeypatch, *, rotated_access="rotated-access",
                        rotated_refresh="rotated-refresh"):
    """Stub the network POST to return a deterministic rotated pair.

    Returns a dict tracking invocation count so concurrency tests can assert
    exactly one POST occurred.
    """
    calls = {"count": 0}
    calls_lock = threading.Lock()

    def fake_refresh(state, **kwargs):
        with calls_lock:
            calls["count"] += 1
        # Simulate a small delay so concurrent threads overlap before the
        # lock-serialized path serializes them.
        time.sleep(0.05)
        now_dt = datetime.now(timezone.utc)
        expires_at_unix = (now_dt + timedelta(seconds=3600)).timestamp()
        return {
            "access_token": rotated_access,
            "refresh_token": rotated_refresh,
            "obtained_at": now_dt.isoformat(),
            "expires_at": datetime.fromtimestamp(
                expires_at_unix, tz=timezone.utc
            ).isoformat(),
            "expires_in": 3600,
        }

    monkeypatch.setattr(A, "refresh_minimax_oauth_pure", fake_refresh)
    monkeypatch.setattr(CP, "refresh_minimax_oauth_pure", fake_refresh)
    return calls


# ---------------------------------------------------------------------------
# 1. Write-through to root when profile reads root fallback
# ---------------------------------------------------------------------------

def test_minimax_oauth_pool_refresh_writes_through_to_root(profile_and_root, monkeypatch):
    """A profile reading root's grant must push rotated tokens back to root.

    Mirrors ``test_pool_refresh_writes_through_to_root_when_profile_reads_root``
    for Codex/xAI.  The profile has no own ``providers.minimax-oauth`` block;
    it resolves the grant from the global-root fallback.  When the pool
    refresh rotates the single-use refresh token, the rotated chain MUST
    land back in root — otherwise root keeps a revoked refresh token and
    every other profile reading root dies with ``refresh_token_reused``.
    """
    profile_path, root_path = profile_and_root
    _write_store(profile_path, {"version": 1, "providers": {}})
    root_state = _minimax_state()
    _write_store(
        root_path,
        {"version": 1, "providers": {"minimax-oauth": dict(root_state)}},
    )

    _patch_refresh_pure(monkeypatch)

    pool = CredentialPool("minimax-oauth", [_entry()])
    # Stub get_provider_auth_state so the refresh impl can pull client_id /
    # portal_base_url from the persisted singleton (the pool entry does not
    # carry routing metadata).
    monkeypatch.setattr(
        A, "get_provider_auth_state", lambda _pid: dict(root_state)
    )

    refreshed = pool._refresh_entry(pool._entries[0], force=True)

    assert refreshed is not None
    assert refreshed.access_token == "rotated-access"
    assert refreshed.refresh_token == "rotated-refresh"

    # Profile store received the rotated state (set_active=False — must NOT
    # flip active_provider, which the user did not choose).
    profile = _read_store(profile_path)
    mm = profile["providers"]["minimax-oauth"]
    assert mm["access_token"] == "rotated-access"
    assert mm["refresh_token"] == "rotated-refresh"
    assert profile.get("active_provider") != "minimax-oauth"

    # AND the global root no longer holds the revoked refresh token.
    root = _read_store(root_path)
    assert root["providers"]["minimax-oauth"]["access_token"] == "rotated-access"
    assert root["providers"]["minimax-oauth"]["refresh_token"] == "rotated-refresh"


# ---------------------------------------------------------------------------
# 2. Profile-local shadow is NOT promoted to root
# ---------------------------------------------------------------------------

def test_minimax_oauth_profile_local_login_not_promoted_to_root(profile_and_root, monkeypatch):
    """A profile that genuinely owns its own minimax-oauth block shadows root.

    Its refresh must update the profile store but must NOT clobber the root
    grant (the profile owns its own login; root's grant belongs to root).
    """
    profile_path, root_path = profile_and_root
    _write_store(
        profile_path,
        {
            "version": 1,
            "providers": {
                "minimax-oauth": _minimax_state(
                    access_token="profile-old",
                    refresh_token="profile-old-refresh",
                )
            },
        },
    )
    _write_store(
        root_path,
        {
            "version": 1,
            "providers": {
                "minimax-oauth": _minimax_state(
                    access_token="root-untouched",
                    refresh_token="root-untouched-refresh",
                )
            },
        },
    )

    _patch_refresh_pure(
        monkeypatch,
        rotated_access="profile-new",
        rotated_refresh="profile-new-refresh",
    )

    pool = CredentialPool(
        "minimax-oauth",
        [
            _entry(
                access_token="profile-old",
                refresh_token="profile-old-refresh",
            )
        ],
    )
    monkeypatch.setattr(
        A,
        "get_provider_auth_state",
        lambda _pid: _minimax_state(
            access_token="profile-old", refresh_token="profile-old-refresh"
        ),
    )

    refreshed = pool._refresh_entry(pool._entries[0], force=True)
    assert refreshed is not None
    assert refreshed.refresh_token == "profile-new-refresh"

    # Profile store updated.
    profile = _read_store(profile_path)
    assert (
        profile["providers"]["minimax-oauth"]["refresh_token"]
        == "profile-new-refresh"
    )

    # Root keeps its own grant — write-through must not run when the profile
    # owns the block.
    root = _read_store(root_path)
    assert (
        root["providers"]["minimax-oauth"]["refresh_token"]
        == "root-untouched-refresh"
    )


# ---------------------------------------------------------------------------
# 3. Lock held across the POST (concurrency safety invariant)
# ---------------------------------------------------------------------------

def test_minimax_oauth_pool_refresh_holds_auth_store_lock_across_post(monkeypatch, tmp_path):
    """The MiniMax OAuth pool refresh must POST under the cross-process lock.

    MiniMax refresh tokens are single-use.  Serializing the
    sync -> refresh POST -> write-back through the shared ``_auth_store_lock``
    closes the window where two processes both POST the same token and the
    loser gets ``refresh_token_reused``.

    Asserts the invariant directly — that ``refresh_minimax_oauth_pure`` is
    only ever called while the auth-store lock is held.
    """
    profile_path = tmp_path / "auth.json"
    monkeypatch.setattr(A, "_auth_file_path", lambda: profile_path)
    monkeypatch.setattr(A, "_global_auth_file_path", lambda: None)
    monkeypatch.setenv("HOME", str(tmp_path / "not-the-root"))

    lock_held: dict = {"during_post": None}
    real_lock = A._auth_store_lock
    depth = {"n": 0}

    @contextlib.contextmanager
    def tracking_lock(*args, **kwargs):
        depth["n"] += 1
        try:
            with real_lock(*args, **kwargs):
                yield
        finally:
            depth["n"] -= 1

    monkeypatch.setattr(A, "_auth_store_lock", tracking_lock)
    monkeypatch.setattr(CP, "_auth_store_lock", tracking_lock)

    def fake_refresh(state, **kwargs):
        lock_held["during_post"] = depth["n"] > 0
        now_dt = datetime.now(timezone.utc)
        return {
            "access_token": "rotated-access",
            "refresh_token": "rotated-refresh",
            "obtained_at": now_dt.isoformat(),
            "expires_at": (now_dt + timedelta(seconds=3600)).isoformat(),
            "expires_in": 3600,
        }

    monkeypatch.setattr(A, "refresh_minimax_oauth_pure", fake_refresh)
    monkeypatch.setattr(CP, "refresh_minimax_oauth_pure", fake_refresh)
    monkeypatch.setattr(
        A,
        "get_provider_auth_state",
        lambda _pid: _minimax_state(),
    )

    entry = _entry()
    pool = CredentialPool("minimax-oauth", [entry])
    refreshed = pool._refresh_entry(entry, force=True)

    assert refreshed is not None
    assert refreshed.access_token == "rotated-access"
    # The invariant: the single-use token POST ran inside the auth-store lock.
    assert lock_held["during_post"] is True


# ---------------------------------------------------------------------------
# 4. Concurrent refresh produces exactly one POST
# ---------------------------------------------------------------------------

def test_minimax_oauth_concurrent_refresh_single_rotation(profile_and_root, monkeypatch):
    """Two profile pools sharing one root grant refresh concurrently.

    Because the POST is serialized through ``_auth_store_lock`` and the
    in-lock re-sync adopts the rotated token the winner persisted, the loser
    must NOT re-POST — exactly one token-endpoint exchange occurs.  Without
    serialization both would POST the same single-use refresh token and the
    loser would get ``refresh_token_reused``, revoking the whole chain.
    """
    profile_path, root_path = profile_and_root
    root_state = _minimax_state()
    _write_store(
        root_path,
        {"version": 1, "providers": {"minimax-oauth": dict(root_state)}},
    )
    # Both "profiles" share the same profile_path for this test (the
    # contention is over the root grant + the auth-store lock).
    _write_store(profile_path, {"version": 1, "providers": {}})

    calls = _patch_refresh_pure(monkeypatch)
    monkeypatch.setattr(A, "get_provider_auth_state", lambda _pid: dict(root_state))

    barrier = threading.Barrier(2)
    results: dict = {"p1": None, "p2": None}
    errors: dict = {"p1": None, "p2": None}

    def refresh_worker(key):
        pool = CredentialPool("minimax-oauth", [_entry(id=key)])
        # Synchronize both threads so they race for the lock.
        barrier.wait(timeout=5)
        try:
            results[key] = pool._refresh_entry(pool._entries[0], force=True)
        except Exception as exc:
            errors[key] = exc

    t1 = threading.Thread(target=refresh_worker, args=("p1",), name="minimax-p1")
    t2 = threading.Thread(target=refresh_worker, args=("p2",), name="minimax-p2")
    t1.start()
    t2.start()
    t1.join(timeout=10)
    t2.join(timeout=10)

    assert not t1.is_alive()
    assert not t2.is_alive()
    assert errors["p1"] is None, f"p1 errored: {errors['p1']}"
    assert errors["p2"] is None, f"p2 errored: {errors['p2']}"

    # Exactly one POST exchange occurred.
    assert calls["count"] == 1, (
        f"expected exactly 1 refresh POST, got {calls['count']}"
    )

    # At least one worker produced a refreshed entry.  The loser may have
    # short-circuited (adopted the winner's rotated token without POSTing)
    # OR also returned the rotated entry — both are acceptable so long as
    # only one POST happened.
    refreshed_any = [r for r in results.values() if r is not None]
    assert refreshed_any, "at least one worker should have a refreshed entry"


# ---------------------------------------------------------------------------
# 5. Terminal refresh failure quarantines profile and root
# ---------------------------------------------------------------------------

def test_minimax_oauth_terminal_refresh_quarantines_root_and_profile(profile_and_root, monkeypatch):
    """An invalid_grant terminal failure must wipe tokens from both stores.

    Mirrors ``test_resolve_credentials_quarantines_on_terminal_refresh_failure``
    for the eager-resolve path, but exercises the POOL refresh path.  After
    quarantine, the pool entry is removed and a subsequent ``load_pool``
    would not re-seed the dead singleton.
    """
    profile_path, root_path = profile_and_root
    root_state = _minimax_state()
    _write_store(
        root_path,
        {"version": 1, "providers": {"minimax-oauth": dict(root_state)}},
    )
    _write_store(profile_path, {"version": 1, "providers": {}})

    def terminal_refresh(state, **kwargs):
        raise AuthError(
            "invalid_grant",
            provider="minimax-oauth",
            code="refresh_failed",
            relogin_required=True,
        )

    monkeypatch.setattr(A, "refresh_minimax_oauth_pure", terminal_refresh)
    monkeypatch.setattr(CP, "refresh_minimax_oauth_pure", terminal_refresh)
    monkeypatch.setattr(A, "get_provider_auth_state", lambda _pid: dict(root_state))

    pool = CredentialPool("minimax-oauth", [_entry()])
    result = pool._refresh_entry(pool._entries[0], force=True)

    # Pool entry removed (quarantined to DEAD then dropped from rotation).
    assert result is None
    assert all(e.source != "oauth" for e in pool._entries), (
        "singleton-seeded minimax entry should have been removed"
    )

    # Root store: dead OAuth fields cleared, diagnostic blob written.
    root = _read_store(root_path)
    mm = root["providers"]["minimax-oauth"]
    assert "access_token" not in mm
    assert "refresh_token" not in mm
    assert "expires_at" not in mm
    err = mm.get("last_auth_error")
    assert isinstance(err, dict)
    assert err["provider"] == "minimax-oauth"
    assert err["relogin_required"] is True
    # Routing metadata preserved.
    assert mm["portal_base_url"] == root_state["portal_base_url"]
    assert mm["client_id"] == root_state["client_id"]


# ---------------------------------------------------------------------------
# 6. Transient refresh failure retains credentials (no quarantine)
# ---------------------------------------------------------------------------

def test_minimax_oauth_transient_refresh_failure_retains_credentials(profile_and_root, monkeypatch):
    """A non-terminal failure (relogin_required=False) must NOT quarantine.

    Transient failures (429 / 5xx / network) leave the stored refresh token
    intact so the next attempt can retry.  Quarantine is reserved for
    terminal states that cannot recover.
    """
    profile_path, root_path = profile_and_root
    root_state = _minimax_state()
    _write_store(
        root_path,
        {"version": 1, "providers": {"minimax-oauth": dict(root_state)}},
    )
    _write_store(profile_path, {"version": 1, "providers": {}})

    def transient_refresh(state, **kwargs):
        raise AuthError(
            "service unavailable",
            provider="minimax-oauth",
            code="refresh_failed",
            relogin_required=False,
        )

    monkeypatch.setattr(A, "refresh_minimax_oauth_pure", transient_refresh)
    monkeypatch.setattr(CP, "refresh_minimax_oauth_pure", transient_refresh)
    monkeypatch.setattr(A, "get_provider_auth_state", lambda _pid: dict(root_state))

    pool = CredentialPool("minimax-oauth", [_entry()])
    result = pool._refresh_entry(pool._entries[0], force=True)

    # Transient failure marks the entry exhausted (not quarantined/dead).
    assert result is None

    # Root store: tokens RETAINED — no quarantine write.
    root = _read_store(root_path)
    mm = root["providers"]["minimax-oauth"]
    assert mm["access_token"] == root_state["access_token"]
    assert mm["refresh_token"] == root_state["refresh_token"]
    assert "last_auth_error" not in mm


# ---------------------------------------------------------------------------
# 7. Terminal error detector unit test
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "code,relogin,expected",
    [
        ("refresh_failed", True, True),
        ("no_refresh_token", True, True),
        ("invalid_grant", True, True),
        ("refresh_token_reused", True, True),
        ("refresh_failed", False, False),  # transient
        ("refresh_failed", True, True),
    ],
)
def test_is_terminal_minimax_oauth_refresh_error(code, relogin, expected):
    exc = AuthError(
        "msg", provider="minimax-oauth", code=code, relogin_required=relogin
    )
    assert A._is_terminal_minimax_oauth_refresh_error(exc) is expected


def test_is_terminal_minimax_oauth_refresh_error_wrong_provider():
    exc = AuthError(
        "msg", provider="openai-codex", code="refresh_failed", relogin_required=True
    )
    assert A._is_terminal_minimax_oauth_refresh_error(exc) is False
