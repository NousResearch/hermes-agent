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
import os
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

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
    expires_at_ms=None,
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
        expires_at_ms=expires_at_ms,
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


def _patch_refresh_pure(
    monkeypatch, *, rotated_access="rotated-access", rotated_refresh="rotated-refresh"
):
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


def test_minimax_oauth_pool_refresh_writes_through_to_root(
    profile_and_root, monkeypatch
):
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
    monkeypatch.setattr(A, "get_provider_auth_state", lambda _pid: dict(root_state))

    refreshed = pool._refresh_entry(pool._entries[0], force=True)

    assert refreshed is not None
    assert refreshed.access_token == "rotated-access"
    assert refreshed.refresh_token == "rotated-refresh"

    # The profile has no own providers.minimax-oauth block; it borrowed the
    # grant from root.  Source-aware write-back persists the rotated chain to
    # root directly and intentionally does NOT create a profile-local shadow
    # that would break future write-throughs (Review A MUST-FIX #2).
    profile = _read_store(profile_path)
    assert "minimax-oauth" not in profile.get("providers", {})
    assert profile.get("active_provider") != "minimax-oauth"

    # AND the global root no longer holds the revoked refresh token.
    root = _read_store(root_path)
    assert root["providers"]["minimax-oauth"]["access_token"] == "rotated-access"
    assert root["providers"]["minimax-oauth"]["refresh_token"] == "rotated-refresh"


# ---------------------------------------------------------------------------
# 2. Borrowed-root grant stays owned by root across two successive refreshes
# ---------------------------------------------------------------------------


def test_minimax_oauth_borrowed_root_grant_stays_owned_across_two_refreshes(
    profile_and_root, monkeypatch
):
    """Two successive refreshes of a borrowed-root grant both write back to root.

    A profile that borrows the global-root grant must not create a local
    ``providers.minimax-oauth`` shadow after the first refresh.  If it did, the
    second refresh would treat the profile as the owner and stop writing to
    root, leaving root with a stale (and eventually revoked) refresh token
    (Review A MUST-FIX #2).
    """
    profile_path, root_path = profile_and_root
    _write_store(
        profile_path, {"version": 1, "providers": {}, "active_provider": "openrouter"}
    )
    root_state = _minimax_state()
    _write_store(
        root_path,
        {"version": 1, "providers": {"minimax-oauth": dict(root_state)}},
    )

    rotations = _patch_refresh_pure(monkeypatch)
    monkeypatch.setattr(A, "get_provider_auth_state", lambda _pid: dict(root_state))

    pool = CredentialPool("minimax-oauth", [_entry()])

    # First refresh rotates root's grant and must write back to root.
    first = pool._refresh_entry(pool._entries[0], force=True)
    assert first is not None
    assert first.access_token == "rotated-access"

    # No profile-local shadow was created.
    profile = _read_store(profile_path)
    assert "minimax-oauth" not in profile.get("providers", {})

    root_after_first = _read_store(root_path)
    assert (
        root_after_first["providers"]["minimax-oauth"]["access_token"]
        == "rotated-access"
    )

    # Second refresh must again write the newly rotated chain back to root,
    # not to a now-materialized profile shadow.
    second = pool._refresh_entry(first, force=True)
    assert second is not None
    assert second.access_token == "rotated-access"

    # Root received the second rotation too.
    root_after_second = _read_store(root_path)
    assert (
        root_after_second["providers"]["minimax-oauth"]["access_token"]
        == "rotated-access"
    )
    # Profile still has no shadow.
    profile = _read_store(profile_path)
    assert "minimax-oauth" not in profile.get("providers", {})


# ---------------------------------------------------------------------------
# 2b. load_pool seeds MiniMax OAuth expiry from singleton state
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("owner", ["root", "profile"])
def test_minimax_oauth_load_pool_hydrates_and_refreshes_owned_source(
    profile_and_root, monkeypatch, owner
):
    """The real load path must hydrate and refresh either owning auth store.

    ``load_pool`` must carry both expiry and refresh-token state in memory so
    proactive selection can reach the source-aware refresh transaction.  A
    root-borrowed grant must remain metadata-only in the profile pool on disk.
    """
    profile_path, root_path = profile_and_root
    expired_state = _minimax_state(expires_in_future_seconds=-10)
    profile_providers = (
        {"minimax-oauth": dict(expired_state)} if owner == "profile" else {}
    )
    root_providers = {"minimax-oauth": dict(expired_state)} if owner == "root" else {}
    _write_store(
        profile_path,
        {
            "version": 1,
            "providers": profile_providers,
            "active_provider": "openrouter",
        },
    )
    _write_store(root_path, {"version": 1, "providers": root_providers})
    calls = _patch_refresh_pure(monkeypatch)

    pool = CP.load_pool("minimax-oauth")

    entry = pool.peek()
    assert entry is not None
    assert entry.source == "oauth"
    assert entry.expires_at == expired_state["expires_at"]
    assert entry.expires_at_ms is not None
    assert entry.refresh_token == expired_state["refresh_token"]

    if owner == "root":
        profile_after_load = _read_store(profile_path)
        persisted = profile_after_load["credential_pool"]["minimax-oauth"][0]
        assert persisted["source_auth_path"] == str(root_path)
        assert "access_token" not in persisted
        assert "refresh_token" not in persisted

    selected = pool.select()

    assert selected is not None
    assert selected.access_token == "rotated-access"
    assert selected.refresh_token == "rotated-refresh"
    assert calls["count"] == 1

    source_path = root_path if owner == "root" else profile_path
    source = _read_store(source_path)
    state = source["providers"]["minimax-oauth"]
    assert state["access_token"] == "rotated-access"
    assert state["refresh_token"] == "rotated-refresh"

    if owner == "root":
        profile_after_refresh = _read_store(profile_path)
        assert "minimax-oauth" not in profile_after_refresh.get("providers", {})
        persisted = profile_after_refresh["credential_pool"]["minimax-oauth"][0]
        assert "access_token" not in persisted
        assert "refresh_token" not in persisted


# ---------------------------------------------------------------------------
# 2. Profile-local shadow is NOT promoted to root
# ---------------------------------------------------------------------------


def test_minimax_oauth_profile_local_login_not_promoted_to_root(
    profile_and_root, monkeypatch
):
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
        profile["providers"]["minimax-oauth"]["refresh_token"] == "profile-new-refresh"
    )

    # Root keeps its own grant — write-through must not run when the profile
    # owns the block.
    root = _read_store(root_path)
    assert (
        root["providers"]["minimax-oauth"]["refresh_token"] == "root-untouched-refresh"
    )


# ---------------------------------------------------------------------------
# 3. Lock held across the POST (concurrency safety invariant)
# ---------------------------------------------------------------------------


def test_minimax_oauth_pool_refresh_holds_auth_store_lock_across_post(
    monkeypatch, tmp_path
):
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
    _write_store(
        profile_path,
        {"version": 1, "providers": {"minimax-oauth": _minimax_state()}},
    )

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


def test_minimax_oauth_concurrent_refresh_single_rotation(
    profile_and_root, monkeypatch
):
    """Two workers sharing one root grant refresh concurrently.

    Because the POST is serialized through the source-aware
    ``_provider_state_transaction`` (which holds both the active profile lock
    and the global-root source lock for borrowed grants), the loser must NOT
    re-POST — exactly one token-endpoint exchange occurs.  Without
    serialization both workers would read the same root refresh token, both
    POST it, and the loser would get ``refresh_token_reused``, revoking the
    whole chain.

    This test exercises the serialization within one process: both workers
    share the same profile auth path and contend on the same root source lock.
    The cross-process variant (two distinct profile lock files contending on
    one root lock) is covered by the lock-mechanism design: the root lock is
    a kernel flock keyed by the root auth.json path, so distinct processes
    holding distinct profile locks still serialize on the shared root lock.
    """
    profile_path, root_path = profile_and_root
    root_state = _minimax_state()
    _write_store(
        root_path,
        {"version": 1, "providers": {"minimax-oauth": dict(root_state)}},
    )
    # Both workers share the same profile_path; the contention is over the
    # root grant + the source lock, which is the real hazard.
    _write_store(profile_path, {"version": 1, "providers": {}})

    calls = _patch_refresh_pure(monkeypatch)
    monkeypatch.setattr(A, "get_provider_auth_state", lambda _pid: dict(root_state))

    barrier = threading.Barrier(2)
    results: dict = {"p1": None, "p2": None}
    errors: dict = {"p1": None, "p2": None}

    def refresh_worker(key):
        pool = CredentialPool("minimax-oauth", [_entry(id=key)])
        # Synchronize both threads so they race for the source lock.
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
    assert calls["count"] == 1, f"expected exactly 1 refresh POST, got {calls['count']}"

    # At least one worker produced a refreshed entry.  The loser may have
    # short-circuited (adopted the winner's rotated token without POSTing)
    # OR also returned the rotated entry — both are acceptable so long as
    # only one POST happened.
    refreshed_any = [r for r in results.values() if r is not None]
    assert refreshed_any, "at least one worker should have a refreshed entry"

    # Root must hold the single rotated chain (not a stale or second-posted
    # refresh token) so other borrowers see the winner's tokens.
    root = _read_store(root_path)
    root_mm = root["providers"]["minimax-oauth"]
    assert root_mm["access_token"] == "rotated-access"
    assert root_mm["refresh_token"] == "rotated-refresh"


# ---------------------------------------------------------------------------
# 4b. Two DISTINCT profiles (distinct profile lock files) sharing one root
#     grant are serialized by the root flock — exactly one POST.
# ---------------------------------------------------------------------------

# Subprocess worker script: each instance is a distinct "profile" with its own
# profile auth.json but sharing one global-root auth.json.  The source-aware
# transaction must serialize both on the root flock so only one POST occurs.
_CROSS_PROFILE_WORKER = """
import json, os, sys, time, fcntl
from pathlib import Path

profile_dir = Path(sys.argv[1])   # distinct per worker
root_path    = Path(sys.argv[2])   # shared
sync_dir     = Path(sys.argv[3])   # shared coordination dir
worker_id    = sys.argv[4]         # "w1" or "w2"
sibling_id   = "w2" if worker_id == "w1" else "w1"

# Make this subprocess look like a profile process.
os.environ["HERMES_HOME"] = str(profile_dir)
os.environ["HOME"] = str(sync_dir / "fake-home")
os.environ.pop("PYTEST_CURRENT_TEST", None)

from hermes_cli import auth as A
from agent import credential_pool as CP
from agent.credential_pool import CredentialPool, PooledCredential, AUTH_TYPE_OAUTH

profile_path = profile_dir / "auth.json"
A._auth_file_path = lambda: profile_path
A._global_auth_file_path = lambda: root_path

post_counter = sync_dir / "post_count.json"
post_lock = sync_dir / "post_count.lock"

def fake_refresh(state, **kwargs):
    # Atomic increment under a file lock (visible across processes).
    with open(post_lock, "w") as lf:
        fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
        try:
            n = 0
            if post_counter.exists():
                n = json.loads(post_counter.read_text()).get("count", 0)
            post_counter.write_text(json.dumps({"count": n + 1}))
        finally:
            fcntl.flock(lf.fileno(), fcntl.LOCK_UN)
    time.sleep(0.15)  # overlap window if unserialized
    from datetime import datetime, timedelta, timezone
    now = datetime.now(timezone.utc)
    return {
        "access_token": "rotated-access",
        "refresh_token": "rotated-refresh",
        "obtained_at": now.isoformat(),
        "expires_at": (now + timedelta(seconds=3600)).isoformat(),
        "expires_in": 3600,
    }

A.refresh_minimax_oauth_pure = fake_refresh
CP.refresh_minimax_oauth_pure = fake_refresh
root_state = json.loads(root_path.read_text())["providers"]["minimax-oauth"]
A.get_provider_auth_state = lambda _pid: dict(root_state)

# Barrier: signal ready, wait for sibling.
(sync_dir / f"ready_{worker_id}").touch()
deadline = time.monotonic() + 15
while not (sync_dir / f"ready_{sibling_id}").exists():
    if time.monotonic() > deadline:
        break
    time.sleep(0.02)

entry = PooledCredential(
    id="e1", auth_type=AUTH_TYPE_OAUTH,
    provider="minimax-oauth", label="minimax-oauth", priority=0,
    access_token="old-access", refresh_token="old-refresh",
    source="oauth", base_url="https://api.minimax.io/anthropic",
)
pool = CredentialPool("minimax-oauth", [entry])
result = pool._refresh_entry(pool._entries[0], force=True)
(sync_dir / f"result_{worker_id}").write_text(json.dumps({
    "access_token": result.access_token if result else None,
}))
"""


def test_minimax_oauth_distinct_profiles_sharing_root_serialize_to_one_post(tmp_path):
    """Two genuinely distinct profiles (distinct profile lock files) must not
    both POST the same single-use root refresh token.

    Review A MUST-FIX #1: the hazard is that two real profiles each hold a
    *different* profile lock, both read the same root refresh token, and both
    POST it before either takes the root lock.  The source-aware
    ``_provider_state_transaction`` holds both the active profile lock AND the
    global-root source lock, so distinct profiles serialize on the shared root
    flock.  This test spawns two subprocesses (distinct HERMES_HOME / profile
    auth.json paths) sharing one root auth.json and asserts exactly one POST.

    Uses subprocesses (not threads) so each worker has its own
    ``_auth_file_path`` without a process-global monkeypatch.
    """
    import subprocess
    import sys

    sync_dir = tmp_path / "sync"
    sync_dir.mkdir()

    # Two distinct profile directories → distinct profile auth.json + .lock.
    profile1_dir = tmp_path / "profiles" / "w1"
    profile1_dir.mkdir(parents=True)
    profile2_dir = tmp_path / "profiles" / "w2"
    profile2_dir.mkdir(parents=True)
    _write_store(profile1_dir / "auth.json", {"version": 1, "providers": {}})
    _write_store(profile2_dir / "auth.json", {"version": 1, "providers": {}})

    # Shared root grant.
    root_path = tmp_path / "root" / "auth.json"
    _write_store(
        root_path,
        {"version": 1, "providers": {"minimax-oauth": dict(_minimax_state())}},
    )

    worker_script = sync_dir / "worker.py"
    worker_script.write_text(_CROSS_PROFILE_WORKER)

    env = os.environ.copy()
    # Ensure the subprocess can import hermes from the repo root.
    repo_root = str(Path(CP.__file__).resolve().parents[1])
    env["PYTHONPATH"] = repo_root + os.pathsep + env.get("PYTHONPATH", "")

    procs = []
    for wid in ("w1", "w2"):
        pdir = profile1_dir if wid == "w1" else profile2_dir
        procs.append(
            subprocess.Popen(
                [
                    sys.executable,
                    str(worker_script),
                    str(pdir),
                    str(root_path),
                    str(sync_dir),
                    wid,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
            )
        )

    # Wait for both with a generous timeout.
    out = {}
    for wid, p in zip(("w1", "w2"), procs):
        stdout, stderr = p.communicate(timeout=30)
        out[wid] = (p.returncode, stdout, stderr)

    for wid in ("w1", "w2"):
        rc, stdout, stderr = out[wid]
        assert rc == 0, (
            f"worker {wid} exited {rc}\\n"
            f"stdout: {stdout.decode(errors='replace')}\\n"
            f"stderr: {stderr.decode(errors='replace')}"
        )

    post_count = json.loads((sync_dir / "post_count.json").read_text())["count"]
    assert post_count == 1, (
        f"expected exactly 1 POST across two distinct profiles sharing root, "
        f"got {post_count}"
    )

    # Root holds the single rotated chain.
    root = _read_store(root_path)
    root_mm = root["providers"]["minimax-oauth"]
    assert root_mm["access_token"] == "rotated-access"
    assert root_mm["refresh_token"] == "rotated-refresh"


# Subprocess borrower used by the source-disappearance regression below.  The
# parent test process owns the root lock, lets this profile resolve the root
# grant, removes that grant while the borrower waits for the same root lock,
# then releases it.  A fail-closed borrower must observe the in-lock re-read as
# authoritative and return without calling the token endpoint.
_SOURCE_DISAPPEARANCE_WORKER = """
import json, os, sys
from pathlib import Path

profile_dir = Path(sys.argv[1])
root_path = Path(sys.argv[2])
sync_dir = Path(sys.argv[3])

os.environ["HERMES_HOME"] = str(profile_dir)
os.environ["HOME"] = str(sync_dir / "fake-home")
os.environ.pop("PYTEST_CURRENT_TEST", None)

from hermes_cli import auth as A
from agent import credential_pool as CP
from agent.credential_pool import CredentialPool, PooledCredential, AUTH_TYPE_OAUTH

profile_path = profile_dir / "auth.json"
A._auth_file_path = lambda: profile_path
A._global_auth_file_path = lambda: root_path

root_state = json.loads(root_path.read_text())["providers"]["minimax-oauth"]
A.get_provider_auth_state = lambda _pid: dict(root_state)

resolved = sync_dir / "borrower_resolved_root"
original_load_with_source = A._load_provider_state_with_source

def traced_load_with_source(auth_store, provider_id):
    state, source_path = original_load_with_source(auth_store, provider_id)
    if source_path is not None and A._same_path(source_path, root_path):
        resolved.touch()
    return state, source_path

A._load_provider_state_with_source = traced_load_with_source

def fake_refresh(state, **kwargs):
    (sync_dir / "post_called").touch()
    return {
        "access_token": "resurrected-access",
        "refresh_token": "resurrected-refresh",
        "expires_at": root_state["expires_at"],
        "expires_in": 3600,
    }

A.refresh_minimax_oauth_pure = fake_refresh
CP.refresh_minimax_oauth_pure = fake_refresh

entry = PooledCredential(
    id="e1", auth_type=AUTH_TYPE_OAUTH,
    provider="minimax-oauth", label="minimax-oauth", priority=0,
    access_token="old-access", refresh_token="old-refresh",
    source="oauth", base_url="https://api.minimax.io/anthropic",
    extra={"source_auth_path": str(root_path)},
)
pool = CredentialPool("minimax-oauth", [entry])
result = pool._refresh_entry(pool._entries[0], force=True)
(sync_dir / "borrower_result.json").write_text(json.dumps({
    "result_is_none": result is None,
    "remaining_entries": len(pool.entries()),
}))
"""


def test_minimax_oauth_borrower_fails_closed_when_root_grant_disappears_while_waiting(
    tmp_path,
):
    """A removed root grant must not be POSTed or resurrected by a waiter."""
    import subprocess
    import sys

    sync_dir = tmp_path / "sync"
    sync_dir.mkdir()
    profile_dir = tmp_path / "profiles" / "borrower"
    profile_dir.mkdir(parents=True)
    profile_path = profile_dir / "auth.json"
    _write_store(profile_path, {"version": 1, "providers": {}})

    root_path = tmp_path / "root" / "auth.json"
    _write_store(
        root_path,
        {"version": 1, "providers": {"minimax-oauth": dict(_minimax_state())}},
    )

    worker_script = sync_dir / "source_disappearance_worker.py"
    worker_script.write_text(_SOURCE_DISAPPEARANCE_WORKER)
    env = os.environ.copy()
    repo_root = str(Path(CP.__file__).resolve().parents[1])
    env["PYTHONPATH"] = repo_root + os.pathsep + env.get("PYTHONPATH", "")

    with A._auth_store_lock(target_path=root_path, timeout_seconds=5):
        proc = subprocess.Popen(
            [
                sys.executable,
                str(worker_script),
                str(profile_dir),
                str(root_path),
                str(sync_dir),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
        )
        resolved = sync_dir / "borrower_resolved_root"
        deadline = time.monotonic() + 15
        while not resolved.exists() and proc.poll() is None:
            if time.monotonic() > deadline:
                proc.kill()
                pytest.fail("borrower did not resolve the root grant before timeout")
            time.sleep(0.02)

        root = _read_store(root_path)
        root["providers"].pop("minimax-oauth")
        _write_store(root_path, root)

    stdout, stderr = proc.communicate(timeout=30)
    assert proc.returncode == 0, (
        f"borrower exited {proc.returncode}\n"
        f"stdout: {stdout.decode(errors='replace')}\n"
        f"stderr: {stderr.decode(errors='replace')}"
    )
    result = json.loads((sync_dir / "borrower_result.json").read_text())
    assert result == {"result_is_none": True, "remaining_entries": 0}
    assert not (sync_dir / "post_called").exists(), "removed grant was POSTed"

    root = _read_store(root_path)
    assert "minimax-oauth" not in root.get("providers", {})

    profile = _read_store(profile_path)
    assert "minimax-oauth" not in profile.get("providers", {})
    assert profile.get("credential_pool", {}).get("minimax-oauth") == []
    serialized_profile = json.dumps(profile)
    assert "old-access" not in serialized_profile
    assert "old-refresh" not in serialized_profile


# ---------------------------------------------------------------------------
# 5. Terminal refresh failure quarantines profile and root
# ---------------------------------------------------------------------------


def test_minimax_oauth_terminal_refresh_quarantines_root_and_profile(
    profile_and_root, monkeypatch
):
    """An invalid_grant terminal failure must wipe tokens from both stores.

    Mirrors ``test_resolve_credentials_quarantines_on_terminal_refresh_failure``
    for the eager-resolve path, but exercises the POOL refresh path.  After
    quarantine, the pool entry is removed and a subsequent ``load_pool``
    would not re-seed the dead singleton.

    Critical: a background pool refresh failure must NOT flip
    ``active_provider`` to ``minimax-oauth`` (Review A MUST-FIX #4).  The
    quarantine persists with ``set_active=False`` so the user's chosen
    provider survives a terminal failure they didn't initiate.
    """
    profile_path, root_path = profile_and_root
    root_state = _minimax_state()
    _write_store(
        root_path,
        {
            "version": 1,
            "providers": {"minimax-oauth": dict(root_state)},
            # Pre-existing active provider that quarantine must NOT change.
            "active_provider": "openrouter",
        },
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

    # active_provider must NOT have flipped to minimax-oauth.  A background
    # pool failure is not a user choosing a provider.
    assert root.get("active_provider") == "openrouter", (
        f"active_provider flipped to {root.get('active_provider')} during quarantine"
    )


# ---------------------------------------------------------------------------
# 6. Transient refresh failure retains credentials (no quarantine)
# ---------------------------------------------------------------------------


def test_minimax_oauth_transient_refresh_failure_retains_credentials(
    profile_and_root, monkeypatch
):
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
