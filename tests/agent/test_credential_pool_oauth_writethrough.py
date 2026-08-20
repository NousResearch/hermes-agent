"""Regression tests for credential-pool OAuth refresh write-through to root.

Companion to ``tests/hermes_cli/test_xai_oauth_writethrough.py``. That file
covers the *non-pool* xAI refresh path (``_save_xai_oauth_tokens``). These
cover the **credential-pool** refresh path
(``CredentialPool._sync_device_code_entry_to_auth_store``): when a profile
that has no own ``providers.<id>`` block refreshes — via the pool — a rotating
OAuth grant it resolved from the global-root fallback, the rotated chain must
be written back to the global root too. Otherwise root keeps a revoked refresh
token and every other profile reading root's stale grant dies with
``refresh_token_reused`` / ``invalid_grant`` once its access token expires
(issue #48415, the Codex/xAI analog of #43589).

The tests drive the real ``_sync_device_code_entry_to_auth_store`` against
real on-disk auth stores (profile + root under ``tmp_path``) rather than
mocking the save boundary, so they exercise the actual atomic write path.
"""

import json
import multiprocessing
import threading
import time

import pytest

from agent import credential_pool as CP
from agent.credential_pool import (
    AUTH_TYPE_OAUTH,
    CredentialPool,
    PooledCredential,
    STATUS_EXHAUSTED,
)
from hermes_cli import auth as A


def _write_store(path, store):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(store), encoding="utf-8")


def _read_store(path):
    return json.loads(path.read_text(encoding="utf-8"))


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


def _codex_entry(
    *, id, access_token, refresh_token, source="manual:device_code", grant_id=None
):
    return PooledCredential(
        provider="openai-codex",
        id=id,
        label="codex",
        auth_type=AUTH_TYPE_OAUTH,
        priority=0,
        source=source,
        access_token=access_token,
        refresh_token=refresh_token,
        extra={} if grant_id is None else {"shared_grant_id": grant_id},
    )


@pytest.fixture
def profile_and_root(tmp_path, monkeypatch):
    """Wire a profile auth store + a distinct global-root auth store on disk.

    The pytest seat belt in ``_write_through_provider_state_to_global_root``
    only refuses the *real* user's ``$HOME/.hermes/auth.json``; a tmp_path
    root is allowed, so point HOME away from the tmp root to keep the guard
    from tripping on these fixtures.
    """
    profile_path = tmp_path / "profiles" / "work" / "auth.json"
    root_path = tmp_path / "root" / "auth.json"

    monkeypatch.setattr(A, "_auth_file_path", lambda: profile_path)
    monkeypatch.setattr(A, "_global_auth_file_path", lambda: root_path)
    monkeypatch.setenv("HOME", str(tmp_path / "not-the-root"))
    return profile_path, root_path








def test_global_write_through_preserves_concurrent_root_update(
    profile_and_root, monkeypatch
):
    """A stale profile write-through must not erase a concurrent root login."""
    _profile_path, root_path = profile_and_root
    _write_store(
        root_path,
        {
            "version": 1,
            "providers": {
                "xai-oauth": {
                    "tokens": {"access_token": "old-xai", "refresh_token": "old-r"}
                }
            },
            "credential_pool": {
                "anthropic": [{"id": "anthropic-existing"}],
                "openrouter": [{"id": "openrouter-existing"}],
            },
        },
    )

    helper_loaded = threading.Event()
    helper_has_target_lock = threading.Event()
    allow_helper_save = threading.Event()
    writer_started = threading.Event()
    writer_done = threading.Event()
    real_auth_load = A._load_auth_store

    def paused_helper_load(path=None):
        store = real_auth_load(path)
        if threading.current_thread().name == "profile-write-through":
            target_holder = A._auth_lock_holder_for(root_path)
            if getattr(target_holder, "depth", 0) > 0:
                helper_has_target_lock.set()
            helper_loaded.set()
            assert allow_helper_save.wait(timeout=5)
        return store

    monkeypatch.setattr(A, "_load_auth_store", paused_helper_load)
    # The pre-fix implementation imported the loader directly; patch both
    # bindings so reverting the safe helper still exercises the stale ordering.
    monkeypatch.setattr(CP, "_load_auth_store", paused_helper_load)

    def profile_write_through():
        CP._write_through_provider_state_to_global_root(
            "xai-oauth",
            {"tokens": {"access_token": "new-xai", "refresh_token": "new-r"}},
        )

    def concurrent_codex_login():
        writer_started.set()
        with A._auth_store_lock(target_path=root_path):
            store = A._load_auth_store(root_path)
            A._store_provider_state(
                store,
                "openai-codex",
                {"tokens": {"access_token": "codex-a", "refresh_token": "codex-r"}},
                set_active=False,
            )
            pool = store.setdefault("credential_pool", {})
            pool["openai-codex"] = [{"id": "codex-login"}]
            A._save_auth_store(store, target_path=root_path)
        writer_done.set()

    helper = threading.Thread(target=profile_write_through, name="profile-write-through")
    helper.start()
    assert helper_loaded.wait(timeout=5)

    writer = threading.Thread(target=concurrent_codex_login, name="concurrent-login")
    writer.start()
    assert writer_started.wait(timeout=5)
    # A fixed helper already owns the target lock, so the writer will merge
    # after release. A reverted unlocked helper must first let the competing
    # login finish; only then do we release its stale save. This makes the
    # losing pre-fix ordering deterministic rather than scheduler-dependent.
    if not helper_has_target_lock.is_set():
        assert writer_done.wait(timeout=5)
    allow_helper_save.set()
    helper.join(timeout=5)
    writer.join(timeout=5)
    assert not helper.is_alive()
    assert not writer.is_alive()

    root = _read_store(root_path)
    assert root["providers"]["xai-oauth"]["tokens"]["refresh_token"] == "new-r"
    assert root["providers"]["openai-codex"]["tokens"]["refresh_token"] == "codex-r"
    assert root["credential_pool"]["openai-codex"] == [{"id": "codex-login"}]
    assert root["credential_pool"]["anthropic"] == [{"id": "anthropic-existing"}]
    assert root["credential_pool"]["openrouter"] == [{"id": "openrouter-existing"}]


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
    monkeypatch.setattr(A, "_global_auth_file_path", lambda: None)
    monkeypatch.setenv("HOME", str(tmp_path / "not-the-root"))

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


def test_write_through_fires_on_every_refresh_not_just_first(
    profile_and_root, monkeypatch
):
    """Write-through to root must fire on the 2nd, 3rd, … refresh too (#74339).

    The old key-presence check decided write-through on whether the *profile*
    store had ``providers.<id>`` BEFORE the save — a key that
    ``_store_provider_state()`` unconditionally created.  Net effect: first
    refresh → write-through fires; every later refresh → silently disabled
    because the profile now "owned" the block, even though it never
    performed its own OAuth grant.

    The fix skips ``_store_provider_state`` entirely when the grant was
    resolved from root, so the profile never accrues a shadowing key and
    ``_load_provider_state_with_source`` always resolves from root.
    """
    profile_path, root_path = profile_and_root
    _write_store(
        root_path,
        {
            "version": 1,
            "providers": {
                "openai-codex": {
                    "tokens": {"access_token": "root-ac", "refresh_token": "root-rf"}
                }
            },
        },
    )

    provider = "openai-codex"
    # After patching A's module-level attributes, the bare-name imports in
    # credential_pool.py still hold references to the original functions
    # (``from X import Y`` creates a local binding that does not update when
    # ``X.Y`` is reassigned).  Patch CP's bindings separately so the
    # ``_sync_device_code_entry_to_auth_store`` method — whose __globals__
    # are ``agent.credential_pool.__dict__`` — sees the mocked paths.
    monkeypatch.setattr(CP, "_global_auth_file_path", lambda: root_path)
    monkeypatch.setattr(CP, "_same_path", lambda a, b: a == b)
    # Let _write_through_provider_state_to_global_root run for real so it
    # persists the rotated token pair to the root auth.json — the test
    # asserts the on-disk values after each refresh.

    # ---- REFRESH 1 ----
    _write_store(profile_path, {"version": 1})
    entry1 = _entry(
        provider, id="c1", access_token="ac1", refresh_token="rf1"
    )
    pool1 = CredentialPool(provider, [entry1])
    pool1._sync_device_code_entry_to_auth_store(entry1)

    # Verify root was updated with the rotated tokens from refresh 1.
    root_store = _read_store(root_path)
    root_tokens = root_store["providers"]["openai-codex"]["tokens"]
    assert root_tokens["access_token"] == "ac1"
    assert root_tokens["refresh_token"] == "rf1"

    # After refresh 1 the profile should NOT have a providers.openai-codex
    # block (the fix skipped _store_provider_state because the grant came
    # from root).  This prevents the self-sealing that broke refresh 2+.
    profile_store = _read_store(profile_path)
    assert "openai-codex" not in profile_store.get("providers", {}), (
        "profile must NOT accrue a shadowing providers.<id> block when the "
        "grant was resolved from root — that key would disable write-through "
        "on the next refresh (#74339)"
    )

    # ---- REFRESH 2 (same scenario, rotated tokens) ----
    entry2 = _entry(
        provider, id="c2", access_token="ac2", refresh_token="rf2"
    )
    pool2 = CredentialPool(provider, [entry2])
    pool2._sync_device_code_entry_to_auth_store(entry2)

    # Verify root was updated with the rotated tokens from refresh 2.
    # The old key-presence check would have silently skipped this write.
    root_store = _read_store(root_path)
    root_tokens = root_store["providers"]["openai-codex"]["tokens"]
    assert root_tokens["access_token"] == "ac2", (
        "refresh 2: root must carry the rotated token pair. "
        "The old code self-disabled write-through here (#74339)"
    )
    assert root_tokens["refresh_token"] == "rf2"


def test_exact_idless_alias_gets_random_shared_grant_id(profile_and_root):
    profile_path, root_path = profile_and_root
    _write_store(profile_path, {"version": 1, "providers": {}})
    _write_store(root_path, {"version": 1, "providers": {"openai-codex": {
        "tokens": {"access_token": "a", "refresh_token": "r"}}}})
    alias = _codex_entry(id="alias", access_token="a", refresh_token="r")
    second = _codex_entry(id="second", access_token="a", refresh_token="r")
    pool = CredentialPool("openai-codex", [alias, second])
    synced = pool._sync_codex_entry_from_auth_store(alias)
    second_synced = pool._sync_codex_entry_from_auth_store(second)
    grant = _read_store(root_path)["providers"]["openai-codex"]["shared_grant_id"]
    assert len(grant) == 32 and all(c in "0123456789abcdef" for c in grant)
    assert synced.extra["shared_grant_id"] == grant
    assert second_synced.extra["shared_grant_id"] == grant


def test_existing_grant_id_is_adopted_only_for_exact_provenance(profile_and_root):
    profile_path, root_path = profile_and_root
    _write_store(profile_path, {"version": 1, "providers": {}})
    _write_store(root_path, {"version": 1, "providers": {"openai-codex": {
        "shared_grant_id": "g0", "tokens": {"access_token": "a", "refresh_token": "r"}}}})
    exact = _codex_entry(id="exact", access_token="a", refresh_token="r")
    mismatch = _codex_entry(id="mismatch", access_token="a", refresh_token="other")
    wrong_id = _codex_entry(
        id="wrong-id", access_token="a", refresh_token="r", grant_id="other-grant"
    )
    pool = CredentialPool("openai-codex", [exact, mismatch])
    assert pool._sync_codex_entry_from_auth_store(exact).extra["shared_grant_id"] == "g0"
    assert pool._sync_codex_entry_from_auth_store(mismatch) is mismatch
    assert pool._sync_codex_entry_from_auth_store(wrong_id) is wrong_id


def test_stale_alias_refresh_uses_current_canonical_tokens(profile_and_root, monkeypatch):
    profile_path, root_path = profile_and_root
    _write_store(profile_path, {"version": 1, "providers": {}})
    _write_store(root_path, {"version": 1, "providers": {"openai-codex": {
        "shared_grant_id": "g", "tokens": {"access_token": "a1", "refresh_token": "r1"}}}})
    calls = []
    monkeypatch.setattr(A, "refresh_codex_oauth_pure", lambda access, refresh: (
        calls.append((access, refresh)) or {"access_token": "a2", "refresh_token": "r2"}
    ))
    alias = _codex_entry(
        id="alias", access_token="stale", refresh_token="revoked", grant_id="g"
    )
    updated = CredentialPool("openai-codex", [alias])._refresh_entry(alias, force=True)
    assert calls == [("a1", "r1")]
    assert (updated.access_token, updated.refresh_token) == ("a2", "r2")
    canonical = _read_store(root_path)["providers"]["openai-codex"]["tokens"]
    assert (canonical["access_token"], canonical["refresh_token"]) == ("a2", "r2")


def test_canonical_rotations_converge_aliases(profile_and_root):
    profile_path, root_path = profile_and_root
    _write_store(profile_path, {"version": 1, "providers": {}})
    pool = CredentialPool("openai-codex", [])
    alias = _codex_entry(id="alias", access_token="a0", refresh_token="r0", grant_id="g")
    for generation in ("1", "2"):
        _write_store(root_path, {"version": 1, "providers": {"openai-codex": {
            "shared_grant_id": "g", "tokens": {
                "access_token": f"a{generation}", "refresh_token": f"r{generation}"
            }}}})
        alias = pool._sync_codex_entry_from_auth_store(alias)
        assert (alias.access_token, alias.refresh_token) == (f"a{generation}", f"r{generation}")


def test_aliases_loaded_before_rotation_adopt_canonical_without_replay(
    profile_and_root, monkeypatch
):
    """A stale singleton-seeded alias must converge by grant identity."""
    profile_path, root_path = profile_and_root
    _write_store(profile_path, {"version": 1, "providers": {}})
    _write_store(root_path, {"version": 1, "providers": {"openai-codex": {
        "tokens": {"access_token": "a0", "refresh_token": "r0"}}}})
    calls = []

    def rotate(access, refresh):
        calls.append((access, refresh))
        return {"access_token": "a1", "refresh_token": "r1"}

    monkeypatch.setattr(A, "refresh_codex_oauth_pure", rotate)
    alias_a = CP.load_pool("openai-codex")
    alias_b = CP.load_pool("openai-codex")
    entry_a = alias_a._entries[0]
    entry_b = alias_b._entries[0]
    assert entry_a.extra["shared_grant_id"] == entry_b.extra["shared_grant_id"]

    alias_a._refresh_entry(entry_a, force=True)
    assert calls == [("a0", "r0")]
    alias_b._refresh_entry(entry_b, force=True)
    assert calls == [("a0", "r0")]
    assert alias_b._entries[0].refresh_token == "r1"


def test_terminal_shared_refresh_quarantines_matching_aliases_only(profile_and_root, monkeypatch):
    profile_path, root_path = profile_and_root
    _write_store(profile_path, {"version": 1, "providers": {"openai-codex": {
        "shared_grant_id": "g", "tokens": {"access_token": "shadow", "refresh_token": "shadow-r"}}}})
    _write_store(root_path, {"version": 1, "providers": {
        "openai-codex": {
            "shared_grant_id": "g", "tokens": {"access_token": "a", "refresh_token": "r"}
        },
        "openrouter": {"api_key": "independent-provider"},
    }, "credential_pool": {
        "openai-codex": [_codex_entry(
            id="root-same", access_token="a", refresh_token="r", grant_id="g"
        ).to_dict(), _codex_entry(
            id="root-other", access_token="x", refresh_token="y", grant_id="independent"
        ).to_dict()],
        "openrouter": [{"id": "keep-provider"}],
    }})

    def rejected(*args, **kwargs):
        raise A.AuthError(
            "revoked", provider="openai-codex", code="invalid_grant", relogin_required=True
        )

    monkeypatch.setattr(A, "refresh_codex_oauth_pure", rejected)
    same = _codex_entry(id="same", access_token="a", refresh_token="r", grant_id="g")
    other = _codex_entry(id="other", access_token="x", refresh_token="y", grant_id="independent")
    pool = CredentialPool("openai-codex", [same, other])
    assert pool._refresh_entry(same, force=True) is None
    assert [entry.id for entry in pool._entries] == ["other"]
    root = _read_store(root_path)
    assert root["providers"]["openai-codex"]["tokens"] == {}
    assert root["providers"]["openrouter"] == {"api_key": "independent-provider"}
    assert root["credential_pool"]["openai-codex"] == [
        _codex_entry(id="root-other", access_token="x", refresh_token="y", grant_id="independent").to_dict()
    ]
    assert root["credential_pool"]["openrouter"] == [{"id": "keep-provider"}]
    assert "openai-codex" not in _read_store(profile_path).get("providers", {})


def test_terminal_shared_refresh_removes_canonical_root_pool_alias_on_reload(
    profile_and_root, monkeypatch
):
    profile_path, root_path = profile_and_root
    _write_store(profile_path, {"version": 1, "providers": {}})
    _write_store(root_path, {
        "version": 1,
        "providers": {"openai-codex": {
            "shared_grant_id": "grant-q",
            "tokens": {"access_token": "a", "refresh_token": "r"},
        }},
        "credential_pool": {"openai-codex": [
            _codex_entry(id="root-alias", access_token="a", refresh_token="r",
                         grant_id="grant-q").to_dict(),
            _codex_entry(id="independent", access_token="ia", refresh_token="ir",
                         grant_id="grant-independent").to_dict(),
        ]},
    })

    def rejected(*args, **kwargs):
        raise A.AuthError(
            "revoked", provider="openai-codex", code="invalid_grant", relogin_required=True
        )

    monkeypatch.setattr(A, "refresh_codex_oauth_pure", rejected)
    doomed = _codex_entry(id="doomed", access_token="a", refresh_token="r", grant_id="grant-q")
    pool = CredentialPool("openai-codex", [doomed])
    assert pool._refresh_entry(doomed, force=True) is None
    assert all(
        item.get("shared_grant_id") != "grant-q"
        for item in _read_store(root_path)["credential_pool"]["openai-codex"]
    )
    reloaded = CP.load_pool("openai-codex")
    assert reloaded.select() is not None
    assert reloaded.select().extra["shared_grant_id"] == "grant-independent"


def test_terminal_canonical_grant_fail_closed_for_copied_profile_alias(
    profile_and_root,
):
    profile_path, root_path = profile_and_root
    copied = _codex_entry(
        id="copied-alias", access_token="stale-a", refresh_token="stale-r", grant_id="grant-q"
    ).to_dict()
    copied.pop("shared_grant_id", None)
    copied["extra"] = {"shared_grant_id": "grant-q"}
    _write_store(profile_path, {"version": 1, "providers": {}, "credential_pool": {
        "openai-codex": [copied, _codex_entry(
            id="independent", access_token="ia", refresh_token="ir",
            grant_id="grant-independent",
        ).to_dict()],
    }})
    _write_store(root_path, {"version": 1, "providers": {"openai-codex": {
        "shared_grant_id": "grant-q",
        "tokens": {},
        "last_auth_error": {"relogin_required": True, "reason": "credential_pool_refresh_failure"},
    }}})

    pool = CP.load_pool("openai-codex")
    selected = pool.select()
    assert selected is not None
    assert selected.extra["shared_grant_id"] == "grant-independent"
    assert all(item.extra.get("shared_grant_id") != "grant-q" for item in pool._entries)


def test_terminal_marker_without_empty_tokens_does_not_filter_alias(profile_and_root):
    profile_path, root_path = profile_and_root
    _write_store(profile_path, {"version": 1, "providers": {}, "credential_pool": {
        "openai-codex": [_codex_entry(
            id="alias", access_token="a", refresh_token="r", grant_id="g"
        ).to_dict()],
    }})
    _write_store(root_path, {"version": 1, "providers": {"openai-codex": {
        "shared_grant_id": "g",
        "tokens": {"access_token": "a", "refresh_token": "r"},
        "last_auth_error": {"relogin_required": True},
    }}})
    pool = CP.load_pool("openai-codex")
    assert pool.select() is not None


def test_transient_refresh_failure_keeps_shared_grant_available(profile_and_root, monkeypatch):
    profile_path, root_path = profile_and_root
    _write_store(profile_path, {"version": 1, "providers": {}})
    _write_store(root_path, {"version": 1, "providers": {"openai-codex": {
        "shared_grant_id": "g", "tokens": {"access_token": "a", "refresh_token": "r"}}}})

    def transient(*args, **kwargs):
        raise A.AuthError(
            "busy", provider="openai-codex", code="codex_rate_limited", relogin_required=False
        )

    monkeypatch.setattr(A, "refresh_codex_oauth_pure", transient)
    entry = _codex_entry(id="same", access_token="a", refresh_token="r", grant_id="g")
    pool = CredentialPool("openai-codex", [entry])
    assert pool._refresh_entry(entry, force=True) is None
    assert pool._entries[0].last_status == STATUS_EXHAUSTED
    assert pool._entries[0].extra.get("shared_grant_id") == "g"
    assert _read_store(root_path)["providers"]["openai-codex"]["tokens"]["refresh_token"] == "r"

