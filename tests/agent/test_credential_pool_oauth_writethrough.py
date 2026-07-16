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

import pytest

from agent import credential_pool as CP
from agent.credential_pool import (
    AUTH_TYPE_OAUTH,
    CredentialPool,
    PooledCredential,
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


@pytest.mark.parametrize(
    "provider",
    ["openai-codex", "xai-oauth"],
)
def test_pool_refresh_writes_through_to_root_when_profile_reads_root(
    profile_and_root, provider
):
    """A profile reading root's grant must push rotated tokens back to root."""
    profile_path, root_path = profile_and_root
    # Profile has NO own provider block (reads root via fallback).
    _write_store(profile_path, {"version": 1, "providers": {}})
    _write_store(
        root_path,
        {
            "version": 1,
            "providers": {
                provider: {
                    "tokens": {
                        "access_token": "old-access",
                        "refresh_token": "old-refresh",
                    }
                }
            },
        },
    )

    pool = CredentialPool(provider, [])
    pool._sync_device_code_entry_to_auth_store(
        _entry(provider, id="e1", access_token="new-access", refresh_token="new-refresh")
    )

    # Profile got the rotated chain (existing behavior).
    profile = _read_store(profile_path)
    assert (
        profile["providers"][provider]["tokens"]["refresh_token"] == "new-refresh"
    )

    # AND the global root no longer holds the revoked refresh token (#48415).
    root = _read_store(root_path)
    assert root["providers"][provider]["tokens"]["access_token"] == "new-access"
    assert root["providers"][provider]["tokens"]["refresh_token"] == "new-refresh"


@pytest.mark.parametrize(
    "provider",
    ["openai-codex", "xai-oauth"],
)
def test_pool_refresh_does_not_touch_root_when_profile_shadows(
    profile_and_root, provider
):
    """A profile that genuinely shadows root must NOT clobber the root grant."""
    profile_path, root_path = profile_and_root
    # Profile has its OWN provider block: it shadows root legitimately.
    _write_store(
        profile_path,
        {
            "version": 1,
            "providers": {
                provider: {
                    "tokens": {
                        "access_token": "profile-old",
                        "refresh_token": "profile-old-refresh",
                    }
                }
            },
        },
    )
    _write_store(
        root_path,
        {
            "version": 1,
            "providers": {
                provider: {
                    "tokens": {
                        "access_token": "root-untouched",
                        "refresh_token": "root-untouched-refresh",
                    }
                }
            },
        },
    )

    pool = CredentialPool(provider, [])
    pool._sync_device_code_entry_to_auth_store(
        _entry(
            provider,
            id="e2",
            access_token="profile-new",
            refresh_token="profile-new-refresh",
        )
    )

    profile = _read_store(profile_path)
    assert (
        profile["providers"][provider]["tokens"]["refresh_token"]
        == "profile-new-refresh"
    )

    # Root keeps its own grant — write-through must not run when the profile
    # owns the block.
    root = _read_store(root_path)
    assert (
        root["providers"][provider]["tokens"]["refresh_token"]
        == "root-untouched-refresh"
    )


def test_write_through_helper_is_noop_in_classic_mode(monkeypatch, tmp_path):
    """When profile == root (classic mode), the helper must be a no-op.

    ``_global_auth_file_path`` returns None in classic mode; the profile save
    already wrote to root, so a second write would be redundant (and the
    helper has nothing to target).
    """
    monkeypatch.setattr(A, "_global_auth_file_path", lambda: None)
    # Must not raise and must not attempt any write.
    CP._write_through_provider_state_to_global_root(
        "openai-codex", {"tokens": {"access_token": "a", "refresh_token": "r"}}
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


def test_inherited_codex_pool_refresh_updates_only_root(
    profile_and_root, monkeypatch
):
    """A loaded inherited Codex pool keeps root as its canonical owner."""
    profile_path, root_path = profile_and_root
    _write_store(
        profile_path,
        {"version": 1, "providers": {}, "credential_pool": {}},
    )
    _write_store(
        root_path,
        {
            "version": 1,
            "providers": {
                "openai-codex": {
                    "tokens": {
                        "access_token": "root-access-old",
                        "refresh_token": "root-refresh-old",
                    },
                    "last_refresh": "2020-01-01T00:00:00Z",
                }
            },
            "credential_pool": {
                "openai-codex": [{
                    "id": "default",
                    "label": "default",
                    "auth_type": "oauth",
                    "priority": 0,
                    "source": "device_code",
                    "access_token": "root-access-old",
                    "refresh_token": "root-refresh-old",
                }]
            },
        },
    )
    monkeypatch.setattr(
        A,
        "refresh_codex_oauth_pure",
        lambda access_token, refresh_token, **kwargs: {
            "access_token": "root-access-new",
            "refresh_token": "root-refresh-new",
            "last_refresh": "2020-01-02T00:00:00Z",
        },
    )

    pool = CP.load_pool("openai-codex")
    entry = pool.entries()[0]
    refreshed = pool._refresh_entry(entry, force=True)

    assert refreshed is not None
    assert refreshed.refresh_token == "root-refresh-new"
    root = _read_store(root_path)
    assert (
        root["providers"]["openai-codex"]["tokens"]["refresh_token"]
        == "root-refresh-new"
    )
    assert (
        root["credential_pool"]["openai-codex"][0]["refresh_token"]
        == "root-refresh-new"
    )
    profile = _read_store(profile_path)
    assert "openai-codex" not in profile.get("providers", {})
    assert "openai-codex" not in profile.get("credential_pool", {})


def test_two_loaded_inherited_manual_codex_pools_reload_rotated_row(
    profile_and_root, monkeypatch
):
    """A waiter must not replay a manual pool row's consumed refresh token."""
    profile_path, root_path = profile_and_root
    _write_store(
        profile_path,
        {"version": 1, "providers": {}, "credential_pool": {}},
    )
    _write_store(
        root_path,
        {
            "version": 1,
            "providers": {},
            "credential_pool": {
                "openai-codex": [{
                    "id": "manual-1",
                    "label": "manual",
                    "auth_type": "oauth",
                    "priority": 0,
                    "source": "manual:device_code",
                    "access_token": "access-0",
                    "refresh_token": "refresh-0",
                }]
            },
        },
    )
    submitted = []

    def fake_refresh(access_token, refresh_token, **kwargs):
        submitted.append((access_token, refresh_token))
        generation = len(submitted)
        return {
            "access_token": f"access-{generation}",
            "refresh_token": f"refresh-{generation}",
            "last_refresh": f"2020-01-0{generation + 1}T00:00:00Z",
        }

    monkeypatch.setattr(A, "refresh_codex_oauth_pure", fake_refresh)
    first_pool = CP.load_pool("openai-codex")
    second_pool = CP.load_pool("openai-codex")

    assert first_pool._refresh_entry(first_pool.entries()[0], force=True)
    assert second_pool._refresh_entry(second_pool.entries()[0], force=True)

    assert submitted == [
        ("access-0", "refresh-0"),
        ("access-1", "refresh-1"),
    ]
    root = _read_store(root_path)
    assert (
        root["credential_pool"]["openai-codex"][0]["refresh_token"]
        == "refresh-2"
    )
    profile = _read_store(profile_path)
    assert "openai-codex" not in profile.get("credential_pool", {})


def test_remove_from_inherited_codex_pool_updates_root_without_shadow(
    profile_and_root,
):
    profile_path, root_path = profile_and_root
    _write_store(
        profile_path,
        {"version": 1, "providers": {}, "credential_pool": {}},
    )
    rows = [
        {
            "id": entry_id,
            "label": entry_id,
            "auth_type": "oauth",
            "priority": priority,
            "source": "manual:device_code",
            "access_token": f"access-{priority}",
            "refresh_token": f"refresh-{priority}",
        }
        for priority, entry_id in enumerate(("first", "second"))
    ]
    _write_store(
        root_path,
        {
            "version": 1,
            "providers": {},
            "credential_pool": {"openai-codex": rows},
        },
    )

    pool = CP.load_pool("openai-codex")
    removed = pool.remove_index(1)

    assert removed is not None and removed.id == "first"
    root = _read_store(root_path)
    assert [
        item["id"] for item in root["credential_pool"]["openai-codex"]
    ] == ["second"]
    profile = _read_store(profile_path)
    assert "openai-codex" not in profile.get("credential_pool", {})


def test_refreshing_different_rows_preserves_concurrent_rotation(
    profile_and_root, monkeypatch
):
    """Writing row B must not overwrite a newer on-disk version of row A."""
    profile_path, root_path = profile_and_root
    _write_store(profile_path, {"version": 1, "providers": {}, "credential_pool": {}})
    rows = [
        {
            "id": entry_id,
            "label": entry_id,
            "auth_type": "oauth",
            "priority": priority,
            "source": "manual:device_code",
            "access_token": f"access-{entry_id}-0",
            "refresh_token": f"refresh-{entry_id}-0",
        }
        for priority, entry_id in enumerate(("a", "b"))
    ]
    _write_store(root_path, {
        "version": 1,
        "providers": {},
        "credential_pool": {"openai-codex": rows},
    })

    def fake_refresh(access_token, refresh_token, **kwargs):
        return {
            "access_token": f"{access_token}-new",
            "refresh_token": f"{refresh_token}-new",
            "last_refresh": "2020-01-02T00:00:00Z",
        }

    monkeypatch.setattr(A, "refresh_codex_oauth_pure", fake_refresh)
    first_pool = CP.load_pool("openai-codex")
    second_pool = CP.load_pool("openai-codex")

    assert first_pool._refresh_entry(first_pool.entries()[0], force=True)
    assert second_pool._refresh_entry(second_pool.entries()[1], force=True)

    root = _read_store(root_path)
    final = {
        item["id"]: item["refresh_token"]
        for item in root["credential_pool"]["openai-codex"]
    }
    assert final == {
        "a": "refresh-a-0-new",
        "b": "refresh-b-0-new",
    }


def test_stale_snapshot_removal_preserves_other_row_rotation(
    profile_and_root, monkeypatch
):
    """Removing row B must not roll back concurrently refreshed row A."""
    profile_path, root_path = profile_and_root
    _write_store(profile_path, {"version": 1, "providers": {}, "credential_pool": {}})
    rows = [
        {
            "id": entry_id,
            "label": entry_id,
            "auth_type": "oauth",
            "priority": priority,
            "source": "manual:device_code",
            "access_token": f"access-{entry_id}-0",
            "refresh_token": f"refresh-{entry_id}-0",
        }
        for priority, entry_id in enumerate(("a", "b"))
    ]
    _write_store(root_path, {
        "version": 1,
        "providers": {},
        "credential_pool": {"openai-codex": rows},
    })
    monkeypatch.setattr(
        A,
        "refresh_codex_oauth_pure",
        lambda access_token, refresh_token, **kwargs: {
            "access_token": f"{access_token}-new",
            "refresh_token": f"{refresh_token}-new",
            "last_refresh": "2020-01-02T00:00:00Z",
        },
    )
    refreshing_pool = CP.load_pool("openai-codex")
    removing_pool = CP.load_pool("openai-codex")

    assert refreshing_pool._refresh_entry(refreshing_pool.entries()[1], force=True)
    removed = removing_pool.remove_index(1)

    assert removed is not None and removed.id == "a"
    root = _read_store(root_path)
    final_rows = root["credential_pool"]["openai-codex"]
    assert len(final_rows) == 1
    assert final_rows[0]["id"] == "b"
    assert final_rows[0]["priority"] == 0
    assert final_rows[0]["refresh_token"] == "refresh-b-0-new"
