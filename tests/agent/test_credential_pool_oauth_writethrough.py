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
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import pytest

from agent import anthropic_adapter
from agent import credential_pool as CP
from agent.credential_pool import (
    AUTH_TYPE_OAUTH,
    CredentialPool,
    PooledCredential,
)
from hermes_constants import reset_hermes_home_override, set_hermes_home_override
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


def _root_codex_store(*, access_token="root-old-access", refresh_token="root-old-refresh"):
    return {
        "version": 1,
        "active_provider": "openrouter",
        "providers": {
            "openai-codex": {
                "tokens": {
                    "access_token": access_token,
                    "refresh_token": refresh_token,
                },
                "last_refresh": "2026-08-01T00:00:00Z",
                "auth_mode": "chatgpt",
            },
            "anthropic": {"api_key": "root-unrelated-provider"},
        },
        "credential_pool": {
            "openai-codex": [
                {
                    "id": "root-device",
                    "label": "root singleton",
                    "source": "device_code",
                    "auth_type": "oauth",
                    "priority": 0,
                    "access_token": access_token,
                    "refresh_token": refresh_token,
                },
                {
                    "id": "root-independent",
                    "label": "root independent",
                    "source": "manual:device_code",
                    "auth_type": "oauth",
                    "priority": 1,
                    "access_token": "root-independent-access",
                    "refresh_token": "root-independent-refresh",
                    "last_status": "exhausted",
                    "last_error_code": 429,
                },
            ],
            "openrouter": [{"id": "root-unrelated-pool"}],
        },
        "root_marker": {"preserve": True},
    }


def _profile_codex_store(marker: str):
    return {
        "version": 1,
        "active_provider": "anthropic",
        "providers": {
            "anthropic": {"api_key": f"profile-{marker}-provider"},
        },
        "credential_pool": {
            "openai-codex": [
                {
                    "id": f"profile-{marker}-manual",
                    "label": f"profile {marker} manual",
                    "source": "manual:device_code",
                    "auth_type": "oauth",
                    "priority": 0,
                    "access_token": f"profile-{marker}-access",
                    "refresh_token": f"profile-{marker}-refresh",
                    "last_status": "exhausted",
                    "last_status_at": "2026-08-01T00:00:00+00:00",
                    "last_error_code": 429,
                    "last_error_reason": "usage_limit",
                }
            ],
            "anthropic": [{"id": f"profile-{marker}-unrelated"}],
        },
        "profile_marker": marker,
    }


def _profile_without_codex_store(marker: str):
    return {
        "version": 1,
        "active_provider": "anthropic",
        "providers": {
            "anthropic": {"api_key": f"profile-{marker}-provider"},
        },
        "credential_pool": {
            "anthropic": [{"id": f"profile-{marker}-unrelated"}],
        },
        "profile_marker": marker,
    }


def _healthy_root_manual_store():
    store = _root_codex_store()
    manual = next(
        item
        for item in store["credential_pool"]["openai-codex"]
        if item["id"] == "root-independent"
    )
    for field in (
        "last_status",
        "last_status_at",
        "last_error_code",
        "last_error_reason",
        "last_error_message",
        "last_error_reset_at",
    ):
        manual.pop(field, None)
    return store


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


def test_load_pool_borrows_root_alias_without_persisting_profile_shadow(
    profile_and_root,
):
    """A non-empty manual profile pool must not fork the root singleton."""
    profile_path, root_path = profile_and_root
    _write_store(root_path, _root_codex_store())
    _write_store(profile_path, _profile_codex_store("work"))
    profile_before = profile_path.read_bytes()

    pool = CP.load_pool("openai-codex")

    entries = pool.entries()
    borrowed = [entry for entry in entries if entry.source == "device_code"]
    manual = [entry for entry in entries if entry.source == "manual:device_code"]
    assert [entry.id for entry in borrowed] == ["root-device"]
    assert [entry.id for entry in manual] == ["profile-work-manual"]
    assert manual[0].last_status == "exhausted"
    assert manual[0].last_error_code == 429
    assert profile_path.read_bytes() == profile_before


def test_borrowed_root_alias_status_routes_to_owner_without_profile_shadow(
    profile_and_root,
):
    """Borrowed status persists at the root while the profile keeps only manual rows."""
    profile_path, root_path = profile_and_root
    _write_store(root_path, _root_codex_store())
    _write_store(profile_path, _profile_codex_store("work"))
    pool = CP.load_pool("openai-codex")
    profile_before = profile_path.read_bytes()
    borrowed = next(item for item in pool.entries() if item.id == "root-device")
    assert borrowed.source_store_path == root_path

    pool._mark_exhausted(
        borrowed,
        429,
        failure_reason="rate_limit",
    )

    root = _read_store(root_path)
    root_alias = next(
        item
        for item in root["credential_pool"]["openai-codex"]
        if item.get("id") == "root-device"
    )
    assert root_alias["last_status"] == "exhausted"
    assert root_alias["last_error_code"] == 429
    profile = _read_store(profile_path)
    assert [
        item["id"] for item in profile["credential_pool"]["openai-codex"]
    ] == ["profile-work-manual"]
    assert profile_path.read_bytes() == profile_before


def test_load_pool_refresh_serializes_root_source_across_processes(
    monkeypatch, tmp_path
):
    """Two real profile pools share one root lock, chain, and alias identity."""
    if "fork" not in multiprocessing.get_all_start_methods():
        pytest.skip("requires fork so patched fake refresh stays network-free")

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.delenv("HERMES_HOME", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "not-the-root"))
    root_home = tmp_path / ".hermes"
    root_path = root_home / "auth.json"
    profile_paths = [
        root_home / "profiles" / "alpha" / "auth.json",
        root_home / "profiles" / "beta" / "auth.json",
    ]
    _write_store(root_path, _root_codex_store())
    for marker, profile_path in zip(("alpha", "beta"), profile_paths):
        _write_store(profile_path, _profile_codex_store(marker))
    profile_before = {path: path.read_bytes() for path in profile_paths}

    ctx = multiprocessing.get_context("fork")
    start = ctx.Event()
    post_entered = ctx.Event()
    allow_post_return = ctx.Event()
    post_count = ctx.Value("i", 0)
    reports = ctx.Queue()

    def fake_refresh(access_token, refresh_token, **_kwargs):
        assert access_token == "root-old-access"
        assert refresh_token == "root-old-refresh"
        with post_count.get_lock():
            post_count.value += 1
        post_entered.set()
        assert allow_post_return.wait(timeout=5)
        return {
            "access_token": "root-new-access",
            "refresh_token": "root-new-refresh",
            "last_refresh": "2026-08-06T00:00:00Z",
        }

    monkeypatch.setattr(A, "refresh_codex_oauth_pure", fake_refresh)

    def refresh_from_profile(profile_path):
        token = set_hermes_home_override(profile_path.parent)
        try:
            pool = CP.load_pool("openai-codex")
            entry = next(item for item in pool.entries() if item.source == "device_code")
            reports.put(("loaded", profile_path.name, entry.id))
            assert start.wait(timeout=5)
            refreshed = pool._refresh_entry(entry, force=True)
            reports.put(
                (
                    "result",
                    profile_path.name,
                    refreshed.id if refreshed else None,
                    refreshed.access_token if refreshed else None,
                    refreshed.refresh_token if refreshed else None,
                )
            )
        except BaseException as exc:
            reports.put(("error", profile_path.name, repr(exc)))
        finally:
            reset_hermes_home_override(token)

    processes = [
        ctx.Process(
            target=refresh_from_profile,
            args=(profile_path,),
            name=f"codex-profile-{index}",
        )
        for index, profile_path in enumerate(profile_paths)
    ]
    for process in processes:
        process.start()

    loaded = [reports.get(timeout=5), reports.get(timeout=5)]
    assert all(report[0] == "loaded" for report in loaded), loaded
    start.set()
    assert post_entered.wait(timeout=5)
    allow_post_return.set()
    results = [reports.get(timeout=5), reports.get(timeout=5)]
    for process in processes:
        process.join(timeout=5)

    assert all(process.exitcode == 0 for process in processes)
    assert all(report[0] == "result" for report in results), results
    assert {report[2] for report in loaded} == {"root-device"}
    assert post_count.value == 1
    assert {report[2] for report in results} == {"root-device"}
    assert {report[3] for report in results} == {"root-new-access"}
    assert {report[4] for report in results} == {"root-new-refresh"}

    root = _read_store(root_path)
    assert root["active_provider"] == "openrouter"
    assert root["providers"]["openai-codex"]["tokens"] == {
        "access_token": "root-new-access",
        "refresh_token": "root-new-refresh",
    }
    root_entries = {
        entry["id"]: entry for entry in root["credential_pool"]["openai-codex"]
    }
    assert root_entries["root-device"]["access_token"] == "root-new-access"
    assert root_entries["root-device"]["refresh_token"] == "root-new-refresh"
    assert root_entries["root-independent"]["access_token"] == (
        "root-independent-access"
    )
    assert root_entries["root-independent"]["last_status"] == "exhausted"
    assert root_entries["root-independent"]["last_error_code"] == 429
    for path in profile_paths:
        assert path.read_bytes() == profile_before[path]


def test_root_manual_refresh_serializes_across_profile_processes(
    monkeypatch, tmp_path
):
    """Every root-fallback Codex row keeps root ownership, not just singleton."""
    if "fork" not in multiprocessing.get_all_start_methods():
        pytest.skip("requires fork so patched fake refresh stays network-free")

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.delenv("HERMES_HOME", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "not-the-root"))
    root_home = tmp_path / ".hermes"
    root_path = root_home / "auth.json"
    profile_paths = [
        root_home / "profiles" / "alpha" / "auth.json",
        root_home / "profiles" / "beta" / "auth.json",
    ]
    _write_store(root_path, _healthy_root_manual_store())
    for marker, profile_path in zip(("alpha", "beta"), profile_paths):
        _write_store(profile_path, _profile_without_codex_store(marker))
    profile_before = {path: path.read_bytes() for path in profile_paths}

    ctx = multiprocessing.get_context("fork")
    start = ctx.Event()
    second_post_entered = ctx.Event()
    post_count = ctx.Value("i", 0)
    reports = ctx.Queue()

    def fake_refresh(access_token, refresh_token, **_kwargs):
        assert access_token == "root-independent-access"
        assert refresh_token == "root-independent-refresh"
        with post_count.get_lock():
            post_count.value += 1
            call_number = post_count.value
        if call_number == 1:
            second_post_entered.wait(timeout=1)
        else:
            second_post_entered.set()
        return {
            "access_token": "root-manual-new-access",
            "refresh_token": "root-manual-new-refresh",
            "last_refresh": "2026-08-06T01:00:00Z",
        }

    monkeypatch.setattr(A, "refresh_codex_oauth_pure", fake_refresh)

    def refresh_from_profile(profile_path):
        token = set_hermes_home_override(profile_path.parent)
        try:
            pool = CP.load_pool("openai-codex")
            entry = next(item for item in pool.entries() if item.id == "root-independent")
            reports.put(
                (
                    "loaded",
                    entry.id,
                    str(entry.source_store_path) if entry.source_store_path else None,
                )
            )
            assert start.wait(timeout=5)
            refreshed = pool._refresh_entry(entry, force=True)
            reports.put(
                (
                    "result",
                    refreshed.id if refreshed else None,
                    refreshed.access_token if refreshed else None,
                    refreshed.refresh_token if refreshed else None,
                )
            )
        except BaseException as exc:
            reports.put(("error", repr(exc)))
        finally:
            reset_hermes_home_override(token)

    processes = [
        ctx.Process(target=refresh_from_profile, args=(path,))
        for path in profile_paths
    ]
    for process in processes:
        process.start()
    loaded = [reports.get(timeout=5), reports.get(timeout=5)]
    assert all(report[0] == "loaded" for report in loaded), loaded
    start.set()
    results = [reports.get(timeout=5), reports.get(timeout=5)]
    for process in processes:
        process.join(timeout=5)

    assert all(process.exitcode == 0 for process in processes)
    assert all(report[0] == "result" for report in results), results
    assert {report[1] for report in loaded} == {"root-independent"}
    assert {report[2] for report in loaded} == {str(root_path)}
    assert post_count.value == 1
    assert {report[1] for report in results} == {"root-independent"}
    assert {report[2] for report in results} == {"root-manual-new-access"}
    assert {report[3] for report in results} == {"root-manual-new-refresh"}

    root = _read_store(root_path)
    assert root["providers"]["openai-codex"]["tokens"] == {
        "access_token": "root-old-access",
        "refresh_token": "root-old-refresh",
    }
    root_entries = {
        item["id"]: item for item in root["credential_pool"]["openai-codex"]
    }
    assert root_entries["root-device"]["access_token"] == "root-old-access"
    assert root_entries["root-device"]["refresh_token"] == "root-old-refresh"
    assert root_entries["root-independent"]["access_token"] == (
        "root-manual-new-access"
    )
    assert root_entries["root-independent"]["refresh_token"] == (
        "root-manual-new-refresh"
    )
    for path in profile_paths:
        assert path.read_bytes() == profile_before[path]


def test_root_manual_waiter_adopts_rotated_exact_alias_without_post(
    profile_and_root, monkeypatch
):
    """A stale borrower adopts the exact manual alias and leaves singleton alone."""
    profile_path, root_path = profile_and_root
    _write_store(root_path, _healthy_root_manual_store())
    _write_store(profile_path, _profile_without_codex_store("work"))
    profile_before = profile_path.read_bytes()

    pool = CP.load_pool("openai-codex")
    stale = next(item for item in pool.entries() if item.id == "root-independent")
    assert stale.source_store_path == root_path

    root = _read_store(root_path)
    exact = next(
        item
        for item in root["credential_pool"]["openai-codex"]
        if item["id"] == "root-independent"
    )
    exact["access_token"] = "winner-access"
    exact["refresh_token"] = "winner-refresh"
    exact["last_refresh"] = "2026-08-06T02:00:00Z"
    _write_store(root_path, root)

    post_count = 0

    def unexpected_post(*_args, **_kwargs):
        nonlocal post_count
        post_count += 1
        raise AssertionError("waiter replayed the already-rotated refresh token")

    monkeypatch.setattr(A, "refresh_codex_oauth_pure", unexpected_post)
    refreshed = pool._refresh_entry(stale, force=True)

    assert refreshed is not None
    assert refreshed.id == "root-independent"
    assert refreshed.access_token == "winner-access"
    assert refreshed.refresh_token == "winner-refresh"
    assert post_count == 0
    root_after = _read_store(root_path)
    assert root_after["providers"]["openai-codex"]["tokens"] == {
        "access_token": "root-old-access",
        "refresh_token": "root-old-refresh",
    }
    assert profile_path.read_bytes() == profile_before


def test_root_manual_status_persists_exact_alias_without_profile_churn(
    profile_and_root,
):
    profile_path, root_path = profile_and_root
    _write_store(root_path, _healthy_root_manual_store())
    _write_store(profile_path, _profile_without_codex_store("work"))
    profile_before = profile_path.read_bytes()
    singleton_before = _read_store(root_path)["providers"]["openai-codex"]

    pool = CP.load_pool("openai-codex")
    rotated = pool.mark_exhausted_and_rotate(
        status_code=429,
        error_context={"reason": "rate_limit", "message": "fake root manual limit"},
        credential_id="root-independent",
    )

    assert rotated is not None
    assert rotated.id == "root-device"
    root_after = _read_store(root_path)
    manual = next(
        item
        for item in root_after["credential_pool"]["openai-codex"]
        if item["id"] == "root-independent"
    )
    assert manual["last_status"] == "exhausted"
    assert manual["last_error_code"] == 429
    assert manual["last_error_reason"] == "rate_limit"
    assert root_after["providers"]["openai-codex"] == singleton_before
    assert profile_path.read_bytes() == profile_before


def test_root_manual_terminal_refresh_removes_only_exact_alias(
    profile_and_root, monkeypatch
):
    profile_path, root_path = profile_and_root
    root_store = _healthy_root_manual_store()
    root_store["credential_pool"]["openai-codex"].append(
        {
            "id": "root-manual-survivor",
            "label": "independent manual survivor",
            "source": "manual:device_code",
            "auth_type": "oauth",
            "priority": 2,
            "access_token": "manual-survivor-access",
            "refresh_token": "manual-survivor-refresh",
        }
    )
    _write_store(root_path, root_store)
    _write_store(profile_path, _profile_without_codex_store("work"))
    profile_before = profile_path.read_bytes()
    singleton_before = _read_store(root_path)["providers"]["openai-codex"]

    def reject_refresh(*_args, **_kwargs):
        raise A.AuthError(
            "fake invalid grant",
            provider="openai-codex",
            code="invalid_grant",
            relogin_required=True,
        )

    monkeypatch.setattr(A, "refresh_codex_oauth_pure", reject_refresh)
    pool = CP.load_pool("openai-codex")
    target = next(item for item in pool.entries() if item.id == "root-independent")
    assert pool._refresh_entry(target, force=True) is None

    root_after = _read_store(root_path)
    root_ids = {
        item["id"]
        for item in root_after["credential_pool"]["openai-codex"]
    }
    assert "root-independent" not in root_ids
    assert {"root-device", "root-manual-survivor"} <= root_ids
    assert root_after["providers"]["openai-codex"] == singleton_before
    assert profile_path.read_bytes() == profile_before

    reloaded = CP.load_pool("openai-codex")
    reloaded_ids = {item.id for item in reloaded.entries()}
    assert "root-independent" not in reloaded_ids
    assert {"root-device", "root-manual-survivor"} <= reloaded_ids


@pytest.mark.parametrize(
    ("status_code", "error_context", "expected_status"),
    [
        (429, {"reason": "rate_limit"}, "exhausted"),
        (401, {"reason": "invalid_grant"}, "dead"),
    ],
)
def test_stale_profile_local_write_preserves_newer_root_status(
    monkeypatch,
    tmp_path,
    status_code,
    error_context,
    expected_status,
):
    """A stale borrower changing only local rows cannot rewrite root metadata."""
    if "fork" not in multiprocessing.get_all_start_methods():
        pytest.skip("requires fork for deterministic stale-process coverage")

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.delenv("HERMES_HOME", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "not-the-root"))
    root_home = tmp_path / ".hermes"
    root_path = root_home / "auth.json"
    profile_a = root_home / "profiles" / "alpha" / "auth.json"
    profile_b = root_home / "profiles" / "beta" / "auth.json"
    _write_store(root_path, _healthy_root_manual_store())
    _write_store(profile_a, _profile_without_codex_store("alpha"))
    _write_store(profile_b, _profile_without_codex_store("beta"))
    profile_a_before = profile_a.read_bytes()

    ctx = multiprocessing.get_context("fork")
    loaded = ctx.Event()
    allow_local_write = ctx.Event()
    reports = ctx.Queue()

    def stale_profile_writer():
        token = set_hermes_home_override(profile_b.parent)
        try:
            pool = CP.load_pool("openai-codex")
            root_entry = next(item for item in pool.entries() if item.id == "root-device")
            reports.put(
                (
                    "loaded",
                    root_entry.last_status,
                    str(root_entry.source_store_path)
                    if root_entry.source_store_path
                    else None,
                )
            )
            loaded.set()
            assert allow_local_write.wait(timeout=5)
            pool.add_entry(
                PooledCredential(
                    provider="openai-codex",
                    id="profile-beta-added",
                    label="profile beta added",
                    auth_type=AUTH_TYPE_OAUTH,
                    priority=99,
                    source="manual:device_code",
                    access_token="profile-beta-access",
                    refresh_token="profile-beta-refresh",
                )
            )
            reports.put(("written", [item.id for item in pool.entries()]))
        except BaseException as exc:
            reports.put(("error", repr(exc)))
        finally:
            reset_hermes_home_override(token)

    writer = ctx.Process(target=stale_profile_writer)
    writer.start()
    assert loaded.wait(timeout=5)
    loaded_report = reports.get(timeout=5)
    assert loaded_report == ("loaded", None, str(root_path))

    token = set_hermes_home_override(profile_a.parent)
    try:
        owner_pool = CP.load_pool("openai-codex")
        owner_pool.mark_exhausted_and_rotate(
            status_code=status_code,
            error_context=error_context,
            credential_id="root-device",
        )
    finally:
        reset_hermes_home_override(token)

    marked_root = _read_store(root_path)
    marked_alias = next(
        item
        for item in marked_root["credential_pool"]["openai-codex"]
        if item["id"] == "root-device"
    )
    assert marked_alias["last_status"] == expected_status
    marked_at = marked_alias["last_status_at"]

    allow_local_write.set()
    written_report = reports.get(timeout=5)
    writer.join(timeout=5)
    assert writer.exitcode == 0
    assert written_report[0] == "written", written_report

    root_after = _read_store(root_path)
    root_alias = next(
        item
        for item in root_after["credential_pool"]["openai-codex"]
        if item["id"] == "root-device"
    )
    assert root_alias["last_status"] == expected_status
    assert root_alias["last_status_at"] == marked_at
    root_manual = next(
        item
        for item in root_after["credential_pool"]["openai-codex"]
        if item["id"] == "root-independent"
    )
    assert root_manual["access_token"] == "root-independent-access"
    assert root_manual["refresh_token"] == "root-independent-refresh"
    profile_b_ids = {
        item["id"]
        for item in _read_store(profile_b)["credential_pool"]["openai-codex"]
    }
    assert profile_b_ids == {"profile-beta-added"}
    assert profile_a.read_bytes() == profile_a_before


def test_load_pool_terminal_refresh_quarantines_only_root_source(
    profile_and_root, monkeypatch
):
    """Terminal refresh failure clears the root chain without a profile alias."""
    profile_path, root_path = profile_and_root
    _write_store(root_path, _root_codex_store())
    _write_store(profile_path, _profile_codex_store("work"))
    profile_before = profile_path.read_bytes()
    post_count = 0

    def reject_refresh(*_args, **_kwargs):
        nonlocal post_count
        post_count += 1
        raise A.AuthError(
            "fake invalid grant",
            provider="openai-codex",
            code="invalid_grant",
            relogin_required=True,
        )

    monkeypatch.setattr(A, "refresh_codex_oauth_pure", reject_refresh)
    pool = CP.load_pool("openai-codex")
    entry = next(item for item in pool.entries() if item.source == "device_code")

    assert pool._refresh_entry(entry, force=True) is None
    assert post_count == 1
    assert all(item.source != "device_code" for item in pool.entries())

    root = _read_store(root_path)
    root_tokens = root["providers"]["openai-codex"]["tokens"]
    assert "access_token" not in root_tokens
    assert "refresh_token" not in root_tokens
    assert all(
        item.get("source") != "device_code"
        for item in root["credential_pool"]["openai-codex"]
    )
    assert any(
        item.get("id") == "root-independent"
        for item in root["credential_pool"]["openai-codex"]
    )
    assert profile_path.read_bytes() == profile_before

    monkeypatch.setattr(A, "_auth_file_path", lambda: root_path)
    monkeypatch.setattr(A, "_global_auth_file_path", lambda: None)
    root_pool = CP.load_pool("openai-codex")
    assert all(item.source != "device_code" for item in root_pool.entries())
    assert post_count == 1


def test_terminal_singleton_quarantine_removes_only_matching_device_aliases(
    profile_and_root, monkeypatch
):
    """Duplicate aliases on one dead chain disappear; independent rows survive."""
    profile_path, root_path = profile_and_root
    root_store = _healthy_root_manual_store()
    root_store["credential_pool"]["openai-codex"].extend(
        [
            {
                "id": "root-device-duplicate",
                "label": "duplicate singleton alias",
                "source": "device_code",
                "auth_type": "oauth",
                "priority": 2,
                "access_token": "root-old-access",
                "refresh_token": "root-old-refresh",
            },
            {
                "id": "root-device-unrelated",
                "label": "different device chain",
                "source": "device_code",
                "auth_type": "oauth",
                "priority": 3,
                "access_token": "different-access",
                "refresh_token": "different-refresh",
            },
            {
                "id": "root-manual-same-chain",
                "label": "manual row sharing token literals",
                "source": "manual:device_code",
                "auth_type": "oauth",
                "priority": 4,
                "access_token": "root-old-access",
                "refresh_token": "root-old-refresh",
            },
        ]
    )
    _write_store(root_path, root_store)
    _write_store(profile_path, _profile_without_codex_store("work"))
    profile_before = profile_path.read_bytes()

    def reject_refresh(*_args, **_kwargs):
        raise A.AuthError(
            "fake invalid grant",
            provider="openai-codex",
            code="invalid_grant",
            relogin_required=True,
        )

    monkeypatch.setattr(A, "refresh_codex_oauth_pure", reject_refresh)
    pool = CP.load_pool("openai-codex")
    selected = next(item for item in pool.entries() if item.id == "root-device")

    assert pool._refresh_entry(selected, force=True) is None

    root_after = _read_store(root_path)
    remaining_ids = {
        item["id"]
        for item in root_after["credential_pool"]["openai-codex"]
    }
    assert "root-device" not in remaining_ids
    assert "root-device-duplicate" not in remaining_ids
    assert {
        "root-device-unrelated",
        "root-independent",
        "root-manual-same-chain",
    } <= remaining_ids
    assert profile_path.read_bytes() == profile_before

    monkeypatch.setattr(A, "_auth_file_path", lambda: root_path)
    monkeypatch.setattr(A, "_global_auth_file_path", lambda: None)
    root_pool = CP.load_pool("openai-codex")
    reloaded_ids = {item.id for item in root_pool.entries()}
    assert "root-device" not in reloaded_ids
    assert "root-device-duplicate" not in reloaded_ids
    assert {
        "root-device-unrelated",
        "root-independent",
        "root-manual-same-chain",
    } <= reloaded_ids


@pytest.mark.parametrize(
    "operation",
    ["status", "reset", "rotation", "selection", "persistence"],
)
@pytest.mark.parametrize(
    "source_removal",
    ["alias_only", "provider_and_alias"],
)
def test_removed_source_revokes_cached_borrower_on_every_pool_path(
    profile_and_root, operation, source_removal
):
    """A cached root row cannot survive once its exact owning source is gone."""
    profile_path, root_path = profile_and_root
    root_store = _healthy_root_manual_store()
    if operation == "reset":
        root_device = next(
            item
            for item in root_store["credential_pool"]["openai-codex"]
            if item["id"] == "root-device"
        )
        root_device.update(
            {
                "last_status": "exhausted",
                "last_status_at": "2026-08-06T03:00:00+00:00",
                "last_error_code": 429,
            }
        )
    _write_store(root_path, root_store)
    _write_store(profile_path, _profile_without_codex_store("work"))
    profile_before = profile_path.read_bytes()

    pool = CP.load_pool("openai-codex")
    cached = next(item for item in pool.entries() if item.id == "root-device")
    assert cached.source_store_path == root_path

    removed = _read_store(root_path)
    if source_removal == "provider_and_alias":
        removed["providers"].pop("openai-codex")
    removed["credential_pool"]["openai-codex"] = [
        item
        for item in removed["credential_pool"]["openai-codex"]
        if item["id"] != "root-device"
    ]
    _write_store(root_path, removed)
    root_after_removal = root_path.read_bytes()

    selected = None
    if operation == "status":
        pool._mark_exhausted(cached, 429)
    elif operation == "reset":
        pool.reset_statuses()
    elif operation == "rotation":
        selected = pool.mark_exhausted_and_rotate(
            status_code=429,
            credential_id="root-device",
        )
    elif operation == "selection":
        selected = pool.select()
    else:
        pool.add_entry(
            PooledCredential(
                provider="openai-codex",
                id="profile-work-added",
                label="profile work added",
                auth_type=AUTH_TYPE_OAUTH,
                priority=99,
                source="manual:device_code",
                access_token="profile-work-added-access",
                refresh_token="profile-work-added-refresh",
            )
        )

    assert "root-device" not in {item.id for item in pool.entries()}
    current = pool.current()
    assert current is None or current.id != "root-device"
    peeked = pool.peek()
    assert peeked is None or peeked.id != "root-device"
    assert pool.acquire_lease("root-device") is None
    if selected is not None:
        assert selected.id == "root-independent"
    assert root_path.read_bytes() == root_after_removal

    profile_after = _read_store(profile_path)
    profile_ids = {
        item["id"]
        for item in profile_after.get("credential_pool", {}).get(
            "openai-codex", []
        )
    }
    if operation == "persistence":
        assert profile_ids == {"profile-work-added"}
    else:
        assert profile_path.read_bytes() == profile_before

    remaining_root_ids = {
        item["id"]
        for item in _read_store(root_path)["credential_pool"]["openai-codex"]
    }
    assert remaining_root_ids == {"root-independent"}


@pytest.mark.parametrize(
    "operation",
    [
        "status",
        "reset",
        "rotation",
        "select",
        "current",
        "peek",
        "lease",
        "refresh",
        "persistence",
    ],
)
def test_provider_only_removal_revokes_cached_singleton_but_keeps_root_manual(
    profile_and_root, monkeypatch, operation
):
    """A borrowed singleton never downgrades to an independent root row."""
    profile_path, root_path = profile_and_root
    root_store = _healthy_root_manual_store()
    if operation == "reset":
        singleton = next(
            item
            for item in root_store["credential_pool"]["openai-codex"]
            if item["id"] == "root-device"
        )
        singleton.update(
            {
                "last_status": "exhausted",
                "last_status_at": "2026-08-06T03:00:00+00:00",
                "last_error_code": 429,
            }
        )
    _write_store(root_path, root_store)
    _write_store(profile_path, _profile_without_codex_store("work"))
    profile_before = profile_path.read_bytes()

    pool = CP.load_pool("openai-codex")
    cached = next(item for item in pool.entries() if item.id == "root-device")
    assert cached.source_store_path == root_path
    if operation == "current":
        pool._current_id = cached.id

    removed = _read_store(root_path)
    removed["providers"].pop("openai-codex")
    _write_store(root_path, removed)
    root_after_removal = root_path.read_bytes()
    post_count = 0

    def fake_refresh(*_args, **_kwargs):
        nonlocal post_count
        post_count += 1
        return {
            "access_token": "must-not-be-used-access",
            "refresh_token": "must-not-be-used-refresh",
        }

    monkeypatch.setattr(A, "refresh_codex_oauth_pure", fake_refresh)

    result = None
    if operation == "status":
        pool._mark_exhausted(cached, 429)
    elif operation == "reset":
        pool.reset_statuses()
    elif operation == "rotation":
        result = pool.mark_exhausted_and_rotate(
            status_code=429,
            credential_id="root-device",
        )
    elif operation == "select":
        result = pool.select()
    elif operation == "current":
        result = pool.current()
    elif operation == "peek":
        result = pool.peek()
    elif operation == "lease":
        result = pool.acquire_lease("root-device")
    elif operation == "refresh":
        result = pool._refresh_entry(cached, force=True)
    else:
        pool.add_entry(
            PooledCredential(
                provider="openai-codex",
                id="profile-work-added",
                label="profile work added",
                auth_type=AUTH_TYPE_OAUTH,
                priority=99,
                source="manual:device_code",
                access_token="profile-work-added-access",
                refresh_token="profile-work-added-refresh",
            )
        )

    assert post_count == 0
    assert "root-device" not in {item.id for item in pool.entries()}
    if operation in {"current", "lease", "refresh", "rotation"}:
        assert result is None
    elif operation in {"select", "peek"}:
        assert result is not None
        assert result.id == "root-independent"

    survivor = pool.select()
    assert survivor is not None
    assert survivor.id == "root-independent"
    assert survivor.access_token == "root-independent-access"
    assert root_path.read_bytes() == root_after_removal

    if operation == "persistence":
        profile_ids = {
            item["id"]
            for item in _read_store(profile_path)["credential_pool"]["openai-codex"]
        }
        assert profile_ids == {"profile-work-added"}
    else:
        assert profile_path.read_bytes() == profile_before


def _run_refresh_overlap(
    *,
    profile_path,
    root_path,
    monkeypatch,
    operation,
    manual_only=False,
):
    root_store = _healthy_root_manual_store()
    credential_id = "root-independent" if manual_only else "root-device"
    old_access = (
        "root-independent-access" if manual_only else "root-old-access"
    )
    old_refresh = (
        "root-independent-refresh" if manual_only else "root-old-refresh"
    )
    if manual_only:
        root_store["providers"].pop("openai-codex")
        root_store["credential_pool"]["openai-codex"] = [
            item
            for item in root_store["credential_pool"]["openai-codex"]
            if item["id"] == credential_id
        ]
        root_store["credential_pool"]["openai-codex"][0]["priority"] = 0
    elif operation == "reset":
        singleton = next(
            item
            for item in root_store["credential_pool"]["openai-codex"]
            if item["id"] == credential_id
        )
        singleton.update(
            {
                "last_status": "exhausted",
                "last_status_at": "2026-08-06T03:00:00+00:00",
                "last_error_code": 429,
            }
        )

    _write_store(root_path, root_store)
    _write_store(profile_path, _profile_without_codex_store("overlap"))
    profile_before = profile_path.read_bytes()
    pool = CP.load_pool("openai-codex")
    cached = next(item for item in pool.entries() if item.id == credential_id)
    if operation == "current":
        pool._current_id = credential_id

    post_entered = threading.Event()
    allow_post_return = threading.Event()
    operation_transaction_entered = threading.Event()
    validation_timed_out = threading.Event()
    refresh_done = threading.Event()
    operation_done = threading.Event()
    post_count = 0
    results = {}
    errors = []

    def fake_refresh(access_token, refresh_token, **_kwargs):
        nonlocal post_count
        assert access_token == old_access
        assert refresh_token == old_refresh
        post_count += 1
        post_entered.set()
        assert allow_post_return.wait(timeout=3)
        return {
            "access_token": "rotated-access",
            "refresh_token": "rotated-refresh",
            "last_refresh": "2026-08-06T04:00:00Z",
        }

    monkeypatch.setattr(A, "refresh_codex_oauth_pure", fake_refresh)
    real_transaction = A._provider_state_transaction
    real_auth_lock = A._auth_store_lock

    @contextmanager
    def bounded_auth_lock(*args, **kwargs):
        is_operation = threading.current_thread().name == "pool-operation"
        if is_operation:
            kwargs["timeout_seconds"] = 1.0
        try:
            with real_auth_lock(*args, **kwargs):
                yield
        except TimeoutError:
            if is_operation:
                validation_timed_out.set()
            raise

    monkeypatch.setattr(A, "_auth_store_lock", bounded_auth_lock)

    @contextmanager
    def tracked_transaction(*args, **kwargs):
        is_operation = threading.current_thread().name == "pool-operation"
        if is_operation:
            operation_transaction_entered.set()
            kwargs["timeout_seconds"] = 1.0
        try:
            with real_transaction(*args, **kwargs) as transaction:
                yield transaction
        except TimeoutError:
            if is_operation:
                validation_timed_out.set()
            raise

    monkeypatch.setattr(A, "_provider_state_transaction", tracked_transaction)

    def refresh_worker():
        try:
            results["refresh"] = pool._refresh_entry(cached, force=True)
        except BaseException as exc:
            errors.append(("refresh", exc))
        finally:
            refresh_done.set()

    def operation_worker():
        try:
            if operation == "select":
                results["operation"] = pool.select()
            elif operation == "peek":
                results["operation"] = pool.peek()
            elif operation == "current":
                results["operation"] = pool.current()
            elif operation == "lease":
                results["lease_id"] = pool.acquire_lease(credential_id)
                results["operation"] = pool.current()
            elif operation == "rotation":
                results["operation"] = pool.mark_exhausted_and_rotate(
                    status_code=429,
                    credential_id=credential_id,
                )
            elif operation == "status":
                results["operation"] = pool._mark_exhausted(cached, 429)
            elif operation == "reset":
                results["operation"] = pool.reset_statuses()
            else:
                results["operation"] = pool.add_entry(
                    PooledCredential(
                        provider="openai-codex",
                        id="profile-overlap-added",
                        label="profile overlap added",
                        auth_type=AUTH_TYPE_OAUTH,
                        priority=99,
                        source="manual:device_code",
                        access_token="profile-overlap-access",
                        refresh_token="profile-overlap-refresh",
                    )
                )
        except BaseException as exc:
            errors.append(("operation", exc))
        finally:
            operation_done.set()

    refresher = threading.Thread(target=refresh_worker, name="pool-refresh")
    operator = threading.Thread(target=operation_worker, name="pool-operation")
    refresher.start()
    assert post_entered.wait(timeout=3)
    operator.start()
    assert operation_transaction_entered.wait(timeout=3)
    allow_post_return.set()
    refresher.join(timeout=3)
    operator.join(timeout=3)

    assert refresh_done.is_set()
    assert operation_done.is_set()
    assert not refresher.is_alive()
    assert not operator.is_alive()
    assert not errors
    assert not validation_timed_out.is_set()
    assert post_count == 1
    assert results["refresh"] is not None
    assert results["refresh"].access_token == "rotated-access"

    root_after = _read_store(root_path)
    root_entry = next(
        item
        for item in root_after["credential_pool"]["openai-codex"]
        if item["id"] == credential_id
    )
    assert root_entry["access_token"] == "rotated-access"
    assert root_entry["refresh_token"] == "rotated-refresh"
    in_memory = next(item for item in pool.entries() if item.id == credential_id)
    assert in_memory.access_token == "rotated-access"
    assert in_memory.refresh_token == "rotated-refresh"

    if operation in {"select", "peek", "current", "lease"}:
        assert results["operation"] is not None
        assert results["operation"].id == credential_id
        assert results["operation"].access_token == "rotated-access"
    elif operation == "rotation":
        assert results["operation"] is not None
        assert results["operation"].id == "root-independent"

    if operation == "persistence":
        profile_ids = {
            item["id"]
            for item in _read_store(profile_path)["credential_pool"]["openai-codex"]
        }
        assert profile_ids == {"profile-overlap-added"}
    else:
        assert profile_path.read_bytes() == profile_before


@pytest.mark.parametrize(
    "operation",
    ["select", "peek", "current", "lease", "rotation", "status", "reset", "persistence"],
)
def test_singleton_refresh_overlap_never_inverts_pool_and_auth_locks(
    profile_and_root, monkeypatch, operation
):
    profile_path, root_path = profile_and_root
    _run_refresh_overlap(
        profile_path=profile_path,
        root_path=root_path,
        monkeypatch=monkeypatch,
        operation=operation,
    )


def test_source_owned_manual_refresh_overlap_never_inverts_selection_lock(
    profile_and_root, monkeypatch
):
    profile_path, root_path = profile_and_root
    _run_refresh_overlap(
        profile_path=profile_path,
        root_path=root_path,
        monkeypatch=monkeypatch,
        operation="select",
        manual_only=True,
    )


@pytest.mark.parametrize("failure", [TimeoutError, RuntimeError])
@pytest.mark.parametrize("operation", ["select", "peek", "current", "lease", "rotation"])
def test_source_validation_failure_fails_borrowed_selection_closed(
    profile_and_root, monkeypatch, operation, failure
):
    profile_path, root_path = profile_and_root
    _write_store(root_path, _healthy_root_manual_store())
    _write_store(profile_path, _profile_without_codex_store("validation"))
    profile_before = profile_path.read_bytes()
    root_before = root_path.read_bytes()
    pool = CP.load_pool("openai-codex")
    if operation == "current":
        pool._current_id = "root-device"

    @contextmanager
    def fail_validation(*_args, **_kwargs):
        raise failure("fake source validation failure")
        yield

    monkeypatch.setattr(A, "_provider_state_transaction", fail_validation)

    if operation == "select":
        result = pool.select()
    elif operation == "peek":
        result = pool.peek()
    elif operation == "current":
        result = pool.current()
    elif operation == "lease":
        result = pool.acquire_lease("root-device")
    else:
        result = pool.mark_exhausted_and_rotate(
            status_code=429,
            credential_id="root-device",
        )

    assert result is None
    assert root_path.read_bytes() == root_before
    assert profile_path.read_bytes() == profile_before


@pytest.mark.parametrize(
    "operation",
    [
        "status",
        "reset",
        "rotation",
        "select",
        "current",
        "peek",
        "lease",
        "refresh",
        "persistence",
    ],
)
def test_load_time_provider_removal_never_reclassifies_singleton_as_pool_owned(
    profile_and_root, monkeypatch, operation
):
    """Owner kind comes from the hydrated row, not a later unlocked reread."""
    profile_path, root_path = profile_and_root
    root_store = _root_codex_store()
    root_store["credential_pool"]["openai-codex"] = [
        root_store["credential_pool"]["openai-codex"][0]
    ]
    if operation == "reset":
        root_store["credential_pool"]["openai-codex"][0].update(
            {
                "last_status": "exhausted",
                "last_status_at": 1.0,
                "last_error_code": 429,
            }
        )
    _write_store(root_path, root_store)
    _write_store(profile_path, _profile_without_codex_store("load-race"))
    profile_before = profile_path.read_bytes()

    real_init = CredentialPool.__init__
    provider_removed = threading.Event()

    def init_then_remove_provider(self, provider, entries):
        real_init(self, provider, entries)
        if provider != "openai-codex":
            return
        source = _read_store(root_path)
        source["providers"].pop("openai-codex")
        _write_store(root_path, source)
        provider_removed.set()

    monkeypatch.setattr(CredentialPool, "__init__", init_then_remove_provider)
    post_calls = []

    def unexpected_post(access_token, refresh_token, **_kwargs):
        post_calls.append((access_token, refresh_token))
        return {
            "access_token": "must-not-be-used-access",
            "refresh_token": "must-not-be-used-refresh",
        }

    monkeypatch.setattr(A, "refresh_codex_oauth_pure", unexpected_post)
    pool = CP.load_pool("openai-codex")
    assert provider_removed.is_set()
    cached = next(item for item in pool.entries() if item.id == "root-device")
    if operation == "current":
        pool._current_id = cached.id
    root_after_removal = root_path.read_bytes()

    result = None
    if operation == "status":
        pool._mark_exhausted(cached, 429)
    elif operation == "reset":
        pool.reset_statuses()
    elif operation == "rotation":
        result = pool.mark_exhausted_and_rotate(
            status_code=429,
            credential_id="root-device",
        )
    elif operation == "select":
        result = pool.select()
    elif operation == "current":
        result = pool.current()
    elif operation == "peek":
        result = pool.peek()
    elif operation == "lease":
        result = pool.acquire_lease("root-device")
    elif operation == "refresh":
        result = pool._refresh_entry(cached, force=True)
    else:
        pool.add_entry(
            PooledCredential(
                provider="openai-codex",
                id="profile-load-race-added",
                label="profile load race added",
                auth_type=AUTH_TYPE_OAUTH,
                priority=99,
                source="manual:device_code",
                access_token="profile-load-race-access",
                refresh_token="profile-load-race-refresh",
            )
        )

    assert result is None
    assert post_calls == []
    assert "root-device" not in {item.id for item in pool.entries()}
    current = pool.current()
    assert current is None or current.id != "root-device"
    peeked = pool.peek()
    assert peeked is None or peeked.id != "root-device"
    assert pool.acquire_lease("root-device") is None
    assert root_path.read_bytes() == root_after_removal
    if operation == "persistence":
        profile_ids = {
            item["id"]
            for item in _read_store(profile_path)["credential_pool"]["openai-codex"]
        }
        assert profile_ids == {"profile-load-race-added"}
    else:
        assert profile_path.read_bytes() == profile_before


def test_unreadable_load_time_owner_classification_fails_singleton_closed(
    profile_and_root, monkeypatch
):
    profile_path, root_path = profile_and_root
    root_store = _root_codex_store()
    root_store["credential_pool"]["openai-codex"] = [
        root_store["credential_pool"]["openai-codex"][0]
    ]
    _write_store(root_path, root_store)
    _write_store(profile_path, _profile_without_codex_store("unreadable-kind"))
    root_before = root_path.read_bytes()
    profile_before = profile_path.read_bytes()

    real_init = CredentialPool.__init__
    fail_next_root_read = threading.Event()

    def init_then_arm_unreadable_source(self, provider, entries):
        real_init(self, provider, entries)
        if provider == "openai-codex":
            fail_next_root_read.set()

    real_load = A._load_auth_store

    def fail_one_root_read(path=None):
        if (
            fail_next_root_read.is_set()
            and path is not None
            and A._same_path(path, root_path)
        ):
            fail_next_root_read.clear()
            raise RuntimeError("fake unreadable owner classification")
        return real_load(path)

    monkeypatch.setattr(CredentialPool, "__init__", init_then_arm_unreadable_source)
    monkeypatch.setattr(A, "_load_auth_store", fail_one_root_read)

    pool = CP.load_pool("openai-codex")
    assert pool.select() is None
    assert not fail_next_root_read.is_set()
    assert root_path.read_bytes() == root_before
    assert profile_path.read_bytes() == profile_before


def test_root_manual_row_remains_independently_pool_owned_without_singleton(
    profile_and_root,
):
    profile_path, root_path = profile_and_root
    root_store = _healthy_root_manual_store()
    root_store["providers"].pop("openai-codex")
    root_store["credential_pool"]["openai-codex"] = [
        item
        for item in root_store["credential_pool"]["openai-codex"]
        if item["id"] == "root-independent"
    ]
    root_store["credential_pool"]["openai-codex"][0]["priority"] = 0
    _write_store(root_path, root_store)
    _write_store(profile_path, _profile_without_codex_store("manual-control"))

    pool = CP.load_pool("openai-codex")
    manual = next(item for item in pool.entries() if item.id == "root-independent")
    owner = pool._trusted_codex_source_owner(manual)
    assert owner is not None
    assert owner.owner_kind == "pool"
    assert pool.select() == manual


@pytest.mark.parametrize("operation", ["rotation", "refresh", "lease"])
def test_exact_failed_source_id_never_falls_back_to_unrelated_survivors(
    profile_and_root, monkeypatch, operation
):
    profile_path, root_path = profile_and_root
    root_store = _root_codex_store()
    root_store["credential_pool"]["openai-codex"] = [
        root_store["credential_pool"]["openai-codex"][0]
    ]
    _write_store(root_path, root_store)
    _write_store(profile_path, _profile_without_codex_store("exact-failed"))
    pool = CP.load_pool("openai-codex")
    for suffix in ("a", "b"):
        pool.add_entry(
            PooledCredential(
                provider="openai-codex",
                id=f"local-{suffix}",
                label=f"local {suffix}",
                auth_type=AUTH_TYPE_OAUTH,
                priority=99,
                source="manual:device_code",
                access_token=f"local-{suffix}-access",
                refresh_token=f"local-{suffix}-refresh",
            )
        )
    survivor_before = {
        item.id: item
        for item in pool.entries()
        if item.id in {"local-a", "local-b"}
    }

    @contextmanager
    def fail_validation(*_args, **_kwargs):
        raise TimeoutError("fake source validation timeout")
        yield

    post_calls = []

    def unexpected_post(access_token, refresh_token, **_kwargs):
        post_calls.append((access_token, refresh_token))
        return {
            "access_token": "must-not-be-used-access",
            "refresh_token": "must-not-be-used-refresh",
        }

    monkeypatch.setattr(A, "_provider_state_transaction", fail_validation)
    monkeypatch.setattr(A, "refresh_codex_oauth_pure", unexpected_post)

    if operation == "rotation":
        result = pool.mark_exhausted_and_rotate(
            status_code=429,
            credential_id="root-device",
        )
    elif operation == "refresh":
        result = pool.try_refresh_matching(credential_id="root-device")
    else:
        result = pool.acquire_lease("root-device")

    assert result is None
    assert post_calls == []
    assert pool.current() is None
    assert {
        item.id: item
        for item in pool.entries()
        if item.id in {"local-a", "local-b"}
    } == survivor_before


def _local_pool_for_persistence_race(
    tmp_path,
    monkeypatch,
    *,
    initial_status=None,
):
    auth_path = tmp_path / "profile" / "auth.json"
    monkeypatch.setattr(A, "_auth_file_path", lambda: auth_path)
    monkeypatch.setattr(A, "_global_auth_file_path", lambda: None)
    entry = PooledCredential(
        provider="openrouter",
        id="local-entry",
        label="local entry",
        auth_type="api_key",
        priority=0,
        source="manual:api_key",
        access_token="local-access",
        last_status=initial_status,
        last_status_at=1.0 if initial_status else None,
        last_error_code=429 if initial_status else None,
        last_error_reason="old_rate_limit" if initial_status else None,
    )
    _write_store(
        auth_path,
        {
            "version": 1,
            "credential_pool": {"openrouter": [entry.to_dict()]},
        },
    )
    return auth_path, CredentialPool("openrouter", [entry])


def _run_two_generation_persistence_race(
    tmp_path,
    monkeypatch,
    *,
    first_mutation,
    initial_status=None,
):
    auth_path, pool = _local_pool_for_persistence_race(
        tmp_path,
        monkeypatch,
        initial_status=initial_status,
    )
    first_write_entered = threading.Event()
    allow_first_write = threading.Event()
    newer_mutation_done = threading.Event()
    writes = []
    errors = []
    real_write = CP.write_credential_pool

    def paused_write(provider, entries, *, removed_ids=None):
        writes.append(
            {
                "statuses": [entry.get("last_status") for entry in entries],
                "reasons": [entry.get("last_error_reason") for entry in entries],
                "removed_ids": list(removed_ids or []),
            }
        )
        if len(writes) == 1:
            first_write_entered.set()
            assert allow_first_write.wait(timeout=5)
        return real_write(provider, entries, removed_ids=removed_ids)

    real_replace = pool._replace_entry

    def tracked_replace(old, new, **kwargs):
        result = real_replace(old, new, **kwargs)
        if threading.current_thread().name == "newer-terminal-mutation":
            newer_mutation_done.set()
        return result

    monkeypatch.setattr(CP, "write_credential_pool", paused_write)
    pool._replace_entry = tracked_replace

    def first_worker():
        try:
            if first_mutation == "reset":
                assert pool.reset_statuses() == 1
            else:
                current = pool.entries()[0]
                pool._mark_exhausted(
                    current,
                    429,
                    {"reason": "rate_limit"},
                )
        except BaseException as exc:
            errors.append(("first", exc))

    def newer_worker():
        try:
            current = pool.entries()[0]
            pool._mark_exhausted(
                current,
                401,
                {"reason": "invalid_grant"},
            )
        except BaseException as exc:
            errors.append(("newer", exc))

    first = threading.Thread(target=first_worker, name="older-persistence")
    newer = threading.Thread(target=newer_worker, name="newer-terminal-mutation")
    first.start()
    assert first_write_entered.wait(timeout=5)
    assert pool._lock.acquire(timeout=1)
    pool._lock.release()
    newer.start()
    assert newer_mutation_done.wait(timeout=5)
    allow_first_write.set()
    first.join(timeout=5)
    newer.join(timeout=5)

    assert not first.is_alive()
    assert not newer.is_alive()
    assert errors == []
    memory_entry = pool.entries()[0]
    disk_entry = _read_store(auth_path)["credential_pool"]["openrouter"][0]
    assert memory_entry.last_status == "dead"
    assert memory_entry.last_error_reason == "invalid_grant"
    assert disk_entry["last_status"] == "dead"
    assert disk_entry["last_error_reason"] == "invalid_grant"
    expected_first_status = None if first_mutation == "reset" else "exhausted"
    expected_first_reason = None if first_mutation == "reset" else "rate_limit"
    assert [write["statuses"] for write in writes] == [
        [expected_first_status],
        ["dead"],
    ]
    assert [write["reasons"] for write in writes] == [
        [expected_first_reason],
        ["invalid_grant"],
    ]
    pool._persist_pending_changes()
    assert len(writes) == 2


def test_reset_snapshot_cannot_clear_newer_terminal_dirty_generation(
    tmp_path, monkeypatch
):
    _run_two_generation_persistence_race(
        tmp_path,
        monkeypatch,
        first_mutation="reset",
        initial_status="exhausted",
    )


def test_cooldown_snapshot_cannot_clear_newer_terminal_dirty_generation(
    tmp_path, monkeypatch
):
    _run_two_generation_persistence_race(
        tmp_path,
        monkeypatch,
        first_mutation="cooldown",
    )


def test_removal_persistence_converges_after_same_id_readd(
    tmp_path, monkeypatch
):
    auth_path, pool = _local_pool_for_persistence_race(tmp_path, monkeypatch)
    first_write_entered = threading.Event()
    allow_first_write = threading.Event()
    readd_mutated = threading.Event()
    writes = []
    errors = []
    real_write = CP.write_credential_pool
    real_pending = pool._persist_pending_changes

    def paused_write(provider, entries, *, removed_ids=None):
        writes.append(
            {
                "ids": [entry.get("id") for entry in entries],
                "tokens": [entry.get("access_token") for entry in entries],
                "removed_ids": list(removed_ids or []),
            }
        )
        if len(writes) == 1:
            first_write_entered.set()
            assert allow_first_write.wait(timeout=5)
        return real_write(provider, entries, removed_ids=removed_ids)

    def tracked_pending():
        if threading.current_thread().name == "same-id-readd":
            readd_mutated.set()
        return real_pending()

    monkeypatch.setattr(CP, "write_credential_pool", paused_write)
    pool._persist_pending_changes = tracked_pending

    def remove_worker():
        try:
            removed = pool.remove_index(1)
            assert removed is not None
            assert removed.id == "local-entry"
        except BaseException as exc:
            errors.append(("remove", exc))

    def readd_worker():
        try:
            pool.add_entry(
                PooledCredential(
                    provider="openrouter",
                    id="local-entry",
                    label="replacement entry",
                    auth_type="api_key",
                    priority=0,
                    source="manual:api_key",
                    access_token="replacement-access",
                )
            )
        except BaseException as exc:
            errors.append(("readd", exc))

    remover = threading.Thread(target=remove_worker, name="remove-entry")
    readd = threading.Thread(target=readd_worker, name="same-id-readd")
    remover.start()
    assert first_write_entered.wait(timeout=5)
    assert pool._lock.acquire(timeout=1)
    pool._lock.release()
    readd.start()
    assert readd_mutated.wait(timeout=5)
    allow_first_write.set()
    remover.join(timeout=5)
    readd.join(timeout=5)

    assert not remover.is_alive()
    assert not readd.is_alive()
    assert errors == []
    memory_entries = pool.entries()
    assert [(entry.id, entry.access_token) for entry in memory_entries] == [
        ("local-entry", "replacement-access")
    ]
    disk_entries = _read_store(auth_path)["credential_pool"]["openrouter"]
    assert [(entry["id"], entry["access_token"]) for entry in disk_entries] == [
        ("local-entry", "replacement-access")
    ]
    assert writes == [
        {"ids": [], "tokens": [], "removed_ids": ["local-entry"]},
        {
            "ids": ["local-entry"],
            "tokens": ["replacement-access"],
            "removed_ids": [],
        },
    ]
    pool._persist_pending_changes()
    assert len(writes) == 2


def _pool_lock_owned_by_current_thread(pool):
    is_owned = getattr(pool._lock, "_is_owned", None)
    return bool(is_owned and is_owned())


def _anthropic_pool_for_lock_order_test(tmp_path, monkeypatch):
    auth_path = tmp_path / "profile" / "auth.json"
    monkeypatch.setattr(A, "_auth_file_path", lambda: auth_path)
    monkeypatch.setattr(A, "_global_auth_file_path", lambda: None)
    entry = PooledCredential(
        provider="anthropic",
        id="claude-entry",
        label="Claude entry",
        auth_type=AUTH_TYPE_OAUTH,
        priority=0,
        source="claude_code",
        access_token="stale-access",
        refresh_token="stale-refresh",
        expires_at_ms=9_999_999_999_999,
        last_status="exhausted",
        last_status_at=1.0,
        last_error_code=429,
        last_error_reason="rate_limit",
    )
    _write_store(
        auth_path,
        {
            "version": 1,
            "credential_pool": {"anthropic": [entry.to_dict()]},
        },
    )
    return auth_path, CredentialPool("anthropic", [entry])


def test_anthropic_status_sync_cannot_deadlock_with_concurrent_persistence(
    tmp_path, monkeypatch
):
    """A real provider status sync must not oppose the persistence lock."""
    from agent import anthropic_adapter

    auth_path, pool = _anthropic_pool_for_lock_order_test(tmp_path, monkeypatch)
    original = pool.entries()[0]
    pool._replace_entry(original, replace(original, request_count=1))

    status_sync_entered = threading.Event()
    persist_lock_acquired = threading.Event()
    raw_persist_lock = threading.Lock()
    errors = []

    class SignalingPersistLock:
        def __enter__(self):
            raw_persist_lock.acquire()
            persist_lock_acquired.set()
            return self

        def __exit__(self, *_exc_info):
            raw_persist_lock.release()

    pool.__dict__["_persist_lock"] = SignalingPersistLock()

    def controlled_credentials_read():
        status_sync_entered.set()
        assert persist_lock_acquired.wait(timeout=5)
        return {
            "accessToken": "fresh-access",
            "refreshToken": "fresh-refresh",
            "expiresAt": 9_999_999_999_999,
        }

    monkeypatch.setattr(
        anthropic_adapter,
        "read_claude_code_credentials",
        controlled_credentials_read,
    )

    def status_worker():
        try:
            assert pool.has_available() is True
        except BaseException as exc:
            errors.append(("status", exc))

    def persistence_worker():
        try:
            assert status_sync_entered.wait(timeout=5)
            pool._persist()
        except BaseException as exc:
            errors.append(("persistence", exc))

    status_thread = threading.Thread(target=status_worker, daemon=True)
    persistence_thread = threading.Thread(target=persistence_worker, daemon=True)
    status_thread.start()
    persistence_thread.start()
    assert status_sync_entered.wait(timeout=5)
    assert persist_lock_acquired.wait(timeout=5)
    status_thread.join(timeout=0.5)
    persistence_thread.join(timeout=0.5)

    deadlock_reproduced = status_thread.is_alive() and persistence_thread.is_alive()
    if status_thread.is_alive() or persistence_thread.is_alive():
        if raw_persist_lock.locked():
            raw_persist_lock.release()
        status_thread.join(timeout=5)
        persistence_thread.join(timeout=5)

    assert not deadlock_reproduced, (
        f"status_thread_alive={status_thread.is_alive()} "
        f"persistence_thread_alive={persistence_thread.is_alive()}"
    )
    assert not status_thread.is_alive()
    assert not persistence_thread.is_alive()
    assert errors == []
    memory = pool.entries()[0]
    disk = _read_store(auth_path)["credential_pool"]["anthropic"][0]
    assert memory.access_token == "fresh-access"
    assert memory.refresh_token == "fresh-refresh"
    assert memory.last_status is None
    assert disk["last_status"] is None
    assert disk["request_count"] == 1


@pytest.mark.parametrize(
    "operation",
    [
        "has_available",
        "next_available_at",
        "peek",
        "select",
        "lease",
        "rotation",
        "refresh",
        "status",
    ],
)
def test_public_pool_mutations_drain_persistence_after_releasing_pool_lock(
    tmp_path, monkeypatch, operation
):
    from agent import anthropic_adapter

    auth_path, pool = _anthropic_pool_for_lock_order_test(tmp_path, monkeypatch)
    io_lock_states = []
    real_write = CP.write_credential_pool
    credentials = {
        "accessToken": "fresh-access",
        "refreshToken": "fresh-refresh",
        "expiresAt": 9_999_999_999_999,
    }

    def fresh_credentials():
        io_lock_states.append(("credentials_read", _pool_lock_owned_by_current_thread(pool)))
        return dict(credentials)

    def recording_write(provider, entries, *, removed_ids=None):
        io_lock_states.append(("pool_write", _pool_lock_owned_by_current_thread(pool)))
        return real_write(provider, entries, removed_ids=removed_ids)

    def refresh_credentials(refresh_token, *, use_json):
        io_lock_states.append(("refresh", _pool_lock_owned_by_current_thread(pool)))
        assert refresh_token == "fresh-refresh"
        assert use_json is False
        return {
            "access_token": "rotated-access",
            "refresh_token": "rotated-refresh",
            "expires_at_ms": 9_999_999_999_999,
        }

    def write_credentials(access_token, refresh_token, expires_at_ms):
        io_lock_states.append(
            ("credentials_write", _pool_lock_owned_by_current_thread(pool))
        )
        assert (access_token, refresh_token, expires_at_ms) == (
            "rotated-access",
            "rotated-refresh",
            9_999_999_999_999,
        )
        credentials.update(
            accessToken=access_token,
            refreshToken=refresh_token,
            expiresAt=expires_at_ms,
        )
        return True

    def write_credentials_locked(
        access_token,
        refresh_token,
        expires_at_ms,
        *,
        expected_refresh_token,
        allow_missing=False,
    ):
        del allow_missing
        assert credentials["refreshToken"] == expected_refresh_token
        return write_credentials(access_token, refresh_token, expires_at_ms)

    def refresh_source(observed):
        refreshed = anthropic_adapter.refresh_anthropic_oauth_pure(
            observed["refreshToken"],
            use_json=False,
        )
        anthropic_adapter._write_claude_code_credentials_locked(
            refreshed["access_token"],
            refreshed["refresh_token"],
            refreshed["expires_at_ms"],
            expected_refresh_token=observed["refreshToken"],
            allow_missing=False,
        )
        return {
            **credentials,
            "source": "claude_code_credentials_file",
        }

    monkeypatch.setattr(
        anthropic_adapter,
        "read_claude_code_credentials",
        fresh_credentials,
    )
    monkeypatch.setattr(
        anthropic_adapter,
        "_read_claude_code_credentials_from_file",
        fresh_credentials,
    )
    monkeypatch.setattr(
        anthropic_adapter,
        "refresh_anthropic_oauth_pure",
        refresh_credentials,
    )
    monkeypatch.setattr(
        anthropic_adapter,
        "_write_claude_code_credentials",
        write_credentials,
    )
    monkeypatch.setattr(
        anthropic_adapter,
        "_write_claude_code_credentials_locked",
        write_credentials_locked,
    )
    monkeypatch.setattr(
        anthropic_adapter,
        "_refresh_claude_code_source_credentials",
        refresh_source,
    )
    monkeypatch.setattr(CP, "write_credential_pool", recording_write)

    if operation == "has_available":
        assert pool.has_available() is True
    elif operation == "next_available_at":
        assert pool.next_available_at() is None
    elif operation == "peek":
        assert pool.peek() is not None
    elif operation == "select":
        assert pool.select() is not None
    elif operation == "lease":
        assert pool.acquire_lease() == "claude-entry"
    elif operation == "rotation":
        assert (
            pool.mark_exhausted_and_rotate(
                status_code=429,
                credential_id="claude-entry",
            )
            is None
        )
    elif operation == "refresh":
        refreshed = pool.try_refresh_matching(credential_id="claude-entry")
        assert refreshed is not None
        assert refreshed.access_token == "rotated-access"
    else:
        assert pool.reset_statuses() == 1

    assert io_lock_states
    assert [kind for kind, owned in io_lock_states if owned] == []
    memory = [entry.to_dict() for entry in pool.entries()]
    disk = _read_store(auth_path)["credential_pool"]["anthropic"]
    assert disk == memory
    assert pool._dirty_entry_ids == set()
    assert pool._pending_removed_entries == []
    if operation == "refresh":
        monkeypatch.setattr(
            A,
            "is_provider_explicitly_configured",
            lambda provider: provider == "anthropic",
        )
        reloaded = next(
            entry
            for entry in CP.load_pool("anthropic").entries()
            if entry.source == "claude_code"
        )
        assert reloaded.access_token == "rotated-access"
        assert reloaded.refresh_token == "rotated-refresh"
        assert reloaded.last_status == CP.STATUS_OK


@pytest.mark.parametrize("provider", ["xai-oauth", "nous"])
def test_terminal_provider_cleanup_persists_removed_ids_outside_pool_lock(
    tmp_path, monkeypatch, provider
):
    auth_path = tmp_path / provider / "auth.json"
    monkeypatch.setattr(A, "_auth_file_path", lambda: auth_path)
    monkeypatch.setattr(A, "_global_auth_file_path", lambda: None)
    device = _entry(
        provider,
        id=f"{provider}-device",
        access_token="device-access",
        refresh_token="device-refresh",
    )
    extra_removed = replace(
        device,
        id=f"{provider}-manual-device",
        source="manual:device_code",
        priority=1,
        access_token="manual-device-access",
        refresh_token="manual-device-refresh",
    )
    survivor = replace(
        device,
        id=f"{provider}-survivor",
        source="manual:oauth",
        priority=2,
        access_token="survivor-access",
        refresh_token="survivor-refresh",
    )
    entries = [device, extra_removed, survivor]
    if provider == "xai-oauth":
        provider_state = {
            "tokens": {
                "access_token": device.access_token,
                "refresh_token": device.refresh_token,
            }
        }
    else:
        provider_state = {
            "access_token": device.access_token,
            "refresh_token": device.refresh_token,
        }
    _write_store(
        auth_path,
        {
            "version": 1,
            "providers": {provider: provider_state},
            "credential_pool": {provider: [entry.to_dict() for entry in entries]},
        },
    )
    pool = CredentialPool(provider, entries)
    write_lock_states = []
    real_write = CP.write_credential_pool

    def recording_write(provider_name, persisted, *, removed_ids=None):
        write_lock_states.append(_pool_lock_owned_by_current_thread(pool))
        return real_write(provider_name, persisted, removed_ids=removed_ids)

    def reject_refresh(*_args, **_kwargs):
        raise RuntimeError("terminal refresh failure")

    monkeypatch.setattr(CP, "write_credential_pool", recording_write)
    if provider == "xai-oauth":
        monkeypatch.setattr(A, "refresh_xai_oauth_pure", reject_refresh)
        monkeypatch.setattr(A, "_is_terminal_xai_oauth_refresh_error", lambda _exc: True)
        expected_ids = {device.id, extra_removed.id, survivor.id}
    else:
        monkeypatch.setattr(A, "resolve_nous_runtime_credentials", reject_refresh)
        monkeypatch.setattr(A, "_is_terminal_nous_refresh_error", lambda _exc: True)
        expected_ids = {survivor.id}

    errors = []

    def cleanup_worker():
        try:
            assert pool._refresh_entry(device, force=True) is None
        except BaseException as exc:
            errors.append(exc)

    worker = threading.Thread(target=cleanup_worker, daemon=True)
    worker.start()
    worker.join(timeout=5)

    assert not worker.is_alive()
    assert errors == []
    if provider == "xai-oauth":
        assert write_lock_states == []
    else:
        assert write_lock_states
        assert not any(write_lock_states)
    assert {entry.id for entry in pool.entries()} == expected_ids
    persisted = _read_store(auth_path)["credential_pool"][provider]
    assert {entry["id"] for entry in persisted} == expected_ids


@pytest.mark.parametrize("construction", ["direct", "replace_then_add"])
@pytest.mark.parametrize("target_kind", ["arbitrary", "known_global"])
def test_untrusted_runtime_source_path_never_gains_owner_write_authority(
    profile_and_root, construction, target_kind
):
    profile_path, root_path = profile_and_root
    if target_kind == "arbitrary":
        source_path = profile_path.parent / "arbitrary-owner" / "auth.json"
        source_store = _healthy_root_manual_store()
        source_store["credential_pool"]["openai-codex"] = [
            item
            for item in source_store["credential_pool"]["openai-codex"]
            if item["id"] == "root-independent"
        ]
        _write_store(source_path, source_store)
    else:
        source_path = root_path
        _write_store(source_path, _healthy_root_manual_store())
    source_before = source_path.read_bytes()
    _write_store(profile_path, _profile_without_codex_store("untrusted"))

    untrusted = PooledCredential(
        provider="openai-codex",
        id="root-independent",
        label="untrusted runtime path",
        auth_type=AUTH_TYPE_OAUTH,
        priority=0,
        source="manual:device_code",
        access_token="root-independent-access",
        refresh_token="root-independent-refresh",
        source_store_path=source_path,
    )
    if construction == "direct":
        pool = CredentialPool("openai-codex", [untrusted])
        entry = untrusted
    else:
        pool = CredentialPool("openai-codex", [])
        entry = pool.add_entry(replace(untrusted))

    pool._mark_exhausted(entry, 429)

    assert source_path.read_bytes() == source_before
    persisted = _read_store(profile_path)["credential_pool"]["openai-codex"]
    assert [item["id"] for item in persisted] == ["root-independent"]
    assert persisted[0]["last_status"] == "exhausted"
    assert persisted[0]["last_error_code"] == 429
    assert "source_store_path" not in persisted[0]


def test_load_pool_waiter_treats_removed_root_source_as_revoked(
    monkeypatch, tmp_path
):
    """A waiter must not refresh stale memory after its owning chain is removed."""
    if "fork" not in multiprocessing.get_all_start_methods():
        pytest.skip("requires fork so patched fake refresh stays network-free")

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.delenv("HERMES_HOME", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "not-the-root"))
    root_home = tmp_path / ".hermes"
    root_path = root_home / "auth.json"
    profile_path = root_home / "profiles" / "waiter" / "auth.json"
    _write_store(root_path, _root_codex_store())
    _write_store(profile_path, _profile_codex_store("waiter"))
    profile_before = profile_path.read_bytes()

    ctx = multiprocessing.get_context("fork")
    refresh_start = ctx.Event()
    source_lock_held = ctx.Event()
    waiter_attempting_source_lock = ctx.Event()
    post_count = ctx.Value("i", 0)
    reports = ctx.Queue()
    real_auth_lock = A._auth_store_lock

    @contextmanager
    def tracking_auth_lock(*args, **kwargs):
        target_path = kwargs.get("target_path")
        if (
            multiprocessing.current_process().name == "codex-revocation-waiter"
            and target_path is not None
            and A._same_path(target_path, root_path)
        ):
            waiter_attempting_source_lock.set()
        with real_auth_lock(*args, **kwargs):
            yield

    monkeypatch.setattr(A, "_auth_store_lock", tracking_auth_lock)

    def fake_refresh(*_args, **_kwargs):
        with post_count.get_lock():
            post_count.value += 1
        return {
            "access_token": "must-not-be-used-access",
            "refresh_token": "must-not-be-used-refresh",
            "last_refresh": "2026-08-06T00:00:00Z",
        }

    monkeypatch.setattr(A, "refresh_codex_oauth_pure", fake_refresh)

    def wait_then_refresh():
        token = set_hermes_home_override(profile_path.parent)
        try:
            pool = CP.load_pool("openai-codex")
            entry = next(item for item in pool.entries() if item.source == "device_code")
            reports.put(("loaded", entry.id))
            assert refresh_start.wait(timeout=5)
            refreshed = pool._refresh_entry(entry, force=True)
            usable_device_entries = [
                item.id
                for item in pool.entries()
                if item.source == "device_code" and item.access_token
            ]
            reports.put(
                (
                    "result",
                    refreshed.id if refreshed else None,
                    usable_device_entries,
                )
            )
        except BaseException as exc:
            reports.put(("error", repr(exc)))
        finally:
            reset_hermes_home_override(token)

    def remove_source_while_holding_lock():
        with A._auth_store_lock(target_path=root_path):
            source_lock_held.set()
            assert waiter_attempting_source_lock.wait(timeout=5)
            store = A._load_auth_store(root_path)
            store["providers"].pop("openai-codex", None)
            entries = store["credential_pool"]["openai-codex"]
            store["credential_pool"]["openai-codex"] = [
                item for item in entries if item.get("source") != "device_code"
            ]
            A._save_auth_store(store, target_path=root_path)

    waiter = ctx.Process(
        target=wait_then_refresh,
        name="codex-revocation-waiter",
    )
    owner = ctx.Process(
        target=remove_source_while_holding_lock,
        name="codex-revocation-owner",
    )
    waiter.start()
    loaded = reports.get(timeout=5)
    assert loaded[0] == "loaded", loaded
    owner.start()
    assert source_lock_held.wait(timeout=5)
    refresh_start.set()

    result = reports.get(timeout=5)
    waiter.join(timeout=5)
    owner.join(timeout=5)

    assert waiter.exitcode == 0
    assert owner.exitcode == 0
    assert result == ("result", None, [])
    assert post_count.value == 0
    assert loaded == ("loaded", "root-device")
    root = _read_store(root_path)
    assert "openai-codex" not in root["providers"]
    assert all(
        item.get("source") != "device_code"
        for item in root["credential_pool"]["openai-codex"]
    )
    assert profile_path.read_bytes() == profile_before








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
    _write_store(
        profile_path,
        {
            "version": 1,
            "providers": {
                provider: {
                    "tokens": {
                        "access_token": "stale-access",
                        "refresh_token": "stale-refresh",
                    }
                }
            },
            "credential_pool": {
                provider: [
                    {
                        "id": "codex-1",
                        "source": "device_code",
                        "access_token": "stale-access",
                        "refresh_token": "stale-refresh",
                    }
                ]
            },
        },
    )

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


def test_codex_pool_refresh_serializes_borrowed_root_chain_across_profiles(
    monkeypatch, tmp_path
):
    """Profiles borrowing one root grant must serialize on the root store."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.delenv("HERMES_HOME", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "not-the-root"))

    root_home = tmp_path / ".hermes"
    root_path = root_home / "auth.json"
    profile_paths = [
        root_home / "profiles" / "alpha" / "auth.json",
        root_home / "profiles" / "beta" / "auth.json",
    ]
    _write_store(
        root_path,
        {
            "version": 1,
            "active_provider": "openrouter",
            "providers": {
                "openai-codex": {
                    "tokens": {
                        "access_token": "root-old-access",
                        "refresh_token": "root-old-refresh",
                    },
                    "last_refresh": "2026-08-01T00:00:00Z",
                    "auth_mode": "chatgpt",
                },
                "anthropic": {"api_key": "root-unrelated-provider"},
            },
            "credential_pool": {
                "openai-codex": [
                    {
                        "id": "root-codex",
                        "label": "root singleton",
                        "source": "device_code",
                        "auth_type": "oauth",
                        "priority": 0,
                        "access_token": "root-old-access",
                        "refresh_token": "root-old-refresh",
                    },
                    {
                        "id": "independent-codex",
                        "label": "independent account",
                        "source": "manual:device_code",
                        "auth_type": "oauth",
                        "priority": 1,
                        "access_token": "independent-access",
                        "refresh_token": "independent-refresh",
                        "last_status": "exhausted",
                        "last_error_code": 429,
                    },
                ],
                "openrouter": [{"id": "root-unrelated-pool"}],
            },
            "root_marker": {"preserve": True},
        },
    )
    for index, profile_path in enumerate(profile_paths):
        _write_store(
            profile_path,
            {
                "version": 1,
                "active_provider": "anthropic",
                "providers": {
                    "anthropic": {"api_key": f"profile-{index}-provider"}
                },
                "credential_pool": {
                    "anthropic": [{"id": f"profile-{index}-pool"}]
                },
                "profile_marker": index,
            },
        )
    profile_before = {path: path.read_bytes() for path in profile_paths}

    calls = []
    calls_lock = threading.Lock()
    second_post_entered = threading.Event()

    def fake_refresh(access_token, refresh_token, **_kwargs):
        with calls_lock:
            calls.append((access_token, refresh_token))
            call_number = len(calls)
        if call_number == 1:
            # On the broken code the other profile owns a different lock and
            # enters the second POST immediately. On fixed code it waits on
            # the root lock, then adopts this rotated pair without POSTing.
            second_post_entered.wait(timeout=1)
        else:
            second_post_entered.set()
        return {
            "access_token": "root-new-access",
            "refresh_token": "root-new-refresh",
            "last_refresh": "2026-08-06T00:00:00Z",
        }

    monkeypatch.setattr(A, "refresh_codex_oauth_pure", fake_refresh)

    start = threading.Barrier(3)
    results = []
    errors = []

    def refresh_from_profile(profile_path):
        token = set_hermes_home_override(profile_path.parent)
        try:
            pool = CP.load_pool("openai-codex")
            entry = next(
                item for item in pool.entries() if item.source == "device_code"
            )
            start.wait(timeout=5)
            results.append(pool._refresh_entry(entry, force=True))
        except BaseException as exc:  # surfaced in the main test thread below
            errors.append(exc)
        finally:
            reset_hermes_home_override(token)

    threads = [
        threading.Thread(target=refresh_from_profile, args=(path,))
        for path in profile_paths
    ]
    for thread in threads:
        thread.start()
    start.wait(timeout=5)
    for thread in threads:
        thread.join(timeout=5)

    assert all(not thread.is_alive() for thread in threads)
    assert errors == []
    assert calls == [("root-old-access", "root-old-refresh")]
    assert len(results) == 2
    assert {result.access_token for result in results if result is not None} == {
        "root-new-access"
    }
    assert {result.refresh_token for result in results if result is not None} == {
        "root-new-refresh"
    }

    root = _read_store(root_path)
    assert root["active_provider"] == "openrouter"
    assert root["providers"]["openai-codex"]["tokens"] == {
        "access_token": "root-new-access",
        "refresh_token": "root-new-refresh",
    }
    root_pool = {
        entry["id"]: entry for entry in root["credential_pool"]["openai-codex"]
    }
    assert root_pool["root-codex"]["access_token"] == "root-new-access"
    assert root_pool["root-codex"]["refresh_token"] == "root-new-refresh"
    assert root_pool["independent-codex"]["access_token"] == "independent-access"
    assert root_pool["independent-codex"]["refresh_token"] == "independent-refresh"
    assert root_pool["independent-codex"]["last_status"] == "exhausted"
    assert root_pool["independent-codex"]["last_error_code"] == 429
    assert root["providers"]["anthropic"] == {
        "api_key": "root-unrelated-provider"
    }
    assert root["credential_pool"]["openrouter"] == [
        {"id": "root-unrelated-pool"}
    ]
    assert root["root_marker"] == {"preserve": True}
    for path in profile_paths:
        assert path.read_bytes() == profile_before[path]


def test_codex_pool_refresh_keeps_profile_owned_chain_local(monkeypatch, tmp_path):
    profile_home = tmp_path / "profile"
    profile_path = profile_home / "auth.json"
    monkeypatch.setenv("HERMES_HOME", str(profile_home))
    monkeypatch.setattr(A, "_global_auth_file_path", lambda: None)
    _write_store(
        profile_path,
        {
            "version": 1,
            "active_provider": "openrouter",
            "providers": {
                "openai-codex": {
                    "tokens": {
                        "access_token": "profile-old-access",
                        "refresh_token": "profile-old-refresh",
                    }
                },
                "anthropic": {"api_key": "profile-unrelated"},
            },
            "credential_pool": {
                "openai-codex": [
                    {
                        "id": "profile-codex",
                        "source": "device_code",
                        "auth_type": "oauth",
                        "priority": 0,
                        "access_token": "profile-old-access",
                        "refresh_token": "profile-old-refresh",
                    }
                ],
                "openrouter": [{"id": "profile-unrelated-pool"}],
            },
            "profile_marker": {"preserve": True},
        },
    )

    monkeypatch.setattr(
        A,
        "refresh_codex_oauth_pure",
        lambda *_args, **_kwargs: {
            "access_token": "profile-new-access",
            "refresh_token": "profile-new-refresh",
            "last_refresh": "2026-08-06T00:00:00Z",
        },
    )
    payload = A.read_credential_pool("openai-codex")[0]
    entry = PooledCredential.from_dict("openai-codex", payload)
    refreshed = CredentialPool("openai-codex", [entry])._refresh_entry(
        entry, force=True
    )

    assert refreshed is not None
    assert refreshed.access_token == "profile-new-access"
    stored = _read_store(profile_path)
    assert stored["active_provider"] == "openrouter"
    assert stored["providers"]["openai-codex"]["tokens"] == {
        "access_token": "profile-new-access",
        "refresh_token": "profile-new-refresh",
    }
    assert stored["credential_pool"]["openai-codex"][0][
        "access_token"
    ] == "profile-new-access"
    assert stored["credential_pool"]["openai-codex"][0][
        "refresh_token"
    ] == "profile-new-refresh"
    assert stored["providers"]["anthropic"] == {
        "api_key": "profile-unrelated"
    }
    assert stored["credential_pool"]["openrouter"] == [
        {"id": "profile-unrelated-pool"}
    ]
    assert stored["profile_marker"] == {"preserve": True}


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
    # Patch only the runtime auth-module bindings. credential_pool must resolve
    # source paths through that binding instead of a stale direct import.
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


@pytest.mark.parametrize("provider", ["anthropic", "xai-oauth", "nous"])
def test_source_write_failure_fails_refresh_closed_and_fresh_load_uses_source(
    tmp_path,
    monkeypatch,
    provider,
):
    auth_path = tmp_path / "auth.json"
    monkeypatch.setattr(A, "_auth_file_path", lambda: auth_path)
    monkeypatch.setattr(A, "_global_auth_file_path", lambda: None)
    monkeypatch.setattr(A, "is_provider_explicitly_configured", lambda _provider: True)
    monkeypatch.setattr(CP, "load_env", lambda: {})
    monkeypatch.setattr(CP, "_get_secret", lambda *_args: "")

    old = replace(
        _entry(
            provider,
            id=f"{provider}-source",
            access_token="old-access",
            refresh_token="old-refresh",
        ),
        expires_at_ms=1,
        source="claude_code" if provider == "anthropic" else "device_code",
    )
    store = {
        "version": 1,
        "credential_pool": {provider: [old.to_dict()]},
        "providers": {},
    }
    if provider == "xai-oauth":
        store["providers"][provider] = {
            "tokens": {
                "access_token": "old-access",
                "refresh_token": "old-refresh",
            },
            "discovery": {"token_endpoint": "https://auth.x.ai/oauth/token"},
        }
    elif provider == "nous":
        store["providers"][provider] = {
            "access_token": "old-access",
            "refresh_token": "old-refresh",
            "expires_at": "2000-01-01T00:00:00+00:00",
        }
    _write_store(auth_path, store)

    claude_credentials = {}
    if provider == "anthropic":
        claude_credentials.update({
            "accessToken": "old-access",
            "refreshToken": "old-refresh",
            "expiresAt": 1,
        })
        monkeypatch.setattr(
            anthropic_adapter,
            "read_claude_code_credentials",
            lambda: dict(claude_credentials),
        )
        monkeypatch.setattr(
            anthropic_adapter,
            "_read_claude_code_credentials_from_file",
            lambda: dict(claude_credentials),
        )
        pool = CredentialPool(provider, [old])
        monkeypatch.setattr(
            anthropic_adapter,
            "refresh_anthropic_oauth_pure",
            lambda *_args, **_kwargs: {
                "access_token": "orphan-access",
                "refresh_token": "orphan-refresh",
                "expires_at_ms": 9_999_999_999_999,
            },
        )

        def reject_claude_source_write(_observed):
            claude_credentials.clear()
            raise A.SourceCredentialPersistenceError(
                "anthropic",
                source_path=None,
                consumed_refresh_token="old-refresh",
            )

        monkeypatch.setattr(
            anthropic_adapter,
            "_refresh_claude_code_source_credentials",
            reject_claude_source_write,
        )
    elif provider == "xai-oauth":
        pool = CredentialPool(provider, [old])
        monkeypatch.setattr(
            A,
            "refresh_xai_oauth_pure",
            lambda *_args, **_kwargs: {
                "access_token": "orphan-access",
                "refresh_token": "orphan-refresh",
                "last_refresh": "2026-08-06T05:00:00Z",
            },
        )
        real_save = A._save_auth_store
        xai_saves = 0

        def reject_xai_source_write(store, target_path=None):
            nonlocal xai_saves
            if target_path is not None and A._same_path(Path(target_path), auth_path):
                xai_saves += 1
                if xai_saves > 1:
                    raise OSError("simulated source persistence failure")
            return real_save(store, target_path)

        monkeypatch.setattr(
            A,
            "_save_auth_store",
            reject_xai_source_write,
        )
    else:
        pool = CredentialPool(provider, [old])

        def fake_nous_refresh(**_kwargs):
            reserved = _read_store(auth_path)
            reserved["providers"]["nous"] = {
                A.SOURCE_REFRESH_RESERVATION_KEY: {"status": "reserved"},
            }
            for item in reserved["credential_pool"]["nous"]:
                item.pop("access_token", None)
                item.pop("refresh_token", None)
                item["last_status"] = CP.STATUS_DEAD
            _write_store(auth_path, reserved)
            raise A.SourceCredentialPersistenceError(
                "nous",
                source_path=auth_path,
                consumed_refresh_token="old-refresh",
            )

        monkeypatch.setattr(
            A,
            "resolve_nous_runtime_credentials",
            fake_nous_refresh,
        )

    refreshed = pool._refresh_entry(old, force=True)
    reloaded = CP.load_pool(provider)
    source = "claude_code" if provider == "anthropic" else "device_code"

    assert refreshed is None
    assert reloaded.has_available() is False
    assert all(entry.access_token != "orphan-access" for entry in reloaded.entries())
    if provider in {"anthropic", "nous"}:
        assert all(entry.source != source for entry in reloaded.entries())
    else:
        source_entries = [
            entry for entry in reloaded.entries() if entry.source == source
        ]
        assert all(entry.refresh_token != "old-refresh" for entry in source_entries)
        assert all(entry.last_status == CP.STATUS_DEAD for entry in source_entries)

    if provider == "anthropic":
        claude_credentials.update(
            accessToken="owner-access",
            refreshToken="owner-refresh",
            expiresAt=9_999_999_999_999,
        )
        adopted = next(
            entry
            for entry in CP.load_pool(provider).entries()
            if entry.source == "claude_code"
        )
        assert adopted.access_token == "owner-access"
        assert adopted.refresh_token == "owner-refresh"
        assert adopted.last_status is None


def test_two_profile_xai_refreshes_lock_the_shared_global_source(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.delenv("HERMES_HOME", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "not-the-root"))
    root_home = tmp_path / ".hermes"
    root_path = root_home / "auth.json"
    profile_paths = [
        root_home / "profiles" / "alpha" / "auth.json",
        root_home / "profiles" / "beta" / "auth.json",
    ]
    root_entry = replace(
        _entry(
            "xai-oauth",
            id="root-xai",
            access_token="root-old-access",
            refresh_token="root-old-refresh",
        ),
        expires_at_ms=1,
    )
    _write_store(
        root_path,
        {
            "version": 1,
            "providers": {
                "xai-oauth": {
                    "tokens": {
                        "access_token": "root-old-access",
                        "refresh_token": "root-old-refresh",
                    },
                    "discovery": {
                        "token_endpoint": "https://auth.x.ai/oauth/token"
                    },
                }
            },
            "credential_pool": {"xai-oauth": [root_entry.to_dict()]},
        },
    )
    for marker, profile_path in zip(("alpha", "beta"), profile_paths):
        _write_store(
            profile_path,
            {
                "version": 1,
                "providers": {"anthropic": {"api_key": f"{marker}-key"}},
                "credential_pool": {},
            },
        )

    calls = []
    calls_lock = threading.Lock()
    second_post_entered = threading.Event()

    def fake_refresh(access_token, refresh_token, **_kwargs):
        with calls_lock:
            calls.append((access_token, refresh_token))
            call_number = len(calls)
        if call_number == 1:
            second_post_entered.wait(timeout=1)
        else:
            second_post_entered.set()
        return {
            "access_token": "root-new-access",
            "refresh_token": "root-new-refresh",
            "last_refresh": "2026-08-06T06:00:00Z",
        }

    monkeypatch.setattr(A, "refresh_xai_oauth_pure", fake_refresh)
    monkeypatch.setattr(
        CredentialPool,
        "_entry_needs_refresh",
        lambda self, entry: (
            self.provider == "xai-oauth"
            and entry.access_token == "root-old-access"
        ),
    )
    start = threading.Barrier(3)
    results = []
    errors = []

    def select_from_profile(profile_path):
        token = set_hermes_home_override(profile_path.parent)
        try:
            pool = CP.load_pool("xai-oauth")
            start.wait(timeout=5)
            results.append(pool.select())
        except BaseException as exc:
            errors.append(exc)
        finally:
            reset_hermes_home_override(token)

    selectors = [
        threading.Thread(target=select_from_profile, args=(path,))
        for path in profile_paths
    ]
    for selector in selectors:
        selector.start()
    start.wait(timeout=5)
    for selector in selectors:
        selector.join(timeout=5)

    assert all(not selector.is_alive() for selector in selectors)
    assert errors == []
    assert calls == [("root-old-access", "root-old-refresh")]
    assert len(results) == 2 and all(result is not None for result in results)
    assert {result.access_token for result in results} == {"root-new-access"}
    assert {result.refresh_token for result in results} == {"root-new-refresh"}
    root = _read_store(root_path)
    assert root["providers"]["xai-oauth"]["tokens"] == {
        "access_token": "root-new-access",
        "refresh_token": "root-new-refresh",
    }


def test_refresh_persistence_does_not_invert_auth_store_lock(
    monkeypatch,
    tmp_path,
):
    auth_path = tmp_path / "auth.json"
    monkeypatch.setattr(A, "_auth_file_path", lambda: auth_path)
    monkeypatch.setattr(A, "_global_auth_file_path", lambda: None)
    entry = replace(
        _entry(
            "xai-oauth",
            id="xai-source",
            access_token="old-access",
            refresh_token="old-refresh",
        ),
        expires_at_ms=1,
    )
    _write_store(
        auth_path,
        {
            "version": 1,
            "providers": {
                "xai-oauth": {
                    "tokens": {
                        "access_token": "old-access",
                        "refresh_token": "old-refresh",
                    },
                    "discovery": {
                        "token_endpoint": "https://auth.x.ai/oauth/token"
                    },
                }
            },
            "credential_pool": {"xai-oauth": [entry.to_dict()]},
        },
    )
    pool = CredentialPool("xai-oauth", [entry])
    with pool._lock:
        pool._record_entry_mutation_unlocked(entry.id, dirty=True)

    post_entered = threading.Event()
    persistence_has_lock = threading.Event()
    allow_auth_attempt = threading.Event()
    errors = []
    results = []
    real_write = CP.write_credential_pool

    def fake_refresh(*_args, **_kwargs):
        post_entered.set()
        assert persistence_has_lock.wait(timeout=5)
        allow_auth_attempt.set()
        return {
            "access_token": "new-access",
            "refresh_token": "new-refresh",
            "last_refresh": "2026-08-06T07:00:00Z",
        }

    def barrier_write(provider, entries, *, removed_ids=None):
        if threading.current_thread().name == "persistence-owner":
            persistence_has_lock.set()
            assert allow_auth_attempt.wait(timeout=5)
            with A._auth_store_lock(target_path=auth_path, timeout_seconds=1):
                return real_write(provider, entries, removed_ids=removed_ids)
        return real_write(provider, entries, removed_ids=removed_ids)

    monkeypatch.setattr(A, "refresh_xai_oauth_pure", fake_refresh)
    monkeypatch.setattr(CP, "write_credential_pool", barrier_write)

    def refresh_worker():
        try:
            results.append(pool._refresh_entry(entry, force=True))
        except BaseException as exc:
            errors.append(("refresh", exc))

    def persistence_worker():
        try:
            pool._persist_pending_changes()
        except BaseException as exc:
            errors.append(("persistence", exc))

    refresher = threading.Thread(target=refresh_worker, name="refresh-owner")
    persister = threading.Thread(target=persistence_worker, name="persistence-owner")
    refresher.start()
    assert post_entered.wait(timeout=5)
    persister.start()
    refresher.join(timeout=5)
    persister.join(timeout=5)

    assert not refresher.is_alive()
    assert not persister.is_alive()
    assert errors == []
    assert len(results) == 1 and results[0] is not None
    assert results[0].access_token == "new-access"
    stored = _read_store(auth_path)
    assert stored["providers"]["xai-oauth"]["tokens"]["refresh_token"] == (
        "new-refresh"
    )
    assert stored["credential_pool"]["xai-oauth"][0]["refresh_token"] == (
        "new-refresh"
    )

