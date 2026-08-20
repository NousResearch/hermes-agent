"""Production-path regressions for source-owned rotating OAuth credentials."""

from __future__ import annotations

import base64
import json
import threading
import time
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import pytest

from agent import anthropic_adapter
from agent import credential_pool as CP
from agent.credential_persistence import sanitize_borrowed_credential_payload
from agent.credential_pool import AUTH_TYPE_OAUTH, CredentialPool, PooledCredential
from hermes_cli import auth as A
from hermes_constants import reset_hermes_home_override, set_hermes_home_override


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _jwt(*, seconds: int, scope: str | None = None) -> str:
    claims: dict[str, Any] = {"sub": "test-user", "exp": int(time.time() + seconds)}
    if scope is not None:
        claims["scope"] = scope

    def encode(value: dict[str, Any]) -> str:
        raw = json.dumps(value, separators=(",", ":")).encode("utf-8")
        return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")

    return f"{encode({'alg': 'none', 'typ': 'JWT'})}.{encode(claims)}.signature"


def _pool_entry(
    provider: str,
    *,
    entry_id: str,
    access_token: str,
    refresh_token: str,
    source: str = "device_code",
    expires_at_ms: int | None = 1,
    last_status: str | None = None,
) -> PooledCredential:
    return PooledCredential(
        provider=provider,
        id=entry_id,
        label=entry_id,
        auth_type=AUTH_TYPE_OAUTH,
        priority=0,
        source=source,
        access_token=access_token,
        refresh_token=refresh_token,
        expires_at_ms=expires_at_ms,
        last_status=last_status,
        last_status_at=time.time() if last_status else None,
    )


def _xai_store(
    access_token: str,
    refresh_token: str,
    *,
    entry_id: str = "xai-source",
) -> dict[str, Any]:
    entry = _pool_entry(
        "xai-oauth",
        entry_id=entry_id,
        access_token=access_token,
        refresh_token=refresh_token,
    )
    return {
        "version": 1,
        "providers": {
            "xai-oauth": {
                "tokens": {
                    "access_token": access_token,
                    "refresh_token": refresh_token,
                },
                "discovery": {"token_endpoint": "https://auth.x.ai/oauth/token"},
                "auth_mode": "oauth_device_code",
            }
        },
        "credential_pool": {"xai-oauth": [entry.to_dict()]},
    }


def _install_xai_refresh(
    monkeypatch: pytest.MonkeyPatch,
    *,
    new_access: str,
    new_refresh: str,
) -> tuple[
    list[tuple[str, str]],
    threading.Event,
    threading.Event,
    threading.Event,
]:
    calls: list[tuple[str, str]] = []
    calls_lock = threading.Lock()
    first_post = threading.Event()
    second_post = threading.Event()
    release_post = threading.Event()

    def fake_refresh(access_token: str, refresh_token: str, **_kwargs: Any) -> dict[str, Any]:
        with calls_lock:
            calls.append((access_token, refresh_token))
            call_number = len(calls)
        if call_number == 1:
            first_post.set()
        else:
            second_post.set()
        assert release_post.wait(timeout=5)
        return {
            "access_token": new_access,
            "refresh_token": new_refresh,
            "last_refresh": "2026-08-06T12:00:00Z",
        }

    monkeypatch.setattr(A, "refresh_xai_oauth_pure", fake_refresh)
    return calls, first_post, second_post, release_post


def _run_threads(workers: list[threading.Thread]) -> None:
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=5)
    assert all(not worker.is_alive() for worker in workers)


def test_two_profile_xai_runtime_resolvers_share_the_root_transaction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.delenv("HERMES_HOME", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "isolated-home"))
    root_home = tmp_path / ".hermes"
    root_path = root_home / "auth.json"
    profile_homes = [
        root_home / "profiles" / "one",
        root_home / "profiles" / "two",
    ]
    old_access = _jwt(seconds=-60)
    new_access = _jwt(seconds=86_400)
    _write_json(root_path, _xai_store(old_access, "root-refresh-old"))
    for profile_home in profile_homes:
        _write_json(profile_home / "auth.json", {"version": 1, "providers": {}})

    calls, first_post, second_post, release_post = _install_xai_refresh(
        monkeypatch,
        new_access=new_access,
        new_refresh="root-refresh-new",
    )
    start = threading.Barrier(3)
    results: list[dict[str, Any]] = []
    errors: list[BaseException] = []

    def resolve(profile_home: Path) -> None:
        token = set_hermes_home_override(profile_home)
        try:
            start.wait(timeout=5)
            results.append(
                A.resolve_xai_oauth_runtime_credentials(refresh_skew_seconds=0)
            )
        except BaseException as exc:
            errors.append(exc)
        finally:
            reset_hermes_home_override(token)

    workers = [threading.Thread(target=resolve, args=(home,)) for home in profile_homes]
    for worker in workers:
        worker.start()
    start.wait(timeout=5)
    assert first_post.wait(timeout=5)
    second_post.wait(timeout=1)
    release_post.set()
    for worker in workers:
        worker.join(timeout=5)

    assert all(not worker.is_alive() for worker in workers)
    assert errors == []
    assert calls == [(old_access, "root-refresh-old")]
    assert len(results) == 2
    assert {result["api_key"] for result in results} == {new_access}
    root_tokens = _read_json(root_path)["providers"]["xai-oauth"]["tokens"]
    assert root_tokens["access_token"] == new_access
    assert root_tokens["refresh_token"] == "root-refresh-new"


def test_xai_pool_and_runtime_resolver_share_the_root_transaction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.delenv("HERMES_HOME", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "isolated-home"))
    monkeypatch.setattr(A, "is_provider_explicitly_configured", lambda _provider: True)
    monkeypatch.setattr(CP, "load_env", lambda: {})
    monkeypatch.setattr(CP, "_get_secret", lambda *_args: "")
    root_home = tmp_path / ".hermes"
    root_path = root_home / "auth.json"
    pool_home = root_home / "profiles" / "pool"
    runtime_home = root_home / "profiles" / "runtime"
    old_access = _jwt(seconds=-60)
    new_access = _jwt(seconds=86_400)
    _write_json(root_path, _xai_store(old_access, "root-refresh-old"))
    for profile_home in (pool_home, runtime_home):
        _write_json(profile_home / "auth.json", {"version": 1, "providers": {}})

    calls, first_post, second_post, release_post = _install_xai_refresh(
        monkeypatch,
        new_access=new_access,
        new_refresh="root-refresh-new",
    )
    start = threading.Barrier(3)
    pool_results: list[PooledCredential | None] = []
    runtime_results: list[dict[str, Any]] = []
    errors: list[BaseException] = []

    def select_pool() -> None:
        token = set_hermes_home_override(pool_home)
        try:
            pool = CP.load_pool("xai-oauth")
            start.wait(timeout=5)
            pool_results.append(pool.select())
        except BaseException as exc:
            errors.append(exc)
        finally:
            reset_hermes_home_override(token)

    def resolve_runtime() -> None:
        token = set_hermes_home_override(runtime_home)
        try:
            start.wait(timeout=5)
            runtime_results.append(
                A.resolve_xai_oauth_runtime_credentials(refresh_skew_seconds=0)
            )
        except BaseException as exc:
            errors.append(exc)
        finally:
            reset_hermes_home_override(token)

    workers = [threading.Thread(target=select_pool), threading.Thread(target=resolve_runtime)]
    for worker in workers:
        worker.start()
    start.wait(timeout=5)
    assert first_post.wait(timeout=5)
    second_post.wait(timeout=1)
    release_post.set()
    for worker in workers:
        worker.join(timeout=5)

    assert all(not worker.is_alive() for worker in workers)
    assert errors == []
    assert calls == [(old_access, "root-refresh-old")]
    assert len(pool_results) == 1 and pool_results[0] is not None
    assert pool_results[0].access_token == new_access
    assert len(runtime_results) == 1 and runtime_results[0]["api_key"] == new_access
    root = _read_json(root_path)
    assert root["providers"]["xai-oauth"]["tokens"]["refresh_token"] == "root-refresh-new"
    persisted = root["credential_pool"]["xai-oauth"][0]
    assert persisted["access_token"] == new_access
    assert persisted["refresh_token"] == "root-refresh-new"


def test_xai_pool_restarts_when_local_owner_disappears_before_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.delenv("HERMES_HOME", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "isolated-home"))
    monkeypatch.setattr(A, "is_provider_explicitly_configured", lambda _provider: True)
    monkeypatch.setattr(CP, "load_env", lambda: {})
    monkeypatch.setattr(CP, "_get_secret", lambda *_args: "")
    root_home = tmp_path / ".hermes"
    root_path = root_home / "auth.json"
    profile_home = root_home / "profiles" / "local"
    root_peer_home = root_home / "profiles" / "root-peer"
    profile_path = profile_home / "auth.json"
    old_access = _jwt(seconds=-60)
    new_access = _jwt(seconds=86_400)
    _write_json(root_path, _xai_store(old_access, "shared-refresh-old", entry_id="root-xai"))
    _write_json(profile_path, _xai_store(old_access, "shared-refresh-old", entry_id="local-xai"))
    _write_json(root_peer_home / "auth.json", {"version": 1, "providers": {}})

    removed = threading.Event()
    transaction_paths: list[tuple[str, Path | None]] = []
    real_transaction = A._provider_state_transaction

    @contextmanager
    def scheduled_transaction(provider_id: str, *args: Any, **kwargs: Any):
        expected = kwargs.get("expected_source_path")
        if (
            provider_id == "xai-oauth"
            and threading.current_thread().name == "local-selector"
            and expected is not None
            and A._same_path(Path(expected), profile_path)
            and not removed.is_set()
        ):
            local = _read_json(profile_path)
            local.get("providers", {}).pop("xai-oauth", None)
            _write_json(profile_path, local)
            removed.set()
        with real_transaction(provider_id, *args, **kwargs) as transaction:
            transaction_paths.append(
                (threading.current_thread().name, transaction[2])
            )
            yield transaction

    monkeypatch.setattr(A, "_provider_state_transaction", scheduled_transaction)
    calls, first_post, _second_post, release_post = _install_xai_refresh(
        monkeypatch,
        new_access=new_access,
        new_refresh="shared-refresh-new",
    )
    results: list[PooledCredential | None] = []
    errors: list[BaseException] = []
    root_empty_selection = threading.Event()
    release_empty_selection = threading.Event()
    real_select_under_lock = CP.CredentialPool._select_under_lock

    def scheduled_select_under_lock(
        pool: CP.CredentialPool,
        excluded_source_ids: set[str] | None = None,
    ) -> tuple[PooledCredential | None, list[tuple[Any, Any]]]:
        selected = real_select_under_lock(pool, excluded_source_ids)
        if (
            threading.current_thread().name == "root-selector"
            and selected == (None, [])
            and not root_empty_selection.is_set()
        ):
            root_empty_selection.set()
            assert release_empty_selection.wait(timeout=5)
        return selected

    monkeypatch.setattr(
        CP.CredentialPool,
        "_select_under_lock",
        scheduled_select_under_lock,
    )

    def select(profile: Path, *, wait_for_removal: bool) -> None:
        token = set_hermes_home_override(profile)
        try:
            if wait_for_removal:
                assert removed.wait(timeout=5)
                assert first_post.wait(timeout=5)
            pool = CP.load_pool("xai-oauth")
            results.append(pool.select())
        except BaseException as exc:
            errors.append(exc)
        finally:
            reset_hermes_home_override(token)

    workers = [
        threading.Thread(
            target=select,
            kwargs={"profile": profile_home, "wait_for_removal": False},
            name="local-selector",
        ),
        threading.Thread(
            target=select,
            kwargs={"profile": root_peer_home, "wait_for_removal": True},
            name="root-selector",
        ),
    ]
    for worker in workers:
        worker.start()
    assert removed.wait(timeout=5)
    assert first_post.wait(timeout=5)
    assert root_empty_selection.wait(timeout=5)
    release_post.set()
    workers[0].join(timeout=5)
    assert not workers[0].is_alive()
    release_empty_selection.set()
    workers[1].join(timeout=5)

    assert all(not worker.is_alive() for worker in workers)
    assert errors == []
    assert calls == [(old_access, "shared-refresh-old")]
    local_paths = [path for name, path in transaction_paths if name == "local-selector"]
    assert any(path is not None and A._same_path(Path(path), profile_path) for path in local_paths)
    assert any(path is not None and A._same_path(Path(path), root_path) for path in local_paths)
    assert len(results) == 2 and all(result is not None for result in results)
    assert {result.access_token for result in results if result is not None} == {new_access}


def test_xai_root_write_failure_never_returns_or_reloads_consumed_lineage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.delenv("HERMES_HOME", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "isolated-home"))
    monkeypatch.setattr(A, "is_provider_explicitly_configured", lambda _provider: True)
    monkeypatch.setattr(CP, "load_env", lambda: {})
    monkeypatch.setattr(CP, "_get_secret", lambda *_args: "")
    root_home = tmp_path / ".hermes"
    root_path = root_home / "auth.json"
    profile_home = root_home / "profiles" / "runtime"
    profile_path = profile_home / "auth.json"
    old_access = _jwt(seconds=-60)
    orphan_access = _jwt(seconds=3600)
    _write_json(root_path, _xai_store(old_access, "root-refresh-old"))
    _write_json(profile_path, {"version": 1, "providers": {}})
    monkeypatch.setattr(
        A,
        "refresh_xai_oauth_pure",
        lambda *_args, **_kwargs: {
            "access_token": orphan_access,
            "refresh_token": "orphan-refresh",
            "last_refresh": "2026-08-06T12:00:00Z",
        },
    )
    real_save = A._save_auth_store
    failed = False
    root_saves = 0

    def fail_rotated_root_write(store: dict[str, Any], target_path: Path | None = None):
        nonlocal failed, root_saves
        if target_path is not None and A._same_path(Path(target_path), root_path):
            root_saves += 1
            if root_saves > 1:
                failed = True
                raise OSError("simulated source persistence failure")
        return real_save(store, target_path)

    monkeypatch.setattr(A, "_save_auth_store", fail_rotated_root_write)
    token = set_hermes_home_override(profile_home)
    try:
        with pytest.raises(Exception):
            A.resolve_xai_oauth_runtime_credentials(force_refresh=True)
    finally:
        reset_hermes_home_override(token)

    assert failed is True
    reload_home = root_home / "profiles" / "reload"
    _write_json(reload_home / "auth.json", {"version": 1, "providers": {}})
    token = set_hermes_home_override(reload_home)
    try:
        reloaded = CP.load_pool("xai-oauth")
    finally:
        reset_hermes_home_override(token)
    assert reloaded.has_available() is False
    assert all(entry.access_token != orphan_access for entry in reloaded.entries())


def _write_claude_credentials(
    path: Path,
    *,
    access_token: str,
    refresh_token: str,
    expires_at_ms: int,
) -> None:
    _write_json(
        path,
        {
            "claudeAiOauth": {
                "accessToken": access_token,
                "refreshToken": refresh_token,
                "expiresAt": expires_at_ms,
            }
        },
    )


def _claude_tokens(path: Path) -> tuple[str, str]:
    oauth = _read_json(path)["claudeAiOauth"]
    return oauth["accessToken"], oauth["refreshToken"]


@pytest.mark.parametrize("peer", ["direct", "pool"])
def test_anthropic_direct_refresh_shares_credentials_transaction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    peer: str,
) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    auth_path = tmp_path / "hermes" / "auth.json"
    credentials_path = tmp_path / ".claude" / ".credentials.json"
    monkeypatch.setattr(A, "_auth_file_path", lambda: auth_path)
    monkeypatch.setattr(A, "_global_auth_file_path", lambda: None)
    monkeypatch.setattr(A, "is_provider_explicitly_configured", lambda _provider: True)
    monkeypatch.setattr(CP, "load_env", lambda: {})
    monkeypatch.setattr(CP, "_get_secret", lambda *_args: "")
    old = {
        "accessToken": "claude-access-old",
        "refreshToken": "claude-refresh-old",
        "expiresAt": 1,
    }
    _write_claude_credentials(
        credentials_path,
        access_token=old["accessToken"],
        refresh_token=old["refreshToken"],
        expires_at_ms=old["expiresAt"],
    )
    pool_entry = _pool_entry(
        "anthropic",
        entry_id="claude-source",
        access_token=old["accessToken"],
        refresh_token=old["refreshToken"],
        source="claude_code",
    )
    _write_json(
        auth_path,
        {
            "version": 1,
            "credential_pool": {"anthropic": [pool_entry.to_dict()]},
        },
    )
    pool = CredentialPool("anthropic", [pool_entry])
    calls: list[str] = []
    calls_lock = threading.Lock()
    first_post = threading.Event()
    second_post = threading.Event()
    release_post = threading.Event()

    def fake_refresh(refresh_token: str, *, use_json: bool) -> dict[str, Any]:
        assert use_json is False
        with calls_lock:
            calls.append(refresh_token)
            call_number = len(calls)
        if call_number == 1:
            first_post.set()
        else:
            second_post.set()
        assert release_post.wait(timeout=5)
        return {
            "access_token": "claude-access-new",
            "refresh_token": "claude-refresh-new",
            "expires_at_ms": 9_999_999_999_999,
        }

    monkeypatch.setattr(anthropic_adapter, "refresh_anthropic_oauth_pure", fake_refresh)
    start = threading.Barrier(3)
    results: list[Any] = []
    errors: list[BaseException] = []

    def direct_worker() -> None:
        try:
            start.wait(timeout=5)
            results.append(anthropic_adapter._refresh_oauth_token(dict(old)))
        except BaseException as exc:
            errors.append(exc)

    def peer_worker() -> None:
        try:
            start.wait(timeout=5)
            if peer == "direct":
                results.append(anthropic_adapter._refresh_oauth_token(dict(old)))
            else:
                results.append(pool.select())
        except BaseException as exc:
            errors.append(exc)

    workers = [threading.Thread(target=direct_worker), threading.Thread(target=peer_worker)]
    for worker in workers:
        worker.start()
    start.wait(timeout=5)
    assert first_post.wait(timeout=5)
    second_post.wait(timeout=1)
    release_post.set()
    for worker in workers:
        worker.join(timeout=5)

    assert all(not worker.is_alive() for worker in workers)
    assert errors == []
    assert calls == ["claude-refresh-old"]
    assert _claude_tokens(credentials_path) == (
        "claude-access-new",
        "claude-refresh-new",
    )
    assert len(results) == 2
    assert all(
        result == "claude-access-new"
        or (
            isinstance(result, PooledCredential)
            and result.access_token == "claude-access-new"
            and result.refresh_token == "claude-refresh-new"
        )
        for result in results
    )


def test_anthropic_replacement_after_post_survives_conditional_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    credentials_path = tmp_path / ".claude" / ".credentials.json"
    old = {
        "accessToken": "claude-access-old",
        "refreshToken": "claude-refresh-old",
        "expiresAt": 1,
    }
    _write_claude_credentials(
        credentials_path,
        access_token=old["accessToken"],
        refresh_token=old["refreshToken"],
        expires_at_ms=old["expiresAt"],
    )
    write_boundary = threading.Event()
    release_write = threading.Event()

    def fake_refresh(_refresh_token: str, *, use_json: bool) -> dict[str, Any]:
        assert use_json is False
        return {
            "access_token": "obsolete-access",
            "refresh_token": "obsolete-refresh",
            "expires_at_ms": 9_999_999_999_999,
        }

    monkeypatch.setattr(anthropic_adapter, "refresh_anthropic_oauth_pure", fake_refresh)
    def pause_at_write_boundary() -> None:
        write_boundary.set()
        assert release_write.wait(timeout=5)

    monkeypatch.setattr(
        anthropic_adapter,
        "_claude_reservation_commit_hook",
        pause_at_write_boundary,
    )
    results: list[str | None] = []
    worker = threading.Thread(
        target=lambda: results.append(anthropic_adapter._refresh_oauth_token(dict(old)))
    )
    worker.start()
    assert write_boundary.wait(timeout=5)
    _write_claude_credentials(
        credentials_path,
        access_token="winner-access",
        refresh_token="winner-refresh",
        expires_at_ms=9_999_999_999_999,
    )
    release_write.set()
    worker.join(timeout=5)

    assert not worker.is_alive()
    assert _claude_tokens(credentials_path) == ("winner-access", "winner-refresh")
    assert results[0] != "obsolete-access"


def _nous_entry(
    *,
    access_token: str,
    refresh_token: str,
    status: str | None = None,
) -> PooledCredential:
    expires_at = datetime.fromtimestamp(time.time() + 3600, tz=timezone.utc).isoformat()
    return replace(
        _pool_entry(
            "nous",
            entry_id="nous-source",
            access_token=access_token,
            refresh_token=refresh_token,
            source="device_code",
            expires_at_ms=None,
            last_status=status,
        ),
        expires_at=expires_at,
        agent_key=access_token,
        agent_key_expires_at=expires_at,
        inference_base_url=A.DEFAULT_NOUS_INFERENCE_URL,
        extra={
            "scope": A.DEFAULT_NOUS_SCOPE,
            "client_id": A.DEFAULT_NOUS_CLIENT_ID,
            "portal_base_url": A.DEFAULT_NOUS_PORTAL_URL,
        },
    )


def test_nous_resolver_persistence_failure_reserves_consumed_lineage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    auth_path = tmp_path / "hermes" / "auth.json"
    monkeypatch.setattr(A, "_auth_file_path", lambda: auth_path)
    monkeypatch.setattr(A, "_global_auth_file_path", lambda: None)
    monkeypatch.setattr(A, "is_provider_explicitly_configured", lambda _provider: True)
    monkeypatch.setattr(CP, "load_env", lambda: {})
    monkeypatch.setattr(CP, "_get_secret", lambda *_args: "")
    monkeypatch.setattr(A, "_read_shared_nous_state", lambda: None)
    monkeypatch.setattr(A, "_reserve_shared_nous_lineage", lambda _token: None)
    monkeypatch.setattr(A, "_write_shared_nous_state", lambda _state: None)
    old_access = _jwt(seconds=3600, scope=A.DEFAULT_NOUS_SCOPE)
    new_access = _jwt(seconds=7200, scope=A.DEFAULT_NOUS_SCOPE)
    old = _nous_entry(access_token=old_access, refresh_token="nous-refresh-old")
    _write_json(
        auth_path,
        {
            "version": 1,
            "providers": {
                "nous": {
                    "access_token": old_access,
                    "refresh_token": "nous-refresh-old",
                    "expires_at": old.expires_at,
                    "scope": A.DEFAULT_NOUS_SCOPE,
                    "portal_base_url": A.DEFAULT_NOUS_PORTAL_URL,
                    "inference_base_url": A.DEFAULT_NOUS_INFERENCE_URL,
                    "client_id": A.DEFAULT_NOUS_CLIENT_ID,
                    "agent_key": old_access,
                    "agent_key_expires_at": old.agent_key_expires_at,
                }
            },
            "credential_pool": {"nous": [old.to_dict()]},
        },
    )
    requests: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request.headers["x-nous-refresh-token"])
        return httpx.Response(
            200,
            request=request,
            json={
                "access_token": new_access,
                "refresh_token": "nous-refresh-new",
                "expires_in": 7200,
                "token_type": "Bearer",
                "scope": A.DEFAULT_NOUS_SCOPE,
                "inference_base_url": A.DEFAULT_NOUS_INFERENCE_URL,
            },
        )

    real_client = httpx.Client
    transport = httpx.MockTransport(handler)

    def client_factory(*args: Any, **kwargs: Any) -> httpx.Client:
        return real_client(*args, transport=transport, **kwargs)

    monkeypatch.setattr(A.httpx, "Client", client_factory)
    real_save = A._save_auth_store
    failed = False
    auth_saves = 0

    def fail_post_refresh_save(
        store: dict[str, Any],
        target_path: Path | None = None,
    ) -> None:
        nonlocal failed, auth_saves
        if target_path is not None and A._same_path(Path(target_path), auth_path):
            auth_saves += 1
            if auth_saves > 1:
                failed = True
                raise OSError("simulated provider-state save failure")
        real_save(store, target_path)

    monkeypatch.setattr(A, "_save_auth_store", fail_post_refresh_save)
    pool = CredentialPool("nous", [old])
    refreshed = pool._refresh_entry(old, force=True)

    assert failed is True
    assert requests == ["nous-refresh-old"]
    assert refreshed is None
    fresh = CP.load_pool("nous")
    assert fresh.has_available() is False
    assert all(
        entry.refresh_token != "nous-refresh-old"
        for entry in fresh.entries()
    )

    newer = _read_json(auth_path)
    newer_state = newer["providers"]["nous"]
    newer_state.pop(A.SOURCE_REFRESH_RESERVATION_KEY, None)
    newer_state["access_token"] = new_access
    newer_state["refresh_token"] = "nous-refresh-newer"
    newer_state["agent_key"] = new_access
    newer_state["agent_key_expires_at"] = datetime.fromtimestamp(
        time.time() + 7200,
        tz=timezone.utc,
    ).isoformat()
    _write_json(auth_path, newer)
    recovered = CP.load_pool("nous")
    recovered_entry = next(
        entry for entry in recovered.entries() if entry.source == "device_code"
    )
    assert recovered_entry.refresh_token == "nous-refresh-newer"
    assert recovered_entry.last_status is None
    assert recovered.has_available() is True


@pytest.mark.parametrize(
    ("source_access", "source_refresh", "expected_status"),
    [
        ("claude-access-old", "claude-refresh-new", None),
        ("claude-access-new", "claude-refresh-old", CP.STATUS_DEAD),
    ],
)
def test_claude_dead_state_tracks_refresh_lineage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source_access: str,
    source_refresh: str,
    expected_status: str | None,
) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    auth_path = tmp_path / "hermes" / "auth.json"
    credentials_path = tmp_path / ".claude" / ".credentials.json"
    monkeypatch.setattr(A, "_auth_file_path", lambda: auth_path)
    monkeypatch.setattr(A, "_global_auth_file_path", lambda: None)
    monkeypatch.setattr(A, "is_provider_explicitly_configured", lambda _provider: True)
    monkeypatch.setattr(CP, "load_env", lambda: {})
    monkeypatch.setattr(CP, "_get_secret", lambda *_args: "")
    consumed = _pool_entry(
        "anthropic",
        entry_id="claude-source",
        access_token="claude-access-old",
        refresh_token="claude-refresh-old",
        source="claude_code",
        last_status=CP.STATUS_DEAD,
    )
    persisted = sanitize_borrowed_credential_payload(consumed.to_dict(), "anthropic")
    _write_json(
        auth_path,
        {"version": 1, "credential_pool": {"anthropic": [persisted]}},
    )
    _write_claude_credentials(
        credentials_path,
        access_token=source_access,
        refresh_token=source_refresh,
        expires_at_ms=9_999_999_999_999,
    )

    loaded = CP.load_pool("anthropic")
    claude = next(entry for entry in loaded.entries() if entry.source == "claude_code")

    assert claude.access_token == source_access
    assert claude.refresh_token == source_refresh
    assert claude.last_status == expected_status
    assert loaded.has_available() is (expected_status is None)


def test_claude_normal_rotation_reloads_available(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    auth_path = tmp_path / "hermes" / "auth.json"
    credentials_path = tmp_path / ".claude" / ".credentials.json"
    monkeypatch.setattr(A, "_auth_file_path", lambda: auth_path)
    monkeypatch.setattr(A, "_global_auth_file_path", lambda: None)
    monkeypatch.setattr(A, "is_provider_explicitly_configured", lambda _provider: True)
    monkeypatch.setattr(CP, "load_env", lambda: {})
    monkeypatch.setattr(CP, "_get_secret", lambda *_args: "")
    previous = _pool_entry(
        "anthropic",
        entry_id="claude-source",
        access_token="claude-access-old",
        refresh_token="claude-refresh-old",
        source="claude_code",
    )
    persisted = sanitize_borrowed_credential_payload(previous.to_dict(), "anthropic")
    _write_json(
        auth_path,
        {"version": 1, "credential_pool": {"anthropic": [persisted]}},
    )
    _write_claude_credentials(
        credentials_path,
        access_token="claude-access-new",
        refresh_token="claude-refresh-new",
        expires_at_ms=9_999_999_999_999,
    )

    loaded = CP.load_pool("anthropic")
    claude = next(entry for entry in loaded.entries() if entry.source == "claude_code")

    assert claude.access_token == "claude-access-new"
    assert claude.refresh_token == "claude-refresh-new"
    assert claude.last_status is None
    assert loaded.has_available() is True
