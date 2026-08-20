"""Schedules for durable reservation of rotating source credentials."""

from __future__ import annotations

import base64
import json
import os
import threading
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import httpx
import pytest

from agent import anthropic_adapter
from agent import credential_pool as CP
from agent.credential_pool import AUTH_TYPE_OAUTH, CredentialPool, PooledCredential
from hermes_cli import auth as A
from hermes_constants import reset_hermes_home_override, set_hermes_home_override


_RESERVATION_KEY = "_oauth_refresh_reservation"


def _assert_non_secret_reservation(
    metadata: dict[str, Any],
    *secrets: str,
) -> None:
    encoded = json.dumps(metadata, sort_keys=True)
    assert metadata["status"] == "reserved"
    assert metadata["nonce"]
    assert metadata["owner_fingerprint"].startswith("sha256:")
    assert metadata["refresh_fingerprint"].startswith("sha256:")
    assert all(secret not in encoded for secret in secrets)


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


def _nous_entry(access_token: str, refresh_token: str) -> PooledCredential:
    expires_at = datetime.fromtimestamp(
        time.time() + 3600,
        tz=timezone.utc,
    ).isoformat()
    return replace(
        _pool_entry(
            "nous",
            entry_id="nous-source",
            access_token=access_token,
            refresh_token=refresh_token,
            expires_at_ms=None,
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


def _nous_store(access_token: str, refresh_token: str) -> dict[str, Any]:
    entry = _nous_entry(access_token, refresh_token)
    return {
        "version": 1,
        "providers": {
            "nous": {
                "access_token": access_token,
                "refresh_token": refresh_token,
                "expires_at": entry.expires_at,
                "scope": A.DEFAULT_NOUS_SCOPE,
                "portal_base_url": A.DEFAULT_NOUS_PORTAL_URL,
                "inference_base_url": A.DEFAULT_NOUS_INFERENCE_URL,
                "client_id": A.DEFAULT_NOUS_CLIENT_ID,
                "agent_key": access_token,
                "agent_key_expires_at": entry.agent_key_expires_at,
            }
        },
        "credential_pool": {"nous": [entry.to_dict()]},
    }


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
                "scopes": ["user:inference"],
            },
            "unrelated": {"preserved": True},
        },
    )


def _configure_profiles(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path]:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.delenv("HERMES_HOME", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "isolated-home"))
    shared = tmp_path / "shared"
    monkeypatch.setenv("HERMES_SHARED_AUTH_DIR", str(shared))
    return tmp_path / ".hermes", shared


def _configure_pool(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(A, "is_provider_explicitly_configured", lambda _provider: True)
    monkeypatch.setattr(CP, "load_env", lambda: {})
    monkeypatch.setattr(CP, "_get_secret", lambda *_args: "")


def _install_xai_refresh(
    monkeypatch: pytest.MonkeyPatch,
    *,
    same_access: bool = False,
) -> tuple[list[tuple[str, str]], threading.Event, threading.Event, threading.Event]:
    calls: list[tuple[str, str]] = []
    calls_lock = threading.Lock()
    first_post = threading.Event()
    second_post = threading.Event()
    release_post = threading.Event()

    def fake_refresh(access_token: str, refresh_token: str, **_kwargs: Any) -> dict[str, Any]:
        with calls_lock:
            calls.append((access_token, refresh_token))
            call_number = len(calls)
        (first_post if call_number == 1 else second_post).set()
        assert release_post.wait(timeout=5)
        return {
            "access_token": access_token if same_access else _jwt(seconds=86_400),
            "refresh_token": f"winner-refresh-{call_number}",
            "last_refresh": "2026-08-06T12:00:00Z",
        }

    monkeypatch.setattr(A, "refresh_xai_oauth_pure", fake_refresh)
    return calls, first_post, second_post, release_post


def _join_threads(workers: list[threading.Thread]) -> None:
    for worker in workers:
        worker.join(timeout=5)
    assert all(not worker.is_alive() for worker in workers)


def _install_nous_transport(
    monkeypatch: pytest.MonkeyPatch,
    *,
    new_access: str,
    new_refresh: str,
    entered: threading.Event | None = None,
    release: threading.Event | None = None,
    fail_after_enter: bool = False,
) -> list[str]:
    requests: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request.headers["x-nous-refresh-token"])
        if entered is not None:
            entered.set()
        if release is not None:
            assert release.wait(timeout=5)
        if fail_after_enter:
            raise httpx.ConnectError("simulated interrupted exchange", request=request)
        return httpx.Response(
            200,
            request=request,
            json={
                "access_token": new_access,
                "refresh_token": new_refresh,
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
    monkeypatch.setattr(A, "_read_shared_nous_state", lambda: None)
    monkeypatch.setattr(A, "_write_shared_nous_state", lambda _state: None)
    return requests


def test_forced_xai_runtime_waiter_adopts_rotated_root_lineage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root_home, _shared = _configure_profiles(tmp_path, monkeypatch)
    root_path = root_home / "auth.json"
    homes = [root_home / "profiles" / name for name in ("one", "two")]
    old_access = _jwt(seconds=3600)
    _write_json(root_path, _xai_store(old_access, "root-refresh-old"))
    for home in homes:
        _write_json(home / "auth.json", {"version": 1, "providers": {}})

    calls, first_post, second_post, release_post = _install_xai_refresh(monkeypatch)
    start = threading.Barrier(3)
    results: list[dict[str, Any]] = []
    errors: list[BaseException] = []

    def worker(home: Path) -> None:
        token = set_hermes_home_override(home)
        try:
            start.wait(timeout=5)
            results.append(A.resolve_xai_oauth_runtime_credentials(force_refresh=True))
        except BaseException as exc:
            errors.append(exc)
        finally:
            reset_hermes_home_override(token)

    workers = [threading.Thread(target=worker, args=(home,)) for home in homes]
    for item in workers:
        item.start()
    start.wait(timeout=5)
    assert first_post.wait(timeout=5)
    assert not second_post.wait(timeout=0.5)
    release_post.set()
    _join_threads(workers)

    assert errors == []
    assert len(calls) == 1
    assert len(results) == 2
    winner_access = _read_json(root_path)["providers"]["xai-oauth"]["tokens"]["access_token"]
    assert {result["api_key"] for result in results} == {winner_access}


def test_forced_xai_pool_and_runtime_waiters_share_root_reservation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root_home, _shared = _configure_profiles(tmp_path, monkeypatch)
    _configure_pool(monkeypatch)
    root_path = root_home / "auth.json"
    pool_home = root_home / "profiles" / "pool"
    runtime_home = root_home / "profiles" / "runtime"
    old_access = _jwt(seconds=3600)
    _write_json(root_path, _xai_store(old_access, "root-refresh-old"))
    for home in (pool_home, runtime_home):
        _write_json(home / "auth.json", {"version": 1, "providers": {}})

    pool_token = set_hermes_home_override(pool_home)
    try:
        pool = CP.load_pool("xai-oauth")
        entry = next(item for item in pool.entries() if item.source == "device_code")
    finally:
        reset_hermes_home_override(pool_token)

    calls, first_post, second_post, release_post = _install_xai_refresh(monkeypatch)
    results: list[Any] = []
    errors: list[BaseException] = []

    def pool_worker() -> None:
        token = set_hermes_home_override(pool_home)
        try:
            results.append(pool._refresh_entry(entry, force=True))
        except BaseException as exc:
            errors.append(exc)
        finally:
            reset_hermes_home_override(token)

    def runtime_worker() -> None:
        token = set_hermes_home_override(runtime_home)
        try:
            results.append(A.resolve_xai_oauth_runtime_credentials(force_refresh=True))
        except BaseException as exc:
            errors.append(exc)
        finally:
            reset_hermes_home_override(token)

    workers = [threading.Thread(target=pool_worker), threading.Thread(target=runtime_worker)]
    workers[0].start()
    assert first_post.wait(timeout=5)
    workers[1].start()
    assert not second_post.wait(timeout=0.5)
    release_post.set()
    _join_threads(workers)

    assert errors == []
    assert len(calls) == 1
    assert len(results) == 2
    winner_access = _read_json(root_path)["providers"]["xai-oauth"]["tokens"]["access_token"]
    assert all(
        result is not None
        and (
            result.get("api_key") == winner_access
            if isinstance(result, dict)
            else result.access_token == winner_access
        )
        for result in results
    )


def test_forced_nous_pool_adopts_changed_root_lineage_without_post(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root_home, _shared = _configure_profiles(tmp_path, monkeypatch)
    _configure_pool(monkeypatch)
    root_path = root_home / "auth.json"
    profile_home = root_home / "profiles" / "worker"
    old_access = _jwt(seconds=3600, scope=A.DEFAULT_NOUS_SCOPE)
    winner_access = _jwt(seconds=86_400, scope=A.DEFAULT_NOUS_SCOPE)
    _write_json(root_path, _nous_store(old_access, "nous-refresh-old"))
    _write_json(profile_home / "auth.json", {"version": 1, "providers": {}})
    stale = _pool_entry(
        "nous",
        entry_id="nous-source",
        access_token=old_access,
        refresh_token="nous-refresh-old",
        source="device_code",
    )
    pool = CredentialPool("nous", [stale])
    _write_json(root_path, _nous_store(winner_access, "nous-refresh-winner"))
    requests = _install_nous_transport(
        monkeypatch,
        new_access=_jwt(seconds=7200, scope=A.DEFAULT_NOUS_SCOPE),
        new_refresh="nous-refresh-obsolete",
    )

    token = set_hermes_home_override(profile_home)
    try:
        adopted = pool._refresh_entry(stale, force=True)
    finally:
        reset_hermes_home_override(token)

    assert adopted is not None
    assert adopted.access_token == winner_access
    assert adopted.refresh_token == "nous-refresh-winner"
    assert requests == []


def test_xai_persistent_finalization_failure_leaves_root_reserved(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root_home, _shared = _configure_profiles(tmp_path, monkeypatch)
    _configure_pool(monkeypatch)
    root_path = root_home / "auth.json"
    profile_home = root_home / "profiles" / "runtime"
    reload_home = root_home / "profiles" / "reload"
    old_access = _jwt(seconds=3600)
    _write_json(root_path, _xai_store(old_access, "root-refresh-old"))
    for home in (profile_home, reload_home):
        _write_json(home / "auth.json", {"version": 1, "providers": {}})

    post_calls = 0

    def fake_refresh(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        nonlocal post_calls
        post_calls += 1
        return {
            "access_token": _jwt(seconds=86_400),
            "refresh_token": "root-refresh-new",
            "last_refresh": "2026-08-06T12:00:00Z",
        }

    monkeypatch.setattr(A, "refresh_xai_oauth_pure", fake_refresh)
    real_save = A._save_auth_store
    root_save_count = 0

    def fail_after_reservation(store: dict[str, Any], target_path: Path | None = None, **kwargs: Any):
        nonlocal root_save_count
        if target_path is not None and A._same_path(Path(target_path), root_path):
            root_save_count += 1
            if root_save_count > 1:
                raise OSError("persistent owner finalization failure")
        return real_save(store, target_path, **kwargs)

    monkeypatch.setattr(A, "_save_auth_store", fail_after_reservation)
    token = set_hermes_home_override(profile_home)
    try:
        with pytest.raises(A.SourceCredentialPersistenceError) as raised:
            A.resolve_xai_oauth_runtime_credentials(force_refresh=True)
    finally:
        reset_hermes_home_override(token)

    assert raised.value.source_path is not None
    assert A._same_path(Path(raised.value.source_path), root_path)
    assert post_calls == 1
    root_text = root_path.read_text(encoding="utf-8")
    assert "root-refresh-old" not in root_text
    assert _RESERVATION_KEY in root_text

    token = set_hermes_home_override(reload_home)
    try:
        fresh = CP.load_pool("xai-oauth")
        with pytest.raises(A.AuthError):
            A.resolve_xai_oauth_runtime_credentials(force_refresh=True)
    finally:
        reset_hermes_home_override(token)
    assert fresh.has_available() is False
    assert post_calls == 1
    assert root_save_count >= 2


def test_nous_root_finalization_failure_is_not_redirected_to_borrower(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root_home, _shared = _configure_profiles(tmp_path, monkeypatch)
    _configure_pool(monkeypatch)
    root_path = root_home / "auth.json"
    borrower_home = root_home / "profiles" / "borrower"
    reload_home = root_home / "profiles" / "reload"
    borrower_path = borrower_home / "auth.json"
    old_access = _jwt(seconds=3600, scope=A.DEFAULT_NOUS_SCOPE)
    new_access = _jwt(seconds=7200, scope=A.DEFAULT_NOUS_SCOPE)
    _write_json(root_path, _nous_store(old_access, "nous-refresh-old"))
    for home in (borrower_home, reload_home):
        _write_json(home / "auth.json", {"version": 1, "providers": {}})

    requests = _install_nous_transport(
        monkeypatch,
        new_access=new_access,
        new_refresh="nous-refresh-new",
    )
    token = set_hermes_home_override(borrower_home)
    try:
        pool = CP.load_pool("nous")
        entry = next(item for item in pool.entries() if item.source == "device_code")
    finally:
        reset_hermes_home_override(token)

    real_save = A._save_auth_store
    root_save_count = 0

    def fail_after_reservation(store: dict[str, Any], target_path: Path | None = None, **kwargs: Any):
        nonlocal root_save_count
        if target_path is not None and A._same_path(Path(target_path), root_path):
            root_save_count += 1
            if root_save_count > 1:
                raise OSError("persistent root finalization failure")
        return real_save(store, target_path, **kwargs)

    monkeypatch.setattr(A, "_save_auth_store", fail_after_reservation)
    token = set_hermes_home_override(borrower_home)
    try:
        assert pool._refresh_entry(entry, force=True) is None
    finally:
        reset_hermes_home_override(token)

    assert requests == ["nous-refresh-old"]
    assert "nous-refresh-old" not in root_path.read_text(encoding="utf-8")
    assert _RESERVATION_KEY in root_path.read_text(encoding="utf-8")
    borrower = _read_json(borrower_path)
    assert "nous-refresh-old" not in json.dumps(borrower)

    token = set_hermes_home_override(reload_home)
    try:
        fresh = CP.load_pool("nous")
        assert fresh.has_available() is False
        with pytest.raises(A.AuthError):
            A.resolve_nous_runtime_credentials(force_refresh=True)
    finally:
        reset_hermes_home_override(token)
    assert requests == ["nous-refresh-old"]
    assert root_save_count >= 2


def test_anthropic_persistent_finalization_failure_cannot_be_replayed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    credentials_path = tmp_path / ".claude" / ".credentials.json"
    old = {
        "accessToken": "claude-access-old",
        "refreshToken": "claude-refresh-old",
        "expiresAt": 1,
        "source": "claude_code_credentials_file",
    }
    _write_claude_credentials(
        credentials_path,
        access_token=old["accessToken"],
        refresh_token=old["refreshToken"],
        expires_at_ms=old["expiresAt"],
    )
    calls: list[str] = []

    def fake_refresh(refresh_token: str, *, use_json: bool) -> dict[str, Any]:
        calls.append(refresh_token)
        return {
            "access_token": "claude-access-new",
            "refresh_token": "claude-refresh-new",
            "expires_at_ms": 9_999_999_999_999,
        }

    monkeypatch.setattr(anthropic_adapter, "refresh_anthropic_oauth_pure", fake_refresh)
    real_replace = os.replace
    credential_replaces = 0

    def fail_after_reservation(src: str | os.PathLike[str], dst: str | os.PathLike[str]) -> None:
        nonlocal credential_replaces
        if Path(src) == credentials_path or Path(dst) == credentials_path:
            credential_replaces += 1
            if credential_replaces > 1:
                raise OSError("persistent Claude finalization failure")
        real_replace(src, dst)

    monkeypatch.setattr(anthropic_adapter.os, "replace", fail_after_reservation)

    assert anthropic_adapter._refresh_oauth_token(dict(old)) is None
    for _ in range(2):
        assert anthropic_adapter._resolve_claude_code_token_from_credentials() is None
    assert calls == ["claude-refresh-old"]
    payload = _read_json(credentials_path)
    assert payload["unrelated"] == {"preserved": True}
    serialized = json.dumps(payload)
    assert "claude-refresh-old" not in serialized
    assert _RESERVATION_KEY in serialized
    assert credential_replaces >= 2


def test_newer_keychain_owner_never_falls_back_to_stale_file_refresh(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    credentials_path = tmp_path / ".claude" / ".credentials.json"
    _write_claude_credentials(
        credentials_path,
        access_token="stale-file-access",
        refresh_token="stale-file-refresh",
        expires_at_ms=1,
    )
    keychain = {
        "accessToken": "newer-keychain-access",
        "refreshToken": "newer-keychain-refresh",
        "expiresAt": 2,
        "source": "macos_keychain",
    }
    monkeypatch.setattr(
        anthropic_adapter,
        "_read_claude_code_credentials_from_keychain",
        lambda: dict(keychain),
    )
    calls: list[str] = []
    monkeypatch.setattr(
        anthropic_adapter,
        "refresh_anthropic_oauth_pure",
        lambda refresh_token, **_kwargs: calls.append(refresh_token),
    )

    selected = anthropic_adapter.read_claude_code_credentials()
    assert selected is not None and selected["source"] == "macos_keychain"
    assert anthropic_adapter._refresh_oauth_token(selected) is None
    assert calls == []
    assert _read_json(credentials_path)["claudeAiOauth"]["refreshToken"] == "stale-file-refresh"


def _run_anthropic_waiter_schedule(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    peer: str,
) -> tuple[list[str], list[Any], list[BaseException], Path]:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    auth_path = tmp_path / "hermes" / "auth.json"
    credentials_path = tmp_path / ".claude" / ".credentials.json"
    monkeypatch.setattr(A, "_auth_file_path", lambda: auth_path)
    monkeypatch.setattr(A, "_global_auth_file_path", lambda: None)
    _configure_pool(monkeypatch)
    old = {
        "accessToken": "same-claude-access",
        "refreshToken": "claude-refresh-old",
        "expiresAt": 1,
        "source": "claude_code_credentials_file",
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
        {"version": 1, "credential_pool": {"anthropic": [pool_entry.to_dict()]}},
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
        (first_post if call_number == 1 else second_post).set()
        assert release_post.wait(timeout=5)
        return {
            "access_token": "same-claude-access",
            "refresh_token": f"claude-refresh-new-{call_number}",
            "expires_at_ms": 9_999_999_999_999,
        }

    monkeypatch.setattr(anthropic_adapter, "refresh_anthropic_oauth_pure", fake_refresh)
    results: list[Any] = []
    errors: list[BaseException] = []

    def direct_worker() -> None:
        try:
            results.append(anthropic_adapter._refresh_oauth_token(dict(old)))
        except BaseException as exc:
            errors.append(exc)

    def peer_worker() -> None:
        try:
            if peer == "direct":
                results.append(anthropic_adapter._refresh_oauth_token(dict(old)))
            else:
                results.append(pool._refresh_entry(pool_entry, force=True))
        except BaseException as exc:
            errors.append(exc)

    workers = (
        [threading.Thread(target=peer_worker), threading.Thread(target=direct_worker)]
        if peer == "pool"
        else [threading.Thread(target=direct_worker), threading.Thread(target=peer_worker)]
    )
    workers[0].start()
    assert first_post.wait(timeout=5)
    workers[1].start()
    assert not second_post.wait(timeout=0.5)
    release_post.set()
    _join_threads(workers)
    return calls, results, errors, credentials_path


def test_anthropic_same_access_new_refresh_direct_waiter_coalesces(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls, results, errors, credentials_path = _run_anthropic_waiter_schedule(
        tmp_path,
        monkeypatch,
        peer="direct",
    )
    assert errors == []
    assert calls == ["claude-refresh-old"]
    assert results == ["same-claude-access", "same-claude-access"]
    assert _read_json(credentials_path)["claudeAiOauth"]["refreshToken"] == "claude-refresh-new-1"


def test_anthropic_same_access_new_refresh_direct_pool_waiter_coalesces(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls, results, errors, credentials_path = _run_anthropic_waiter_schedule(
        tmp_path,
        monkeypatch,
        peer="pool",
    )
    assert errors == []
    assert calls == ["claude-refresh-old"]
    assert len(results) == 2
    assert all(
        result == "same-claude-access"
        or (
            isinstance(result, PooledCredential)
            and result.access_token == "same-claude-access"
            and result.refresh_token == "claude-refresh-new-1"
        )
        for result in results
    )
    assert _read_json(credentials_path)["claudeAiOauth"]["refreshToken"] == "claude-refresh-new-1"


def test_claude_external_replacement_after_final_reservation_check_survives(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    credentials_path = tmp_path / ".claude" / ".credentials.json"
    old = {
        "accessToken": "claude-access-old",
        "refreshToken": "claude-refresh-old",
        "expiresAt": 1,
        "source": "claude_code_credentials_file",
    }
    _write_claude_credentials(
        credentials_path,
        access_token=old["accessToken"],
        refresh_token=old["refreshToken"],
        expires_at_ms=old["expiresAt"],
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
    final_check = threading.Event()
    release_commit = threading.Event()

    def commit_hook() -> None:
        final_check.set()
        assert release_commit.wait(timeout=5)

    monkeypatch.setattr(
        anthropic_adapter,
        "_claude_reservation_commit_hook",
        commit_hook,
        raising=False,
    )
    results: list[str | None] = []
    worker = threading.Thread(
        target=lambda: results.append(anthropic_adapter._refresh_oauth_token(dict(old)))
    )
    worker.start()
    reached = final_check.wait(timeout=2)
    if reached:
        winner_path = credentials_path.with_suffix(".external-winner")
        _write_claude_credentials(
            winner_path,
            access_token="winner-access",
            refresh_token="winner-refresh",
            expires_at_ms=9_999_999_999_999,
        )
        os.replace(winner_path, credentials_path)
    release_commit.set()
    worker.join(timeout=5)

    assert reached is True
    assert not worker.is_alive()
    oauth = _read_json(credentials_path)["claudeAiOauth"]
    assert (oauth["accessToken"], oauth["refreshToken"]) == (
        "winner-access",
        "winner-refresh",
    )
    assert results == ["winner-access"]


@pytest.mark.parametrize("provider", ["xai-oauth", "nous"])
def test_external_newer_auth_owner_state_wins_over_stale_finalizer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
) -> None:
    root_home, _shared = _configure_profiles(tmp_path, monkeypatch)
    root_path = root_home / "auth.json"
    profile_home = root_home / "profiles" / "worker"
    _write_json(profile_home / "auth.json", {"version": 1, "providers": {}})
    entered = threading.Event()
    release = threading.Event()
    results: list[dict[str, Any]] = []
    errors: list[BaseException] = []

    if provider == "xai-oauth":
        old_access = _jwt(seconds=3600)
        winner_access = _jwt(seconds=86_400)
        obsolete_access = _jwt(seconds=7200)
        _write_json(root_path, _xai_store(old_access, "xai-refresh-old"))

        def blocked_refresh(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
            entered.set()
            assert release.wait(timeout=5)
            return {
                "access_token": obsolete_access,
                "refresh_token": "xai-refresh-obsolete",
                "last_refresh": "2026-08-06T12:00:00Z",
            }

        monkeypatch.setattr(A, "refresh_xai_oauth_pure", blocked_refresh)

        def invoke() -> dict[str, Any]:
            return A.resolve_xai_oauth_runtime_credentials(force_refresh=True)

        winner_store = _xai_store(winner_access, "xai-refresh-winner")
        expected_key = winner_access
    else:
        old_access = _jwt(seconds=3600, scope=A.DEFAULT_NOUS_SCOPE)
        winner_access = _jwt(seconds=86_400, scope=A.DEFAULT_NOUS_SCOPE)
        obsolete_access = _jwt(seconds=7200, scope=A.DEFAULT_NOUS_SCOPE)
        _write_json(root_path, _nous_store(old_access, "nous-refresh-old"))
        _install_nous_transport(
            monkeypatch,
            new_access=obsolete_access,
            new_refresh="nous-refresh-obsolete",
            entered=entered,
            release=release,
        )

        def invoke() -> dict[str, Any]:
            return A.resolve_nous_runtime_credentials(force_refresh=True)

        winner_store = _nous_store(winner_access, "nous-refresh-winner")
        expected_key = winner_access

    def worker() -> None:
        token = set_hermes_home_override(profile_home)
        try:
            results.append(invoke())
        except BaseException as exc:
            errors.append(exc)
        finally:
            reset_hermes_home_override(token)

    thread = threading.Thread(target=worker)
    thread.start()
    assert entered.wait(timeout=5)
    external_path = root_path.with_suffix(".external-winner")
    _write_json(external_path, winner_store)
    os.replace(external_path, root_path)
    release.set()
    thread.join(timeout=5)

    assert not thread.is_alive()
    assert errors == []
    assert len(results) == 1
    assert results[0]["api_key"] == expected_key
    persisted = _read_json(root_path)["providers"][provider]
    persisted_refresh = (
        persisted["tokens"]["refresh_token"]
        if provider == "xai-oauth"
        else persisted["refresh_token"]
    )
    assert persisted_refresh == f"{provider.removesuffix('-oauth')}-refresh-winner"


@pytest.mark.parametrize("provider", ["xai-oauth", "nous", "anthropic"])
def test_reservation_persistence_failure_prevents_post(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
) -> None:
    post_calls = 0
    requests: list[str] = []

    if provider == "anthropic":
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        path = tmp_path / ".claude" / ".credentials.json"
        creds = {
            "accessToken": "claude-access-old",
            "refreshToken": "claude-refresh-old",
            "expiresAt": 1,
            "source": "claude_code_credentials_file",
        }
        _write_claude_credentials(
            path,
            access_token=creds["accessToken"],
            refresh_token=creds["refreshToken"],
            expires_at_ms=creds["expiresAt"],
        )

        def fake_refresh(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
            nonlocal post_calls
            post_calls += 1
            return {}

        monkeypatch.setattr(anthropic_adapter, "refresh_anthropic_oauth_pure", fake_refresh)
        monkeypatch.setattr(
            anthropic_adapter.os,
            "replace",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("reservation failed")),
        )
        assert anthropic_adapter._refresh_oauth_token(creds) is None
    else:
        root_home, _shared = _configure_profiles(tmp_path, monkeypatch)
        profile_home = root_home / "profiles" / "worker"
        root_path = root_home / "auth.json"
        _write_json(profile_home / "auth.json", {"version": 1, "providers": {}})
        if provider == "xai-oauth":
            _write_json(root_path, _xai_store(_jwt(seconds=3600), "xai-refresh-old"))

            def fake_xai(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
                nonlocal post_calls
                post_calls += 1
                return {}

            monkeypatch.setattr(A, "refresh_xai_oauth_pure", fake_xai)
        else:
            old = _jwt(seconds=3600, scope=A.DEFAULT_NOUS_SCOPE)
            new = _jwt(seconds=7200, scope=A.DEFAULT_NOUS_SCOPE)
            _write_json(root_path, _nous_store(old, "nous-refresh-old"))
            requests = _install_nous_transport(
                monkeypatch,
                new_access=new,
                new_refresh="nous-refresh-new",
            )

        real_save = A._save_auth_store

        def fail_root_save(store: dict[str, Any], target_path: Path | None = None, **kwargs: Any):
            if target_path is not None and A._same_path(Path(target_path), root_path):
                raise OSError("reservation failed")
            return real_save(store, target_path, **kwargs)

        monkeypatch.setattr(A, "_save_auth_store", fail_root_save)
        token = set_hermes_home_override(profile_home)
        try:
            with pytest.raises(Exception):
                if provider == "xai-oauth":
                    A.resolve_xai_oauth_runtime_credentials(force_refresh=True)
                else:
                    A.resolve_nous_runtime_credentials(force_refresh=True)
        finally:
            reset_hermes_home_override(token)
        if provider == "nous":
            post_calls = len(requests)

    assert post_calls == 0


def test_xai_reserved_crash_state_reloads_fail_closed_and_new_lineage_recovers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root_home, _shared = _configure_profiles(tmp_path, monkeypatch)
    _configure_pool(monkeypatch)
    root_path = root_home / "auth.json"
    worker_home = root_home / "profiles" / "worker"
    fresh_home = root_home / "profiles" / "fresh"
    old_access = _jwt(seconds=3600)
    _write_json(root_path, _xai_store(old_access, "xai-refresh-old"))
    for home in (worker_home, fresh_home):
        _write_json(home / "auth.json", {"version": 1, "providers": {}})
    entered = threading.Event()
    release = threading.Event()
    calls = 0

    def interrupted_post(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        entered.set()
        assert release.wait(timeout=5)
        raise RuntimeError("simulated interruption")

    monkeypatch.setattr(A, "refresh_xai_oauth_pure", interrupted_post)
    real_save = A._save_auth_store
    root_saves = 0

    def reject_cleanup_writes(
        store: dict[str, Any],
        target_path: Path | None = None,
        **kwargs: Any,
    ):
        nonlocal root_saves
        if target_path is not None and A._same_path(Path(target_path), root_path):
            root_saves += 1
            if root_saves > 1:
                raise OSError("persistent cleanup failure")
        return real_save(store, target_path, **kwargs)

    monkeypatch.setattr(A, "_save_auth_store", reject_cleanup_writes)
    errors: list[BaseException] = []

    def worker() -> None:
        token = set_hermes_home_override(worker_home)
        try:
            A.resolve_xai_oauth_runtime_credentials(force_refresh=True)
        except BaseException as exc:
            errors.append(exc)
        finally:
            reset_hermes_home_override(token)

    thread = threading.Thread(target=worker)
    thread.start()
    assert entered.wait(timeout=5)
    during = root_path.read_text(encoding="utf-8")
    release.set()
    thread.join(timeout=5)

    assert not thread.is_alive()
    assert errors
    assert root_saves == 1
    assert _RESERVATION_KEY in during
    assert "xai-refresh-old" not in during
    assert old_access not in during
    xai_reservation = json.loads(during)["providers"]["xai-oauth"][
        _RESERVATION_KEY
    ]
    _assert_non_secret_reservation(
        xai_reservation,
        old_access,
        "xai-refresh-old",
    )
    token = set_hermes_home_override(fresh_home)
    try:
        assert CP.load_pool("xai-oauth").has_available() is False
    finally:
        reset_hermes_home_override(token)
    assert calls == 1

    newer_access = _jwt(seconds=86_400)
    _write_json(root_path, _xai_store(newer_access, "xai-refresh-newer"))
    token = set_hermes_home_override(fresh_home)
    try:
        recovered = CP.load_pool("xai-oauth")
    finally:
        reset_hermes_home_override(token)
    assert recovered.has_available() is True
    assert any(item.refresh_token == "xai-refresh-newer" for item in recovered.entries())


def test_nous_reserved_crash_state_reloads_fail_closed_and_new_lineage_recovers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root_home, _shared = _configure_profiles(tmp_path, monkeypatch)
    _configure_pool(monkeypatch)
    root_path = root_home / "auth.json"
    worker_home = root_home / "profiles" / "worker"
    fresh_home = root_home / "profiles" / "fresh"
    old_access = _jwt(seconds=3600, scope=A.DEFAULT_NOUS_SCOPE)
    new_access = _jwt(seconds=7200, scope=A.DEFAULT_NOUS_SCOPE)
    _write_json(root_path, _nous_store(old_access, "nous-refresh-old"))
    for home in (worker_home, fresh_home):
        _write_json(home / "auth.json", {"version": 1, "providers": {}})
    entered = threading.Event()
    release = threading.Event()
    requests = _install_nous_transport(
        monkeypatch,
        new_access=new_access,
        new_refresh="nous-refresh-new",
        entered=entered,
        release=release,
        fail_after_enter=True,
    )
    real_save = A._save_auth_store
    root_saves = 0

    def reject_cleanup_writes(
        store: dict[str, Any],
        target_path: Path | None = None,
        **kwargs: Any,
    ):
        nonlocal root_saves
        if target_path is not None and A._same_path(Path(target_path), root_path):
            root_saves += 1
            if root_saves > 1:
                raise OSError("persistent cleanup failure")
        return real_save(store, target_path, **kwargs)

    monkeypatch.setattr(A, "_save_auth_store", reject_cleanup_writes)
    errors: list[BaseException] = []

    def worker() -> None:
        token = set_hermes_home_override(worker_home)
        try:
            A.resolve_nous_runtime_credentials(force_refresh=True)
        except BaseException as exc:
            errors.append(exc)
        finally:
            reset_hermes_home_override(token)

    thread = threading.Thread(target=worker)
    thread.start()
    assert entered.wait(timeout=5)
    during = root_path.read_text(encoding="utf-8")
    release.set()
    thread.join(timeout=5)

    assert not thread.is_alive()
    assert errors
    assert root_saves == 1
    assert _RESERVATION_KEY in during
    assert "nous-refresh-old" not in during
    assert old_access not in during
    nous_reservation = json.loads(during)["providers"]["nous"][_RESERVATION_KEY]
    _assert_non_secret_reservation(
        nous_reservation,
        old_access,
        "nous-refresh-old",
    )
    token = set_hermes_home_override(fresh_home)
    try:
        assert CP.load_pool("nous").has_available() is False
    finally:
        reset_hermes_home_override(token)
    assert requests == ["nous-refresh-old"]

    _write_json(root_path, _nous_store(new_access, "nous-refresh-newer"))
    token = set_hermes_home_override(fresh_home)
    try:
        recovered = CP.load_pool("nous")
    finally:
        reset_hermes_home_override(token)
    assert recovered.has_available() is True
    assert any(item.refresh_token == "nous-refresh-newer" for item in recovered.entries())


def test_claude_reserved_crash_state_reloads_fail_closed_and_new_lineage_recovers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    credentials_path = tmp_path / ".claude" / ".credentials.json"
    old = {
        "accessToken": "claude-access-old",
        "refreshToken": "claude-refresh-old",
        "expiresAt": 1,
        "source": "claude_code_credentials_file",
    }
    _write_claude_credentials(
        credentials_path,
        access_token=old["accessToken"],
        refresh_token=old["refreshToken"],
        expires_at_ms=old["expiresAt"],
    )
    entered = threading.Event()
    release = threading.Event()
    calls = 0

    def interrupted_post(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        entered.set()
        assert release.wait(timeout=5)
        raise RuntimeError("simulated interruption")

    monkeypatch.setattr(anthropic_adapter, "refresh_anthropic_oauth_pure", interrupted_post)
    real_replace = anthropic_adapter.os.replace
    source_commits = 0

    def reject_cleanup_writes(
        src: str | os.PathLike[str],
        dst: str | os.PathLike[str],
    ) -> None:
        nonlocal source_commits
        if Path(src) == credentials_path or Path(dst) == credentials_path:
            source_commits += 1
            if source_commits > 1:
                raise OSError("persistent cleanup failure")
        real_replace(src, dst)

    monkeypatch.setattr(anthropic_adapter.os, "replace", reject_cleanup_writes)
    results: list[str | None] = []
    thread = threading.Thread(
        target=lambda: results.append(anthropic_adapter._refresh_oauth_token(dict(old)))
    )
    thread.start()
    assert entered.wait(timeout=5)
    during = credentials_path.read_text(encoding="utf-8")
    release.set()
    thread.join(timeout=5)

    assert not thread.is_alive()
    assert results == [None]
    assert source_commits == 1
    assert _RESERVATION_KEY in during
    assert "claude-refresh-old" not in during
    assert "claude-access-old" not in during
    claude_reservation = json.loads(during)["claudeAiOauth"][_RESERVATION_KEY]
    _assert_non_secret_reservation(
        claude_reservation,
        "claude-access-old",
        "claude-refresh-old",
    )
    assert anthropic_adapter.read_claude_code_credentials() is None
    assert calls == 1

    _write_claude_credentials(
        credentials_path,
        access_token="claude-access-newer",
        refresh_token="claude-refresh-newer",
        expires_at_ms=9_999_999_999_999,
    )
    recovered = anthropic_adapter.read_claude_code_credentials()
    assert recovered is not None
    assert recovered["refreshToken"] == "claude-refresh-newer"
