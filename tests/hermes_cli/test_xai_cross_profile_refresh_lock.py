"""Cross-profile serialization for xAI's rotating refresh-token chain."""

from __future__ import annotations

import base64
import json
import threading
import time
from pathlib import Path

import pytest

from hermes_cli import auth


def _jwt_with_exp(exp_epoch: int) -> str:
    payload = base64.urlsafe_b64encode(
        json.dumps({"exp": exp_epoch}).encode("utf-8")
    ).decode("ascii").rstrip("=")
    return f"h.{payload}.s"


def _write_root_state(root_path: Path, access_token: str, refresh_token: str) -> None:
    root_path.parent.mkdir(parents=True, exist_ok=True)
    root_path.write_text(
        json.dumps(
            {
                "version": 1,
                "providers": {
                    "xai-oauth": {
                        "tokens": {
                            "access_token": access_token,
                            "refresh_token": refresh_token,
                            "token_type": "Bearer",
                        },
                        "discovery": {
                            "token_endpoint": "https://auth.x.ai/oauth2/token"
                        },
                        "auth_mode": "oauth_device_code",
                    }
                },
            }
        ),
        encoding="utf-8",
    )


def _install_thread_scoped_profile_paths(
    monkeypatch, tmp_path: Path
) -> tuple[threading.local, Path]:
    thread_state = threading.local()
    root_path = tmp_path / "root" / "auth.json"

    def _active_auth_path() -> Path:
        return thread_state.home / "auth.json"

    def _active_lock_path() -> Path:
        return thread_state.home / "auth.lock"

    monkeypatch.setattr(auth, "_auth_file_path", _active_auth_path)
    monkeypatch.setattr(auth, "_auth_lock_path", _active_lock_path)
    monkeypatch.setattr(auth, "_global_auth_file_path", lambda: root_path)
    return thread_state, root_path


def _run_two_profiles(worker, tmp_path: Path, thread_state: threading.local) -> list[object]:
    start = threading.Barrier(2)
    results: list[object] = [None, None]

    def _run(index: int) -> None:
        profile_home = tmp_path / "profiles" / f"p{index}"
        profile_home.mkdir(parents=True, exist_ok=True)
        (profile_home / "auth.json").write_text(
            json.dumps({"version": 1, "providers": {}}), encoding="utf-8"
        )
        thread_state.home = profile_home
        start.wait(timeout=2)
        try:
            results[index] = worker()
        except BaseException as exc:  # surface thread failures in the test process
            results[index] = exc

    threads = [threading.Thread(target=_run, args=(index,)) for index in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)
        assert not thread.is_alive(), "cross-profile refresh worker deadlocked"
    for result in results:
        if isinstance(result, BaseException):
            raise result
    return results


def _install_rotating_refresh_stub(monkeypatch) -> tuple[dict[str, int], str]:
    calls = {"count": 0}
    calls_lock = threading.Lock()
    fresh_access = _jwt_with_exp(int(time.time()) + 2 * 60 * 60)

    def _refresh(access_token: str, refresh_token: str, **_kwargs):
        assert refresh_token == "refresh-0"
        with calls_lock:
            calls["count"] += 1
        # Make the broken implementation's distinct profile locks overlap.
        time.sleep(0.15)
        return {
            "access_token": fresh_access,
            "refresh_token": "refresh-1",
            "id_token": "",
            "expires_in": 7200,
            "token_type": "Bearer",
            "last_refresh": "2026-08-07T23:00:00Z",
        }

    monkeypatch.setattr(auth, "refresh_xai_oauth_pure", _refresh)
    return calls, fresh_access


def _install_single_profile(
    tmp_path: Path,
    thread_state: threading.local,
    store: dict,
) -> Path:
    profile_home = tmp_path / "profiles" / "single"
    profile_home.mkdir(parents=True, exist_ok=True)
    (profile_home / "auth.json").write_text(json.dumps(store), encoding="utf-8")
    thread_state.home = profile_home
    return profile_home / "auth.json"


def test_direct_refresh_serializes_on_shared_root_source(tmp_path, monkeypatch):
    thread_state, root_path = _install_thread_scoped_profile_paths(
        monkeypatch, tmp_path
    )
    _write_root_state(
        root_path,
        _jwt_with_exp(int(time.time()) - 10),
        "refresh-0",
    )
    calls, fresh_access = _install_rotating_refresh_stub(monkeypatch)

    results = _run_two_profiles(
        lambda: auth.resolve_xai_oauth_runtime_credentials()["api_key"],
        tmp_path,
        thread_state,
    )

    assert calls["count"] == 1
    assert results == [fresh_access, fresh_access]
    root_tokens = json.loads(root_path.read_text(encoding="utf-8"))["providers"][
        "xai-oauth"
    ]["tokens"]
    assert root_tokens["refresh_token"] == "refresh-1"


def test_pool_refresh_serializes_on_shared_root_source(tmp_path, monkeypatch):
    from agent import credential_pool as pool_mod

    thread_state, root_path = _install_thread_scoped_profile_paths(
        monkeypatch, tmp_path
    )
    monkeypatch.setattr(pool_mod, "_global_auth_file_path", lambda: root_path)
    _write_root_state(
        root_path,
        _jwt_with_exp(int(time.time()) - 10),
        "refresh-0",
    )
    calls, fresh_access = _install_rotating_refresh_stub(monkeypatch)

    def _select_access_token() -> str:
        selected = pool_mod.load_pool("xai-oauth").select()
        assert selected is not None
        return selected.access_token

    results = _run_two_profiles(_select_access_token, tmp_path, thread_state)

    assert calls["count"] == 1
    assert results == [fresh_access, fresh_access]
    root_tokens = json.loads(root_path.read_text(encoding="utf-8"))["providers"][
        "xai-oauth"
    ]["tokens"]
    assert root_tokens["refresh_token"] == "refresh-1"


def test_direct_refresh_ignores_unusable_profile_shadow(tmp_path, monkeypatch):
    thread_state, root_path = _install_thread_scoped_profile_paths(
        monkeypatch, tmp_path
    )
    _write_root_state(
        root_path,
        _jwt_with_exp(int(time.time()) - 10),
        "refresh-0",
    )
    profile_path = _install_single_profile(
        tmp_path,
        thread_state,
        {
            "version": 1,
            "providers": {"xai-oauth": {"tokens": {}, "last_auth_error": {}}},
        },
    )
    _calls, fresh_access = _install_rotating_refresh_stub(monkeypatch)

    resolved = auth.resolve_xai_oauth_runtime_credentials()

    assert resolved["api_key"] == fresh_access
    root_tokens = json.loads(root_path.read_text(encoding="utf-8"))["providers"][
        "xai-oauth"
    ]["tokens"]
    assert root_tokens["refresh_token"] == "refresh-1"
    profile_tokens = json.loads(profile_path.read_text(encoding="utf-8"))[
        "providers"
    ]["xai-oauth"]["tokens"]
    assert profile_tokens == {}


def test_pool_refresh_ignores_unusable_profile_shadow(tmp_path, monkeypatch):
    from agent import credential_pool as pool_mod

    thread_state, root_path = _install_thread_scoped_profile_paths(
        monkeypatch, tmp_path
    )
    monkeypatch.setattr(pool_mod, "_global_auth_file_path", lambda: root_path)
    _write_root_state(
        root_path,
        _jwt_with_exp(int(time.time()) - 10),
        "refresh-0",
    )
    profile_path = _install_single_profile(
        tmp_path,
        thread_state,
        {
            "version": 1,
            "providers": {"xai-oauth": {"tokens": {}, "last_auth_error": {}}},
        },
    )
    _calls, fresh_access = _install_rotating_refresh_stub(monkeypatch)

    selected = pool_mod.load_pool("xai-oauth").select()

    assert selected is not None
    assert selected.access_token == fresh_access
    root_tokens = json.loads(root_path.read_text(encoding="utf-8"))["providers"][
        "xai-oauth"
    ]["tokens"]
    assert root_tokens["refresh_token"] == "refresh-1"
    profile_tokens = json.loads(profile_path.read_text(encoding="utf-8"))[
        "providers"
    ]["xai-oauth"]["tokens"]
    assert profile_tokens == {}


def test_direct_refresh_surfaces_required_root_persistence_failure(
    tmp_path, monkeypatch
):
    thread_state, root_path = _install_thread_scoped_profile_paths(
        monkeypatch, tmp_path
    )
    _write_root_state(
        root_path,
        _jwt_with_exp(int(time.time()) - 10),
        "refresh-0",
    )
    _install_single_profile(tmp_path, thread_state, {"version": 1, "providers": {}})
    _install_rotating_refresh_stub(monkeypatch)

    def _fail_persist(*_args, **_kwargs):
        raise OSError("simulated root persistence failure")

    monkeypatch.setattr(auth, "_persist_provider_state_to_store", _fail_persist)

    with pytest.raises(OSError, match="simulated root persistence failure"):
        auth.resolve_xai_oauth_runtime_credentials()


def test_pool_refresh_surfaces_required_root_persistence_failure(
    tmp_path, monkeypatch
):
    from agent import credential_pool as pool_mod

    thread_state, root_path = _install_thread_scoped_profile_paths(
        monkeypatch, tmp_path
    )
    monkeypatch.setattr(pool_mod, "_global_auth_file_path", lambda: root_path)
    _write_root_state(
        root_path,
        _jwt_with_exp(int(time.time()) - 10),
        "refresh-0",
    )
    _install_single_profile(tmp_path, thread_state, {"version": 1, "providers": {}})
    _install_rotating_refresh_stub(monkeypatch)

    def _fail_persist(*_args, **_kwargs):
        raise OSError("simulated root persistence failure")

    monkeypatch.setattr(auth, "_persist_provider_state_to_store", _fail_persist)

    with pytest.raises(OSError, match="simulated root persistence failure"):
        pool_mod.load_pool("xai-oauth").select()
