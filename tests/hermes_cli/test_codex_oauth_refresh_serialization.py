"""Focused tests for serializing OpenAI Codex OAuth refreshes.

These tests describe the first phase of the focused auth race fix:

* a profile using root fallback must refresh the root canonical store, not
  create a profile-owned provider block;
* an explicit profile provider block, even an empty one, shadows root;
* concurrent in-process callers should single-flight one rotating refresh.
"""

import base64
import json
import threading
import time
from pathlib import Path

import pytest

from hermes_cli import auth
from hermes_cli.auth import AuthError, resolve_codex_runtime_credentials


def _jwt_with_exp(exp_epoch: int) -> str:
    payload = {"exp": exp_epoch}
    encoded = (
        base64.urlsafe_b64encode(json.dumps(payload).encode("utf-8"))
        .rstrip(b"=")
        .decode("utf-8")
    )
    return f"h.{encoded}.s"


def _write_store(path: Path, store: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(store, indent=2), encoding="utf-8")


def _read_store(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture
def profile_and_root_auth_paths(tmp_path, monkeypatch):
    """Wire distinct profile and root auth stores without touching real auth."""
    profile_path = tmp_path / "profiles" / "work" / "auth.json"
    root_path = tmp_path / "root" / "auth.json"

    monkeypatch.setattr(auth, "_auth_file_path", lambda: profile_path)
    monkeypatch.setattr(auth, "_global_auth_file_path", lambda: root_path)
    monkeypatch.setenv("HOME", str(tmp_path / "not-the-real-home"))
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "missing-codex-cli"))
    return profile_path, root_path


def test_root_fallback_refresh_persists_to_root_without_profile_shadow(
    profile_and_root_auth_paths,
    monkeypatch,
):
    """A root-fallback refresh must update root and not create profile state."""
    profile_path, root_path = profile_and_root_auth_paths
    old_access = _jwt_with_exp(int(time.time()) - 60)
    new_access = _jwt_with_exp(int(time.time()) + 3600)

    _write_store(
        profile_path,
        {
            "version": 1,
            "active_provider": "anthropic",
            "providers": {},
        },
    )
    _write_store(
        root_path,
        {
            "version": 1,
            "active_provider": "anthropic",
            "providers": {
                "openai-codex": {
                    "tokens": {
                        "access_token": old_access,
                        "refresh_token": "root-refresh-old",
                    },
                    "last_refresh": "2026-01-01T00:00:00Z",
                    "auth_mode": "chatgpt",
                }
            },
        },
    )

    refresh_calls = {"count": 0}

    def _fake_refresh(access_token, refresh_token, *, timeout_seconds=20.0):
        refresh_calls["count"] += 1
        assert access_token == old_access
        assert refresh_token == "root-refresh-old"
        return {
            "access_token": new_access,
            "refresh_token": "root-refresh-new",
            "last_refresh": "2026-06-21T00:00:00Z",
        }

    monkeypatch.setattr(auth, "refresh_codex_oauth_pure", _fake_refresh)

    resolved = resolve_codex_runtime_credentials()

    assert refresh_calls["count"] == 1
    assert resolved["api_key"] == new_access

    root = _read_store(root_path)
    root_state = root["providers"]["openai-codex"]
    assert root_state["tokens"]["access_token"] == new_access
    assert root_state["tokens"]["refresh_token"] == "root-refresh-new"
    assert root_state["last_refresh"] == "2026-06-21T00:00:00Z"
    assert root["active_provider"] == "anthropic"

    profile = _read_store(profile_path)
    assert "openai-codex" not in profile.get("providers", {})
    assert profile["active_provider"] == "anthropic"


def test_profile_owned_empty_codex_block_shadows_root(profile_and_root_auth_paths):
    """An empty profile block is own state and must not fall back to root."""
    profile_path, root_path = profile_and_root_auth_paths
    root_access = _jwt_with_exp(int(time.time()) + 3600)

    _write_store(
        profile_path,
        {
            "version": 1,
            "active_provider": "anthropic",
            "providers": {"openai-codex": {}},
        },
    )
    _write_store(
        root_path,
        {
            "version": 1,
            "active_provider": "openai-codex",
            "providers": {
                "openai-codex": {
                    "tokens": {
                        "access_token": root_access,
                        "refresh_token": "root-refresh",
                    },
                    "auth_mode": "chatgpt",
                }
            },
        },
    )

    with pytest.raises(AuthError) as exc:
        resolve_codex_runtime_credentials()

    assert exc.value.code in {"codex_auth_missing", "codex_auth_invalid_shape"}
    profile = _read_store(profile_path)
    assert profile["providers"]["openai-codex"] == {}
    root = _read_store(root_path)
    assert root["providers"]["openai-codex"]["tokens"]["refresh_token"] == "root-refresh"


def test_same_process_concurrent_codex_refresh_single_flights(tmp_path, monkeypatch):
    """Concurrent callers with the same expired chain should spend it once."""
    hermes_home = tmp_path / "hermes"
    auth_path = hermes_home / "auth.json"
    old_access = _jwt_with_exp(int(time.time()) - 60)
    new_access = _jwt_with_exp(int(time.time()) + 3600)
    _write_store(
        auth_path,
        {
            "version": 1,
            "active_provider": "anthropic",
            "providers": {
                "openai-codex": {
                    "tokens": {
                        "access_token": old_access,
                        "refresh_token": "refresh-old",
                    },
                    "last_refresh": "2026-01-01T00:00:00Z",
                    "auth_mode": "chatgpt",
                }
            },
        },
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "missing-codex-cli"))
    monkeypatch.setenv("HERMES_CODEX_REFRESH_TIMEOUT_SECONDS", "5")

    refresh_calls = {"count": 0}
    refresh_lock = threading.Lock()
    start = threading.Barrier(10)

    def _fake_refresh(access_token, refresh_token, *, timeout_seconds=20.0):
        assert access_token == old_access
        assert refresh_token == "refresh-old"
        with refresh_lock:
            refresh_calls["count"] += 1
        time.sleep(0.05)
        return {
            "access_token": new_access,
            "refresh_token": "refresh-new",
            "last_refresh": "2026-06-21T00:00:00Z",
        }

    monkeypatch.setattr(auth, "refresh_codex_oauth_pure", _fake_refresh)

    results = []
    errors = []

    def _worker():
        try:
            start.wait(timeout=5)
            results.append(resolve_codex_runtime_credentials()["api_key"])
        except Exception as exc:  # pragma: no cover - assertion reports details below
            errors.append(exc)

    threads = [threading.Thread(target=_worker) for _ in range(10)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)

    assert not errors
    assert len(results) == 10
    assert set(results) == {new_access}
    assert refresh_calls["count"] == 1

    store = _read_store(auth_path)
    state = store["providers"]["openai-codex"]
    assert state["tokens"]["access_token"] == new_access
    assert state["tokens"]["refresh_token"] == "refresh-new"
    assert store["active_provider"] == "anthropic"


def test_refresh_failure_adopts_newer_fresh_canonical_store(tmp_path, monkeypatch):
    """Any refresh exception should reread and adopt a fresh changed store."""
    hermes_home = tmp_path / "hermes"
    auth_path = hermes_home / "auth.json"
    old_access = _jwt_with_exp(int(time.time()) - 60)
    new_access = _jwt_with_exp(int(time.time()) + 3600)
    _write_store(
        auth_path,
        {
            "version": 1,
            "active_provider": "anthropic",
            "providers": {
                "openai-codex": {
                    "tokens": {
                        "access_token": old_access,
                        "refresh_token": "refresh-old",
                    },
                    "last_refresh": "2026-01-01T00:00:00Z",
                    "auth_mode": "chatgpt",
                }
            },
        },
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "missing-codex-cli"))

    def _fake_refresh(access_token, refresh_token, *, timeout_seconds=20.0):
        assert access_token == old_access
        assert refresh_token == "refresh-old"
        _write_store(
            auth_path,
            {
                "version": 1,
                "active_provider": "anthropic",
                "providers": {
                    "openai-codex": {
                        "tokens": {
                            "access_token": new_access,
                            "refresh_token": "refresh-new",
                        },
                        "last_refresh": "2026-06-21T00:00:00Z",
                        "auth_mode": "chatgpt",
                    }
                },
            },
        )
        raise AuthError(
            "refresh token reused",
            provider="openai-codex",
            code="refresh_token_reused",
            relogin_required=True,
        )

    monkeypatch.setattr(auth, "refresh_codex_oauth_pure", _fake_refresh)
    monkeypatch.setattr(
        auth,
        "_recover_codex_tokens_from_cli",
        lambda reason: pytest.fail("runtime refresh must not import Codex CLI tokens"),
    )

    resolved = resolve_codex_runtime_credentials()

    assert resolved["api_key"] == new_access
    assert resolved["last_refresh"] == "2026-06-21T00:00:00Z"
    assert _read_store(auth_path)["providers"]["openai-codex"]["tokens"]["refresh_token"] == "refresh-new"


def test_changed_unknown_exp_token_is_not_adopted_after_refresh_failure(tmp_path, monkeypatch):
    """Recovery adoption requires a parseable future exp claim."""
    hermes_home = tmp_path / "hermes"
    auth_path = hermes_home / "auth.json"
    old_access = _jwt_with_exp(int(time.time()) - 60)
    _write_store(
        auth_path,
        {
            "version": 1,
            "providers": {
                "openai-codex": {
                    "tokens": {
                        "access_token": old_access,
                        "refresh_token": "refresh-old",
                    },
                    "last_refresh": "2026-01-01T00:00:00Z",
                    "auth_mode": "chatgpt",
                }
            },
        },
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "missing-codex-cli"))

    def _fake_refresh(access_token, refresh_token, *, timeout_seconds=20.0):
        _write_store(
            auth_path,
            {
                "version": 1,
                "providers": {
                    "openai-codex": {
                        "tokens": {
                            "access_token": "not-a-jwt",
                            "refresh_token": "refresh-new",
                        },
                        "last_refresh": "2026-06-21T00:00:00Z",
                        "auth_mode": "chatgpt",
                    }
                },
            },
        )
        raise AuthError(
            "refresh token reused",
            provider="openai-codex",
            code="refresh_token_reused",
            relogin_required=True,
        )

    monkeypatch.setattr(auth, "refresh_codex_oauth_pure", _fake_refresh)

    with pytest.raises(AuthError) as exc:
        resolve_codex_runtime_credentials()

    assert exc.value.code == "refresh_token_reused"


def test_persist_failure_after_http_success_is_terminal_and_not_retried(tmp_path, monkeypatch):
    """Once refresh succeeds, save failure must not trigger a second refresh."""
    hermes_home = tmp_path / "hermes"
    auth_path = hermes_home / "auth.json"
    old_access = _jwt_with_exp(int(time.time()) - 60)
    new_access = _jwt_with_exp(int(time.time()) + 3600)
    _write_store(
        auth_path,
        {
            "version": 1,
            "providers": {
                "openai-codex": {
                    "tokens": {
                        "access_token": old_access,
                        "refresh_token": "refresh-old",
                    },
                    "last_refresh": "2026-01-01T00:00:00Z",
                    "auth_mode": "chatgpt",
                }
            },
        },
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "missing-codex-cli"))

    refresh_calls = {"count": 0}

    def _fake_refresh(access_token, refresh_token, *, timeout_seconds=20.0):
        refresh_calls["count"] += 1
        return {
            "access_token": new_access,
            "refresh_token": "refresh-new",
            "last_refresh": "2026-06-21T00:00:00Z",
        }

    def _failing_save(store, target_path=None):
        if target_path == auth_path:
            raise OSError("simulated save failure")
        return auth._save_auth_store(store, target_path)

    monkeypatch.setattr(auth, "refresh_codex_oauth_pure", _fake_refresh)
    monkeypatch.setattr(auth, "_save_auth_store", _failing_save)

    with pytest.raises(AuthError) as exc:
        resolve_codex_runtime_credentials()

    assert exc.value.code == "codex_refresh_persist_failed"
    assert exc.value.relogin_required is False
    assert refresh_calls["count"] == 1


def test_runtime_missing_access_token_does_not_auto_import_codex_cli(tmp_path, monkeypatch):
    """Runtime recovery must not silently switch to ~/.codex/auth.json."""
    hermes_home = tmp_path / "hermes"
    _write_store(
        hermes_home / "auth.json",
        {
            "version": 1,
            "providers": {
                "openai-codex": {
                    "tokens": {"refresh_token": "stale-refresh"},
                    "last_refresh": "2026-01-01T00:00:00Z",
                    "auth_mode": "chatgpt",
                }
            },
        },
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    recover_calls = {"count": 0}

    def _recover_spy(reason):
        recover_calls["count"] += 1
        return {"access_token": _jwt_with_exp(int(time.time()) + 3600), "refresh_token": "cli-refresh"}

    monkeypatch.setattr(auth, "_recover_codex_tokens_from_cli", _recover_spy)

    with pytest.raises(AuthError) as exc:
        resolve_codex_runtime_credentials()

    assert exc.value.code == "codex_auth_missing_access_token"
    assert recover_calls["count"] == 0
