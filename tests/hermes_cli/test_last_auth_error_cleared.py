"""Regression tests: ``last_auth_error`` is cleared on successful re-auth.

The quarantine paths (Nous / Codex / xAI / MiniMax / Spotify) write a
``last_auth_error`` marker with ``relogin_required: true`` into the provider
state on a terminal refresh failure. Before this fix nothing ever popped the
marker on the success paths, so a later successful login or refresh left
auth.json advertising a months-old terminal failure alongside live tokens.

The xai-oauth success path is deliberately out of scope here: PR #67304
already pops the marker in ``_save_xai_oauth_tokens``. This change covers the
paths that PR does not reach.

These tests seed a quarantined-then-recovered state and assert the marker is
gone after each success path:

- ``_save_codex_tokens``      (openai-codex login/refresh)
- ``_spotify_token_payload_to_state``  (spotify login/refresh builder)
- ``_refresh_minimax_oauth_state``     (minimax-oauth refresh)
- ``resolve_nous_access_token``        (nous managed refresh)
- ``resolve_nous_runtime_credentials`` (nous runtime refresh)
- ``_merge_shared_nous_oauth_state``   (nous cross-profile shared-token adopt)
- ``CredentialPool._sync_device_code_entry_to_auth_store`` (pool sync-back)

The write side (marker IS written on terminal failure) is already covered by
test_spotify_auth.py, tests/test_minimax_oauth.py, test_auth_xai_oauth_provider.py
and tests/tools/test_docker_rebootstrap_nous_session.py; nothing here relaxes
those contracts.
"""

from __future__ import annotations

import base64
import json
import time
from types import SimpleNamespace

from agent import credential_pool as CP
from agent.credential_pool import AUTH_TYPE_OAUTH, CredentialPool, PooledCredential
from hermes_cli import auth as auth_mod


def _stale_marker(provider: str) -> dict:
    return {
        "provider": provider,
        "code": "invalid_grant",
        "message": "terminal refresh failure (stale)",
        "reason": "runtime_refresh_failure",
        "relogin_required": True,
        "at": "2026-01-01T00:00:00+00:00",
    }


def _seed_provider_state(provider: str, state: dict) -> None:
    with auth_mod._auth_store_lock():
        store = auth_mod._load_auth_store()
        auth_mod._store_provider_state(store, provider, state, set_active=False)
        auth_mod._save_auth_store(store)


# ---------------------------------------------------------------------------
# openai-codex: _save_codex_tokens (login, refresh)
# ---------------------------------------------------------------------------


def test_save_codex_tokens_clears_stale_marker(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _seed_provider_state(
        "openai-codex",
        {"auth_mode": "chatgpt", "last_auth_error": _stale_marker("openai-codex")},
    )

    auth_mod._save_codex_tokens(
        {"access_token": "fresh-access", "refresh_token": "fresh-refresh"}
    )

    persisted = auth_mod.get_provider_auth_state("openai-codex")
    assert persisted is not None
    assert persisted["tokens"]["access_token"] == "fresh-access"
    assert "last_auth_error" not in persisted


# ---------------------------------------------------------------------------
# spotify: _spotify_token_payload_to_state carries previous_state forward
# ---------------------------------------------------------------------------


def test_spotify_token_payload_to_state_drops_stale_marker():
    previous = {
        "refresh_token": "old-refresh",
        "last_auth_error": _stale_marker("spotify"),
    }

    state = auth_mod._spotify_token_payload_to_state(
        {"access_token": "fresh-access", "expires_in": 3600},
        client_id="client",
        redirect_uri="http://127.0.0.1:43827/spotify/callback",
        requested_scope=auth_mod.DEFAULT_SPOTIFY_SCOPE,
        accounts_base_url=auth_mod.DEFAULT_SPOTIFY_ACCOUNTS_BASE_URL,
        api_base_url=auth_mod.DEFAULT_SPOTIFY_API_BASE_URL,
        previous_state=previous,
    )

    assert state["access_token"] == "fresh-access"
    # refresh_token fallback from previous_state must still work.
    assert state["refresh_token"] == "old-refresh"
    assert "last_auth_error" not in state


# ---------------------------------------------------------------------------
# minimax-oauth: _refresh_minimax_oauth_state
# ---------------------------------------------------------------------------


def test_refresh_minimax_oauth_state_drops_stale_marker(monkeypatch):
    saved = {}

    monkeypatch.setattr(
        auth_mod, "_minimax_save_auth_state", lambda state: saved.update(state)
    )
    monkeypatch.setattr(
        auth_mod,
        "_minimax_resolve_token_expiry_unix",
        lambda expired_in, *, now: now.timestamp() + 3600,
    )
    monkeypatch.setattr(
        auth_mod,
        "_minimax_post_form",
        lambda client, url, data, headers: SimpleNamespace(
            status_code=200,
            json=lambda: {
                "status": "success",
                "expired_in": 3600,
                "access_token": "fresh-access",
                "refresh_token": "fresh-refresh",
            },
        ),
    )

    state = {
        "client_id": "client",
        "portal_base_url": "https://minimax.invalid",
        "access_token": "old-access",
        "refresh_token": "old-refresh",
        "expires_at": "2000-01-01T00:00:00+00:00",
        "last_auth_error": _stale_marker("minimax-oauth"),
    }

    new_state = auth_mod._refresh_minimax_oauth_state(state, force=True)

    assert new_state["access_token"] == "fresh-access"
    assert "last_auth_error" not in new_state
    assert "last_auth_error" not in saved


# ---------------------------------------------------------------------------
# nous: resolve_nous_access_token refresh success
# ---------------------------------------------------------------------------


def test_resolve_nous_access_token_refresh_clears_stale_marker(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(auth_mod, "_RESOLVE_TOKEN_CACHE", None)
    _seed_provider_state(
        "nous",
        {
            # Marker + live-but-expired tokens: the coexistence state the
            # anti-flap guard in get_nous_session_validity tolerates.
            "access_token": "expired-access",
            "refresh_token": "old-refresh",
            "expires_at": "2000-01-01T00:00:00+00:00",
            "last_auth_error": _stale_marker("nous"),
        },
    )

    monkeypatch.setattr(
        auth_mod,
        "_refresh_access_token",
        lambda **kwargs: {
            "access_token": "fresh-access",
            "refresh_token": "rotated-refresh",
            "expires_in": 3600,
        },
    )

    token = auth_mod.resolve_nous_access_token()

    assert token == "fresh-access"
    persisted = auth_mod.get_provider_auth_state("nous")
    assert persisted is not None
    assert persisted["access_token"] == "fresh-access"
    assert persisted["refresh_token"] == "rotated-refresh"
    assert "last_auth_error" not in persisted


# ---------------------------------------------------------------------------
# nous: resolve_nous_runtime_credentials refresh success
# ---------------------------------------------------------------------------


def _invoke_jwt(seconds: int = 3600) -> str:
    """Minimal unsigned inference-scoped JWT (claims-only decode, no verify)."""

    def _part(payload: dict) -> str:
        raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")

    claims = {
        "sub": "test-user",
        "scope": "inference:invoke",
        "exp": int(time.time() + seconds),
    }
    return f"{_part({'alg': 'none', 'typ': 'JWT'})}.{_part(claims)}.sig"


def test_resolve_nous_runtime_credentials_refresh_clears_stale_marker(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    # No shared cross-profile state to adopt: force the in-transaction
    # refresh branch (the clear point under test) rather than a merge.
    monkeypatch.setattr(auth_mod, "_read_shared_nous_state", lambda: None)
    monkeypatch.setattr(auth_mod, "_write_shared_nous_state", lambda state: None)

    fresh_jwt = _invoke_jwt()
    monkeypatch.setattr(
        auth_mod,
        "_refresh_access_token",
        lambda **kwargs: {
            "access_token": fresh_jwt,
            "refresh_token": "rotated-refresh",
            "expires_in": 3600,
            "scope": "inference:invoke",
        },
    )

    _seed_provider_state(
        "nous",
        {
            # Non-JWT access token: unusable for inference
            # ("access_token_not_jwt"), so the runtime path must refresh.
            "access_token": "expired-access",
            "refresh_token": "old-refresh",
            "expires_at": "2000-01-01T00:00:00+00:00",
            "last_auth_error": _stale_marker("nous"),
        },
    )

    creds = auth_mod.resolve_nous_runtime_credentials()

    assert creds["api_key"] == fresh_jwt
    persisted = auth_mod.get_provider_auth_state("nous")
    assert persisted is not None
    assert persisted["access_token"] == fresh_jwt
    assert persisted["refresh_token"] == "rotated-refresh"
    assert "last_auth_error" not in persisted


# ---------------------------------------------------------------------------
# nous: _merge_shared_nous_oauth_state adopting fresher shared tokens
# ---------------------------------------------------------------------------


def test_merge_shared_nous_state_drops_stale_marker(monkeypatch):
    monkeypatch.setattr(
        auth_mod,
        "_read_shared_nous_state",
        lambda: {
            "access_token": "shared-access",
            "refresh_token": "shared-refresh",
            "expires_at": "2099-01-01T00:00:00+00:00",
        },
    )

    # Quarantined local state: no tokens, only the marker.
    state = {"last_auth_error": _stale_marker("nous")}

    assert auth_mod._merge_shared_nous_oauth_state(state) is True
    assert state["access_token"] == "shared-access"
    assert "last_auth_error" not in state


def test_merge_shared_nous_state_no_adopt_keeps_marker(monkeypatch):
    """When nothing is adopted the marker must survive (no false clears)."""
    monkeypatch.setattr(auth_mod, "_read_shared_nous_state", lambda: None)

    state = {"last_auth_error": _stale_marker("nous")}

    assert auth_mod._merge_shared_nous_oauth_state(state) is False
    assert state["last_auth_error"] == _stale_marker("nous")


# ---------------------------------------------------------------------------
# credential pool: _sync_device_code_entry_to_auth_store write-back
# ---------------------------------------------------------------------------


def test_pool_sync_back_clears_stale_marker(tmp_path, monkeypatch):
    profile_path = tmp_path / "auth.json"
    monkeypatch.setattr(auth_mod, "_auth_file_path", lambda: profile_path)
    monkeypatch.setattr(CP, "_global_auth_file_path", lambda: None)
    monkeypatch.setenv("HOME", str(tmp_path / "not-the-root"))

    profile_path.write_text(
        json.dumps(
            {
                "version": 1,
                "providers": {
                    "openai-codex": {
                        "tokens": {
                            "access_token": "old-access",
                            "refresh_token": "old-refresh",
                        },
                        "last_auth_error": _stale_marker("openai-codex"),
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    entry = PooledCredential(
        provider="openai-codex",
        id="codex-1",
        label="cred",
        auth_type=AUTH_TYPE_OAUTH,
        priority=0,
        source="device_code",
        access_token="fresh-access",
        refresh_token="fresh-refresh",
    )
    pool = CredentialPool("openai-codex", [entry])

    pool._sync_device_code_entry_to_auth_store(entry)

    store = json.loads(profile_path.read_text(encoding="utf-8"))
    state = store["providers"]["openai-codex"]
    assert state["tokens"]["access_token"] == "fresh-access"
    assert "last_auth_error" not in state
