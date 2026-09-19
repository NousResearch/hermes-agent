"""Regression tests for Codex refresh_token self-heal (cross-store rotation).

Hermes keeps its OWN copy of the Codex OAuth token (per profile + top-level),
separate from the Codex CLI's ``~/.codex/auth.json``. OAuth refresh_tokens are
single-use, so when the Codex CLI (or another Hermes process) rotates the shared
token, the frozen copy's refresh_token goes stale and ``refresh_codex_oauth_pure``
fails with a relogin-required error. ``_refresh_codex_auth_tokens`` must then
recover by re-importing the canonical token from ``~/.codex/auth.json`` instead of
surfacing a hard 401 — but ONLY for relogin-required failures, never for transient
ones (e.g. 429 quota, where the stored token is still valid).
"""

import base64
import json
import time

import pytest

import hermes_cli.auth as auth
import hermes_cli.auth_codex as auth_codex
from hermes_cli.auth import AuthError, _refresh_codex_auth_tokens, resolve_codex_runtime_credentials
from hermes_cli.auth_codex import _pool_codex_access_token

STALE = {"access_token": "stale-access", "refresh_token": "stale-refresh"}


def _jwt_with_exp(exp_epoch: int) -> str:
    payload = {"exp": exp_epoch}
    encoded = base64.urlsafe_b64encode(
        json.dumps(payload).encode("utf-8")
    ).rstrip(b"=").decode("utf-8")
    return f"h.{encoded}.s"


def test_self_heals_on_stale_refresh_token(monkeypatch):
    """invalid_grant (relogin-required) → reimport from ~/.codex and persist it."""
    saved = {}
    fresh = {
        "access_token": "fresh-access",
        "refresh_token": "fresh-refresh",
        "last_refresh": "2026-06-12T00:00:00Z",
    }

    def _rejected(*_a, **_k):
        raise AuthError(
            "refresh token rejected",
            provider="openai-codex",
            code="invalid_grant",
            relogin_required=True,
        )

    monkeypatch.setattr(auth, "refresh_codex_oauth_pure", _rejected)
    monkeypatch.setattr(auth_codex, "refresh_codex_oauth_pure", _rejected)
    monkeypatch.setattr(auth, "_import_codex_cli_tokens", lambda: dict(fresh))
    monkeypatch.setattr(auth_codex, "_import_codex_cli_tokens", lambda: dict(fresh))
    monkeypatch.setattr(auth, "_save_codex_tokens", lambda t, *a, **k: saved.update(t))
    monkeypatch.setattr(auth_codex, "_save_codex_tokens", lambda t, *a, **k: saved.update(t))

    out = _refresh_codex_auth_tokens(STALE, 20.0)

    assert out["access_token"] == "fresh-access"
    assert out["refresh_token"] == "fresh-refresh"
    # the recovered token was persisted to the Hermes auth store
    assert saved["access_token"] == "fresh-access"










def test_self_heals_missing_singleton_access_token_from_codex_cli(tmp_path, monkeypatch):
    """Exact cron failure path: Hermes auth has refresh_token but missing access_token."""
    hermes_home = tmp_path / "hermes"
    codex_home = tmp_path / "codex"
    hermes_home.mkdir()
    codex_home.mkdir()
    (hermes_home / "auth.json").write_text(json.dumps({
        "version": 1,
        "providers": {
            "openai-codex": {
                "tokens": {"refresh_token": "stale-refresh"},
                "last_refresh": "2026-06-01T00:00:00Z",
                "auth_mode": "chatgpt",
            },
        },
    }))
    (codex_home / "auth.json").write_text(json.dumps({
        "tokens": {
            "access_token": "fresh-access",
            "refresh_token": "fresh-refresh",
        },
    }))
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("CODEX_HOME", str(codex_home))

    resolved = resolve_codex_runtime_credentials()

    assert resolved["api_key"] == "fresh-access"
    assert resolved["source"] == "hermes-auth-store"
    stored = json.loads((hermes_home / "auth.json").read_text())
    tokens = stored["providers"]["openai-codex"]["tokens"]
    assert tokens["access_token"] == "fresh-access"
    assert tokens["refresh_token"] == "fresh-refresh"


def _seed_pool(hermes_home, access_token):
    """Write a HERMES_HOME auth.json whose ONLY Codex credential is one pool entry."""
    hermes_home.mkdir(parents=True, exist_ok=True)
    (hermes_home / "auth.json").write_text(json.dumps({
        "version": 1,
        "providers": {},  # force the pool fallback path
        "credential_pool": {
            "openai-codex": [
                {
                    "source": "device_code",
                    "auth_type": "oauth",
                    "access_token": access_token,
                    "refresh_token": "stale-refresh",
                    "last_status": "ok",
                },
            ],
        },
    }), encoding="utf-8")


def test_pool_self_heals_when_access_token_expiring(tmp_path, monkeypatch):
    """Pool fallback recovers from ~/.codex/auth.json when the entry's access_token is expiring.

    The singleton path self-heals in ``_refresh_codex_auth_tokens`` (refresh
    rejected) and ``resolve_codex_runtime_credentials`` (missing/malformed), but
    ``_pool_codex_access_token`` historically returned the first non-empty
    pool entry with no expiry check, so an aged access_token surfaced as a
    bare HTTP 401. Mirror the singleton recovery here: when the selected
    entry's access_token is within the refresh skew window of expiry, adopt
    a fresh pair from the Codex CLI instead of returning the stale JWT.
    """
    hermes_home = tmp_path / "hermes"
    codex_home = tmp_path / "codex"
    _seed_pool(hermes_home, _jwt_with_exp(int(time.time()) + 30))  # within 120s skew
    codex_home.mkdir(parents=True, exist_ok=True)
    (codex_home / "auth.json").write_text(json.dumps({
        "tokens": {
            "access_token": "fresh-access",
            "refresh_token": "fresh-refresh",
        },
    }), encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("CODEX_HOME", str(codex_home))

    saved = {}
    monkeypatch.setattr(auth, "_save_codex_tokens", lambda t, *a, **k: saved.update(t))

    token = _pool_codex_access_token()

    assert token == "fresh-access"
    # the recovered pair was persisted to the Hermes auth store
    assert saved.get("access_token") == "fresh-access"
    assert saved.get("refresh_token") == "fresh-refresh"


def test_pool_returns_stale_token_when_not_expiring(tmp_path, monkeypatch):
    """Pool fallback returns the entry token as-is when it is not near expiry.

    Ensures the self-heal only fires within the refresh skew window — a token
    with plenty of life left must NOT trigger a CLI import on every call.
    """
    hermes_home = tmp_path / "hermes"
    fresh_jwt = _jwt_with_exp(int(time.time()) + 3600)  # well outside skew
    _seed_pool(hermes_home, fresh_jwt)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    import_calls = []
    monkeypatch.setattr(
        auth_codex, "_recover_codex_tokens_from_cli",
        lambda *a, **k: import_calls.append((a, k)) or None,
    )

    token = _pool_codex_access_token()

    assert token == fresh_jwt
    assert import_calls == []  # no recovery attempted


def test_pool_returns_stale_token_when_cli_has_no_fresh_tokens(tmp_path, monkeypatch):
    """Pool fallback falls back to the stale entry when the CLI import yields nothing.

    If ``~/.codex/auth.json`` is itself expired or absent, recovery returns
    None and the pool entry's (expired) token is returned as-is so the caller
    raises the original AuthError rather than a confusing empty-credential one.
    """
    hermes_home = tmp_path / "hermes"
    codex_home = tmp_path / "codex"
    expiring_jwt = _jwt_with_exp(int(time.time()) + 30)
    _seed_pool(hermes_home, expiring_jwt)
    codex_home.mkdir(parents=True, exist_ok=True)
    # Codex CLI auth.json is stale (expired) → _import_codex_cli_tokens returns None
    (codex_home / "auth.json").write_text(json.dumps({
        "tokens": {
            "access_token": _jwt_with_exp(int(time.time()) - 10),
            "refresh_token": "dead-refresh",
        },
    }), encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("CODEX_HOME", str(codex_home))

    token = _pool_codex_access_token()

    # Recovery yielded nothing → fall through to the stale entry token.
    assert token == expiring_jwt


def test_pool_falls_back_to_entry_token_when_recovery_raises(tmp_path, monkeypatch):
    """Pool fallback returns the entry token when recovery raises.

    ``_recover_codex_tokens_from_cli`` calls ``_save_codex_tokens``, which
    acquires the auth-store lock and can raise (lock timeout, disk error).
    Such a failure must NOT swallow the still-present entry token — a token
    within the skew window may have up to ``CODEX_ACCESS_TOKEN_REFRESH_SKEW_SECONDS``
    of life left and work fine on the wire. The recovery attempt is isolated
    from the outer lookup so the entry token is returned on failure.
    """
    hermes_home = tmp_path / "hermes"
    expiring_jwt = _jwt_with_exp(int(time.time()) + 30)  # within 120s skew
    _seed_pool(hermes_home, expiring_jwt)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    def _boom(*_a, **_k):
        raise RuntimeError("auth store lock timed out")

    monkeypatch.setattr(auth_codex, "_recover_codex_tokens_from_cli", _boom)

    token = _pool_codex_access_token()

    # Recovery raised → fall through to the entry token, not "".
    assert token == expiring_jwt


