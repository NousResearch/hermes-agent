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

import json

import pytest

import hermes_cli.auth as auth
from hermes_cli.auth import AuthError, _refresh_codex_auth_tokens, resolve_codex_runtime_credentials

STALE = {"access_token": "stale-access", "refresh_token": "stale-refresh"}


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
    monkeypatch.setattr(auth, "_import_codex_cli_tokens", lambda: dict(fresh))
    monkeypatch.setattr(auth, "_save_codex_tokens", lambda t, *a, **k: saved.update(t))

    out = _refresh_codex_auth_tokens(STALE, 20.0)

    assert out["access_token"] == "fresh-access"
    assert out["refresh_token"] == "fresh-refresh"
    # the recovered token was persisted to the Hermes auth store
    assert saved["access_token"] == "fresh-access"


def test_does_not_self_heal_on_rate_limit(monkeypatch):
    """429 quota keeps relogin_required=False — token still valid, must NOT reimport."""
    import_calls = {"n": 0}

    def _rate_limited(*_a, **_k):
        raise AuthError(
            "quota exhausted",
            provider="openai-codex",
            code="codex_rate_limited",
            relogin_required=False,
        )

    def _import_spy():
        import_calls["n"] += 1
        return {"access_token": "should-not-be-used"}

    monkeypatch.setattr(auth, "refresh_codex_oauth_pure", _rate_limited)
    monkeypatch.setattr(auth, "_import_codex_cli_tokens", _import_spy)
    monkeypatch.setattr(auth, "_save_codex_tokens", lambda *a, **k: None)

    with pytest.raises(AuthError) as ei:
        _refresh_codex_auth_tokens(STALE, 20.0)

    assert ei.value.code == "codex_rate_limited"
    assert import_calls["n"] == 0  # never touched ~/.codex on a transient failure


def test_reraises_when_codex_cli_token_absent(monkeypatch):
    """relogin-required but ~/.codex unavailable/expired → propagate original error."""

    def _reused(*_a, **_k):
        raise AuthError(
            "refresh token reused",
            provider="openai-codex",
            code="refresh_token_reused",
            relogin_required=True,
        )

    monkeypatch.setattr(auth, "refresh_codex_oauth_pure", _reused)
    monkeypatch.setattr(auth, "_import_codex_cli_tokens", lambda: None)
    monkeypatch.setattr(auth, "_save_codex_tokens", lambda *a, **k: None)

    with pytest.raises(AuthError) as ei:
        _refresh_codex_auth_tokens(STALE, 20.0)

    assert ei.value.code == "refresh_token_reused"


def test_happy_path_unchanged(monkeypatch):
    """Normal refresh succeeds → rotated tokens persisted, ~/.codex never consulted."""
    saved = {}
    import_calls = {"n": 0}

    def _import_spy():
        import_calls["n"] += 1
        return None

    monkeypatch.setattr(
        auth,
        "refresh_codex_oauth_pure",
        lambda *a, **k: {"access_token": "rotated", "refresh_token": "rotated-r"},
    )
    monkeypatch.setattr(auth, "_import_codex_cli_tokens", _import_spy)
    monkeypatch.setattr(auth, "_save_codex_tokens", lambda t, *a, **k: saved.update(t))

    out = _refresh_codex_auth_tokens({"access_token": "a", "refresh_token": "b"}, 20.0)

    assert out["access_token"] == "rotated"
    assert out["refresh_token"] == "rotated-r"
    assert saved["access_token"] == "rotated"
    assert import_calls["n"] == 0  # happy path must not consult ~/.codex


def test_reraises_when_imported_token_lacks_refresh_token(monkeypatch):
    """relogin-required, but ~/.codex returns an access_token with NO refresh_token →
    re-raise rather than persist a half-token that would break the next refresh."""
    saved = {}

    def _rejected(*_a, **_k):
        raise AuthError(
            "refresh token rejected",
            provider="openai-codex",
            code="invalid_grant",
            relogin_required=True,
        )

    monkeypatch.setattr(auth, "refresh_codex_oauth_pure", _rejected)
    monkeypatch.setattr(auth, "_import_codex_cli_tokens", lambda: {"access_token": "fresh-only"})
    monkeypatch.setattr(auth, "_save_codex_tokens", lambda t, *a, **k: saved.update(t))

    with pytest.raises(AuthError) as ei:
        _refresh_codex_auth_tokens(STALE, 20.0)

    assert ei.value.code == "invalid_grant"
    assert saved == {}  # nothing was persisted


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


def test_missing_singleton_access_token_reraises_when_codex_cli_half_token(tmp_path, monkeypatch):
    """Missing access_token must not be masked by a malformed Codex CLI import."""
    hermes_home = tmp_path / "hermes"
    codex_home = tmp_path / "codex"
    hermes_home.mkdir()
    codex_home.mkdir()
    (hermes_home / "auth.json").write_text(json.dumps({
        "version": 1,
        "providers": {
            "openai-codex": {
                "tokens": {"refresh_token": "stale-refresh"},
                "auth_mode": "chatgpt",
            },
        },
    }))
    (codex_home / "auth.json").write_text(json.dumps({
        "tokens": {"access_token": "fresh-only"},
    }))
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("CODEX_HOME", str(codex_home))

    with pytest.raises(AuthError) as ei:
        resolve_codex_runtime_credentials()

    assert ei.value.code == "codex_auth_missing_access_token"


class TestResolveCodexReadonly:
    """Issue #68004: status/doctor must stay side-effect-free.

    ``resolve_codex_runtime_credentials(read_only=True)`` must never recover
    tokens from the Codex CLI nor refresh an expiring token — otherwise a
    diagnostic read would reach a second credential store, perform a
    network-backed refresh, and persist auth state.
    """

    def test_read_only_does_not_recover_tokens(self, monkeypatch):
        """A relogin-required read error must NOT trigger CLI recovery when
        read_only=True (it re-raises unchanged instead)."""
        import hermes_cli.auth as auth

        recover_calls = []
        original = auth._recover_codex_tokens_from_cli

        def _spy_recover(reason):
            recover_calls.append(reason)
            return original(reason)

        monkeypatch.setattr(auth, "_recover_codex_tokens_from_cli", _spy_recover)
        monkeypatch.setattr(
            auth,
            "_read_codex_tokens",
            lambda *_a, **_k: (_ for _ in ()).throw(
                AuthError(
                    "missing access token",
                    provider="openai-codex",
                    code="codex_auth_missing_access_token",
                    relogin_required=True,
                )
            ),
        )
        monkeypatch.setattr(auth, "_pool_codex_access_token", lambda: None)

        with pytest.raises(AuthError) as ei:
            resolve_codex_runtime_credentials(read_only=True)

        assert ei.value.code == "codex_auth_missing_access_token"
        assert recover_calls == [], "read_only must not recover tokens"

    def test_read_only_does_not_refresh_expiring_token(self, monkeypatch):
        """An expiring token must be returned as-is when read_only=True — no
        network refresh, no persist.  The same call without read_only WOULD
        refresh, proving the gate actually suppresses the mutation."""
        import hermes_cli.auth as auth
        import base64, json as _json

        # Build a JWT whose exp is in the past so _codex_access_token_is_expiring
        # reports True.
        def _b64(d):
            return base64.urlsafe_b64encode(_json.dumps(d).encode()).rstrip(b"=").decode()

        expired_jwt = f"{_b64({'alg': 'none'})}.{_b64({'exp': 1})}.sig"

        refresh_calls = []

        monkeypatch.setattr(
            auth,
            "_read_codex_tokens",
            lambda *_a, **_k: {
                "tokens": {
                    "access_token": expired_jwt,
                    "refresh_token": "r",
                },
                "last_refresh": None,
            },
        )
        monkeypatch.setattr(
            auth, "_refresh_codex_auth_tokens",
            lambda tokens, *_a: refresh_calls.append(tokens) or dict(tokens, access_token="refreshed"),
        )

        creds = resolve_codex_runtime_credentials(read_only=True)
        assert creds["api_key"] == expired_jwt
        assert refresh_calls == [], "read_only must not refresh expiring token"

        # Contrast: without read_only the same token triggers refresh (the real
        # helper persists internally), proving the gate suppresses the mutation.
        refreshed = resolve_codex_runtime_credentials(read_only=False)
        assert refreshed["api_key"] == "refreshed"
        assert len(refresh_calls) == 1, "non-read-only must refresh expiring token"
