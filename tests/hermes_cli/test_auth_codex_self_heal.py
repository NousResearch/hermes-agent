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

import pytest

import hermes_cli.auth as auth
from hermes_cli.auth import AuthError, _refresh_codex_auth_tokens

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
