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




def test_opt_out_never_adopts_codex_cli_login(tmp_path, monkeypatch):
    """``auth.adopt_external_logins: false`` (#113023): the Codex CLI pair is a single-use refresh-token
    family the user did not hand to Hermes. Both automatic recovery paths must leave it (and Hermes' own
    auth.json) untouched and surface the real error instead."""
    hermes_home = tmp_path / "hermes"
    codex_home = tmp_path / "codex"
    hermes_home.mkdir()
    codex_home.mkdir()
    (hermes_home / "config.yaml").write_text("auth:\n  adopt_external_logins: false\n")
    hermes_auth = {"version": 1, "providers": {"openai-codex": {
        "tokens": {"refresh_token": "stale-refresh"}, "auth_mode": "chatgpt"}}}
    (hermes_home / "auth.json").write_text(json.dumps(hermes_auth))
    (codex_home / "auth.json").write_text(json.dumps({
        "tokens": {"access_token": "fresh-access", "refresh_token": "fresh-refresh"}}))
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("CODEX_HOME", str(codex_home))

    with pytest.raises(AuthError) as info:
        resolve_codex_runtime_credentials()
    assert info.value.code == "codex_auth_missing_access_token"

    def _rejected(*_a, **_k):
        raise AuthError("bad", provider="openai-codex", code="invalid_grant", relogin_required=True)

    monkeypatch.setattr(auth_codex, "refresh_codex_oauth_pure", _rejected)
    with pytest.raises(AuthError) as info:
        _refresh_codex_auth_tokens(dict(STALE), 5.0)
    assert info.value.relogin_required  # surfaced, not papered over with the CLI pair
    assert json.loads((hermes_home / "auth.json").read_text()) == hermes_auth


def _codex_jwt(account_id: str, sub: str = "user-1") -> str:
    def _b64(d):
        return base64.urlsafe_b64encode(json.dumps(d).encode()).rstrip(b"=").decode()
    claims = {"sub": sub, "exp": int(time.time()) + 3600,
              "https://api.openai.com/auth": {"chatgpt_account_id": account_id}}
    return f"{_b64({'alg': 'none'})}.{_b64(claims)}.sig"


def _seed_homes(tmp_path, monkeypatch, hermes_tokens, cli_tokens):
    hermes_home, codex_home = tmp_path / "hermes", tmp_path / "codex"
    hermes_home.mkdir()
    codex_home.mkdir()
    (hermes_home / "auth.json").write_text(json.dumps({"version": 1, "providers": {"openai-codex": {
        "tokens": hermes_tokens, "auth_mode": "chatgpt"}}}))
    (codex_home / "auth.json").write_text(json.dumps({"tokens": cli_tokens}))
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("CODEX_HOME", str(codex_home))
    return hermes_home / "auth.json"


def test_recovery_refuses_codex_cli_login_from_another_workspace(tmp_path, monkeypatch, caplog):
    """#73667: a Codex Desktop/CLI login into ANOTHER ChatGPT workspace must not silently replace the
    Hermes credential it is supposed to repair — the store stays byte-identical and the log says why."""
    personal, team = _codex_jwt("acct-personal"), _codex_jwt("acct-team")
    auth_file = _seed_homes(tmp_path, monkeypatch, {"access_token": personal},
                            {"access_token": team, "refresh_token": "rt-team"})
    before = auth_file.read_bytes()

    with caplog.at_level("WARNING", logger="hermes_cli.auth"), pytest.raises(AuthError) as info:
        resolve_codex_runtime_credentials(refresh_if_expiring=False)

    assert info.value.code == "codex_auth_missing_refresh_token"
    assert auth_file.read_bytes() == before
    assert "different ChatGPT workspace" in caplog.text and team not in caplog.text


def test_recovery_does_not_overwrite_concurrent_reauth(tmp_path, monkeypatch):
    """#73667 (review): an explicit re-auth landing between the failed read and the recovery save must
    win — the save is a compare-and-swap on the observed access_token, not a blind overwrite."""
    personal, reauthed = _codex_jwt("acct-personal"), _codex_jwt("acct-personal", sub="user-1-fresh")
    auth_file = _seed_homes(tmp_path, monkeypatch, {"access_token": personal},
                            {"access_token": _codex_jwt("acct-personal"), "refresh_token": "rt-cli"})
    real_import = auth_codex._import_codex_cli_tokens

    def _import_racing_with_reauth():
        auth_codex._save_codex_tokens({"access_token": reauthed, "refresh_token": "rt-reauthed"})
        return real_import()

    monkeypatch.setattr(auth, "_import_codex_cli_tokens", _import_racing_with_reauth)
    monkeypatch.setattr(auth_codex, "_import_codex_cli_tokens", _import_racing_with_reauth)

    with pytest.raises(AuthError):
        resolve_codex_runtime_credentials(refresh_if_expiring=False)

    tokens = json.loads(auth_file.read_text())["providers"]["openai-codex"]["tokens"]
    assert tokens == {"access_token": reauthed, "refresh_token": "rt-reauthed"}
