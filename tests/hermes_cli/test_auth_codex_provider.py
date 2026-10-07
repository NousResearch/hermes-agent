"""Tests for Codex auth — tokens stored in Hermes auth store (~/.hermes/auth.json)."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli.auth import (
    AuthError,
    DEFAULT_CODEX_BASE_URL,
    _read_codex_tokens,
    _save_codex_tokens,
    refresh_codex_oauth_pure,
    resolve_codex_runtime_credentials,
)


def _setup_hermes_auth(hermes_home: Path, *, access_token: str = "access", refresh_token: str = "refresh"):
    """Write Codex tokens into the Hermes auth store."""
    hermes_home.mkdir(parents=True, exist_ok=True)
    auth_store = {
        "version": 1,
        "active_provider": "openai-codex",
        "providers": {
            "openai-codex": {
                "tokens": {
                    "access_token": access_token,
                    "refresh_token": refresh_token,
                },
                "last_refresh": "2026-02-26T00:00:00Z",
                "auth_mode": "chatgpt",
            },
        },
    }
    auth_file = hermes_home / "auth.json"
    auth_file.write_text(json.dumps(auth_store, indent=2))
    return auth_file


def test_resolve_codex_runtime_credentials_missing_access_token(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes"
    _setup_hermes_auth(hermes_home, access_token="")
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "missing-codex"))

    with pytest.raises(AuthError) as exc:
        resolve_codex_runtime_credentials()
    assert exc.value.code == "codex_auth_missing_access_token"
    assert exc.value.relogin_required is True


def test_resolve_codex_runtime_credentials_falls_back_to_pool_when_singleton_empty(tmp_path, monkeypatch):
    """Regression for #32992 — chat path returns 401 when singleton is empty but pool has creds.

    The chat path historically went through ``resolve_codex_runtime_credentials`` which
    only consulted ``providers.openai-codex.tokens`` and raised ``AuthError`` when that
    was empty.  The auxiliary path went through ``_read_codex_access_token`` which
    checks the pool first.  Users with creds only in the pool (manual seed, partial
    re-auth, restore from backup) hit a bare HTTP 401 on chat but worked fine on
    auxiliary calls.  The fallback closes that divergence.
    """
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    # Singleton: empty tokens (would normally raise AuthError).
    # Pool: valid access_token.
    auth_store = {
        "version": 1,
        "providers": {},  # no openai-codex singleton at all
        "credential_pool": {
            "openai-codex": [
                {
                    "source": "device_code",
                    "access_token": "pool-fallback-token",
                    "refresh_token": "pool-refresh",
                    "last_status": "ok",
                    "auth_type": "oauth",
                },
            ],
        },
    }
    (hermes_home / "auth.json").write_text(json.dumps(auth_store))
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    resolved = resolve_codex_runtime_credentials()
    assert resolved["api_key"] == "pool-fallback-token"
    assert resolved["source"] == "credential_pool"
    assert resolved["base_url"]  # default codex backend URL


def test_save_codex_tokens_syncs_credential_pool(tmp_path, monkeypatch):
    """Re-auth must update the credential_pool device_code entry, not just providers.

    Regression for #33000: the runtime selects from credential_pool, so a
    re-auth that only refreshed providers.openai-codex.tokens left the pool
    holding a consumed refresh token and stale error markers, causing an
    immediate 401 token_invalidated on the next request.
    """
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    (hermes_home / "auth.json").write_text(json.dumps({
        "version": 1,
        "providers": {
            "openai-codex": {
                "tokens": {"access_token": "old-at", "refresh_token": "old-rt"},
                "last_refresh": "2026-01-01T00:00:00Z",
                "auth_mode": "chatgpt",
            },
        },
        "credential_pool": {
            "openai-codex": [
                {
                    "id": "abc123",
                    "source": "device_code",
                    "auth_type": "oauth",
                    "access_token": "old-at",
                    "refresh_token": "old-rt",
                    "last_status": "exhausted",
                    "last_error_code": 401,
                    "last_error_reason": "token_invalidated",
                    "last_error_reset_at": 9999999999,
                },
                {
                    "id": "manual1",
                    "source": "manual:codex",
                    "auth_type": "oauth",
                    "access_token": "manual-at",
                    "refresh_token": "manual-rt",
                },
            ],
        },
    }))
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    _save_codex_tokens({"access_token": "new-at", "refresh_token": "new-rt"},
                       last_refresh="2026-05-27T00:00:00Z")

    auth = json.loads((hermes_home / "auth.json").read_text())
    pool = auth["credential_pool"]["openai-codex"]
    seeded = next(e for e in pool if e["source"] == "device_code")
    assert seeded["access_token"] == "new-at"
    assert seeded["refresh_token"] == "new-rt"
    assert seeded["last_refresh"] == "2026-05-27T00:00:00Z"
    assert seeded["last_status"] is None
    assert seeded["last_error_code"] is None
    assert seeded["last_error_reason"] is None
    assert seeded["last_error_reset_at"] is None

    # Manual entries are independent credentials and must not be overwritten.
    manual = next(e for e in pool if e["source"] == "manual:codex")
    assert manual["access_token"] == "manual-at"
    assert manual["refresh_token"] == "manual-rt"

    # Provider singleton is updated too.
    assert auth["providers"]["openai-codex"]["tokens"]["access_token"] == "new-at"


def test_save_codex_tokens_syncs_manual_device_code_entries(tmp_path, monkeypatch):
    """Re-auth must refresh ``manual:device_code`` entries that are true
    aliases of the singleton, while leaving INDEPENDENT entries alone.

    Original regression for #33538: a user who hit #33000 before the #33164
    fix landed would have run ``hermes auth add openai-codex`` as a
    workaround, leaving a pool entry with ``source="manual:device_code"``.
    On every subsequent re-auth via setup/model picker, the singleton-seeded
    ``device_code`` entry got refreshed but the ``manual:device_code`` entry
    stayed stale, recreating the same 401 token_invalidated symptom that
    #33164 was supposed to fix.

    Narrowed for #39236: the original fix treated every ``manual:device_code``
    entry as a singleton-alias and refreshed them all, which silently
    clobbered independent accounts added via ``hermes auth add openai-codex``.
    The current behavior refreshes only entries whose access_token matches
    the *previous* singleton access_token (true legacy aliases), and leaves
    distinct-token entries alone (independent accounts).
    """
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    (hermes_home / "auth.json").write_text(json.dumps({
        "version": 1,
        "providers": {
            "openai-codex": {
                "tokens": {"access_token": "old-at", "refresh_token": "old-rt"},
                "last_refresh": "2026-01-01T00:00:00Z",
                "auth_mode": "chatgpt",
            },
        },
        "credential_pool": {
            "openai-codex": [
                {
                    "id": "seeded",
                    "source": "device_code",
                    "auth_type": "oauth",
                    "access_token": "old-at",
                    "refresh_token": "old-rt",
                },
                # Legacy alias from the #33000 workaround era — its tokens
                # match the singleton, so it is a true alias and SHOULD be
                # refreshed (preserves #33538 behavior).
                {
                    "id": "legacy-alias",
                    "source": "manual:device_code",
                    "auth_type": "oauth",
                    "access_token": "old-at",
                    "refresh_token": "old-rt",
                    "last_status": "exhausted",
                    "last_error_code": 401,
                    "last_error_reason": "token_invalidated",
                },
                # Independent account from `hermes auth add openai-codex` —
                # its tokens are distinct from the singleton.  Must NOT be
                # overwritten by a re-auth that targeted a different account
                # (#39236).
                {
                    "id": "independent",
                    "source": "manual:device_code",
                    "auth_type": "oauth",
                    "access_token": "independent-at",
                    "refresh_token": "independent-rt",
                },
                {
                    "id": "api-key",
                    "source": "manual:api_key",
                    "auth_type": "api_key",
                    "access_token": "user-api-key",
                },
            ],
        },
    }))
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    _save_codex_tokens({"access_token": "fresh-at", "refresh_token": "fresh-rt"},
                       last_refresh="2026-05-28T00:00:00Z")

    auth = json.loads((hermes_home / "auth.json").read_text())
    pool = auth["credential_pool"]["openai-codex"]

    # Singleton-seeded device_code entry: refreshed and error markers cleared.
    seeded = next(e for e in pool if e["id"] == "seeded")
    assert seeded["access_token"] == "fresh-at"
    assert seeded["refresh_token"] == "fresh-rt"

    # Legacy alias (tokens matched previous singleton): ALSO refreshed.
    legacy = next(e for e in pool if e["id"] == "legacy-alias")
    assert legacy["access_token"] == "fresh-at"
    assert legacy["refresh_token"] == "fresh-rt"
    assert legacy["last_refresh"] == "2026-05-28T00:00:00Z"
    assert legacy["last_status"] is None
    assert legacy["last_error_code"] is None
    assert legacy["last_error_reason"] is None

    # Independent manual:device_code entry: NOT overwritten (#39236).
    independent = next(e for e in pool if e["id"] == "independent")
    assert independent["access_token"] == "independent-at"
    assert independent["refresh_token"] == "independent-rt"

    # manual:api_key entry: untouched — independent credential.
    api_key = next(e for e in pool if e["source"] == "manual:api_key")
    assert api_key["access_token"] == "user-api-key"
    assert "refresh_token" not in api_key or api_key.get("refresh_token") is None


def test_save_codex_tokens_clears_error_markers_only_on_refreshed_entries(tmp_path, monkeypatch):
    """Error markers must be cleared only on entries that were actually
    refreshed by this re-auth.  Independent ``manual:device_code`` entries
    with their own stale-error markers must be left alone (their stale state
    is not the current re-auth's business).
    """
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    (hermes_home / "auth.json").write_text(json.dumps({
        "version": 1,
        "providers": {
            "openai-codex": {
                "tokens": {"access_token": "acctA-at", "refresh_token": "acctA-rt"},
                "auth_mode": "chatgpt",
            },
        },
        "credential_pool": {
            "openai-codex": [
                {
                    "id": "seeded",
                    "source": "device_code",
                    "auth_type": "oauth",
                    "access_token": "acctA-at",
                    "refresh_token": "acctA-rt",
                    "last_status": "exhausted",
                    "last_error_code": 401,
                },
                {
                    "id": "acctB",
                    "source": "manual:device_code",
                    "auth_type": "oauth",
                    "access_token": "acctB-at",
                    "refresh_token": "acctB-rt",
                    "last_status": "exhausted",
                    "last_error_code": 429,
                    "last_error_reason": "quota_exhausted",
                },
            ],
        },
    }))
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    _save_codex_tokens(
        {"access_token": "fresh-at", "refresh_token": "fresh-rt"},
        last_refresh="2026-06-05T00:00:00Z",
    )

    auth = json.loads((hermes_home / "auth.json").read_text())
    pool = auth["credential_pool"]["openai-codex"]

    # Singleton: refreshed AND error markers cleared.
    seeded = next(e for e in pool if e["id"] == "seeded")
    assert seeded["access_token"] == "fresh-at"
    assert seeded["last_status"] is None
    assert seeded["last_error_code"] is None

    # Independent acctB: NOT refreshed AND error markers NOT cleared.
    # (Its 429 quota state belongs to acctB's own account, not acctA's re-auth.)
    acctB = next(e for e in pool if e["id"] == "acctB")
    assert acctB["access_token"] == "acctB-at"  # not overwritten
    assert acctB["last_status"] == "exhausted"  # not cleared
    assert acctB["last_error_code"] == 429
    assert acctB["last_error_reason"] == "quota_exhausted"


def test_codex_tokens_not_written_to_shared_file(tmp_path, monkeypatch):
    """Verify _save_codex_tokens writes only to Hermes auth store, not ~/.codex/."""
    hermes_home = tmp_path / "hermes"
    codex_home = tmp_path / "codex-cli"
    hermes_home.mkdir(parents=True, exist_ok=True)
    codex_home.mkdir(parents=True, exist_ok=True)

    (hermes_home / "auth.json").write_text(json.dumps({"version": 1, "providers": {}}))
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("CODEX_HOME", str(codex_home))

    _save_codex_tokens({"access_token": "hermes-at", "refresh_token": "hermes-rt"})

    # ~/.codex/auth.json should NOT exist — _save_codex_tokens only touches Hermes store
    assert not (codex_home / "auth.json").exists()

    # Hermes auth store should have the tokens
    data = _read_codex_tokens()
    assert data["tokens"]["access_token"] == "hermes-at"


def test_resolve_returns_hermes_auth_store_source(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes"
    _setup_hermes_auth(hermes_home)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    creds = resolve_codex_runtime_credentials()
    assert creds["source"] == "hermes-auth-store"
    assert creds["provider"] == "openai-codex"
    assert creds["base_url"] == DEFAULT_CODEX_BASE_URL


class _StubHTTPResponse:
    def __init__(self, status_code: int, payload, headers=None):
        self.status_code = status_code
        self._payload = payload
        self.headers = headers or {}
        self.text = json.dumps(payload) if isinstance(payload, (dict, list)) else str(payload)

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


class _StubHTTPClient:
    def __init__(self, response):
        self._response = response

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def post(self, *args, **kwargs):
        return self._response


def _patch_httpx(monkeypatch, response):
    def _factory(*args, **kwargs):
        return _StubHTTPClient(response)

    monkeypatch.setattr("hermes_cli.auth.httpx.Client", _factory)


def test_refresh_429_classified_as_quota_not_auth_failure(monkeypatch):
    """429 from the token endpoint is a usage-quota cap, not an auth failure.

    Regression test for #32790: must NOT force relogin and must carry the
    dedicated rate-limit code so callers surface a "retry later" notice rather
    than a misleading "run hermes auth".
    """
    from hermes_cli.auth import (
        CODEX_RATE_LIMITED_CODE,
        format_auth_error,
        is_rate_limited_auth_error,
    )

    response = _StubHTTPResponse(
        429,
        {"error": {"message": "You hit your usage limit.", "code": "usage_limit_reached"}},
        headers={"retry-after": "120"},
    )
    _patch_httpx(monkeypatch, response)

    with pytest.raises(AuthError) as exc_info:
        refresh_codex_oauth_pure("a-tok", "r-tok")

    err = exc_info.value
    assert err.code == CODEX_RATE_LIMITED_CODE
    assert err.relogin_required is False
    assert is_rate_limited_auth_error(err) is True
    # User-facing copy must not tell the operator to re-authenticate.
    rendered = format_auth_error(err)
    assert "re-authenticate" not in rendered
    assert "hermes auth" not in rendered


def test_refresh_429_without_retry_after_header(monkeypatch):
    """429 without a Retry-After header still classifies as quota, no relogin."""
    from hermes_cli.auth import CODEX_RATE_LIMITED_CODE

    response = _StubHTTPResponse(429, {"error": "rate_limited"})
    _patch_httpx(monkeypatch, response)

    with pytest.raises(AuthError) as exc_info:
        refresh_codex_oauth_pure("a-tok", "r-tok")

    err = exc_info.value
    assert err.code == CODEX_RATE_LIMITED_CODE
    assert err.relogin_required is False


def test_pool_only_force_refresh_rotates_the_pool_entry(tmp_path, monkeypatch):
    """Pool-only setup (empty singleton): ``force_refresh`` must refresh the pool credential
    instead of handing back the same token the caller just got a 401 for."""
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    (hermes_home / "auth.json").write_text(json.dumps({
        "version": 1, "providers": {},
        "credential_pool": {"openai-codex": [{
            "source": "device_code", "access_token": "pool-revoked", "refresh_token": "pool-refresh",
            "last_status": "ok", "auth_type": "oauth"}]},
    }))
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    hints = []

    class Pool:
        def try_refresh_matching(self, api_key_hint=None, credential_id=None):
            hints.append(api_key_hint)
            return SimpleNamespace(runtime_api_key="pool-fresh")

    monkeypatch.setattr("agent.credential_pool.load_pool", lambda provider: Pool())

    resolved = resolve_codex_runtime_credentials(force_refresh=True)
    assert resolved["api_key"] == "pool-fresh"
    assert resolved["source"] == "credential_pool"
    assert hints == ["pool-revoked"]
