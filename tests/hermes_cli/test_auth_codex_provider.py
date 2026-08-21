"""Tests for Codex auth — tokens stored in Hermes auth store (~/.hermes/auth.json)."""

import json
import time
import base64
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli.auth import (
    AuthError,
    DEFAULT_CODEX_BASE_URL,
    PROVIDER_REGISTRY,
    _read_codex_tokens,
    _save_codex_tokens,
    _import_codex_cli_tokens,
    _recover_codex_tokens_from_cli,
    _login_openai_codex,
    refresh_codex_oauth_pure,
    resolve_codex_runtime_credentials,
    resolve_provider,
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


def _jwt_with_exp(exp_epoch: int) -> str:
    payload = {"exp": exp_epoch}
    encoded = base64.urlsafe_b64encode(json.dumps(payload).encode("utf-8")).rstrip(b"=").decode("utf-8")
    return f"h.{encoded}.s"






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


def test_save_codex_tokens_does_not_overwrite_independent_manual_entries(tmp_path, monkeypatch):
    """Re-auth must NOT overwrite ``manual:device_code`` entries that hold
    independent token material (different OpenAI/ChatGPT accounts).

    Regression for #39236: ``hermes auth add openai-codex`` for accounts B and C
    routes through ``_save_codex_tokens`` because the singleton path is the
    only Codex OAuth save flow.  The #33538 fix refreshed every
    ``manual:device_code`` entry on every re-auth, which works fine for the
    one-account/legacy-workaround case but silently overwrote distinct
    independent accounts with the latest-authenticated tokens (labels
    preserved, token material clobbered, status/quota readings then lie).

    The safe invariant: an entry is a singleton-alias only when its current
    access_token matches the *previous* singleton access_token.  Manual
    entries whose tokens never matched the singleton are independent accounts
    and must be left alone.
    """
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    (hermes_home / "auth.json").write_text(json.dumps({
        "version": 1,
        "providers": {
            "openai-codex": {
                # Old singleton tokens — represent "account A" which the user
                # logged in with via setup originally.
                "tokens": {"access_token": "acctA-at", "refresh_token": "acctA-rt"},
                "last_refresh": "2026-01-01T00:00:00Z",
                "auth_mode": "chatgpt",
                "label": "account-A",
            },
        },
        "credential_pool": {
            "openai-codex": [
                # The seeded singleton mirror of account A.
                {
                    "id": "seeded",
                    "label": "account-A",
                    "source": "device_code",
                    "auth_type": "oauth",
                    "access_token": "acctA-at",
                    "refresh_token": "acctA-rt",
                },
                # Two INDEPENDENT manual entries added later via
                # ``hermes auth add openai-codex`` (account B and account C).
                # Each has its OWN distinct token material, unrelated to the
                # singleton.
                {
                    "id": "acctB",
                    "label": "account-B",
                    "source": "manual:device_code",
                    "auth_type": "oauth",
                    "access_token": "acctB-at",
                    "refresh_token": "acctB-rt",
                },
                {
                    "id": "acctC",
                    "label": "account-C",
                    "source": "manual:device_code",
                    "auth_type": "oauth",
                    "access_token": "acctC-at",
                    "refresh_token": "acctC-rt",
                },
            ],
        },
    }))
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    # User re-authenticates account A — fresh device-code login produces new
    # tokens.  The legitimate update is the seeded singleton mirror; the
    # independent acctB/acctC entries must be untouched.
    _save_codex_tokens(
        {"access_token": "acctA-new-at", "refresh_token": "acctA-new-rt"},
        last_refresh="2026-06-05T00:00:00Z",
    )

    auth = json.loads((hermes_home / "auth.json").read_text())
    pool = auth["credential_pool"]["openai-codex"]

    # Singleton-seeded entry: refreshed (legitimate sync).
    seeded = next(e for e in pool if e["source"] == "device_code")
    assert seeded["access_token"] == "acctA-new-at"
    assert seeded["refresh_token"] == "acctA-new-rt"
    assert seeded["last_refresh"] == "2026-06-05T00:00:00Z"

    # acctB: INDEPENDENT entry — must NOT be overwritten.
    acctB = next(e for e in pool if e["id"] == "acctB")
    assert acctB["access_token"] == "acctB-at", (
        "acctB was clobbered by acctA re-auth (#39236 regression)"
    )
    assert acctB["refresh_token"] == "acctB-rt"

    # acctC: INDEPENDENT entry — must NOT be overwritten.
    acctC = next(e for e in pool if e["id"] == "acctC")
    assert acctC["access_token"] == "acctC-at", (
        "acctC was clobbered by acctA re-auth (#39236 regression)"
    )
    assert acctC["refresh_token"] == "acctC-rt"


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
    assert "retry after 120s" in str(err)
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
    assert "quota exhausted" in str(err).lower()


def test_is_rate_limited_auth_error_distinguishes_credential_errors():
    """Missing/expired credentials must NOT be treated as rate-limit errors."""
    from hermes_cli.auth import CODEX_RATE_LIMITED_CODE, is_rate_limited_auth_error

    rate_limited = AuthError(
        "quota", provider="openai-codex", code=CODEX_RATE_LIMITED_CODE, relogin_required=False
    )
    missing_creds = AuthError(
        "No Codex credentials stored.",
        provider="openai-codex",
        code="codex_auth_missing",
        relogin_required=True,
    )
    assert is_rate_limited_auth_error(rate_limited) is True
    assert is_rate_limited_auth_error(missing_creds) is False
    assert is_rate_limited_auth_error(ValueError("nope")) is False




class _FakeResp:
    def __init__(self, status_code, json_data=None, headers=None):
        self.status_code = status_code
        self._json = json_data or {}
        self.headers = headers or {}

    def json(self):
        return self._json


def _patch_httpx_post(monkeypatch, responses):
    """Patch hermes_cli.auth.httpx.Client so .post() returns queued responses."""
    seq = iter(responses)

    class _FakeClient:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def post(self, *args, **kwargs):
            return next(seq)

    monkeypatch.setattr("hermes_cli.auth.httpx.Client", lambda *a, **k: _FakeClient())


# ---------------------------------------------------------------------------
# #73667 / #73677 — locked-CAS recovery for Codex CLI token import
# ---------------------------------------------------------------------------

def _codex_jwt_with_account(account_id=None, exp_offset=3600):
    """Build a synthetic Codex OAuth JWT carrying a chatgpt_account_id claim.

    Mirrors the claim layout codex-rs places under the namespaced
    ``https://api.openai.com/auth`` object.  Pass ``account_id=None`` to omit
    the claim entirely (for the missing-identity compatibility cases).

    Each call includes a monotonically-incrementing ``iat`` claim so that
    successive calls under the same second produce distinct JWT strings —
    critical for CAS-race tests that compare token identity.
    """
    import itertools
    _codex_jwt_with_account._counter = getattr(_codex_jwt_with_account, "_counter", itertools.count())
    payload: dict = {"exp": int(time.time()) + exp_offset, "iat": next(_codex_jwt_with_account._counter)}
    if account_id is not None:
        payload["https://api.openai.com/auth"] = {"chatgpt_account_id": account_id}
    encoded = base64.urlsafe_b64encode(json.dumps(payload).encode("utf-8")).rstrip(b"=").decode("utf-8")
    return f"h.{encoded}.s"


def _seed_codex_store(hermes_home, access_token, refresh_token="rt-stored"):
    """Write a Codex provider singleton into a temp HERMES_HOME auth.json."""
    hermes_home.mkdir(parents=True, exist_ok=True)
    auth_store = {
        "version": 1,
        "providers": {
            "openai-codex": {
                "tokens": {
                    "access_token": access_token,
                    "refresh_token": refresh_token,
                },
                "last_refresh": "2026-07-28T00:00:00Z",
                "auth_mode": "chatgpt",
            },
        },
    }
    auth_file = hermes_home / "auth.json"
    auth_file.write_text(json.dumps(auth_store, indent=2))
    return auth_file


# --- _codex_chatgpt_account_id unit tests ---

def test_codex_chatgpt_account_id_extracts_claim():
    """Direct unit test for the identity-extraction helper (#73667)."""
    from hermes_cli.auth import _codex_chatgpt_account_id

    assert _codex_chatgpt_account_id(_codex_jwt_with_account("acct-1")) == "acct-1"
    assert _codex_chatgpt_account_id(_codex_jwt_with_account(None)) is None
    assert _codex_chatgpt_account_id("not-a-jwt") is None
    assert _codex_chatgpt_account_id(None) is None


# --- Workspace guard tests ---

def test_recover_codex_refuses_cross_workspace_mismatch(tmp_path, monkeypatch):
    """A Team-workspace import must NOT overwrite a Personal credential.

    After the fix it refuses, returns None, and leaves the auth store
    byte-for-byte unchanged.
    """
    hermes_home = tmp_path / "hermes"
    personal_jwt = _codex_jwt_with_account("acct-personal")
    team_jwt = _codex_jwt_with_account("acct-team")
    auth_file = _seed_codex_store(hermes_home, personal_jwt)
    before = auth_file.read_text()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    monkeypatch.setattr(
        "hermes_cli.auth._import_codex_cli_tokens",
        lambda: {"access_token": team_jwt, "refresh_token": "***"},
    )

    result = _recover_codex_tokens_from_cli(
        "refresh_token rejected: test",
        observed_access_token=personal_jwt,
    )

    assert result is None  # recovery refused
    assert auth_file.read_text() == before  # store byte-for-byte unchanged
    assert (
        json.loads(before)["providers"]["openai-codex"]["tokens"]["access_token"]
        == personal_jwt
    )


def test_recover_codex_allows_same_workspace(tmp_path, monkeypatch):
    """Same workspace id → recovery proceeds and persists the import."""
    hermes_home = tmp_path / "hermes"
    personal_jwt = _codex_jwt_with_account("acct-personal")
    refreshed_jwt = _codex_jwt_with_account("acct-personal")  # same workspace
    _seed_codex_store(hermes_home, personal_jwt)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    monkeypatch.setattr(
        "hermes_cli.auth._import_codex_cli_tokens",
        lambda: {"access_token": refreshed_jwt, "refresh_token": "***"},
    )

    result = _recover_codex_tokens_from_cli(
        "refresh_token rejected: test",
        observed_access_token=personal_jwt,
    )

    assert result is not None
    assert result["access_token"] == refreshed_jwt
    data = _read_codex_tokens()
    assert data["tokens"]["access_token"] == refreshed_jwt


def test_recover_codex_allows_when_imported_lacks_identity(tmp_path, monkeypatch):
    """Imported token with no chatgpt_account_id → compatibility allow."""
    hermes_home = tmp_path / "hermes"
    personal_jwt = _codex_jwt_with_account("acct-personal")
    no_id_jwt = _codex_jwt_with_account(None)
    _seed_codex_store(hermes_home, personal_jwt)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    monkeypatch.setattr(
        "hermes_cli.auth._import_codex_cli_tokens",
        lambda: {"access_token": no_id_jwt, "refresh_token": "***"},
    )

    assert _recover_codex_tokens_from_cli(
        "test", observed_access_token=personal_jwt,
    ) is not None


def test_recover_codex_allows_when_store_lacks_identity(tmp_path, monkeypatch):
    """Stored token with no chatgpt_account_id → compatibility allow."""
    hermes_home = tmp_path / "hermes"
    no_id_jwt = _codex_jwt_with_account(None)
    team_jwt = _codex_jwt_with_account("acct-team")
    _seed_codex_store(hermes_home, no_id_jwt)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    monkeypatch.setattr(
        "hermes_cli.auth._import_codex_cli_tokens",
        lambda: {"access_token": team_jwt, "refresh_token": "***"},
    )

    assert _recover_codex_tokens_from_cli(
        "test", observed_access_token=no_id_jwt,
    ) is not None


def test_recover_codex_allows_into_empty_store(tmp_path, monkeypatch):
    """No codex provider in store → recovery proceeds (missing-token path)."""
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    (hermes_home / "auth.json").write_text(json.dumps({"version": 1, "providers": {}}))
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    team_jwt = _codex_jwt_with_account("acct-team")
    monkeypatch.setattr(
        "hermes_cli.auth._import_codex_cli_tokens",
        lambda: {"access_token": team_jwt, "refresh_token": "***"},
    )

    assert _recover_codex_tokens_from_cli(
        "test", observed_access_token=None,
    ) is not None


# --- CAS-race regression tests ---

def test_recover_codex_cas_skips_when_token_changed_same_workspace(tmp_path, monkeypatch):
    """Existing-token path: concurrent same-workspace reauth must be preserved.

    Scenario:
    1. Store has token A (workspace=personal).
    2. Caller observes token A under lock, decides recovery is needed.
    3. Before recovery commits, a concurrent ``hermes auth`` writes token B
       (same workspace, different access_token) into the store.
    4. Recovery is called with ``observed_access_token=A``.
    5. Under the reacquired lock, the CAS read sees token B ≠ A.
    6. Recovery is skipped; token B is preserved.
    """
    hermes_home = tmp_path / "hermes"
    token_a = _codex_jwt_with_account("acct-personal", exp_offset=7200)
    token_b = _codex_jwt_with_account("acct-personal", exp_offset=7200)
    _seed_codex_store(hermes_home, token_a)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    # Simulate concurrent reauth: before calling recovery, write token B
    # directly into the store (same workspace, different token).
    _seed_codex_store(hermes_home, token_b)

    token_c = _codex_jwt_with_account("acct-personal", exp_offset=7200)
    monkeypatch.setattr(
        "hermes_cli.auth._import_codex_cli_tokens",
        lambda: {"access_token": token_c, "refresh_token": "***"},
    )

    # Recovery called with the *original* observation (token A)
    result = _recover_codex_tokens_from_cli(
        "refresh_token rejected: test",
        observed_access_token=token_a,
    )

    # CAS check: stored token B ≠ observed token A → recovery skipped
    # Recovery returns the current valid state (token B)
    assert result is not None
    assert result["access_token"] == token_b
    # Token B must still be in the store (not overwritten by token C)
    data = _read_codex_tokens()
    assert data["tokens"]["access_token"] == token_b


def test_recover_codex_cas_skips_when_token_changed_different_workspace(tmp_path, monkeypatch):
    """Existing-token path: concurrent different-workspace reauth survives.

    Scenario:
    1. Store has token A (workspace=personal).
    2. Caller observes token A.
    3. Concurrent reauth writes token B (workspace=team) into the store.
    4. Recovery imports token C (workspace=personal, matching A).
    5. CAS check: stored B ≠ observed A → recovery skipped.
    6. Token B is preserved (the concurrent reauth survives).
    """
    hermes_home = tmp_path / "hermes"
    token_a = _codex_jwt_with_account("acct-personal", exp_offset=7200)
    token_b = _codex_jwt_with_account("acct-team", exp_offset=7200)
    _seed_codex_store(hermes_home, token_a)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    # Simulate concurrent reauth with a different workspace
    _seed_codex_store(hermes_home, token_b)

    token_c = _codex_jwt_with_account("acct-personal", exp_offset=7200)
    monkeypatch.setattr(
        "hermes_cli.auth._import_codex_cli_tokens",
        lambda: {"access_token": token_c, "refresh_token": "***"},
    )

    result = _recover_codex_tokens_from_cli(
        "refresh_token rejected: test",
        observed_access_token=token_a,
    )

    # CAS check fails first (store changed), so workspace guard is never reached
    # Recovery returns the current valid state (the concurrent reauth preserves)
    assert result is not None
    assert result["access_token"] == token_b
    # Concurrent reauth (token B) is preserved
    data = _read_codex_tokens()
    assert data["tokens"]["access_token"] == token_b


def test_recover_codex_cas_preserves_concurrent_reauth_missing_token_path(tmp_path, monkeypatch):
    """Missing-token/None path: concurrent reauth must survive recovery.

    Scenario:
    1. Store has NO Codex tokens (empty providers).
    2. Caller observes None, decides recovery is needed.
    3. Before recovery commits, a concurrent reauth populates token X.
    4. Recovery is called with ``observed_access_token=None``.
    5. CAS check: stored X ≠ None → change detected.
    6. Recovery returns current valid state (token X); never overwrites it.
    """
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    # Initial state: empty store (no Codex provider)
    (hermes_home / "auth.json").write_text(json.dumps({"version": 1, "providers": {}}))
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    # Simulate concurrent reauth: write a valid token into the store
    token_x = _codex_jwt_with_account("acct-personal", exp_offset=7200)
    _seed_codex_store(hermes_home, token_x)

    token_y = _codex_jwt_with_account("acct-personal", exp_offset=7200)
    monkeypatch.setattr(
        "hermes_cli.auth._import_codex_cli_tokens",
        lambda: {"access_token": token_y, "refresh_token": "***"},
    )

    # Recovery called with observed_access_token=None (missing-token path)
    result = _recover_codex_tokens_from_cli(
        "test",
        observed_access_token=None,
    )

    # CAS check: stored X ≠ observed None → recovery skipped
    # Recovery returns the current valid state (the concurrent reauth)
    assert result is not None
    assert result["access_token"] == token_x
    # Store still has token X (not overwritten by token Y)
    data = _read_codex_tokens()
    assert data["tokens"]["access_token"] == token_x


def test_recover_codex_cas_allows_when_none_unchanged(tmp_path, monkeypatch):
    """Missing-token path: recovery proceeds when store is still empty.

    Scenario:
    1. Store has NO Codex tokens (empty providers).
    2. Caller observes None.
    3. No concurrent reauth happens.
    4. Recovery is called with observed_access_token=None.
    5. CAS check: stored None == observed None → proceed.
    6. Recovery saves imported tokens successfully.
    """
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    (hermes_home / "auth.json").write_text(json.dumps({"version": 1, "providers": {}}))
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    token_y = _codex_jwt_with_account("acct-personal", exp_offset=7200)
    monkeypatch.setattr(
        "hermes_cli.auth._import_codex_cli_tokens",
        lambda: {"access_token": token_y, "refresh_token": "***"},
    )

    result = _recover_codex_tokens_from_cli(
        "test",
        observed_access_token=None,
    )

    # CAS check: stored None == observed None → proceed
    assert result is not None
    assert result["access_token"] == token_y
    data = _read_codex_tokens()
    assert data["tokens"]["access_token"] == token_y


def test_recover_codex_cas_allows_when_token_unchanged(tmp_path, monkeypatch):
    """Existing-token path: recovery proceeds when store is unchanged.

    Scenario:
    1. Store has token A.
    2. No concurrent reauth happens.
    3. Recovery is called with observed_access_token=A.
    4. CAS check: stored A == observed A → proceed.
    5. Recovery saves imported tokens successfully.
    """
    hermes_home = tmp_path / "hermes"
    token_a = _codex_jwt_with_account("acct-personal", exp_offset=7200)
    _seed_codex_store(hermes_home, token_a)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    token_b = _codex_jwt_with_account("acct-personal", exp_offset=7200)
    monkeypatch.setattr(
        "hermes_cli.auth._import_codex_cli_tokens",
        lambda: {"access_token": token_b, "refresh_token": "***"},
    )

    result = _recover_codex_tokens_from_cli(
        "refresh_token rejected: test",
        observed_access_token=token_a,
    )

    assert result is not None
    assert result["access_token"] == token_b
    data = _read_codex_tokens()
    assert data["tokens"]["access_token"] == token_b




