"""Tests for Codex auth — tokens stored in Hermes auth store (~/.hermes/auth.json)."""

import json
import time
import base64
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
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


def test_read_codex_tokens_success(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes"
    _setup_hermes_auth(hermes_home)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    data = _read_codex_tokens()
    assert data["tokens"]["access_token"] == "access"
    assert data["tokens"]["refresh_token"] == "refresh"


def test_read_codex_tokens_missing(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    # Empty auth store
    (hermes_home / "auth.json").write_text(json.dumps({"version": 1, "providers": {}}))
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    with pytest.raises(AuthError) as exc:
        _read_codex_tokens()
    assert exc.value.code == "codex_auth_missing"


def test_resolve_codex_runtime_credentials_missing_access_token(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes"
    _setup_hermes_auth(hermes_home, access_token="")
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "missing-codex"))

    with pytest.raises(AuthError) as exc:
        resolve_codex_runtime_credentials()
    assert exc.value.code == "codex_auth_missing_access_token"
    assert exc.value.relogin_required is True


def test_resolve_codex_runtime_credentials_refreshes_expiring_token(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes"
    expiring_token = _jwt_with_exp(int(time.time()) - 10)
    _setup_hermes_auth(hermes_home, access_token=expiring_token, refresh_token="refresh-old")
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    called = {"count": 0}

    def _fake_refresh(tokens, timeout_seconds):
        called["count"] += 1
        return {"access_token": "access-new", "refresh_token": "refresh-new"}

    monkeypatch.setattr("hermes_cli.auth._refresh_codex_auth_tokens", _fake_refresh)

    resolved = resolve_codex_runtime_credentials()

    assert called["count"] == 1
    assert resolved["api_key"] == "access-new"


def test_resolve_codex_runtime_credentials_force_refresh(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes"
    _setup_hermes_auth(hermes_home, access_token="access-current", refresh_token="refresh-old")
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    called = {"count": 0}

    def _fake_refresh(tokens, timeout_seconds):
        called["count"] += 1
        return {"access_token": "access-forced", "refresh_token": "refresh-new"}

    monkeypatch.setattr("hermes_cli.auth._refresh_codex_auth_tokens", _fake_refresh)

    resolved = resolve_codex_runtime_credentials(force_refresh=True, refresh_if_expiring=False)

    assert called["count"] == 1
    assert resolved["api_key"] == "access-forced"


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


def test_resolve_codex_runtime_credentials_pool_fallback_skips_exhausted(tmp_path, monkeypatch):
    """The pool fallback skips entries currently in an exhaustion cooldown window."""
    import time as _time

    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    future_reset = _time.time() + 3600  # 1h cooldown remaining
    auth_store = {
        "version": 1,
        "providers": {},
        "credential_pool": {
            "openai-codex": [
                {
                    "source": "device_code",
                    "access_token": "wedged-token",
                    "last_error_reset_at": future_reset,  # in cooldown
                },
                {
                    "source": "device_code",
                    "access_token": "usable-token",
                    "last_status": "ok",
                },
            ],
        },
    }
    (hermes_home / "auth.json").write_text(json.dumps(auth_store))
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    resolved = resolve_codex_runtime_credentials()
    assert resolved["api_key"] == "usable-token"
    assert resolved["source"] == "credential_pool"


def test_resolve_codex_runtime_credentials_pool_fallback_no_usable_entry(tmp_path, monkeypatch):
    """When both singleton and pool are empty/unusable, the original AuthError propagates."""
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    auth_store = {
        "version": 1,
        "providers": {},
        "credential_pool": {
            "openai-codex": [
                {"source": "device_code", "access_token": ""},  # empty
            ],
        },
    }
    (hermes_home / "auth.json").write_text(json.dumps(auth_store))
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    with pytest.raises(AuthError) as exc:
        resolve_codex_runtime_credentials()
    assert exc.value.code == "codex_auth_missing"


def test_resolve_provider_explicit_codex_does_not_fallback(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    assert resolve_provider("openai-codex") == "openai-codex"


def test_save_codex_tokens_roundtrip(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    (hermes_home / "auth.json").write_text(json.dumps({"version": 1, "providers": {}}))
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    _save_codex_tokens({"access_token": "at123", "refresh_token": "rt456"})
    data = _read_codex_tokens()

    assert data["tokens"]["access_token"] == "at123"
    assert data["tokens"]["refresh_token"] == "rt456"


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


def test_save_codex_tokens_still_refreshes_legacy_manual_alias(tmp_path, monkeypatch):
    """The #33538 legacy use case must keep working.

    A user who hit #33000 before the #33164 fix landed might have run
    ``hermes auth add openai-codex`` as a workaround when there was no
    singleton entry — that created a ``manual:device_code`` pool entry that
    holds the SAME token material as the (later) singleton.  This entry is a
    true alias of the singleton and SHOULD still be refreshed on subsequent
    re-auths, otherwise it goes stale and recreates the #33538 symptom.

    The distinguishing signal: a legacy alias has access_token == previous
    singleton access_token; an independent account does not.
    """
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    (hermes_home / "auth.json").write_text(json.dumps({
        "version": 1,
        "providers": {
            "openai-codex": {
                "tokens": {"access_token": "shared-at", "refresh_token": "shared-rt"},
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
                    "access_token": "shared-at",
                    "refresh_token": "shared-rt",
                },
                {
                    "id": "legacy",
                    "label": "legacy-alias",
                    "source": "manual:device_code",
                    "auth_type": "oauth",
                    # Token material matches the singleton — this is a true
                    # alias from the #33000 workaround era.
                    "access_token": "shared-at",
                    "refresh_token": "shared-rt",
                    "last_status": "exhausted",
                    "last_error_code": 401,
                    "last_error_reason": "token_invalidated",
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

    # Singleton: refreshed.
    seeded = next(e for e in pool if e["source"] == "device_code")
    assert seeded["access_token"] == "fresh-at"

    # Legacy alias: still refreshed (preserves #33538 fix).
    legacy = next(e for e in pool if e["id"] == "legacy")
    assert legacy["access_token"] == "fresh-at"
    assert legacy["refresh_token"] == "fresh-rt"
    assert legacy["last_refresh"] == "2026-06-05T00:00:00Z"
    # Error markers cleared on the refreshed entry.
    assert legacy["last_status"] is None
    assert legacy["last_error_code"] is None
    assert legacy["last_error_reason"] is None


def test_save_codex_tokens_handles_missing_previous_singleton_tokens(tmp_path, monkeypatch):
    """First-ever Codex save (no prior singleton tokens) must not crash.

    Edge case: a user has only pool entries (e.g. via direct auth.json edit
    or a partial state from a corrupted upgrade), no `providers.openai-codex.tokens`
    block at all.  The previous-singleton-tokens guard must handle missing
    state gracefully — fall back to "no previous tokens", which means no
    pool entry can be a true alias and only the singleton-seeded entry gets
    written.
    """
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    (hermes_home / "auth.json").write_text(json.dumps({
        "version": 1,
        "providers": {},
        "credential_pool": {
            "openai-codex": [
                {
                    "id": "preexisting",
                    "label": "pre-existing-manual",
                    "source": "manual:device_code",
                    "auth_type": "oauth",
                    "access_token": "preexisting-at",
                    "refresh_token": "preexisting-rt",
                },
            ],
        },
    }))
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    _save_codex_tokens(
        {"access_token": "first-at", "refresh_token": "first-rt"},
        last_refresh="2026-06-05T00:00:00Z",
    )

    auth = json.loads((hermes_home / "auth.json").read_text())
    pool = auth["credential_pool"]["openai-codex"]
    # Pre-existing independent entry with no relationship to a (now-new)
    # singleton MUST be preserved.
    pre = next(e for e in pool if e["id"] == "preexisting")
    assert pre["access_token"] == "preexisting-at"
    assert pre["refresh_token"] == "preexisting-rt"


def test_save_codex_tokens_alias_match_uses_access_token_only(tmp_path, monkeypatch):
    """A manual entry counts as an alias if its access_token matches the
    previous singleton access_token, regardless of refresh_token presence.

    Some legacy entries (older auth.json schemas, pre-refresh-token versions)
    have access_token but no refresh_token.  These should still be treated as
    aliases when the access_token matches.
    """
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    (hermes_home / "auth.json").write_text(json.dumps({
        "version": 1,
        "providers": {
            "openai-codex": {
                "tokens": {"access_token": "shared-at", "refresh_token": "shared-rt"},
                "auth_mode": "chatgpt",
            },
        },
        "credential_pool": {
            "openai-codex": [
                {
                    "id": "alias-no-refresh",
                    "source": "manual:device_code",
                    "auth_type": "oauth",
                    "access_token": "shared-at",
                    # No refresh_token at all — legacy schema.
                },
            ],
        },
    }))
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    _save_codex_tokens(
        {"access_token": "new-at", "refresh_token": "new-rt"},
        last_refresh="2026-06-05T00:00:00Z",
    )

    auth = json.loads((hermes_home / "auth.json").read_text())
    pool = auth["credential_pool"]["openai-codex"]
    alias = next(e for e in pool if e["id"] == "alias-no-refresh")
    # Treated as alias → refreshed with new tokens.
    assert alias["access_token"] == "new-at"
    assert alias["refresh_token"] == "new-rt"


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


def test_import_codex_cli_tokens(tmp_path, monkeypatch):
    codex_home = tmp_path / "codex-cli"
    codex_home.mkdir(parents=True, exist_ok=True)
    (codex_home / "auth.json").write_text(json.dumps({
        "tokens": {"access_token": "cli-at", "refresh_token": "cli-rt"},
    }))
    monkeypatch.setenv("CODEX_HOME", str(codex_home))

    tokens = _import_codex_cli_tokens()
    assert tokens is not None
    assert tokens["access_token"] == "cli-at"
    assert tokens["refresh_token"] == "cli-rt"


def test_import_codex_cli_tokens_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "nonexistent"))
    assert _import_codex_cli_tokens() is None


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


def test_refresh_parses_openai_nested_error_shape_refresh_token_reused(monkeypatch):
    """OpenAI returns {"error": {"code": "refresh_token_reused", "message": "..."}}
    — parser must surface relogin_required and the dedicated message.
    """
    response = _StubHTTPResponse(
        401,
        {
            "error": {
                "message": "Your refresh token has already been used to generate a new access token. Please try signing in again.",
                "type": "invalid_request_error",
                "param": None,
                "code": "refresh_token_reused",
            }
        },
    )
    _patch_httpx(monkeypatch, response)

    with pytest.raises(AuthError) as exc_info:
        refresh_codex_oauth_pure("a-tok", "r-tok")

    err = exc_info.value
    assert err.code == "refresh_token_reused"
    assert err.relogin_required is True
    # The existing dedicated branch should override the message with actionable guidance.
    assert "already consumed by another client" in str(err)


def test_refresh_parses_openai_nested_error_shape_generic_code(monkeypatch):
    """Nested error with arbitrary code still surfaces code + message."""
    response = _StubHTTPResponse(
        400,
        {
            "error": {
                "message": "Invalid client credentials.",
                "type": "invalid_request_error",
                "code": "invalid_client",
            }
        },
    )
    _patch_httpx(monkeypatch, response)

    with pytest.raises(AuthError) as exc_info:
        refresh_codex_oauth_pure("a-tok", "r-tok")

    err = exc_info.value
    assert err.code == "invalid_client"
    assert "Invalid client credentials." in str(err)


def test_refresh_parses_oauth_spec_flat_error_shape_invalid_grant(monkeypatch):
    """Fallback path: OAuth spec-shape {"error": "invalid_grant", "error_description": "..."}
    must still map to relogin_required=True via the existing code set.
    """
    response = _StubHTTPResponse(
        400,
        {
            "error": "invalid_grant",
            "error_description": "Refresh token is expired or revoked.",
        },
    )
    _patch_httpx(monkeypatch, response)

    with pytest.raises(AuthError) as exc_info:
        refresh_codex_oauth_pure("a-tok", "r-tok")

    err = exc_info.value
    assert err.code == "invalid_grant"
    assert err.relogin_required is True
    assert "Refresh token is expired or revoked." in str(err)


def test_refresh_falls_back_to_generic_message_on_unparseable_body(monkeypatch):
    """No JSON body → generic 'with status 401' message; 401 always forces relogin."""
    response = _StubHTTPResponse(401, ValueError("not json"))
    _patch_httpx(monkeypatch, response)

    with pytest.raises(AuthError) as exc_info:
        refresh_codex_oauth_pure("a-tok", "r-tok")

    err = exc_info.value
    assert err.code == "codex_refresh_failed"
    # 401/403 from the token endpoint always means the refresh token is
    # invalid/expired — force relogin even without a parseable error body.
    assert err.relogin_required is True
    assert "status 401" in str(err)


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


def test_parse_retry_after_http_date():
    """Retry-After may be an HTTP-date, not only delta-seconds."""
    from hermes_cli.auth import _parse_retry_after_seconds

    retry_at = datetime.now(timezone.utc) + timedelta(seconds=90)
    header = format_datetime(retry_at, usegmt=True)

    parsed = _parse_retry_after_seconds({"retry-after": header})

    assert parsed is not None
    assert 0 < parsed <= 90


def test_parse_retry_after_past_http_date_returns_zero():
    from hermes_cli.auth import _parse_retry_after_seconds

    retry_at = datetime.now(timezone.utc) - timedelta(seconds=30)
    header = format_datetime(retry_at, usegmt=True)

    assert _parse_retry_after_seconds({"retry-after": header}) == 0


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


def test_login_openai_codex_force_new_login_skips_existing_reuse_prompt(monkeypatch):
    called = {"device_login": 0}

    monkeypatch.setattr(
        "hermes_cli.auth.resolve_codex_runtime_credentials",
        lambda: {"base_url": DEFAULT_CODEX_BASE_URL},
    )
    monkeypatch.setattr(
        "hermes_cli.auth._import_codex_cli_tokens",
        lambda: {"access_token": "cli-at", "refresh_token": "cli-rt"},
    )
    monkeypatch.setattr(
        "hermes_cli.auth._codex_device_code_login",
        lambda: {
            "tokens": {"access_token": "fresh-at", "refresh_token": "fresh-rt"},
            "last_refresh": "2026-04-01T00:00:00Z",
            "base_url": DEFAULT_CODEX_BASE_URL,
        },
    )

    def _fake_save(tokens, last_refresh=None):
        called["device_login"] += 1
        called["tokens"] = dict(tokens)
        called["last_refresh"] = last_refresh

    monkeypatch.setattr("hermes_cli.auth._save_codex_tokens", _fake_save)
    monkeypatch.setattr("hermes_cli.auth._update_config_for_provider", lambda *args, **kwargs: "/tmp/config.yaml")
    monkeypatch.setattr(
        "builtins.input",
        lambda prompt="": (_ for _ in ()).throw(AssertionError("force_new_login should not prompt for reuse/import")),
    )

    _login_openai_codex(SimpleNamespace(), PROVIDER_REGISTRY["openai-codex"], force_new_login=True)

    assert called["device_login"] == 1
    assert called["tokens"]["access_token"] == "fresh-at"


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


def test_device_code_login_retries_on_429_then_succeeds(monkeypatch):
    """A transient 429 on the device-code request is retried, not surfaced."""
    from hermes_cli import auth as auth_mod

    sleeps = []
    monkeypatch.setattr("time.sleep", lambda s: sleeps.append(s))

    # First call 429 (with Retry-After), second call succeeds. The polling
    # loop then returns the authorization code, and token exchange succeeds.
    _patch_httpx_post(
        monkeypatch,
        [
            _FakeResp(429, headers={"retry-after": "1"}),
            _FakeResp(200, {"user_code": "ABCD", "device_auth_id": "dev-1", "interval": "5"}),
            _FakeResp(200, {"authorization_code": "auth-code", "code_verifier": "verifier"}),
            _FakeResp(200, {"access_token": "at", "refresh_token": "rt", "expires_in": 3600}),
        ],
    )
    # Skip the polling sleep too (shares time.sleep, already patched).

    creds = auth_mod._codex_device_code_login()

    assert creds["tokens"]["access_token"] == "at"
    # The 429 caused exactly one backoff sleep before the retry succeeded.
    assert 1 in sleeps


def test_device_code_login_persistent_429_raises_rate_limited(monkeypatch):
    """A persistent 429 surfaces a clear rate-limit error, not a bare status."""
    from hermes_cli import auth as auth_mod

    monkeypatch.setattr("time.sleep", lambda s: None)
    _patch_httpx_post(monkeypatch, [_FakeResp(429, headers={"retry-after": "30"})] * 4)

    with pytest.raises(AuthError) as exc_info:
        auth_mod._codex_device_code_login()

    err = exc_info.value
    assert err.code == auth_mod.CODEX_RATE_LIMITED_CODE
    assert "rate-limiting" in str(err)
    assert "30s" in str(err)
    assert auth_mod.is_rate_limited_auth_error(err)


def test_device_code_login_non_429_error_unchanged(monkeypatch):
    """Non-429 failures keep the generic device_code_request_error code."""
    from hermes_cli import auth as auth_mod

    monkeypatch.setattr("time.sleep", lambda s: None)
    _patch_httpx_post(monkeypatch, [_FakeResp(500)])

    with pytest.raises(AuthError) as exc_info:
        auth_mod._codex_device_code_login()

    assert exc_info.value.code == "device_code_request_error"
