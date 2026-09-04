"""Regression coverage for Codex singleton/manual credential isolation."""

from __future__ import annotations

import base64
import json

import pytest


def _write_auth_store(tmp_path, payload: dict) -> None:
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    (hermes_home / "auth.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )


def _jwt_with_claims(claims: dict) -> str:
    def _part(payload: dict) -> str:
        raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")

    return f"{_part({'alg': 'none', 'typ': 'JWT'})}.{_part(claims)}.sig"


def _codex_jwt(
    account_id: str,
    *,
    subject: str,
    token_id: str | None = None,
) -> str:
    claims: dict[str, object] = {
        "sub": subject,
        "https://api.openai.com/auth": {
            "chatgpt_account_id": account_id,
        },
    }
    if token_id is not None:
        claims["jti"] = token_id
    return _jwt_with_claims(claims)


def _pool_entry(
    *,
    entry_id: str,
    source: str,
    access_token: str,
    refresh_token: str,
    priority: int = 0,
):
    from agent.credential_pool import PooledCredential

    return PooledCredential(
        provider="openai-codex",
        id=entry_id,
        label=entry_id,
        auth_type="oauth",
        priority=priority,
        source=source,
        access_token=access_token,
        refresh_token=refresh_token,
    )


def _singleton_store(access_token: str, refresh_token: str, pool_entries: list[dict] | None = None) -> dict:
    payload: dict = {
        "version": 1,
        "active_provider": "openai-codex",
        "providers": {
            "openai-codex": {
                "tokens": {
                    "access_token": access_token,
                    "refresh_token": refresh_token,
                }
            }
        },
    }
    if pool_entries is not None:
        payload["credential_pool"] = {"openai-codex": pool_entries}
    return payload


def test_manual_different_principal_does_not_adopt_singleton(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    singleton_access = _codex_jwt("workspace-a", subject="user-a")
    manual_access = _codex_jwt("workspace-b", subject="user-b")
    _write_auth_store(tmp_path, _singleton_store(singleton_access, "refresh-a"))

    from agent.credential_pool import CredentialPool

    manual = _pool_entry(
        entry_id="manual-b",
        source="manual:device_code",
        access_token=manual_access,
        refresh_token="refresh-b",
    )
    pool = CredentialPool("openai-codex", [manual])

    synced = pool._sync_codex_entry_from_auth_store(manual)

    assert synced is manual
    assert synced.access_token == manual_access
    assert synced.refresh_token == "refresh-b"


def test_manual_same_principal_adopts_rotated_singleton(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    stale_access = _codex_jwt("workspace-a", subject="user-a", token_id="stale")
    fresh_access = _codex_jwt("workspace-a", subject="user-a", token_id="fresh")
    _write_auth_store(tmp_path, _singleton_store(fresh_access, "refresh-a2"))

    from agent.credential_pool import CredentialPool, STATUS_EXHAUSTED

    manual = _pool_entry(
        entry_id="manual-a",
        source="manual:device_code",
        access_token=stale_access,
        refresh_token="refresh-a1",
    )
    manual.last_status = STATUS_EXHAUSTED
    manual.last_error_code = 401
    pool = CredentialPool("openai-codex", [manual])

    synced = pool._sync_codex_entry_from_auth_store(manual)

    assert synced is not manual
    assert synced.access_token == fresh_access
    assert synced.refresh_token == "refresh-a2"
    assert synced.last_status is None
    assert synced.last_error_code is None


def test_manual_same_workspace_different_subject_does_not_adopt_singleton(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    singleton_access = _codex_jwt("shared-workspace", subject="user-a")
    manual_access = _codex_jwt("shared-workspace", subject="user-b")
    _write_auth_store(tmp_path, _singleton_store(singleton_access, "refresh-a"))

    from agent.credential_pool import CredentialPool

    manual = _pool_entry(
        entry_id="manual-b",
        source="manual:device_code",
        access_token=manual_access,
        refresh_token="refresh-b",
    )
    pool = CredentialPool("openai-codex", [manual])

    assert pool._sync_codex_entry_from_auth_store(manual) is manual
    assert manual.access_token == manual_access
    assert manual.refresh_token == "refresh-b"


@pytest.mark.parametrize(
    "manual_access",
    [
        "not-a-jwt",
        _jwt_with_claims(
            {"https://api.openai.com/auth": {"chatgpt_account_id": "workspace-a"}}
        ),
        _jwt_with_claims({"sub": "user-a", "https://api.openai.com/auth": {}}),
        _jwt_with_claims(
            {
                "sub": " ",
                "https://api.openai.com/auth": {"chatgpt_account_id": "workspace-a"},
            }
        ),
    ],
)
def test_manual_unknown_principal_fails_closed(tmp_path, monkeypatch, manual_access):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    singleton_access = _codex_jwt("workspace-a", subject="user-a")
    _write_auth_store(tmp_path, _singleton_store(singleton_access, "refresh-a"))

    from agent.credential_pool import CredentialPool

    manual = _pool_entry(
        entry_id="manual-unknown",
        source="manual:device_code",
        access_token=manual_access,
        refresh_token="refresh-manual",
    )
    pool = CredentialPool("openai-codex", [manual])

    assert pool._sync_codex_entry_from_auth_store(manual) is manual
    assert manual.access_token == manual_access
    assert manual.refresh_token == "refresh-manual"


def test_canonical_device_code_still_tracks_singleton_without_identity(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    _write_auth_store(tmp_path, _singleton_store("fresh-non-jwt", "fresh-refresh"))

    from agent.credential_pool import CredentialPool

    canonical = _pool_entry(
        entry_id="canonical",
        source="device_code",
        access_token="stale-non-jwt",
        refresh_token="stale-refresh",
    )
    pool = CredentialPool("openai-codex", [canonical])

    synced = pool._sync_codex_entry_from_auth_store(canonical)

    assert synced.access_token == "fresh-non-jwt"
    assert synced.refresh_token == "fresh-refresh"


def test_independent_manual_refresh_uses_own_pair_and_leaves_singleton_unchanged(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    singleton_access = _codex_jwt("workspace-a", subject="user-a")
    manual_access = _codex_jwt("workspace-b", subject="user-b", token_id="old")
    refreshed_manual_access = _codex_jwt(
        "workspace-b", subject="user-b", token_id="new"
    )
    _write_auth_store(tmp_path, _singleton_store(singleton_access, "refresh-a"))
    refresh_calls: list[tuple[str, str]] = []

    def _refresh(access_token, refresh_token):
        refresh_calls.append((access_token, refresh_token))
        return {
            "access_token": refreshed_manual_access,
            "refresh_token": "refresh-b2",
            "last_refresh": "2026-09-04T00:00:00+00:00",
        }

    monkeypatch.setattr("hermes_cli.auth.refresh_codex_oauth_pure", _refresh)

    from agent.credential_pool import CredentialPool

    manual = _pool_entry(
        entry_id="manual-b",
        source="manual:device_code",
        access_token=manual_access,
        refresh_token="refresh-b1",
    )
    pool = CredentialPool("openai-codex", [manual])

    refreshed = pool._refresh_entry(manual, force=True)

    assert refreshed is not None
    assert refresh_calls == [(manual_access, "refresh-b1")]
    assert refreshed.access_token == refreshed_manual_access
    assert refreshed.refresh_token == "refresh-b2"

    persisted = json.loads(
        (tmp_path / "hermes" / "auth.json").read_text(encoding="utf-8")
    )
    assert persisted["providers"]["openai-codex"]["tokens"] == {
        "access_token": singleton_access,
        "refresh_token": "refresh-a",
    }


@pytest.mark.parametrize(
    "reason",
    [
        "refresh_token_reused",
        "codex_refresh_failed",
        "codex_auth_missing_refresh_token",
    ],
)
def test_independent_manual_terminal_refresh_persists_dead_without_quarantining_singleton(
    tmp_path, monkeypatch, reason
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    monkeypatch.setattr("hermes_cli.auth._import_codex_cli_tokens", lambda: None)
    singleton_access = _codex_jwt("workspace-a", subject="user-a")
    manual_access = _codex_jwt("workspace-b", subject="user-b")
    seeded = _pool_entry(
        entry_id="seeded-a",
        source="device_code",
        access_token=singleton_access,
        refresh_token="refresh-a",
        priority=0,
    )
    manual = _pool_entry(
        entry_id="manual-b",
        source="manual:device_code",
        access_token=manual_access,
        refresh_token="refresh-b",
        priority=1,
    )
    _write_auth_store(
        tmp_path,
        _singleton_store(
            singleton_access,
            "refresh-a",
            [seeded.to_dict(), manual.to_dict()],
        ),
    )

    from hermes_cli.auth import AuthError

    def _terminal_refresh(*_args, **_kwargs):
        raise AuthError(
            "synthetic terminal Codex refresh failure",
            provider="openai-codex",
            code=reason,
            relogin_required=True,
        )

    monkeypatch.setattr("hermes_cli.auth.refresh_codex_oauth_pure", _terminal_refresh)

    from agent.credential_pool import CredentialPool, STATUS_DEAD, load_pool

    pool = CredentialPool("openai-codex", [seeded, manual])
    assert pool._refresh_entry(manual, force=True) is None

    by_id = {entry.id: entry for entry in pool.entries()}
    assert "seeded-a" in by_id
    assert by_id["manual-b"].last_status == STATUS_DEAD
    assert by_id["manual-b"].last_error_reason == reason

    persisted = json.loads(
        (tmp_path / "hermes" / "auth.json").read_text(encoding="utf-8")
    )
    assert persisted["providers"]["openai-codex"]["tokens"] == {
        "access_token": singleton_access,
        "refresh_token": "refresh-a",
    }
    persisted_by_id = {
        entry["id"]: entry
        for entry in persisted["credential_pool"]["openai-codex"]
    }
    assert "seeded-a" in persisted_by_id
    assert persisted_by_id["manual-b"]["last_status"] == STATUS_DEAD
    assert persisted_by_id["manual-b"]["last_error_reason"] == reason

    reloaded = load_pool("openai-codex")
    reloaded_manual = next(
        entry for entry in reloaded.entries() if entry.id == "manual-b"
    )
    assert reloaded_manual.last_status == STATUS_DEAD
    assert reloaded_manual.last_error_reason == reason


def test_429_rotation_does_not_rewrite_fallback_manual_token_pair(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    monkeypatch.setattr("hermes_cli.auth._import_codex_cli_tokens", lambda: None)
    access_a = _codex_jwt("workspace-a", subject="user-a")
    access_b = _codex_jwt("workspace-b", subject="user-b")
    entry_a = _pool_entry(
        entry_id="manual-a",
        source="manual:device_code",
        access_token=access_a,
        refresh_token="refresh-a",
        priority=0,
    )
    entry_b = _pool_entry(
        entry_id="manual-b",
        source="manual:device_code",
        access_token=access_b,
        refresh_token="refresh-b",
        priority=1,
    )
    _write_auth_store(
        tmp_path,
        {
            "version": 1,
            "credential_pool": {
                "openai-codex": [entry_a.to_dict(), entry_b.to_dict()]
            },
        },
    )

    from agent.credential_pool import CredentialPool

    pool = CredentialPool("openai-codex", [entry_a, entry_b])
    assert pool.select().id == "manual-a"

    next_entry = pool.mark_exhausted_and_rotate(
        status_code=429,
        credential_id="manual-a",
        api_key_hint=access_a,
        error_context={
            "reason": "rate_limit_exceeded",
            "message": "synthetic rate limit",
        },
    )

    assert next_entry is not None
    assert next_entry.id == "manual-b"
    assert next_entry.access_token == access_b
    assert next_entry.refresh_token == "refresh-b"

    persisted = json.loads(
        (tmp_path / "hermes" / "auth.json").read_text(encoding="utf-8")
    )
    persisted_b = next(
        entry
        for entry in persisted["credential_pool"]["openai-codex"]
        if entry["id"] == "manual-b"
    )
    assert persisted_b["access_token"] == access_b
    assert persisted_b["refresh_token"] == "refresh-b"
