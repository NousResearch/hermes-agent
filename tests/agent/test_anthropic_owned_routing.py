"""Owned OAuth must win without rotating an unrelated borrowed grant."""
import json
import time
from types import SimpleNamespace

import httpx
import pytest
from openai import AuthenticationError

from agent import anthropic_credentials as ac
from agent import auxiliary_client as aux


def _seed(tmp_path, monkeypatch):
    monkeypatch.setattr(ac.Path, "home", lambda: tmp_path)
    monkeypatch.setattr(ac, "_first_env", lambda *names: "")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    borrowed = tmp_path / ".claude" / ".credentials.json"
    borrowed.parent.mkdir()
    borrowed.write_text(json.dumps({"claudeAiOauth": {
        "accessToken": "borrowed-token", "refreshToken": "borrowed-refresh", "expiresAt": 1,
    }}))
    (tmp_path / "auth.json").write_text(json.dumps({"credential_pool": {"anthropic": [{
        "id": "owned", "source": "manual:hermes_pkce", "auth_type": "oauth",
        "access_token": "owned-token", "refresh_token": "owned-refresh",
        "expires_at": int(time.time()*1000)+3600000, "priority": 0,
    }]}}))
    return borrowed


def test_owned_pool_precedes_expired_borrowed_login(tmp_path, monkeypatch):
    borrowed = _seed(tmp_path, monkeypatch)
    before = borrowed.read_bytes()
    def forbidden(*args, **kwargs):
        pytest.fail("Borrowed refresh was consumed despite owned grant")
    monkeypatch.setattr(ac, "_refresh_oauth_token", forbidden)
    assert ac.resolve_anthropic_token() == "owned-token"
    assert borrowed.read_bytes() == before


def test_auxiliary_owned_refresh_does_not_spend_borrowed_rotation(tmp_path, monkeypatch):
    borrowed = _seed(tmp_path, monkeypatch)
    before = borrowed.read_bytes()
    def refresh(refresh_token, **kwargs):
        assert refresh_token == "owned-refresh"
        return {"access_token": "owned-new-token", "refresh_token": "owned-new-refresh",
                "expires_at_ms": int(time.time()*1000)+3600000}
    def forbidden(*args, **kwargs):
        pytest.fail("Auxiliary refreshed an unrelated borrowed grant")
    monkeypatch.setattr(ac, "refresh_anthropic_oauth_pure", refresh)
    monkeypatch.setattr(ac, "_refresh_oauth_token", forbidden)
    error = AuthenticationError("Invalid API key", response=httpx.Response(
        401, request=httpx.Request("POST", "https://api.anthropic.com/v1/messages")), body={})
    route = SimpleNamespace(client=SimpleNamespace(api_key="owned-token"), task="compression", tag="",
        resolved_provider="anthropic", base_info="https://api.anthropic.com", resolved_model="fixture",
        final_model="fixture", main_runtime=None)
    retry = aux._ladder_credential_rungs(error, route, {}, False)
    assert next(retry).kind == "retry_same_provider"
    retry.close()
    assert borrowed.read_bytes() == before
    assert aux._refresh_anthropic_credentials("unrelated-api-key") is False


def test_suppressed_claude_code_source_is_not_read(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from hermes_cli.auth import suppress_credential_source

    suppress_credential_source("anthropic", "claude_code")

    def forbidden(*args, **kwargs):
        pytest.fail("suppressed Claude Code credentials were read")

    monkeypatch.setattr(ac, "_read_claude_code_credentials_from_keychain", forbidden)
    monkeypatch.setattr(ac, "_read_claude_code_credentials_from_file", forbidden)

    assert ac.read_claude_code_credentials() is None


def test_auxiliary_refresh_respects_claude_code_suppression(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from hermes_cli.auth import suppress_credential_source

    suppress_credential_source("anthropic", "claude_code")

    def forbidden(*args, **kwargs):
        pytest.fail("auxiliary refresh bypassed Claude Code source suppression")

    monkeypatch.setattr(ac, "_read_claude_code_credentials_from_keychain", forbidden)
    monkeypatch.setattr(ac, "_read_claude_code_credentials_from_file", forbidden)
    monkeypatch.setattr(ac, "_refresh_oauth_token", forbidden)

    assert aux._refresh_anthropic_credentials("failed-token") is False


def test_existing_pool_cannot_refresh_after_source_is_suppressed(tmp_path, monkeypatch):
    from agent.credential_pool import AUTH_TYPE_OAUTH, CredentialPool, PooledCredential
    from hermes_cli.auth import suppress_credential_source

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    entry = PooledCredential(
        provider="anthropic", id="borrowed", label="claude_code",
        auth_type=AUTH_TYPE_OAUTH, priority=0, source="claude_code",
        access_token="old-access", refresh_token="old-refresh", expires_at_ms=1,
    )
    pool = CredentialPool("anthropic", [entry])
    suppress_credential_source("anthropic", "claude_code")

    def forbidden(*args, **kwargs):
        pytest.fail("suppressed in-memory credential reached refresh")

    monkeypatch.setattr(pool, "_refresh_entry_impl", forbidden)

    assert pool._refresh_entry(entry, force=True) is None


def test_existing_pool_fails_closed_when_suppression_store_is_unreadable(tmp_path, monkeypatch):
    from agent.credential_pool import AUTH_TYPE_OAUTH, CredentialPool, PooledCredential
    from hermes_cli import auth as auth_mod

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    entry = PooledCredential(
        provider="anthropic", id="borrowed", label="claude_code",
        auth_type=AUTH_TYPE_OAUTH, priority=0, source="claude_code",
        access_token="old-access", refresh_token="old-refresh", expires_at_ms=1,
    )
    pool = CredentialPool("anthropic", [entry])
    monkeypatch.setattr(auth_mod, "_load_auth_store_strict", lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("unreadable")))

    def forbidden(*args, **kwargs):
        pytest.fail("unreadable suppression store re-enabled external refresh")

    monkeypatch.setattr(pool, "_refresh_entry_impl", forbidden)

    assert pool._refresh_entry(entry, force=True) is None


def test_corrupt_auth_store_fails_closed_before_external_claude_reads(tmp_path, monkeypatch):
    from agent import anthropic_credentials as ac

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    auth_path = tmp_path / "auth.json"
    auth_path.write_text("{not-json")

    def forbidden():
        pytest.fail("corrupt auth state allowed an external Claude credential read")

    monkeypatch.setattr(ac, "_read_claude_code_credentials_from_keychain", forbidden)
    monkeypatch.setattr(ac, "_read_claude_code_credentials_from_file", forbidden)

    assert ac.claude_code_source_is_suppressed() is True
    assert ac.read_claude_code_credentials() is None
    assert auth_path.read_text() == "{not-json"


def test_direct_refresh_rechecks_suppression_before_credential_lock(tmp_path, monkeypatch):
    from agent import anthropic_credentials as ac

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    cred_path = tmp_path / "claude" / ".credentials.json"
    monkeypatch.setattr(ac, "claude_code_credentials_path", lambda: cred_path)
    checks = iter([False, True])
    monkeypatch.setattr(ac, "claude_code_source_is_suppressed", lambda: next(checks))

    def forbidden(*args, **kwargs):
        pytest.fail("suppression raced with refresh and the POST still ran")

    monkeypatch.setattr(ac, "refresh_anthropic_oauth_pure", forbidden)

    assert ac._refresh_oauth_token({
        "accessToken": "old-access", "refreshToken": "old-refresh", "expiresAt": 1,
    }) is None
    assert not cred_path.with_suffix(".lock").exists()


def test_pool_refresh_rechecks_suppression_inside_auth_lock(tmp_path, monkeypatch):
    from agent import credential_pool as cp
    from agent.credential_pool import AUTH_TYPE_OAUTH, CredentialPool, PooledCredential

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    entry = PooledCredential(
        provider="anthropic", id="borrowed", label="claude_code",
        auth_type=AUTH_TYPE_OAUTH, priority=0, source="claude_code",
        access_token="old-access", refresh_token="old-refresh", expires_at_ms=1,
    )
    pool = CredentialPool("anthropic", [entry])
    checks = iter([False, True])
    monkeypatch.setattr(cp, "_source_is_suppressed", lambda *_args: next(checks))

    def forbidden(*args, **kwargs):
        pytest.fail("pool spent a refresh token after suppression committed")

    monkeypatch.setattr(pool, "_refresh_entry_impl", forbidden)
    assert pool._refresh_entry(entry, force=True) is None


def test_existing_pool_stops_selecting_source_after_suppression(tmp_path, monkeypatch):
    from agent.credential_pool import AUTH_TYPE_OAUTH, CredentialPool, PooledCredential
    from hermes_cli.auth import suppress_credential_source

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    entry = PooledCredential(
        provider="anthropic", id="borrowed", label="claude_code",
        auth_type=AUTH_TYPE_OAUTH, priority=0, source="claude_code",
        access_token="valid-access", refresh_token="valid-refresh",
        expires_at_ms=9_999_999_999_999,
    )
    pool = CredentialPool("anthropic", [entry])
    selected = pool.select()
    assert selected is not None
    assert selected.id == entry.id

    suppress_credential_source("anthropic", "claude_code")

    assert pool.select() is None


def test_claude_credential_read_rechecks_suppression_after_external_access(tmp_path, monkeypatch):
    from agent import anthropic_credentials as ac

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    # Both snapshots are unsuppressed, but the changed generation proves an
    # intervening suppress -> unsuppress ABA transition occurred.
    states = iter([(False, 7), (False, 9)])
    monkeypatch.setattr(ac, "claude_code_source_policy_state", lambda: next(states))
    monkeypatch.setattr(ac, "_read_claude_code_credentials_from_keychain", lambda: {
        "accessToken": "must-not-escape", "refreshToken": "refresh", "expiresAt": 9_999_999_999_999,
    })
    monkeypatch.setattr(ac, "_read_claude_code_credentials_from_file", lambda: None)

    assert ac.read_claude_code_credentials() is None
