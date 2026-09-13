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
