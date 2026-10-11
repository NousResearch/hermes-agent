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


def _seed_two_healthy(tmp_path, monkeypatch):
    import time as _time
    monkeypatch.setattr(ac.Path, "home", lambda: tmp_path)
    monkeypatch.setattr(ac, "_first_env", lambda *names: "")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "auth.json").write_text(json.dumps({"credential_pool": {"anthropic": [
        {"id": "a", "source": "manual:hermes_pkce", "auth_type": "oauth",
         "access_token": "token-a", "refresh_token": "refresh-a",
         "expires_at": int(_time.time()*1000)+3600000, "priority": 0},
        {"id": "b", "source": "manual:hermes_pkce", "auth_type": "oauth",
         "access_token": "token-b", "refresh_token": "refresh-b",
         "expires_at": int(_time.time()*1000)+3600000, "priority": 1},
    ]}}))


def test_retry_401_attributes_to_stale_key_not_healthy_current(tmp_path, monkeypatch):
    """Regression for #122601: a retry 401 from a stale explicit key must not
    quarantine the healthy entry the first recovery rotated to."""
    _seed_two_healthy(tmp_path, monkeypatch)

    def _err():
        return AuthenticationError("revoked", response=httpx.Response(
            401, request=httpx.Request("POST", "https://api.anthropic.com/v1/messages")), body={})

    route = SimpleNamespace(client=SimpleNamespace(api_key="stale-revoked-token"),
        task="approval", tag="", resolved_provider="anthropic",
        base_info="https://api.anthropic.com", resolved_model="fixture",
        final_model="fixture", main_runtime=None)
    calls = []
    real_recover = aux._recover_provider_pool
    def _spy(provider, exc, **kwargs):
        calls.append(dict(kwargs))
        return real_recover(provider, exc, **kwargs)
    monkeypatch.setattr(aux, "_recover_provider_pool", _spy)

    gen = aux._ladder_credential_rungs(_err(), route, {}, False)
    step = next(gen)
    assert step.kind == "retry_same_provider"
    try:
        gen.throw(_err())
    except StopIteration as stop:
        result = stop.value
    else:
        pytest.fail("ladder should finish after the retry 401")
    assert result[0] is None
    assert len(calls) == 2
    assert calls[1].get("failed_api_key") == "stale-revoked-token"

    from agent.credential_pool import load_pool
    pool = load_pool("anthropic")
    states = {(e.label or e.id): e.last_status for e in pool._entries}
    assert all(status is None for status in states.values()), states
