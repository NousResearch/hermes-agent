"""Regression coverage for rotating pool-only Codex OAuth credentials."""
from __future__ import annotations

import base64
import json
import time
from dataclasses import replace
from types import SimpleNamespace

import pytest


def _future_jwt(subject: str) -> str:
    header = base64.urlsafe_b64encode(b'{"alg":"none"}').rstrip(b"=").decode()
    claims = json.dumps({"sub": subject, "exp": time.time() + 3600}).encode()
    payload = base64.urlsafe_b64encode(claims).rstrip(b"=").decode()
    return f"{header}.{payload}.signature"


def test_cached_agent_adopts_rotated_token_from_its_own_pool_row(monkeypatch):
    from agent.client_lifecycle import ClientLifecycleMixin

    active = SimpleNamespace(
        api_mode="codex_responses", provider="openai-codex", api_key="stale-token",
        _credential_pool_entry_id="credential-a",
    )
    adopted = {}
    active._adopt_openai_credentials = lambda key, url, *, reason: adopted.update(
        api_key=key, base_url=url, reason=reason
    ) or True

    class PersistedPool:
        def entry_id_for_api_key(self, key):
            return "credential-a" if key == "fresh-token" else None

    def resolve(*, force_refresh=False, **_):
        return {
            "api_key": "fresh-token", "base_url": "https://chatgpt.com/backend-api/codex",
            "source": "credential_pool",
        }

    monkeypatch.setattr("hermes_cli.auth.resolve_codex_runtime_credentials", resolve)
    monkeypatch.setattr("agent.credential_pool.load_pool", lambda _provider: PersistedPool())

    assert ClientLifecycleMixin._try_refresh_codex_client_credentials(active, force=True) is True
    assert adopted == {
        "api_key": "fresh-token", "base_url": "https://chatgpt.com/backend-api/codex",
        "reason": "openai-codex_credential_refresh",
    }


def test_cached_agent_refuses_different_pool_row(monkeypatch):
    from agent.client_lifecycle import ClientLifecycleMixin

    active = SimpleNamespace(
        api_mode="codex_responses", provider="openai-codex", api_key="credential-a-token",
        _credential_pool_entry_id="credential-a",
    )
    active._adopt_openai_credentials = lambda *_args, **_kwargs: pytest.fail("must not swap accounts")

    class PersistedPool:
        def entry_id_for_api_key(self, _key):
            return "credential-b"

    calls = []

    def resolve(*, force_refresh=False, **_):
        calls.append(force_refresh)
        return {
            "api_key": "credential-b-token", "base_url": "https://chatgpt.com/backend-api/codex",
            "source": "credential_pool",
        }

    monkeypatch.setattr("hermes_cli.auth.resolve_codex_runtime_credentials", resolve)
    monkeypatch.setattr("agent.credential_pool.load_pool", lambda _provider: PersistedPool())

    assert ClientLifecycleMixin._try_refresh_codex_client_credentials(active, force=True) is False
    assert calls == [False]


def test_cached_agent_refuses_pool_row_change_during_refresh(monkeypatch):
    from agent.client_lifecycle import ClientLifecycleMixin

    active = SimpleNamespace(
        api_mode="codex_responses", provider="openai-codex", api_key="stale-token",
        _credential_pool_entry_id="credential-a",
    )
    active._adopt_openai_credentials = lambda *_args, **_kwargs: pytest.fail("must not swap accounts")

    class PersistedPool:
        def entry_id_for_api_key(self, key):
            return {"fresh-token": "credential-a", "other-token": "credential-b"}.get(key)

    def resolve(*, force_refresh=False, **_):
        token = "other-token" if force_refresh else "fresh-token"
        return {
            "api_key": token, "base_url": "https://chatgpt.com/backend-api/codex",
            "source": "credential_pool",
        }

    monkeypatch.setattr("hermes_cli.auth.resolve_codex_runtime_credentials", resolve)
    monkeypatch.setattr("agent.credential_pool.load_pool", lambda _provider: PersistedPool())

    assert ClientLifecycleMixin._try_refresh_codex_client_credentials(active, force=True) is False


def test_pool_recovery_reads_rotated_manual_entry_before_refresh(tmp_path, monkeypatch):
    from agent.credential_pool import AUTH_TYPE_OAUTH, CredentialPool, PooledCredential
    from hermes_cli.auth import read_credential_pool, write_credential_pool

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    stale_token = _future_jwt("old")
    fresh_token = _future_jwt("new")
    stale = PooledCredential(
        provider="openai-codex", id="credential-a", label="account", auth_type=AUTH_TYPE_OAUTH,
        priority=0, source="manual:device_code", access_token=stale_token,
        refresh_token="stale-refresh-token",
    )
    fresh = replace(stale, access_token=fresh_token, refresh_token="fresh-refresh-token")
    write_credential_pool("openai-codex", [fresh.to_dict()])
    assert read_credential_pool("openai-codex")[0]["access_token"] == fresh_token

    pool = CredentialPool("openai-codex", [stale])
    monkeypatch.setattr(
        pool, "_post_tokens_refresh",
        lambda _entry: pytest.fail("must not replay stale refresh token"),
    )

    recovered = pool.try_refresh_matching(credential_id="credential-a")

    assert recovered is not None
    assert recovered.access_token == fresh_token
    assert recovered.refresh_token == "fresh-refresh-token"
