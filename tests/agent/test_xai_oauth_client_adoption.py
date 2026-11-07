"""A live xAI client catches up with a rotated local grant without changing accounts (#108122).

The auth resolver, on-disk stores, AIAgent and SDK clients are real. Only the token endpoint
and inference HTTP transport are replaced; no live credentials or provider calls are needed.
"""

from __future__ import annotations

import base64
import copy
import json
import time
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from agent import auxiliary_client
from agent.client_lifecycle import ClientLifecycleMixin
from run_agent import AIAgent


_BASE_URL = "https://api.x.ai/v1"


def _jwt(subject, expires_at, *, generation="old"):
    payload = {"exp": expires_at, "jti": generation}
    if subject is not None:
        payload["sub"] = subject
    encoded = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")
    return f"header.{encoded}.signature"


def _store_grant(home, token, *, pool_only=False, provider="xai-oauth"):
    tokens = {"access_token": token, "refresh_token": "synthetic-refresh-token"}
    state = {
        "tokens": tokens,
        "discovery": {"token_endpoint": "https://auth.x.ai/oauth2/token"},
        "auth_mode": "oauth_device_code",
    }
    store = {"version": 1, "providers": {}, "credential_pool": {}}
    if pool_only:
        store["credential_pool"][provider] = [
            {"id": "singleton", "source": "device_code", "auth_type": "oauth", **tokens},
        ]
    else:
        store["providers"][provider] = state
    (home / "auth.json").write_text(json.dumps(store), encoding="utf-8")


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    requests, exchanges, transports, agents = [], [], [], []
    refreshed = {"token": ""}

    def respond(request):
        requests.append(request)
        return httpx.Response(200, json={
            "id": "response-fixture", "object": "response", "created_at": 1,
            "status": "completed", "model": "grok-fixture", "output": [],
        })

    def transport(*_args, **_kwargs):
        if refreshed.get("rebuild_error"):
            raise RuntimeError("synthetic client construction failure")
        client = httpx.Client(transport=httpx.MockTransport(respond))
        transports.append(client)
        return client

    def exchange(access_token, refresh_token, **_kwargs):
        exchanges.append((access_token, refresh_token))
        if refreshed.get("refresh_error"):
            raise httpx.TransportError("synthetic token endpoint unavailable")
        assert refreshed["token"], "a usable stored bearer must not spend a refresh token"
        return {
            "access_token": refreshed["token"], "refresh_token": "synthetic-rotated-refresh",
            "last_refresh": "2026-11-04T00:00:00Z", "token_type": "Bearer",
        }

    monkeypatch.setattr(ClientLifecycleMixin, "_build_keepalive_http_client", staticmethod(transport))
    monkeypatch.setattr("hermes_cli.auth.refresh_xai_oauth_pure", exchange)

    def build(token, *, provider="xai-oauth", base_url=_BASE_URL):
        agent = AIAgent(
            model="grok-fixture", provider=provider, api_mode="codex_responses",
            api_key=token, base_url=base_url, quiet_mode=True, max_iterations=2,
            enabled_toolsets=[], disabled_toolsets=["all"], skip_context_files=True,
            skip_memory=True, skip_background_review=True,
        )
        agents.append(agent)
        return agent

    yield SimpleNamespace(
        home=home, requests=requests, exchanges=exchanges, refreshed=refreshed, build=build,
    )
    for agent in agents:
        agent.close()
    for client in transports:
        client.close()


@pytest.mark.parametrize("case, expected, refresh_count", [
    ("rotated", True, 0),
    ("pool_only", True, 0),
    ("expired_store", True, 1),
    ("missing_expiry", True, 1),
    ("different_account", False, 0),
    ("missing_store_subject", False, 0),
    ("non_string_subject", False, 0),
    ("missing_active_subject", False, 0),
    ("account_changes_during_refresh", False, 1),
    ("expired_refresh_result", False, 1),
    ("foreign_route", False, 0),
    ("codex_scope", False, 0),
    ("rebuild_failure", False, 0),
    ("missing_store", False, 0),
    ("refresh_transport_failure", False, 1),
])
def test_live_client_uses_only_same_account_rotation(runtime, case, expected, refresh_count):
    now = time.time()
    old_subject = None if case == "missing_active_subject" else "account-a"
    subject = {
        "different_account": "account-b", "missing_store_subject": None, "non_string_subject": ["account-a"],
    }.get(case, "account-a")
    old_token = _jwt(old_subject, now - 3600)
    expiry = now - 60 if case in {
        "expired_store", "account_changes_during_refresh", "expired_refresh_result", "refresh_transport_failure",
    } else now + 3600
    if case == "missing_expiry":
        expiry = None
    stored_token = _jwt(subject, expiry, generation="stored")
    refreshed_subject = "account-b" if case == "account_changes_during_refresh" else "account-a"
    refreshed_expiry = now - 30 if case == "expired_refresh_result" else now + 7200
    minted = _jwt(refreshed_subject, refreshed_expiry, generation="refreshed")
    runtime.refreshed["token"] = minted if refresh_count else ""
    runtime.refreshed["refresh_error"] = case == "refresh_transport_failure"
    provider = "openai-codex" if case == "codex_scope" else "xai-oauth"
    base_url = "https://relay.example.test/v1" if case == "foreign_route" else _BASE_URL
    agent = runtime.build(old_token, provider=provider, base_url=base_url)
    if case != "missing_store":
        _store_grant(runtime.home, stored_token, pool_only=case == "pool_only", provider=provider)
    old_client = agent.client
    runtime.refreshed["rebuild_error"] = case == "rebuild_failure"

    assert agent._try_refresh_codex_client_credentials(force=True) is expected
    assert len(runtime.exchanges) == refresh_count
    if not expected:
        assert agent.api_key == agent._client_kwargs["api_key"] == old_token
        assert agent.base_url == agent._client_kwargs["base_url"] == base_url
        assert agent.client is old_client
        return

    adopted = minted if refresh_count else stored_token
    assert agent.api_key == agent._client_kwargs["api_key"] == adopted
    assert agent.client is not old_client
    agent.client.responses.create(model="grok-fixture", input="ping")
    assert runtime.requests[-1].headers["Authorization"] == f"Bearer {adopted}"
    assert runtime.requests[-1].url == f"{_BASE_URL}/responses"


@pytest.mark.parametrize("holder_route", ["matching", "other_key", "other_provider"])
def test_adoption_moves_matching_snapshot_and_aux_keys_only(runtime, holder_route):
    now = time.time()
    old_token = _jwt("account-a", now - 3600)
    stored_token = _jwt("account-a", now + 3600, generation="stored")
    agent = runtime.build(old_token)
    _store_grant(runtime.home, stored_token)
    if holder_route == "other_key":
        agent.context_compressor.api_key = "other-route-key"
        agent._primary_runtime["api_key"] = "other-route-key"
        agent._primary_runtime["client_kwargs"]["api_key"] = "other-route-key"
        agent._primary_runtime["compressor_api_key"] = "other-route-key"
    if holder_route == "other_provider":
        agent.context_compressor.provider = "openrouter"
        agent._primary_runtime["provider"] = "openrouter"
        agent._primary_runtime["compressor_provider"] = "openrouter"
    snapshot = copy.deepcopy(agent._primary_runtime)
    aux_runtime = {
        "provider": "openrouter" if holder_route == "other_provider" else "xai-oauth",
        "api_key": "other-route-key" if holder_route == "other_key" else old_token,
    }
    context_token = auxiliary_client._RUNTIME_MAIN_CONTEXT.set(aux_runtime)
    try:
        assert agent._try_refresh_codex_client_credentials(force=True)
        expected = {"matching": stored_token, "other_key": "other-route-key", "other_provider": old_token}[holder_route]
        assert agent.context_compressor.api_key == expected
        assert agent._primary_runtime["api_key"] == expected
        assert agent._primary_runtime["client_kwargs"]["api_key"] == expected
        assert agent._primary_runtime["compressor_api_key"] == expected
        assert aux_runtime["api_key"] == expected
        for field in ("model", "provider", "base_url", "api_mode", "request_overrides"):
            assert agent._primary_runtime[field] == snapshot[field]

        if holder_route == "matching":
            # A session without a bound pool must restore the adopted bearer from its snapshot.
            agent._credential_pool = None
            agent._fallback_activated = True
            assert agent._restore_primary_runtime()
            agent.client.responses.create(model="grok-fixture", input="after fallback")
            assert runtime.requests[-1].headers["Authorization"] == f"Bearer {stored_token}"
            assert runtime.exchanges == []
    finally:
        auxiliary_client._RUNTIME_MAIN_CONTEXT.reset(context_token)
