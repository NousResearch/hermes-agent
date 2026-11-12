"""A live Codex agent can use a recovered token without changing principals or routes."""
from __future__ import annotations

import base64
import json
import time

import httpx
import pytest
from openai import OpenAI

from agent.agent_runtime_helpers import recover_with_credential_pool
from agent.client_lifecycle import ClientLifecycleMixin
from agent.credential_pool import CredentialPool, PooledCredential
from agent.turn_recovery import _refresh_credentials_after_401
from agent.turn_retry_state import TurnRetryState
from hermes_cli import auth

_BASE = "https://chatgpt.com/backend-api/codex"


def _jwt(account="account", subject="user", *, expires):
    claims = {"sub": subject, "https://api.openai.com/auth": {"chatgpt_account_id": account}}
    if expires is not None:
        claims["exp"] = expires
    payload = base64.urlsafe_b64encode(json.dumps(claims).encode()).rstrip(b"=").decode()
    return f"e30.{payload}.sig"


def _write_auth(home, token):
    home.mkdir(parents=True, exist_ok=True)
    (home / "auth.json").write_text(json.dumps({
        "version": 1,
        "providers": {"openai-codex": {"tokens": token}},
    }), encoding="utf-8")


def _agent(token, transport, *, base_url=_BASE):
    agent = ClientLifecycleMixin()
    agent.api_mode, agent.provider = "codex_responses", "openai-codex"
    agent.model, agent.api_key, agent.base_url = "codex-test", token, base_url
    agent._client_kwargs = {"api_key": token, "base_url": base_url, "http_client": transport}
    agent.client = OpenAI(**agent._client_kwargs)
    agent._buffer_vprint = lambda *_: None
    return agent


@pytest.mark.parametrize("recover_from_cli", [False, True])
def test_401_retries_with_recovered_same_principal_token(tmp_path, monkeypatch, recover_from_cli):
    """Exercise real auth-store/CLI recovery, client replacement and its outgoing bearer."""
    now = time.time()
    old, fresh = _jwt(expires=now - 3600), _jwt(expires=now + 7200)
    home, codex_home = tmp_path / "hermes", tmp_path / "codex"
    tokens = {"access_token": fresh, "refresh_token": "fresh-refresh"}
    _write_auth(home, {} if recover_from_cli else tokens)
    codex_home.mkdir()
    (codex_home / "auth.json").write_text(json.dumps({"tokens": tokens}), encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("CODEX_HOME", str(codex_home))

    def no_refresh(**kwargs):
        pytest.fail("A usable recovered token must not spend another OAuth refresh token")

    monkeypatch.setattr(auth, "_codex_http_client", no_refresh)
    sent = []

    def receive(request):
        sent.append(request.headers["authorization"])
        return httpx.Response(200, json={"ok": True})

    with httpx.Client(transport=httpx.MockTransport(receive)) as transport:
        agent = _agent(old, transport)
        previous_client = agent.client
        retry = TurnRetryState()
        error = RuntimeError("HTTP 401: Provided authentication token is expired")

        assert _refresh_credentials_after_401(agent, error, retry, 401) is True
        assert agent.api_key == agent._client_kwargs["api_key"] == fresh
        assert agent.client is not previous_client
        assert agent.client.get("/credential-probe", cast_to=dict[str, bool]) == {"ok": True}
        assert sent == [f"Bearer {fresh}"]
        assert _refresh_credentials_after_401(agent, error, retry, 401) is False
        assert json.loads((home / "auth.json").read_text())["providers"]["openai-codex"]["tokens"] == tokens
        agent.client.close()
        previous_client.close()


@pytest.mark.parametrize("case", [
    "other-account", "other-subject", "unknown-principal", "other-route",
    "expired-replacement", "unknown-expiry", "live-original", "pool-only",
])
def test_401_does_not_adopt_an_unproven_replacement(tmp_path, monkeypatch, case):
    """Recovery cannot change a principal/route, borrow a pool row, or replay an expired key."""
    now = time.time()
    old = _jwt(expires=now + 3600 if case == "live-original" else now - 3600)
    fresh = _jwt(
        account="other" if case == "other-account" else "account",
        subject="other" if case == "other-subject" else "user",
        expires=None if case == "unknown-expiry" else now - 60 if case == "expired-replacement" else now + 7200,
    )
    if case == "unknown-principal":
        fresh = "opaque-token"
    home = tmp_path / "hermes"
    _write_auth(home, {"access_token": fresh, "refresh_token": "fresh-refresh"})
    if case == "pool-only":
        (home / "auth.json").write_text(json.dumps({
            "version": 1, "providers": {},
            "credential_pool": {"openai-codex": [{
                "id": "independent", "source": "manual:device_code", "auth_type": "oauth",
                "access_token": fresh, "refresh_token": "pool-refresh", "base_url": _BASE,
            }]},
        }), encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "empty-codex"))
    before = (home / "auth.json").read_bytes()

    def no_refresh(**kwargs):
        pytest.fail("An unrelated or unusable replacement must not trigger a singleton refresh")

    monkeypatch.setattr(auth, "_codex_http_client", no_refresh)
    with httpx.Client(transport=httpx.MockTransport(lambda _: pytest.fail("Unexpected network request"))) as transport:
        agent = _agent(old, transport, base_url="https://gateway.example/v1" if case == "other-route" else _BASE)
        previous_client = agent.client

        assert agent._try_refresh_codex_client_credentials(force=True) is False
        assert agent.api_key == agent._client_kwargs["api_key"] == old
        assert agent.client is previous_client
        assert (home / "auth.json").read_bytes() == before
        agent.client.close()


def test_recovered_token_keeps_subsequent_refresh_on_its_own_grant(tmp_path, monkeypatch):
    """A later 401 must refresh the adopted singleton, not the previously selected grant."""
    now = time.time()
    old, fresh, rotated = (_jwt(expires=now + ttl) for ttl in (-60, 7200, 8400))
    home = tmp_path / "hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "empty-codex"))
    tokens = {"access_token": fresh, "refresh_token": "singleton-refresh"}
    _write_auth(home, tokens)

    def iso(ts):
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts))

    rows = [
        {"id": "manual", "source": "manual:device_code", "auth_type": "oauth", "priority": 0,
         "access_token": old, "refresh_token": "manual-refresh", "base_url": _BASE,
         "last_refresh": iso(now - 4000)},
        {"id": "singleton", "source": "device_code", "auth_type": "oauth", "priority": 1,
         **tokens, "base_url": _BASE, "last_refresh": iso(now - 7200)},
    ]
    store = json.loads((home / "auth.json").read_text())
    store["providers"]["openai-codex"]["last_refresh"] = iso(now - 7200)
    store["credential_pool"] = {"openai-codex": rows}
    (home / "auth.json").write_text(json.dumps(store), encoding="utf-8")
    pool = CredentialPool("openai-codex", [PooledCredential.from_dict("openai-codex", row) for row in rows])
    posted = []

    def refresh(access_token, refresh_token):
        posted.append(refresh_token)
        return {"access_token": rotated, "refresh_token": "rotated-refresh", "last_refresh": iso(now)}

    monkeypatch.setattr(auth, "refresh_codex_oauth_pure", refresh)
    with httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(200, json={}))) as transport:
        agent = _agent(old, transport)
        agent.log_prefix = ""
        agent._credential_pool = pool
        agent._credential_pool_entry_id = pool.entry_id_for_api_key(old)
        agent._is_entitlement_failure = lambda *_: False
        previous_client = agent.client

        assert agent._try_refresh_codex_client_credentials(force=True) is True
        assert recover_with_credential_pool(agent, status_code=401, has_retried_429=False) == (True, False)
        assert posted == ["singleton-refresh"]
        assert agent._credential_pool_entry_id == "singleton"
        assert agent.api_key == rotated
        agent.client.close()
        previous_client.close()
