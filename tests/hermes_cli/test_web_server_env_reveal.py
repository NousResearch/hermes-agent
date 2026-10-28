"""Reveal policy, cache prevention and per-caller auditing/rate limits."""

import logging
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from hermes_cli import web_server
from hermes_cli.config import atomic_config_write, load_config
from hermes_cli.dashboard_auth import clear_providers, register_provider
from hermes_cli.web_routers import config_env
from hermes_constants import get_hermes_home
from tests.hermes_cli.conftest_dashboard_auth import StubAuthProvider, _sign


_KEY = "TEST_REVEAL_KEY"
_VALUE = "reveal-test-secret-not-for-logs"


@pytest.fixture(autouse=True)
def reveal_state(monkeypatch, tmp_path):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(web_server.app.state, "auth_required", False, raising=False)
    monkeypatch.setattr(web_server.app.state, "bound_host", "testserver", raising=False)
    monkeypatch.setattr(web_server.app.state, "bound_port", 80, raising=False)
    # Keep synthetic secrets in memory; exercise the real config loader and auth gate.
    monkeypatch.setattr(config_env, "load_env", lambda: {_KEY: _VALUE})
    clock = [1000.0]
    monkeypatch.setattr(config_env, "time", SimpleNamespace(
        time=lambda: clock[0], monotonic=lambda: clock[0]))
    config_env._reveal_timestamps.clear()
    clear_providers()
    yield clock
    config_env._reveal_timestamps.clear()
    clear_providers()


@pytest.fixture
def client():
    client = TestClient(web_server.app)
    client.headers[web_server._SESSION_HEADER_NAME] = web_server._SESSION_TOKEN
    yield client
    client.close()


@pytest.fixture
def gated_client(monkeypatch):
    monkeypatch.setattr(web_server.app.state, "auth_required", True)
    register_provider(StubAuthProvider())
    client = TestClient(web_server.app)
    yield client
    client.close()


def _identity_headers(user_id, *, nonce="first"):
    token = _sign({
        "sub": user_id, "email": "caller@example.test", "name": "Caller",
        "org_id": "test-org", "exp": int(time.time()) + 3600, "nonce": nonce,
    })
    return {"Authorization": f"Bearer {token}"}


def test_reveal_enabled_by_default():
    assert load_config()["security"]["dashboard_reveal_secrets"] is True


def test_reveal_not_cacheable(client):
    response = client.post("/api/env/reveal", json={"key": _KEY})
    assert response.status_code == 200
    assert response.json() == {"key": _KEY, "value": _VALUE}
    assert response.headers.get("cache-control") == "no-store"


def test_reveal_disabled_by_config(client):
    atomic_config_write(get_hermes_home() / "config.yaml", {
        "security": {"dashboard_reveal_secrets": False},
    })
    response = client.post("/api/env/reveal", json={"key": _KEY})
    assert response.status_code == 403
    assert "disabled by security.dashboard_reveal_secrets" in response.json()["detail"]
    assert _VALUE not in response.text


@pytest.mark.parametrize("body_profile", [None, "private"])
def test_reveal_policy_uses_requested_profile(client, body_profile):
    private = get_hermes_home() / "profiles" / "private"
    private.mkdir(parents=True)
    atomic_config_write(private / "config.yaml", {
        "security": {"dashboard_reveal_secrets": False},
    })
    body = {"key": _KEY}
    if body_profile:
        body["profile"] = body_profile
    response = client.post("/api/env/reveal?profile=private", json=body)
    assert response.status_code == 403
    assert client.post("/api/env/reveal", json={"key": _KEY}).status_code == 200


@pytest.mark.parametrize("profile_source", ["default", "query", "body"])
def test_reveal_audits_profile_and_client_without_value(client, caplog, profile_source):
    body = {"key": _KEY}
    url = "/api/env/reveal"
    profile = "default"
    if profile_source != "default":
        profile = "work"
        home = get_hermes_home() / "profiles" / profile
        home.mkdir(parents=True)
        atomic_config_write(home / "config.yaml", {})
        url += "?profile=work" if profile_source == "query" else "?profile=unused"
        if profile_source == "body":
            body["profile"] = profile
    with caplog.at_level(logging.INFO, logger="hermes_cli.web_server"):
        response = client.post(url, json=body)
    assert response.status_code == 200
    audit = [r.getMessage() for r in caplog.records if "env/reveal:" in r.getMessage()]
    assert len(audit) == 1
    assert _KEY in audit[0]
    assert f"profile={profile}" in audit[0]
    assert "principal=testclient" in audit[0]
    assert _VALUE not in caplog.text
    assert web_server._SESSION_TOKEN not in caplog.text


def test_reveal_audits_verified_identity_not_token(gated_client, caplog):
    headers = _identity_headers("user-alice")
    with caplog.at_level(logging.INFO, logger="hermes_cli.web_server"):
        response = gated_client.post("/api/env/reveal", json={"key": _KEY}, headers=headers)
    assert response.status_code == 200
    audit = [r.getMessage() for r in caplog.records if "env/reveal:" in r.getMessage()]
    assert len(audit) == 1
    assert "profile=default" in audit[0]
    assert "principal=user-alice" in audit[0]
    assert _VALUE not in caplog.text
    assert headers["Authorization"].removeprefix("Bearer ") not in caplog.text


def test_reveal_limit_is_per_verified_user_not_host_or_token(gated_client):
    for user in ("user-alice", "user-bob"):
        headers = _identity_headers(user)
        for _ in range(5):
            response = gated_client.post("/api/env/reveal", json={"key": _KEY}, headers=headers)
            assert response.status_code == 200
        # Rotating a session token must not reset the same user's budget.
        response = gated_client.post("/api/env/reveal", json={"key": _KEY},
                                     headers=_identity_headers(user, nonce="rotated"))
        assert response.status_code == 429


def test_reveal_loopback_limit_is_per_client_host(client):
    for host in ("127.0.0.1", "127.0.0.2"):
        with TestClient(web_server.app, client=(host, 12345), headers=client.headers) as caller:
            for _ in range(5):
                assert caller.post("/api/env/reveal", json={"key": _KEY}).status_code == 200
            assert caller.post("/api/env/reveal", json={"key": _KEY}).status_code == 429


def test_reveal_expired_principals_are_pruned(gated_client, reveal_state):
    assert gated_client.post("/api/env/reveal", json={"key": _KEY},
                             headers=_identity_headers("old-user")).status_code == 200
    reveal_state[0] += 30
    assert gated_client.post("/api/env/reveal", json={"key": _KEY},
                             headers=_identity_headers("new-user")).status_code == 200
    assert list(config_env._reveal_timestamps) == ["new-user"]
