"""Authorization regression tests for ``PUT/DELETE /api/env``.

These endpoints have no handler-local auth check — they rely on the shared
dashboard auth layer: ``auth_middleware`` in ``web_server.py`` rejects any
non-public ``/api/*`` request in loopback-token mode, and
``dashboard_auth/middleware.py`` enforces the cookie session in gated mode.
This file pins that protection for both modes so a refactor that accidentally
drops ``/api/env`` out of the gated surface (or adds it to the public
allowlist) fails loudly.

Also pins the Custom Keys contract: under valid auth, a PUT for a key that is
NOT in ``OPTIONAL_ENV_VARS`` or the provider catalog must be accepted — the
Keys page lets users manage arbitrary env vars, so a catalog whitelist on the
write path would be a regression.
"""
from __future__ import annotations

import pytest

from fastapi.testclient import TestClient

import hermes_cli.web_server as web_server
from hermes_cli.dashboard_auth import clear_providers, register_provider
from tests.hermes_cli.conftest_dashboard_auth import StubAuthProvider

TOKEN_HEADERS = {"X-Hermes-Session-Token": web_server._SESSION_TOKEN}

# A key deliberately absent from OPTIONAL_ENV_VARS and the provider catalog.
CUSTOM_KEY = "MY_TOTALLY_CUSTOM_KEY_FOR_TEST"


@pytest.fixture
def env_store(monkeypatch):
    """Route env writes into an in-memory dict so tests never touch .env."""
    store: dict[str, str] = {}

    def fake_save(key: str, value: str) -> None:
        store[key] = value

    def fake_remove(key: str) -> bool:
        return store.pop(key, None) is not None

    monkeypatch.setattr(web_server, "save_env_value", fake_save)
    monkeypatch.setattr(web_server, "remove_env_value", fake_remove)
    return store


@pytest.fixture
def client_loopback():
    """Loopback bind: token auth via auth_middleware (X-Hermes-Session-Token).

    Mirrors the fixture in test_dashboard_auth_gate.py: pin bound_host so
    host_header_middleware's DNS-rebinding check accepts the requests.
    """
    prev_host = getattr(web_server.app.state, "bound_host", None)
    prev_port = getattr(web_server.app.state, "bound_port", None)
    web_server.app.state.bound_host = "127.0.0.1"
    web_server.app.state.bound_port = 9119
    client = TestClient(web_server.app, base_url="http://127.0.0.1:9119")
    yield client
    web_server.app.state.bound_host = prev_host
    web_server.app.state.bound_port = prev_port


@pytest.fixture
def gated_app():
    """Gated (public bind) mode: cookie auth via dashboard_auth middleware.

    Mirrors the fixture in test_dashboard_auth_middleware.py.
    """
    clear_providers()
    register_provider(StubAuthProvider())
    prev_host = getattr(web_server.app.state, "bound_host", None)
    prev_port = getattr(web_server.app.state, "bound_port", None)
    prev_required = getattr(web_server.app.state, "auth_required", None)
    web_server.app.state.bound_host = "fly-app.fly.dev"
    web_server.app.state.bound_port = 443
    web_server.app.state.auth_required = True
    client = TestClient(web_server.app, base_url="https://fly-app.fly.dev")
    yield client
    clear_providers()
    web_server.app.state.bound_host = prev_host
    web_server.app.state.bound_port = prev_port
    web_server.app.state.auth_required = prev_required


def _complete_stub_login(client) -> None:
    """Walk the stub OAuth round trip so ``client`` carries a valid session."""
    r1 = client.get("/auth/login?provider=stub", follow_redirects=False)
    assert r1.status_code == 302
    state = r1.headers["location"].split("state=")[1]
    r2 = client.get(
        f"/auth/callback?code=stub_code&state={state}",
        follow_redirects=False,
    )
    assert r2.status_code == 302


# ---------------------------------------------------------------------------
# Loopback-token mode
# ---------------------------------------------------------------------------


def test_loopback_put_env_without_token_is_rejected(client_loopback, env_store):
    r = client_loopback.put(
        "/api/env", json={"key": "OPENAI_API_KEY", "value": "sk-test"}
    )
    assert r.status_code == 401, (
        f"Unauthenticated PUT /api/env must be rejected by auth_middleware, "
        f"got {r.status_code}: {r.text}"
    )
    assert env_store == {}, "the write must never reach the handler"


def test_loopback_delete_env_without_token_is_rejected(client_loopback, env_store):
    env_store["OPENAI_API_KEY"] = "sk-test"
    r = client_loopback.request(
        "DELETE", "/api/env", json={"key": "OPENAI_API_KEY"}
    )
    assert r.status_code == 401, (
        f"Unauthenticated DELETE /api/env must be rejected by auth_middleware, "
        f"got {r.status_code}: {r.text}"
    )
    assert env_store == {"OPENAI_API_KEY": "sk-test"}, (
        "the delete must never reach the handler"
    )


def test_loopback_put_env_with_token_succeeds(client_loopback, env_store):
    r = client_loopback.put(
        "/api/env",
        json={"key": "OPENAI_API_KEY", "value": "sk-test"},
        headers=TOKEN_HEADERS,
    )
    assert r.status_code == 200, f"got {r.status_code}: {r.text}"
    assert r.json() == {"ok": True, "key": "OPENAI_API_KEY"}
    assert env_store == {"OPENAI_API_KEY": "sk-test"}


def test_loopback_delete_env_with_token_succeeds(client_loopback, env_store):
    env_store["OPENAI_API_KEY"] = "sk-test"
    r = client_loopback.request(
        "DELETE",
        "/api/env",
        json={"key": "OPENAI_API_KEY"},
        headers=TOKEN_HEADERS,
    )
    assert r.status_code == 200, f"got {r.status_code}: {r.text}"
    assert r.json() == {"ok": True, "key": "OPENAI_API_KEY"}
    assert env_store == {}


def test_loopback_put_custom_key_with_token_is_accepted(client_loopback, env_store):
    """Custom Keys contract: a key outside every catalog must still be
    writable under valid auth — no whitelist on the write path."""
    assert CUSTOM_KEY not in web_server.OPTIONAL_ENV_VARS
    r = client_loopback.put(
        "/api/env",
        json={"key": CUSTOM_KEY, "value": "custom-value"},
        headers=TOKEN_HEADERS,
    )
    assert r.status_code == 200, (
        f"PUT of a non-catalog (custom) key must be accepted, "
        f"got {r.status_code}: {r.text}"
    )
    assert env_store == {CUSTOM_KEY: "custom-value"}


# ---------------------------------------------------------------------------
# Gated (cookie) mode
# ---------------------------------------------------------------------------


def test_gated_put_env_without_cookie_is_rejected(gated_app, env_store):
    r = gated_app.put(
        "/api/env", json={"key": "OPENAI_API_KEY", "value": "sk-test"}
    )
    assert r.status_code == 401, (
        f"Unauthenticated PUT /api/env must be rejected by the cookie gate, "
        f"got {r.status_code}: {r.text}"
    )
    assert env_store == {}


def test_gated_delete_env_without_cookie_is_rejected(gated_app, env_store):
    env_store["OPENAI_API_KEY"] = "sk-test"
    r = gated_app.request("DELETE", "/api/env", json={"key": "OPENAI_API_KEY"})
    assert r.status_code == 401, (
        f"Unauthenticated DELETE /api/env must be rejected by the cookie gate, "
        f"got {r.status_code}: {r.text}"
    )
    assert env_store == {"OPENAI_API_KEY": "sk-test"}


def test_gated_put_env_with_cookie_session_succeeds(gated_app, env_store):
    _complete_stub_login(gated_app)
    r = gated_app.put(
        "/api/env", json={"key": "OPENAI_API_KEY", "value": "sk-test"}
    )
    assert r.status_code == 200, f"got {r.status_code}: {r.text}"
    assert r.json() == {"ok": True, "key": "OPENAI_API_KEY"}
    assert env_store == {"OPENAI_API_KEY": "sk-test"}


def test_gated_delete_env_with_cookie_session_succeeds(gated_app, env_store):
    _complete_stub_login(gated_app)
    env_store["OPENAI_API_KEY"] = "sk-test"
    r = gated_app.request("DELETE", "/api/env", json={"key": "OPENAI_API_KEY"})
    assert r.status_code == 200, f"got {r.status_code}: {r.text}"
    assert env_store == {}


def test_gated_put_custom_key_with_cookie_session_is_accepted(gated_app, env_store):
    """Custom Keys contract under the cookie gate: same as loopback."""
    assert CUSTOM_KEY not in web_server.OPTIONAL_ENV_VARS
    _complete_stub_login(gated_app)
    r = gated_app.put(
        "/api/env", json={"key": CUSTOM_KEY, "value": "custom-value"}
    )
    assert r.status_code == 200, (
        f"PUT of a non-catalog (custom) key must be accepted, "
        f"got {r.status_code}: {r.text}"
    )
    assert env_store == {CUSTOM_KEY: "custom-value"}
