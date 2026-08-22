"""Tests for the loopback-only Kanban board read credential."""
from __future__ import annotations

import secrets
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from hermes_cli import kanban_db
from hermes_cli.dashboard_auth import (
    TokenPrincipal,
    assert_protocol_compliance,
    clear_providers,
    register_provider,
)
from hermes_cli.dashboard_auth import token_auth
from tests.hermes_cli.conftest_dashboard_auth import StubAuthProvider

import plugins.dashboard_auth.drain as drain_plugin
import plugins.dashboard_auth.kanban_read as kanban_read


def _strong_secret() -> str:
    return secrets.token_urlsafe(32)


@pytest.fixture(autouse=True)
def _clean_env_and_routes(monkeypatch):
    monkeypatch.delenv("HERMES_DASHBOARD_KANBAN_READ_SECRET", raising=False)
    monkeypatch.delenv("HERMES_DASHBOARD_DRAIN_SECRET", raising=False)
    token_auth.clear_token_routes()
    clear_providers()
    yield
    clear_providers()
    token_auth.clear_token_routes()


def test_protocol_and_capability_flags():
    assert_protocol_compliance(kanban_read.KanbanReadSecretProvider)
    provider = kanban_read.KanbanReadSecretProvider(secret=_strong_secret())
    assert provider.supports_token is True
    assert provider.supports_session is False


def test_verify_token_returns_fixed_principal_and_scope():
    secret = _strong_secret()
    provider = kanban_read.KanbanReadSecretProvider(secret=secret)

    principal = provider.verify_token(token=secret)
    assert isinstance(principal, TokenPrincipal)
    assert principal.principal == "kanban-board-reader"
    assert principal.provider == "kanban-read-secret"
    assert principal.scopes == ("kanban.read",)
    assert provider.verify_token(token="wrong") is None
    assert provider.verify_token(token="") is None


def test_verify_token_uses_constant_time_credential_comparison(monkeypatch):
    secret = _strong_secret()
    provider = kanban_read.KanbanReadSecretProvider(secret=secret)
    calls = []

    def compare_digest(left, right):
        calls.append((left, right))
        return left == right

    monkeypatch.setattr(kanban_read.hmac, "compare_digest", compare_digest)
    assert provider.verify_token(token=secret) is not None
    assert provider.verify_token(token="wrong") is None
    assert len(calls) == 2
    assert all(
        isinstance(left, bytes) and isinstance(right, bytes)
        for left, right in calls
    )


def test_constructor_rejects_weak_secret():
    with pytest.raises(ValueError, match="kanban read secret rejected"):
        kanban_read.KanbanReadSecretProvider(secret="weak")


def test_interactive_methods_remain_unsupported():
    provider = kanban_read.KanbanReadSecretProvider(secret=_strong_secret())
    with pytest.raises(NotImplementedError):
        provider.start_login(redirect_uri="https://example.test/callback")
    with pytest.raises(NotImplementedError):
        provider.complete_login(
            code="c",
            state="s",
            code_verifier="v",
            redirect_uri="https://example.test/callback",
        )
    with pytest.raises(NotImplementedError):
        provider.refresh_session(refresh_token="r")


class TestRegister:
    def test_noop_when_secret_missing(self):
        ctx = MagicMock()
        kanban_read.register(ctx)
        ctx.register_dashboard_auth_provider.assert_not_called()
        assert "HERMES_DASHBOARD_KANBAN_READ_SECRET" in kanban_read.LAST_SKIP_REASON
        assert not token_auth.is_token_route(kanban_read.KANBAN_BOARD_ROUTE_PATH)

    def test_weak_secret_fails_closed(self, monkeypatch):
        monkeypatch.setenv("HERMES_DASHBOARD_KANBAN_READ_SECRET", "too-weak")
        ctx = MagicMock()
        kanban_read.register(ctx)
        ctx.register_dashboard_auth_provider.assert_not_called()
        assert "rejected" in kanban_read.LAST_SKIP_REASON
        assert not token_auth.is_token_route(kanban_read.KANBAN_BOARD_ROUTE_PATH)

    def test_strong_env_secret_registers_exact_get_and_fixed_scope(self, monkeypatch):
        secret = _strong_secret()
        monkeypatch.setenv("HERMES_DASHBOARD_KANBAN_READ_SECRET", f"  {secret}  ")
        ctx = MagicMock()
        kanban_read.register(ctx)

        ctx.register_dashboard_auth_provider.assert_called_once()
        provider = ctx.register_dashboard_auth_provider.call_args.args[0]
        assert isinstance(provider, kanban_read.KanbanReadSecretProvider)
        assert provider.verify_token(token=secret).scopes == ("kanban.read",)
        assert kanban_read.LAST_SKIP_REASON == ""
        assert token_auth.is_token_route(kanban_read.KANBAN_BOARD_ROUTE_PATH, "GET")
        assert not token_auth.is_token_route(kanban_read.KANBAN_BOARD_ROUTE_PATH, "POST")

    def test_route_registration_conflict_skips_provider(self, monkeypatch):
        secret = _strong_secret()
        monkeypatch.setenv("HERMES_DASHBOARD_KANBAN_READ_SECRET", secret)

        def fail_route_registration(*args, **kwargs):
            raise ValueError("route conflict")

        monkeypatch.setattr(
            token_auth, "register_token_route", fail_route_registration
        )
        ctx = MagicMock()

        kanban_read.register(ctx)

        ctx.register_dashboard_auth_provider.assert_not_called()
        assert "route conflict" in kanban_read.LAST_SKIP_REASON
        assert not token_auth.is_token_route(kanban_read.KANBAN_BOARD_ROUTE_PATH)


@pytest.fixture
def mounted_dashboard(tmp_path, monkeypatch):
    from hermes_cli import web_server

    home = tmp_path / "hermes-home"
    kanban_home = tmp_path / "kanban-home"
    home.mkdir()
    kanban_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(kanban_home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kanban_db.init_db()

    kanban_secret = _strong_secret()
    drain_secret = _strong_secret()
    monkeypatch.setenv("HERMES_DASHBOARD_KANBAN_READ_SECRET", kanban_secret)
    monkeypatch.setenv("HERMES_DASHBOARD_DRAIN_SECRET", drain_secret)
    monkeypatch.setattr(drain_plugin, "_load_config_drain_auth_section", lambda: {})

    kanban_ctx = MagicMock()
    kanban_read.register(kanban_ctx)
    drain_ctx = MagicMock()
    drain_plugin.register(drain_ctx)
    register_provider(kanban_ctx.register_dashboard_auth_provider.call_args.args[0])
    register_provider(drain_ctx.register_dashboard_auth_provider.call_args.args[0])
    register_provider(StubAuthProvider())

    previous = {
        "bound_host": getattr(web_server.app.state, "bound_host", None),
        "bound_port": getattr(web_server.app.state, "bound_port", None),
        "auth_required": getattr(web_server.app.state, "auth_required", None),
    }
    web_server.app.state.bound_host = "127.0.0.1"
    web_server.app.state.bound_port = 8765
    web_server.app.state.auth_required = False
    client = TestClient(
        web_server.app,
        base_url="http://127.0.0.1:8765",
        client=("127.0.0.1", 50000),
    )

    yield client, web_server.app, kanban_secret, drain_secret

    clear_providers()
    token_auth.clear_token_routes()
    for key, value in previous.items():
        setattr(web_server.app.state, key, value)


def _count_tasks() -> int:
    conn = kanban_db.connect()
    try:
        return len(kanban_db.list_tasks(conn))
    finally:
        conn.close()


def test_mounted_loopback_routes_are_read_only_and_scope_bound(mounted_dashboard):
    client, app, kanban_secret, drain_secret = mounted_dashboard
    from hermes_cli import web_server

    board_path = kanban_read.KANBAN_BOARD_ROUTE_PATH
    kanban_headers = {"Authorization": f"Bearer {kanban_secret}"}
    browser_headers = {
        web_server._SESSION_HEADER_NAME: web_server._SESSION_TOKEN,
    }

    board = client.get(board_path, headers=kanban_headers)
    assert board.status_code == 200, board.text
    assert {"columns", "tenants", "assignees", "latest_event_id", "now"}.issubset(
        board.json()
    )

    browser_board = client.get(board_path, headers=browser_headers)
    assert browser_board.status_code == 200, browser_board.text
    assert {"columns", "latest_event_id"}.issubset(browser_board.json())
    assert client.get(board_path).status_code == 401
    assert client.get(board_path, headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert client.get(
        board_path, headers={"Authorization": f"Bearer {drain_secret}"}
    ).status_code == 403
    assert client.post(
        "/api/gateway/drain", headers=kanban_headers, json={"action": "drain"}
    ).status_code == 403

    before = _count_tasks()
    task_response = client.post(
        "/api/plugins/kanban/tasks",
        headers=kanban_headers,
        json={"title": "must not exist"},
    )
    assert not 200 <= task_response.status_code < 300
    assert _count_tasks() == before


def test_mounted_kanban_board_accepts_legacy_dashboard_bearer_session(
    mounted_dashboard,
):
    client, _app, _kanban_secret, _drain_secret = mounted_dashboard
    from hermes_cli import web_server

    response = client.get(
        kanban_read.KANBAN_BOARD_ROUTE_PATH,
        headers={"Authorization": f"Bearer {web_server._SESSION_TOKEN}"},
    )

    assert response.status_code == 200, response.text
    assert {"columns", "latest_event_id"}.issubset(response.json())


def test_mounted_non_loopback_and_unknown_bind_fall_back_to_dashboard_session(
    mounted_dashboard,
):
    _loopback_client, app, kanban_secret, _drain_secret = mounted_dashboard

    app.state.bound_host = "10.0.0.7"
    app.state.auth_required = True
    remote = TestClient(
        app,
        base_url="https://10.0.0.7:8765",
        client=("10.0.0.7", 50001),
    )
    service_response = remote.get(
        kanban_read.KANBAN_BOARD_ROUTE_PATH,
        headers={"Authorization": f"Bearer {kanban_secret}"},
    )
    assert service_response.status_code == 401

    login = remote.get("/auth/login?provider=stub", follow_redirects=False)
    assert login.status_code == 302
    state = login.headers["location"].split("state=", 1)[1]
    callback = remote.get(
        f"/auth/callback?code=stub_code&state={state}", follow_redirects=False
    )
    assert callback.status_code == 302
    assert remote.get(kanban_read.KANBAN_BOARD_ROUTE_PATH).status_code == 200
    assert remote.get("/api/auth/me").status_code == 200

    app.state.bound_host = None
    unknown_bind = TestClient(
        app,
        base_url="http://127.0.0.1:8765",
        client=("127.0.0.1", 50002),
    )
    unknown_response = unknown_bind.get(
        kanban_read.KANBAN_BOARD_ROUTE_PATH,
        headers={"Authorization": f"Bearer {kanban_secret}"},
    )
    assert unknown_response.status_code == 401
