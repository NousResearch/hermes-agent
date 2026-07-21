"""Tests for the KanbanApiSecretProvider plugin (non-interactive bearer secret).

Loads the bundled kanban_api auth plugin module directly and exercises:
  * constant-time verify_token returning a kanban-scoped TokenPrincipal,
  * the register(ctx) entry point's env/config resolution, skip reasons, and
    token-route-prefix registration (dashboard subtree excluded),
  * E2E through the real mounted dashboard app in gated mode: an external
    controller's bearer credential reaches the external kanban surface, the
    interactive dashboard subtree stays on the cookie gate, and a
    foreign-scoped service credential (the drain secret) is refused.

The shared entropy gate itself is covered by test_drain_provider.py — it
lives in ``hermes_cli.dashboard_auth.secret_strength`` and both plugins use
the same function.
"""
from __future__ import annotations

import secrets
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import plugins.dashboard_auth.kanban_api as kanban_plugin
from hermes_cli.dashboard_auth import (
    TokenPrincipal,
    assert_protocol_compliance,
    clear_providers,
    register_provider,
)
from hermes_cli.dashboard_auth import token_auth


@pytest.fixture(scope="module")
def plugin():
    return kanban_plugin


@pytest.fixture(autouse=True)
def _clean_env_routes_and_providers(monkeypatch):
    monkeypatch.delenv("HERMES_KANBAN_API_SECRET", raising=False)
    clear_providers()
    token_auth.clear_token_routes()
    yield
    clear_providers()
    token_auth.clear_token_routes()


def _strong_secret() -> str:
    # token_urlsafe(32) → 43 url-safe-b64 chars ≈ 256 bits.
    return secrets.token_urlsafe(32)


# ---------------------------------------------------------------------------
# Provider behaviour
# ---------------------------------------------------------------------------


class TestProvider:
    def test_protocol_compliance(self, plugin):
        assert_protocol_compliance(plugin.KanbanApiSecretProvider)

    def test_supports_token_flag(self, plugin):
        p = plugin.KanbanApiSecretProvider(secret=_strong_secret())
        assert p.supports_token is True

    def test_is_non_interactive(self, plugin):
        p = plugin.KanbanApiSecretProvider(secret=_strong_secret())
        assert p.supports_session is False

    def test_verify_token_accepts_matching_secret(self, plugin):
        s = _strong_secret()
        p = plugin.KanbanApiSecretProvider(secret=s)
        principal = p.verify_token(token=s)
        assert isinstance(principal, TokenPrincipal)
        assert principal.principal == "kanban-api"
        assert principal.provider == "kanban-api-secret"
        assert principal.scopes == ("kanban",)

    def test_verify_token_rejects_wrong_secret(self, plugin):
        p = plugin.KanbanApiSecretProvider(secret=_strong_secret())
        assert p.verify_token(token=_strong_secret()) is None

    def test_verify_token_rejects_empty(self, plugin):
        p = plugin.KanbanApiSecretProvider(secret=_strong_secret())
        assert p.verify_token(token="") is None

    def test_custom_scope_attached(self, plugin):
        s = _strong_secret()
        p = plugin.KanbanApiSecretProvider(secret=s, scope="boards")
        assert p.verify_token(token=s).scopes == ("boards",)

    def test_construction_rejects_weak_secret(self, plugin):
        with pytest.raises(ValueError):
            plugin.KanbanApiSecretProvider(secret="weak")

    def test_verify_session_returns_none_not_raises(self, plugin):
        p = plugin.KanbanApiSecretProvider(secret=_strong_secret())
        assert p.verify_session(access_token="anything") is None

    def test_interactive_methods_raise(self, plugin):
        p = plugin.KanbanApiSecretProvider(secret=_strong_secret())
        with pytest.raises(NotImplementedError):
            p.start_login(redirect_uri="r")
        with pytest.raises(NotImplementedError):
            p.complete_login(code="c", state="s", code_verifier="v", redirect_uri="r")
        with pytest.raises(NotImplementedError):
            p.refresh_session(refresh_token="r")


# ---------------------------------------------------------------------------
# register() entry point
# ---------------------------------------------------------------------------


class TestRegister:
    def test_skips_when_no_secret(self, plugin, monkeypatch):
        monkeypatch.setattr(plugin, "_load_config_kanban_api_auth_section", lambda: {})
        ctx = MagicMock()
        plugin.register(ctx)
        ctx.register_dashboard_auth_provider.assert_not_called()
        assert "HERMES_KANBAN_API_SECRET" in plugin.LAST_SKIP_REASON
        assert not token_auth.is_token_route("/api/plugins/kanban/tasks")

    def test_skips_and_fails_closed_on_weak_secret(self, plugin, monkeypatch):
        monkeypatch.setenv("HERMES_KANBAN_API_SECRET", "tooweak")
        monkeypatch.setattr(plugin, "_load_config_kanban_api_auth_section", lambda: {})
        ctx = MagicMock()
        plugin.register(ctx)
        ctx.register_dashboard_auth_provider.assert_not_called()
        assert "rejected" in plugin.LAST_SKIP_REASON
        # fail-closed: the surface is NOT token-authable, so it stays gated.
        assert not token_auth.is_token_route("/api/plugins/kanban/tasks")

    def test_registers_with_strong_env_secret(self, plugin, monkeypatch):
        s = _strong_secret()
        monkeypatch.setenv("HERMES_KANBAN_API_SECRET", s)
        monkeypatch.setattr(plugin, "_load_config_kanban_api_auth_section", lambda: {})
        ctx = MagicMock()
        plugin.register(ctx)
        ctx.register_dashboard_auth_provider.assert_called_once()
        provider = ctx.register_dashboard_auth_provider.call_args.args[0]
        assert isinstance(provider, plugin.KanbanApiSecretProvider)
        assert provider.verify_token(token=s) is not None
        assert plugin.LAST_SKIP_REASON == ""
        # The external surface (parameterised paths included) is token-authable…
        assert token_auth.is_token_route("/api/plugins/kanban/tasks")
        assert token_auth.is_token_route("/api/plugins/kanban/tasks/t_abc/complete")
        # …while the interactive operator subtree stays on the cookie gate.
        assert not token_auth.is_token_route("/api/plugins/kanban/dashboard/board")

    def test_config_scope_applied(self, plugin, monkeypatch):
        s = _strong_secret()
        monkeypatch.setenv("HERMES_KANBAN_API_SECRET", s)
        monkeypatch.setattr(
            plugin,
            "_load_config_kanban_api_auth_section",
            lambda: {"scope": "boards"},
        )
        ctx = MagicMock()
        plugin.register(ctx)
        provider = ctx.register_dashboard_auth_provider.call_args.args[0]
        assert provider.verify_token(token=s).scopes == ("boards",)

    def test_config_min_secret_chars_can_reject_otherwise_ok_secret(
        self, plugin, monkeypatch
    ):
        s = _strong_secret()  # 43 chars — fine by default, too short at 999
        monkeypatch.setenv("HERMES_KANBAN_API_SECRET", s)
        monkeypatch.setattr(
            plugin,
            "_load_config_kanban_api_auth_section",
            lambda: {"min_secret_chars": 999},
        )
        ctx = MagicMock()
        plugin.register(ctx)
        ctx.register_dashboard_auth_provider.assert_not_called()
        assert "rejected" in plugin.LAST_SKIP_REASON


# ---------------------------------------------------------------------------
# E2E through the mounted dashboard app (gated deployment)
# ---------------------------------------------------------------------------


class _Ctx:
    """Minimal PluginContext stand-in wired to the real provider registry."""

    def register_dashboard_auth_provider(self, provider) -> None:
        register_provider(provider)


@pytest.fixture
def gated_kanban_app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """The real web_server app in gated mode with the kanban credential active.

    Mirrors the deployment the docs describe: a non-loopback bind
    (``auth_required`` True, cookie gate on) where an external controller
    authenticates with ``Authorization: Bearer $HERMES_KANBAN_API_SECRET``
    through the plugin's own ``register()`` wiring — no test-double seams.
    """
    from fastapi.testclient import TestClient

    from hermes_cli import kanban_db, web_server

    monkeypatch.setenv("HERMES_KANBAN_HOME", str(tmp_path))
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_BOARD", raising=False)
    kanban_db._INITIALIZED_PATHS.clear()

    secret = _strong_secret()
    monkeypatch.setenv("HERMES_KANBAN_API_SECRET", secret)
    monkeypatch.setattr(
        kanban_plugin, "_load_config_kanban_api_auth_section", lambda: {}
    )
    kanban_plugin.register(_Ctx())

    # A second stacked service credential with a different capability — the
    # cross-scope case the seam must refuse on kanban routes.
    import plugins.dashboard_auth.drain as drain_plugin

    drain_secret = _strong_secret()
    monkeypatch.setenv("HERMES_DASHBOARD_DRAIN_SECRET", drain_secret)
    monkeypatch.setattr(drain_plugin, "_load_config_drain_auth_section", lambda: {})
    drain_plugin.register(_Ctx())

    prev_host = getattr(web_server.app.state, "bound_host", None)
    prev_port = getattr(web_server.app.state, "bound_port", None)
    prev_required = getattr(web_server.app.state, "auth_required", None)
    web_server.app.state.bound_host = "fly-app.fly.dev"
    web_server.app.state.bound_port = 443
    web_server.app.state.auth_required = True
    client = TestClient(web_server.app, base_url="https://fly-app.fly.dev")
    yield client, secret, drain_secret
    web_server.app.state.bound_host = prev_host
    web_server.app.state.bound_port = prev_port
    web_server.app.state.auth_required = prev_required


class TestGatedEndToEnd:
    def test_bearer_secret_drives_the_external_surface(self, gated_kanban_app):
        client, secret, _ = gated_kanban_app
        auth = {"Authorization": f"Bearer {secret}"}

        health = client.get("/api/plugins/kanban/health", headers=auth)
        assert health.status_code == 200, health.text
        assert health.json()["service"] == "hermes-kanban"

        # The roster endpoint sits under the same token-guarded prefix.
        roster = client.get("/api/plugins/kanban/profiles", headers=auth)
        assert roster.status_code == 200, roster.text
        assert set(roster.json()) == {"profiles", "count"}

        created = client.post(
            "/api/plugins/kanban/tasks",
            headers={**auth, "Idempotency-Key": "e2e-op-1"},
            json={"title": "external e2e operation"},
        )
        assert created.status_code == 201, created.text
        task_id = created.json()["task"]["id"]

        replay = client.post(
            "/api/plugins/kanban/tasks",
            headers={**auth, "Idempotency-Key": "e2e-op-1"},
            json={"title": "external e2e operation"},
        )
        assert replay.status_code == 200, replay.text
        assert replay.json()["created"] is False
        assert replay.json()["task"]["id"] == task_id

    def test_missing_or_wrong_token_is_401(self, gated_kanban_app):
        client, _, _ = gated_kanban_app
        no_token = client.get("/api/plugins/kanban/tasks")
        assert no_token.status_code == 401
        wrong = client.get(
            "/api/plugins/kanban/tasks",
            headers={"Authorization": f"Bearer {_strong_secret()}"},
        )
        assert wrong.status_code == 401

    def test_foreign_scoped_credential_is_403(self, gated_kanban_app):
        # The drain secret authenticates (it's a real provider) but carries
        # scope "drain", not "kanban" — the seam must refuse it here.
        client, _, drain_secret = gated_kanban_app
        r = client.get(
            "/api/plugins/kanban/tasks",
            headers={"Authorization": f"Bearer {drain_secret}"},
        )
        assert r.status_code == 403

    def test_dashboard_subtree_stays_on_cookie_gate(self, gated_kanban_app):
        # The interactive operator surface is excluded from the token seam:
        # the kanban bearer credential must NOT open it, and without a cookie
        # session the gate rejects the request (401/302, never 200).
        client, secret, _ = gated_kanban_app
        with_bearer = client.get(
            "/api/plugins/kanban/dashboard/board",
            headers={"Authorization": f"Bearer {secret}"},
            follow_redirects=False,
        )
        assert with_bearer.status_code in (302, 401), with_bearer.status_code
        without = client.get(
            "/api/plugins/kanban/dashboard/board", follow_redirects=False
        )
        assert without.status_code in (302, 401), without.status_code
