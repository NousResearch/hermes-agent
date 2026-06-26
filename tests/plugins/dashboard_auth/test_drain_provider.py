"""Tests for the DrainSecretProvider plugin (non-interactive bearer secret).

Task 2.0b. Loads the bundled drain plugin module directly and exercises:
  * the entropy gate (assess_secret_strength) — fail-closed on weak secrets,
  * constant-time verify_token returning a scoped TokenPrincipal,
  * the register(ctx) entry point's env/config resolution, skip reasons, and
    token-route registration.
"""
from __future__ import annotations

import secrets
from unittest.mock import MagicMock

import pytest

import plugins.dashboard_auth.drain as drain_plugin
from hermes_cli.dashboard_auth import TokenPrincipal, assert_protocol_compliance
from hermes_cli.dashboard_auth import token_auth


@pytest.fixture(scope="module")
def drain():
    return drain_plugin


@pytest.fixture(autouse=True)
def _clean_env_and_routes(monkeypatch):
    monkeypatch.delenv("HERMES_DASHBOARD_DRAIN_SECRET", raising=False)
    token_auth.clear_token_routes()
    yield
    token_auth.clear_token_routes()


def _strong_secret() -> str:
    # token_urlsafe(32) → 43 url-safe-b64 chars ≈ 256 bits.
    return secrets.token_urlsafe(32)


# ---------------------------------------------------------------------------
# Entropy gate
# ---------------------------------------------------------------------------


class TestEntropyGate:
    def test_strong_secret_passes(self, drain):
        assert drain.assess_secret_strength(_strong_secret()) is None

    def test_empty_rejected(self, drain):
        assert drain.assess_secret_strength("") is not None

    def test_too_short_rejected(self, drain):
        # 42 chars — one under the 43-char bar.
        assert drain.assess_secret_strength("a1B2c3" * 7) is not None

    def test_long_but_repeated_rejected(self, drain):
        # 60 chars, one distinct character → low distinct count + low entropy.
        assert drain.assess_secret_strength("a" * 60) is not None

    def test_long_but_few_distinct_rejected(self, drain):
        # 60 chars cycling through only 4 distinct characters.
        assert drain.assess_secret_strength("abcd" * 15) is not None

    def test_custom_min_chars_enforced(self, drain):
        s = _strong_secret()  # 43 chars
        assert drain.assess_secret_strength(s, min_chars=999) is not None


# ---------------------------------------------------------------------------
# Provider behaviour
# ---------------------------------------------------------------------------


class TestProvider:
    def test_protocol_compliance(self, drain):
        assert_protocol_compliance(drain.DrainSecretProvider)

    def test_supports_token_flag(self, drain):
        p = drain.DrainSecretProvider(secret=_strong_secret())
        assert p.supports_token is True

    def test_is_non_interactive(self, drain):
        # Excluded from interactive surfaces via list_session_providers().
        p = drain.DrainSecretProvider(secret=_strong_secret())
        assert p.supports_session is False

    def test_verify_token_accepts_matching_secret(self, drain):
        s = _strong_secret()
        p = drain.DrainSecretProvider(secret=s, scope="drain")
        principal = p.verify_token(token=s)
        assert isinstance(principal, TokenPrincipal)
        assert principal.principal == "drain-control"
        assert principal.provider == "drain-secret"
        assert principal.scopes == ("drain",)

    def test_verify_token_rejects_wrong_secret(self, drain):
        p = drain.DrainSecretProvider(secret=_strong_secret())
        assert p.verify_token(token=_strong_secret()) is None

    def test_verify_token_rejects_empty(self, drain):
        p = drain.DrainSecretProvider(secret=_strong_secret())
        assert p.verify_token(token="") is None

    def test_custom_scope_attached(self, drain):
        s = _strong_secret()
        p = drain.DrainSecretProvider(secret=s, scope="lifecycle")
        assert p.verify_token(token=s).scopes == ("lifecycle",)

    def test_construction_rejects_weak_secret(self, drain):
        with pytest.raises(ValueError):
            drain.DrainSecretProvider(secret="weak")

    def test_verify_session_returns_none_not_raises(self, drain):
        # Stacks harmlessly in the cookie-verify loop.
        p = drain.DrainSecretProvider(secret=_strong_secret())
        assert p.verify_session(access_token="anything") is None

    def test_interactive_methods_raise(self, drain):
        p = drain.DrainSecretProvider(secret=_strong_secret())
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
    def test_skips_when_no_secret(self, drain, monkeypatch):
        monkeypatch.setattr(drain, "_load_config_drain_auth_section", lambda: {})
        ctx = MagicMock()
        drain.register(ctx)
        ctx.register_dashboard_auth_provider.assert_not_called()
        assert "HERMES_DASHBOARD_DRAIN_SECRET" in drain.LAST_SKIP_REASON
        assert not token_auth.is_token_route(drain.DRAIN_ROUTE_PATH)

    def test_skips_and_fails_closed_on_weak_secret(self, drain, monkeypatch):
        monkeypatch.setenv("HERMES_DASHBOARD_DRAIN_SECRET", "tooweak")
        monkeypatch.setattr(drain, "_load_config_drain_auth_section", lambda: {})
        ctx = MagicMock()
        drain.register(ctx)
        ctx.register_dashboard_auth_provider.assert_not_called()
        assert "rejected" in drain.LAST_SKIP_REASON
        # fail-closed: the route is NOT token-authable, so it stays gated.
        assert not token_auth.is_token_route(drain.DRAIN_ROUTE_PATH)

    def test_registers_with_strong_env_secret(self, drain, monkeypatch):
        s = _strong_secret()
        monkeypatch.setenv("HERMES_DASHBOARD_DRAIN_SECRET", s)
        monkeypatch.setattr(drain, "_load_config_drain_auth_section", lambda: {})
        ctx = MagicMock()
        drain.register(ctx)
        ctx.register_dashboard_auth_provider.assert_called_once()
        provider = ctx.register_dashboard_auth_provider.call_args.args[0]
        assert isinstance(provider, drain.DrainSecretProvider)
        assert provider.verify_token(token=s) is not None
        assert drain.LAST_SKIP_REASON == ""
        # The drain endpoint is now token-authable.
        assert token_auth.is_token_route(drain.DRAIN_ROUTE_PATH)

    def test_config_scope_applied(self, drain, monkeypatch):
        s = _strong_secret()
        monkeypatch.setenv("HERMES_DASHBOARD_DRAIN_SECRET", s)
        monkeypatch.setattr(
            drain, "_load_config_drain_auth_section", lambda: {"scope": "lifecycle"}
        )
        ctx = MagicMock()
        drain.register(ctx)
        provider = ctx.register_dashboard_auth_provider.call_args.args[0]
        assert provider.verify_token(token=s).scopes == ("lifecycle",)

    def test_config_min_secret_chars_can_reject_otherwise_ok_secret(
        self, drain, monkeypatch
    ):
        s = _strong_secret()  # 43 chars — fine by default, too short at 999
        monkeypatch.setenv("HERMES_DASHBOARD_DRAIN_SECRET", s)
        monkeypatch.setattr(
            drain,
            "_load_config_drain_auth_section",
            lambda: {"min_secret_chars": 999},
        )
        ctx = MagicMock()
        drain.register(ctx)
        ctx.register_dashboard_auth_provider.assert_not_called()
        assert "rejected" in drain.LAST_SKIP_REASON
