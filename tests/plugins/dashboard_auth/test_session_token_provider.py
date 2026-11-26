"""Tests for the SessionTokenProvider plugin (the deployment session token as a
gated-mode credential).

Loads the bundled session_token plugin module directly and exercises:
  * registration — registers when HERMES_DASHBOARD_SESSION_TOKEN is provisioned
    (literal or file:<path>, BOM-tolerant), skips when it is absent, when the
    dashboard.session_token_auth.enabled knob opts out, and when the token file
    is unreadable (the file resolver's SystemExit mapped to a skip, never a
    discovery crash),
  * verify_session — constant-time accept of the provisioned token into a
    Session, None for anything else; refresh/revoke/start_login semantics,
  * interactive surfaces — interactive_login False keeps the provider off the
    login page (token-mode notice HTML instead) and out of /api/auth/providers.
"""
from __future__ import annotations

import secrets
from unittest.mock import MagicMock

import pytest

import plugins.dashboard_auth.session_token as session_token_plugin
from hermes_cli.dashboard_auth import RefreshExpiredError, assert_protocol_compliance
from hermes_cli.dashboard_auth import registry


@pytest.fixture(scope="module")
def plugin():
    return session_token_plugin


@pytest.fixture(autouse=True)
def _clean_env_and_registry(monkeypatch):
    monkeypatch.delenv("HERMES_DASHBOARD_SESSION_TOKEN", raising=False)
    registry.clear_providers()
    yield
    registry.clear_providers()


def _token() -> str:
    return secrets.token_urlsafe(32)


# ---------------------------------------------------------------------------
# Provider behaviour
# ---------------------------------------------------------------------------


class TestProvider:
    def test_protocol_compliance(self, plugin):
        assert_protocol_compliance(plugin.SessionTokenProvider)

    def test_is_a_session_credential_but_not_an_interactive_login(self, plugin):
        p = plugin.SessionTokenProvider(token=_token())
        assert p.supports_session is True
        assert p.supports_password is False
        assert p.interactive_login is False

    def test_verify_session_accepts_the_provisioned_token(self, plugin):
        token = _token()
        p = plugin.SessionTokenProvider(token=token)
        session = p.verify_session(access_token=token)
        assert session is not None
        assert session.provider == "session-token"
        assert session.user_id == "deployment-token"
        assert session.access_token == token
        assert session.expires_at > 0

    def test_verify_session_rejects_foreign_and_empty(self, plugin):
        p = plugin.SessionTokenProvider(token=_token())
        assert p.verify_session(access_token=secrets.token_urlsafe(32)) is None
        assert p.verify_session(access_token="") is None

    def test_refresh_raises_expired_and_revoke_is_silent(self, plugin):
        p = plugin.SessionTokenProvider(token=_token())
        with pytest.raises(RefreshExpiredError):
            p.refresh_session(refresh_token="anything")
        assert p.revoke_session(refresh_token="anything") is None

    def test_interactive_methods_raise(self, plugin):
        p = plugin.SessionTokenProvider(token=_token())
        with pytest.raises(NotImplementedError):
            p.start_login(redirect_uri="r")
        with pytest.raises(NotImplementedError):
            p.complete_login(code="c", state="s", code_verifier="v", redirect_uri="r")

    def test_construction_rejects_empty_token(self, plugin):
        with pytest.raises(ValueError):
            plugin.SessionTokenProvider(token="")


# ---------------------------------------------------------------------------
# register() entry point
# ---------------------------------------------------------------------------


class TestRegister:
    def test_skips_when_no_token_provisioned(self, plugin, monkeypatch):
        monkeypatch.setattr(plugin, "_load_session_token_auth_section", dict)
        ctx = MagicMock()
        plugin.register(ctx)
        ctx.register_dashboard_auth_provider.assert_not_called()
        assert "HERMES_DASHBOARD_SESSION_TOKEN" in plugin.LAST_SKIP_REASON

    def test_registers_with_env_token(self, plugin, monkeypatch):
        token = _token()
        monkeypatch.setenv("HERMES_DASHBOARD_SESSION_TOKEN", token)
        monkeypatch.setattr(plugin, "_load_session_token_auth_section", dict)
        ctx = MagicMock()
        plugin.register(ctx)
        ctx.register_dashboard_auth_provider.assert_called_once()
        provider = ctx.register_dashboard_auth_provider.call_args.args[0]
        assert isinstance(provider, plugin.SessionTokenProvider)
        assert provider.verify_session(access_token=token) is not None
        assert plugin.LAST_SKIP_REASON == ""

    def test_resolves_file_prefix(self, plugin, monkeypatch, tmp_path):
        token = _token()
        token_file = tmp_path / "hermes-gateway-token"
        token_file.write_text(f"\ufeff{token}\n", encoding="utf-8")  # BOM + newline
        monkeypatch.setenv("HERMES_DASHBOARD_SESSION_TOKEN", f"file:{token_file}")
        monkeypatch.setattr(plugin, "_load_session_token_auth_section", dict)
        ctx = MagicMock()
        plugin.register(ctx)
        ctx.register_dashboard_auth_provider.assert_called_once()
        provider = ctx.register_dashboard_auth_provider.call_args.args[0]
        assert provider.verify_session(access_token=token) is not None

    def test_skip_on_unreadable_token_file(self, plugin, monkeypatch, tmp_path):
        monkeypatch.setenv(
            "HERMES_DASHBOARD_SESSION_TOKEN", f"file:{tmp_path / 'missing'}")
        monkeypatch.setattr(plugin, "_load_session_token_auth_section", dict)
        ctx = MagicMock()
        plugin.register(ctx)
        ctx.register_dashboard_auth_provider.assert_not_called()
        assert plugin.LAST_SKIP_REASON

    def test_config_opt_out(self, plugin, monkeypatch):
        monkeypatch.setenv("HERMES_DASHBOARD_SESSION_TOKEN", _token())
        monkeypatch.setattr(
            plugin, "_load_session_token_auth_section", lambda: {"enabled": False})
        ctx = MagicMock()
        plugin.register(ctx)
        ctx.register_dashboard_auth_provider.assert_not_called()
        assert "enabled" in plugin.LAST_SKIP_REASON

    def test_short_token_still_registers_with_warning(
            self, plugin, monkeypatch, caplog):
        token = "short-but-mine"
        monkeypatch.setenv("HERMES_DASHBOARD_SESSION_TOKEN", token)
        monkeypatch.setattr(plugin, "_load_session_token_auth_section", dict)
        ctx = MagicMock()
        with caplog.at_level("WARNING"):
            plugin.register(ctx)
        ctx.register_dashboard_auth_provider.assert_called_once()
        assert any("short" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# dashboard.session_token_auth through the REAL config loader
# ---------------------------------------------------------------------------


class TestRealConfigLoader:
    """The opt-out key is read from an actual config.yaml under a temp HERMES_HOME via the same
    ``load_config_section`` path the plugin uses — no loader patch, so a misspelled section or
    key path fails here."""

    @pytest.fixture
    def home(self, monkeypatch, tmp_path):
        home = tmp_path / ".hermes"
        home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(home))
        monkeypatch.setenv("HERMES_DASHBOARD_SESSION_TOKEN", _token())
        return home

    def _register(self, plugin):
        ctx = MagicMock()
        plugin.register(ctx)
        return ctx

    def test_enabled_false_in_config_yaml_skips_registration(self, plugin, home):
        (home / "config.yaml").write_text(
            "dashboard:\n  session_token_auth:\n    enabled: false\n", encoding="utf-8")
        ctx = self._register(plugin)
        ctx.register_dashboard_auth_provider.assert_not_called()
        assert "enabled is false" in plugin.LAST_SKIP_REASON

    def test_enabled_true_in_config_yaml_registers(self, plugin, home):
        (home / "config.yaml").write_text(
            "dashboard:\n  session_token_auth:\n    enabled: true\n", encoding="utf-8")
        ctx = self._register(plugin)
        ctx.register_dashboard_auth_provider.assert_called_once()

    def test_absent_section_defaults_to_enabled(self, plugin, home):
        (home / "config.yaml").write_text("dashboard:\n  theme: default\n", encoding="utf-8")
        ctx = self._register(plugin)
        ctx.register_dashboard_auth_provider.assert_called_once()


# ---------------------------------------------------------------------------
# Interactive surfaces honour interactive_login=False
# ---------------------------------------------------------------------------


class TestInteractiveSurfaces:
    # A real provider subclass: registry.register_global_provider runs
    # assert_protocol_compliance(type(provider)), which needs the credential methods.
    @staticmethod
    def _stub(name: str, display_name: str, *, password: bool = False):
        cls = type("StubProvider", (session_token_plugin.SessionTokenProvider,), {
            "name": name, "display_name": display_name,
            "supports_password": password, "interactive_login": True})
        return cls(token=_token())

    def test_login_page_shows_token_mode_notice_not_a_button(self, plugin):
        from hermes_cli.dashboard_auth.login_page import render_login_html

        registry.register_global_provider(plugin.SessionTokenProvider(token=_token()))
        html = render_login_html(next_path="", prefix="")
        assert "Token-gated dashboard" in html
        assert "Deployment Session Token" in html
        assert "provider-btn" not in html
        assert "/auth/login?provider=" not in html

    def test_login_page_keeps_interactive_providers_clickable(self, plugin):
        from hermes_cli.dashboard_auth.login_page import render_login_html

        registry.register_global_provider(plugin.SessionTokenProvider(token=_token()))
        registry.register_global_provider(
            self._stub("basic", "Username & Password", password=True))
        html = render_login_html(next_path="", prefix="")
        assert "Token-gated dashboard" not in html
        # The password provider renders its form (a link is only for OAuth-style providers).
        assert 'data-provider="basic"' in html

    def test_api_auth_providers_filters_non_interactive(self, plugin):
        import asyncio

        from hermes_cli.dashboard_auth.routes import api_auth_providers

        registry.register_global_provider(plugin.SessionTokenProvider(token=_token()))
        providers = asyncio.run(api_auth_providers())
        assert providers == {"providers": []}

        registry.register_global_provider(self._stub("nous", "Nous Research"))
        providers = asyncio.run(api_auth_providers())
        assert [p["name"] for p in providers["providers"]] == ["nous"]
