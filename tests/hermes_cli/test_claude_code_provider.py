"""Tests for the ``claude-code`` provider (Claude subscription via the Agent SDK).

The contract under test is a boundary, not a feature list: Hermes holds no
Claude credential, asks the official CLI for auth state instead of reading
credential files, and never claims a Claude subscription under the API-key
``anthropic`` provider.

Every ``claude`` invocation is mocked. These tests never spawn the real binary
and never touch the network.
"""

from __future__ import annotations

import subprocess

import pytest

from hermes_cli import claude_code as cc


@pytest.fixture(autouse=True)
def _closed_gate(monkeypatch):
    """Default to a deterministic, closed subscription gate.

    The gate reads config.yaml; pinning it keeps these tests independent of
    whatever the isolated HERMES_HOME happens to contain.
    """
    cc.reset_subscription_gate_cache()
    monkeypatch.setattr(cc, "subscription_enabled", lambda config=None: False)
    yield
    cc.reset_subscription_gate_cache()


# =============================================================================
# Profile identity
# =============================================================================

class TestProfile:
    def _profile(self):
        from providers import get_provider_profile

        prof = get_provider_profile("claude-code")
        assert prof is not None, "claude-code provider profile is not registered"
        return prof

    def test_carries_no_credential_env_vars(self):
        """Load-bearing: the SDK resolves auth, so Hermes declares no credential."""
        assert self._profile().env_vars == ()

    def test_api_mode_and_auth_type(self):
        prof = self._profile()
        assert prof.api_mode == "claude_agent_sdk"
        assert prof.auth_type == "external_process"

    def test_base_url_is_an_internal_scheme_not_http(self):
        """Nothing may mistake this provider for a reachable REST endpoint."""
        base = self._profile().base_url
        assert base and not base.startswith(("http://", "https://"))

    def test_no_rest_model_listing(self):
        assert self._profile().fetch_models() is None

    def test_display_metadata_names_the_login_step(self):
        prof = self._profile()
        assert prof.display_name
        assert "claude auth login" in prof.description.lower()


class TestAnthropicProfileNoLongerClaimsClaudeCode:
    """The API-key provider must not present itself as the subscription path."""

    def _anthropic(self):
        from providers import get_provider_profile

        prof = get_provider_profile("anthropic")
        assert prof is not None
        return prof

    def test_claude_code_aliases_dropped(self):
        aliases = set(self._anthropic().aliases)
        assert "claude-code" not in aliases
        assert "claude-oauth" not in aliases

    def test_subscription_token_env_var_dropped(self):
        assert "CLAUDE_CODE_OAUTH_TOKEN" not in set(self._anthropic().env_vars)

    def test_overlay_drops_the_subscription_token(self):
        from hermes_cli.providers import HERMES_OVERLAYS

        assert "CLAUDE_CODE_OAUTH_TOKEN" not in set(
            HERMES_OVERLAYS["anthropic"].extra_env_vars
        )


# =============================================================================
# Catalog gating + parity
# =============================================================================

class TestCatalogGate:
    """The catalog's parity contract must hold with the gate on OR off.

    The gate is applied at CANONICAL_PROVIDERS (the shared universe), so both
    desktop tabs always see the same provider set. These tests assert that
    relationship rather than a provider count.
    """

    def _catalog_for(self, monkeypatch, enabled: bool):
        import importlib

        import hermes_cli.claude_code as claude_code_mod
        import hermes_cli.models as models_mod
        import hermes_cli.provider_catalog as catalog_mod

        monkeypatch.setattr(
            claude_code_mod, "subscription_enabled", lambda config=None: enabled
        )
        models_mod = importlib.reload(models_mod)
        catalog_mod = importlib.reload(catalog_mod)
        try:
            return (
                [e.slug for e in models_mod.CANONICAL_PROVIDERS],
                catalog_mod.provider_catalog_by_slug(),
            )
        finally:
            # Restore the real gate for anything importing these later.
            monkeypatch.undo()
            importlib.reload(models_mod)
            importlib.reload(catalog_mod)

    def test_absent_from_the_universe_when_the_gate_is_closed(self, monkeypatch):
        slugs, by_slug = self._catalog_for(monkeypatch, enabled=False)
        assert "claude-code" not in slugs
        assert "claude-code" not in by_slug

    def test_lands_on_the_accounts_tab_when_the_gate_is_open(self, monkeypatch):
        slugs, by_slug = self._catalog_for(monkeypatch, enabled=True)
        assert "claude-code" in slugs
        assert by_slug["claude-code"].tab == "accounts"
        assert by_slug["claude-code"].auth_type == "external_process"
        # No credential env var to configure — that is the whole point.
        assert by_slug["claude-code"].api_key_env_vars == ()

    @pytest.mark.parametrize("enabled", [False, True])
    def test_tab_union_still_equals_the_universe(self, monkeypatch, enabled):
        """The locked parity contract, in both gate states."""
        slugs, by_slug = self._catalog_for(monkeypatch, enabled=enabled)
        union = {d.slug for d in by_slug.values() if d.tab in {"keys", "accounts"}}
        assert union == set(slugs)


# =============================================================================
# Legacy alias back-compat
# =============================================================================

class TestLegacyAliases:
    def test_legacy_slugs_still_reach_anthropic_while_the_gate_is_closed(self):
        """An existing ``provider: claude-code`` config must not start erroring."""
        assert cc.legacy_alias_target("claude-code") == "anthropic"
        assert cc.legacy_alias_target("claude-oauth") == "anthropic"

    def test_slug_belongs_to_the_new_provider_once_the_gate_is_open(self, monkeypatch):
        monkeypatch.setattr(cc, "subscription_enabled", lambda config=None: True)
        assert cc.legacy_alias_target("claude-code") is None
        assert cc.legacy_alias_target("claude-oauth") is None

    def test_unrelated_slugs_are_untouched(self):
        assert cc.legacy_alias_target("anthropic") is None
        assert cc.legacy_alias_target("openrouter") is None

    @pytest.mark.parametrize("slug", ["claude-code", "claude-oauth"])
    def test_every_legacy_slug_still_resolves_in_both_gate_states(
        self, monkeypatch, slug
    ):
        """No gate state may leave a previously-valid provider slug unresolvable.

        Regression guard: dropping ``claude-oauth`` from the anthropic profile
        without re-homing it made ``provider: claude-oauth`` raise "Unknown
        provider" the moment a user enabled the subscription gate.
        """
        from hermes_cli.auth import resolve_provider
        from hermes_cli.providers import normalize_provider

        for enabled in (False, True):
            monkeypatch.setattr(cc, "subscription_enabled", lambda config=None: enabled)
            expected = "claude-code" if enabled else "anthropic"
            assert resolve_provider(slug) == expected
            assert normalize_provider(slug) == expected

    def test_gate_open_routes_legacy_slugs_to_a_real_provider_definition(
        self, monkeypatch
    ):
        """The slug must land on a definition, not a dangling alias target."""
        from hermes_cli.providers import get_provider

        monkeypatch.setattr(cc, "subscription_enabled", lambda config=None: True)
        pdef = get_provider("claude-oauth", allow_network=False)
        assert pdef is not None and pdef.id == "claude-code"
        assert pdef.auth_type == "external_process"
        assert pdef.api_key_env_vars == ()


class TestMigrationNotice:
    def test_legacy_slug_gets_both_billing_options(self):
        notice = cc.legacy_provider_notice("claude-code", env={})
        assert "anthropic" in notice
        assert "claude auth login" in notice
        assert "billed" in notice.lower() or "billing" in notice.lower()

    def test_anthropic_with_a_subscription_token_gets_the_notice(self):
        notice = cc.legacy_provider_notice(
            "anthropic", env={"CLAUDE_CODE_OAUTH_TOKEN": "sk-ant-oat01-x"}
        )
        assert notice

    def test_plain_api_key_setup_is_left_alone(self):
        # An API key outranks any Claude login, so there is nothing to warn
        # about and no reason to spawn the auth probe.
        assert cc.legacy_provider_notice(
            "anthropic", env={"ANTHROPIC_API_KEY": "sk-ant-api03-x"}
        ) == ""
        assert cc.legacy_provider_notice("openrouter", env={}) == ""

    def test_anthropic_without_a_key_warns_when_a_claude_login_exists(
        self, monkeypatch
    ):
        # The `anthropic` picker row reports itself authenticated off a Claude
        # Code credential on disk; selecting it then bills extra usage. The
        # probe is stubbed so the assertion does not depend on whether the
        # machine running the suite happens to be signed in.
        cc.reset_probe_cache()
        monkeypatch.setattr(
            cc, "probe_claude_auth_cached", lambda *a, **k: {"logged_in": True}
        )
        assert cc.legacy_provider_notice("anthropic", env={})

    def test_anthropic_without_a_key_is_silent_when_not_signed_in(self, monkeypatch):
        cc.reset_probe_cache()
        monkeypatch.setattr(
            cc, "probe_claude_auth_cached", lambda *a, **k: {"logged_in": False}
        )
        assert cc.legacy_provider_notice("anthropic", env={}) == ""

    def test_notice_states_that_nothing_changed_automatically(self):
        """AGENTS.md: never silently switch a user's billing source."""
        notice = cc.legacy_provider_notice("claude-code", env={})
        assert "automatically" in notice


# =============================================================================
# Status probe — the `claude` subprocess is always mocked
# =============================================================================

def _fake_run(results):
    """Build a subprocess.run stub driven by ``{argv_tail: outcome}``.

    An outcome is a ``(returncode, stdout)`` pair, or an exception instance to
    raise. Asserts stdin is muzzled on every call: Claude Code blocks forever
    on an unusable inherited stdin.
    """
    def _run(cmd, **kwargs):
        assert kwargs.get("stdin") is subprocess.DEVNULL, (
            "claude must be spawned with stdin=DEVNULL or it can block forever"
        )
        assert kwargs.get("timeout"), "every claude probe must be time-bounded"
        key = tuple(cmd[1:])
        outcome = results[key]
        if isinstance(outcome, BaseException):
            raise outcome
        returncode, stdout = outcome
        return subprocess.CompletedProcess(cmd, returncode, stdout, "")

    return _run


AUTH_ARGS = ("auth", "status")
VERSION_ARGS = ("--version",)


@pytest.fixture
def claude_installed(monkeypatch):
    monkeypatch.setattr(cc, "resolve_claude_binary", lambda: "/usr/bin/claude")


class TestProbeClaudeAuth:
    def test_logged_in_via_claude_ai_is_a_subscription(self, monkeypatch, claude_installed):
        monkeypatch.setattr(subprocess, "run", _fake_run({
            AUTH_ARGS: (0, '{"loggedIn": true, "authMethod": "claude.ai", '
                           '"email": "u@example.com", "subscriptionType": "max"}'),
            VERSION_ARGS: (0, "2.1.220 (Claude Code)\n"),
        }))
        info = cc.probe_claude_auth()
        assert info["logged_in"] is True
        assert info["subscription"] is True
        assert info["auth_method"] == "claude.ai"
        assert info["subscription_type"] == "max"
        assert info["cli_version"].startswith("2.1.220")

    def test_logged_in_via_api_key_is_not_a_subscription(self, monkeypatch, claude_installed):
        """A user on an API key is billed as API usage and must be told so."""
        monkeypatch.setattr(subprocess, "run", _fake_run({
            AUTH_ARGS: (0, '{"loggedIn": true, "authMethod": "api-key"}'),
            VERSION_ARGS: (0, "2.1.220\n"),
        }))
        info = cc.probe_claude_auth()
        assert info["logged_in"] is True
        assert info["subscription"] is False
        assert info["auth_method"] == "api-key"
        assert "api" in info["message"].lower()

    def test_not_logged_in_exits_one(self, monkeypatch, claude_installed):
        monkeypatch.setattr(subprocess, "run", _fake_run({
            AUTH_ARGS: (1, '{"loggedIn": false}'),
            VERSION_ARGS: (0, "2.1.220\n"),
        }))
        info = cc.probe_claude_auth()
        assert info["logged_in"] is False
        assert info["status"] == "logged_out"
        assert "claude auth login" in info["message"]

    def test_binary_missing_is_actionable_not_fatal(self, monkeypatch):
        monkeypatch.setattr(cc, "resolve_claude_binary", lambda: None)
        info = cc.probe_claude_auth()
        assert info["status"] == "cli_missing"
        assert info["logged_in"] is False
        assert "install" in info["message"].lower()

    def test_timeout_degrades_to_unknown(self, monkeypatch, claude_installed):
        monkeypatch.setattr(subprocess, "run", _fake_run({
            AUTH_ARGS: subprocess.TimeoutExpired(cmd="claude", timeout=5),
            VERSION_ARGS: subprocess.TimeoutExpired(cmd="claude", timeout=5),
        }))
        info = cc.probe_claude_auth()
        assert info["status"] == "unknown"
        assert info["logged_in"] is False
        assert info["message"]

    def test_malformed_json_falls_back_to_the_exit_code(self, monkeypatch, claude_installed):
        monkeypatch.setattr(subprocess, "run", _fake_run({
            AUTH_ARGS: (0, "Logged in as u@example.com"),
            VERSION_ARGS: (0, "2.1.220\n"),
        }))
        info = cc.probe_claude_auth()
        assert info["logged_in"] is True
        assert info["status"] == "logged_in"

    def test_malformed_json_with_failure_exit_is_logged_out(self, monkeypatch, claude_installed):
        monkeypatch.setattr(subprocess, "run", _fake_run({
            AUTH_ARGS: (1, "Not logged in"),
            VERSION_ARGS: (0, "2.1.220\n"),
        }))
        info = cc.probe_claude_auth()
        assert info["logged_in"] is False

    def test_probe_never_returns_credential_material(self, monkeypatch, claude_installed):
        """The probe reports state, never a token."""
        monkeypatch.setattr(subprocess, "run", _fake_run({
            AUTH_ARGS: (0, '{"loggedIn": true, "authMethod": "claude.ai"}'),
            VERSION_ARGS: (0, "2.1.220\n"),
        }))
        info = cc.probe_claude_auth()
        assert not any(
            k in info for k in ("token", "access_token", "accessToken", "refresh_token")
        )


class TestAuthIntegration:
    """The shared external-process entry points must route claude-code correctly."""

    def test_status_matches_the_external_process_shape(self, monkeypatch, claude_installed):
        from hermes_cli.auth import (
            get_external_process_provider_status,
            get_auth_status,
        )

        monkeypatch.setattr(subprocess, "run", _fake_run({
            AUTH_ARGS: (0, '{"loggedIn": true, "authMethod": "claude.ai"}'),
            VERSION_ARGS: (0, "2.1.220\n"),
        }))
        status = get_external_process_provider_status("claude-code")
        for key in (
            "configured", "provider", "name", "command", "args",
            "resolved_command", "base_url", "logged_in",
        ):
            assert key in status, f"{key} missing from the external-process shape"
        assert status["provider"] == "claude-code"
        assert status["logged_in"] is True
        # Same dict must come back through the generic dispatcher.
        assert get_auth_status("claude-code")["provider"] == "claude-code"

    def test_status_survives_a_broken_cli(self, monkeypatch, claude_installed):
        from hermes_cli.auth import get_external_process_provider_status

        def _boom(*a, **kw):
            raise OSError("exec format error")

        monkeypatch.setattr(subprocess, "run", _boom)
        status = get_external_process_provider_status("claude-code")
        assert status["logged_in"] is False
        assert status["message"]

    def test_credentials_carry_no_secret(self, monkeypatch, claude_installed):
        from hermes_cli.auth import resolve_external_process_provider_credentials

        monkeypatch.setattr(subprocess, "run", _fake_run({
            AUTH_ARGS: (0, '{"loggedIn": true, "authMethod": "claude.ai"}'),
            VERSION_ARGS: (0, "2.1.220\n"),
        }))
        creds = resolve_external_process_provider_credentials("claude-code")
        assert creds["provider"] == "claude-code"
        assert creds["api_key"] == "", "Hermes must not hand out Claude credential material"
        assert not creds["base_url"].startswith(("http://", "https://"))

    def test_missing_cli_raises_an_actionable_auth_error(self, monkeypatch):
        from hermes_cli.auth import (
            AuthError,
            resolve_external_process_provider_credentials,
        )

        monkeypatch.setattr(cc, "resolve_claude_binary", lambda: None)
        with pytest.raises(AuthError) as exc:
            resolve_external_process_provider_credentials("claude-code")
        assert "claude auth login" in str(exc.value)


class TestDashboardCard:
    def test_logout_never_deletes_a_credential_file(self):
        """Regression guard: the dashboard used to rm the credential store."""
        from hermes_cli.web_server import _oauth_provider_disconnect_command

        cmd = _oauth_provider_disconnect_command(
            {"id": "claude-code", "flow": "external"}
        )
        assert cmd == "claude auth logout"

    def test_card_status_exposes_no_token_preview(self, monkeypatch, claude_installed):
        from hermes_cli.web_server import _claude_code_only_status

        monkeypatch.setattr(subprocess, "run", _fake_run({
            AUTH_ARGS: (0, '{"loggedIn": true, "authMethod": "claude.ai"}'),
            VERSION_ARGS: (0, "2.1.220\n"),
        }))
        status = _claude_code_only_status()
        assert status["logged_in"] is True
        assert status["token_preview"] is None

    def test_api_key_login_is_not_reported_as_a_subscription(self, monkeypatch, claude_installed):
        from hermes_cli.web_server import _claude_code_only_status

        monkeypatch.setattr(subprocess, "run", _fake_run({
            AUTH_ARGS: (0, '{"loggedIn": true, "authMethod": "api-key"}'),
            VERSION_ARGS: (0, "2.1.220\n"),
        }))
        status = _claude_code_only_status()
        assert status["subscription"] is False
        assert status["logged_in"] is False, (
            "an API-key login is not subscription mode; the card must not claim it is"
        )
