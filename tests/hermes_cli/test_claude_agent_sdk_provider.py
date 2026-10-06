"""Discovery wiring for the claude-agent-sdk provider (#25267).

The runtime existed (runtime_provider short-circuit + agent runtime) but the
provider was invisible everywhere users look: no CANONICAL_PROVIDERS entry
(so no `hermes model` / setup row, no --provider choice derived from it), no
providers.py identity (label/transport/aliases), no auth status, no doctor
row. These tests pin each discovery surface, mirroring how openai-codex — the
closest twin (subscription, CLI-backed) — is wired.
"""

from __future__ import annotations


# =============================================================================
# CANONICAL_PROVIDERS (hermes model picker / setup wizard / --provider)
# =============================================================================


class TestCanonicalProvider:
    def test_in_canonical_providers(self):
        from hermes_cli.models import CANONICAL_PROVIDERS

        slugs = [p.slug for p in CANONICAL_PROVIDERS]
        assert "claude-agent-sdk" in slugs

    def test_label(self):
        from hermes_cli.models import CANONICAL_PROVIDERS

        entry = next(p for p in CANONICAL_PROVIDERS if p.slug == "claude-agent-sdk")
        assert entry.label == "Claude Agent SDK"

    def test_description_names_the_subscription(self):
        from hermes_cli.models import CANONICAL_PROVIDERS

        entry = next(p for p in CANONICAL_PROVIDERS if p.slug == "claude-agent-sdk")
        assert "subscription" in entry.tui_desc.lower()

    def test_setup_dispatch_routes_to_the_sdk_flow(self, monkeypatch):
        # --provider carries no choices= on main (user-defined providers are valid too), so the
        # discovery seam is the table select_provider_and_model dispatches through: without a row
        # the picker entry would fall through to the generic API-key flow. The row must resolve
        # the flow via hermes_cli.main so existing flow-test monkeypatches keep intercepting.
        import hermes_cli.main as main

        calls = []
        monkeypatch.setattr(main, "_model_flow_claude_agent_sdk", lambda c, m: calls.append((c, m)))
        main._PROVIDER_MODEL_FLOWS["claude-agent-sdk"]({"model": {}}, "claude-opus-4-8", None)
        assert calls == [({"model": {}}, "claude-opus-4-8")]


# =============================================================================
# providers.py identity (overlay, label, transport → api_mode, aliases)
# =============================================================================


class TestProviderIdentity:
    def test_get_provider_resolves_overlay(self):
        from hermes_cli.providers import get_provider

        pdef = get_provider("claude-agent-sdk")
        assert pdef is not None
        assert pdef.transport == "claude_agent_sdk"
        # Subscription OAuth owned by the SDK — not an API-key provider.
        assert pdef.auth_type == "oauth_external"
        # Discovery metadata only; Hermes never sends this anywhere.
        assert "CLAUDE_CODE_OAUTH_TOKEN" in pdef.api_key_env_vars

    def test_label_override(self):
        from hermes_cli.providers import get_label

        assert get_label("claude-agent-sdk") == "Claude Agent SDK"

    def test_transport_maps_to_the_runtime_api_mode(self):
        # Must agree with the runtime_provider short-circuit's api_mode, or
        # determine_api_mode callers would route SDK sessions onto
        # chat_completions.
        from hermes_cli.providers import determine_api_mode

        assert determine_api_mode("claude-agent-sdk") == "claude_agent_sdk"

    def test_api_mode_is_valid_for_config(self):
        from hermes_cli.runtime_provider import _parse_api_mode

        assert _parse_api_mode("claude_agent_sdk") == "claude_agent_sdk"

    def test_aliases_normalize_to_canonical(self):
        # Same accepted spellings as the runtime_provider short-circuit.
        from hermes_cli.providers import normalize_provider

        for alias in ("claude-sdk", "claude-code-sdk", "claude_agent_sdk"):
            assert normalize_provider(alias) == "claude-agent-sdk", alias

    def test_models_aliases_agree(self):
        from hermes_cli.models import _PROVIDER_ALIASES

        for alias in ("claude-sdk", "claude-code-sdk", "claude_agent_sdk"):
            assert _PROVIDER_ALIASES.get(alias) == "claude-agent-sdk", alias

    def test_not_an_aggregator(self):
        from hermes_cli.providers import is_aggregator

        assert is_aggregator("claude-agent-sdk") is False


# =============================================================================
# Desktop provider catalog (derives from CANONICAL_PROVIDERS + overlay)
# =============================================================================


class TestProviderCatalog:
    def test_in_catalog_on_accounts_tab(self):
        from hermes_cli.provider_catalog import provider_catalog_by_slug

        by = provider_catalog_by_slug()
        assert "claude-agent-sdk" in by
        # oauth_external → the accounts tab, like openai-codex.
        assert by["claude-agent-sdk"].tab == "accounts"


# =============================================================================
# Auth status (doctor row / list_available_providers "authenticated" flag)
# =============================================================================


class TestAuthStatus:
    def test_env_token_reports_logged_in(self, monkeypatch, tmp_path):
        from hermes_cli.auth import get_claude_agent_sdk_auth_status

        monkeypatch.setenv("HOME", str(tmp_path))  # no credential files
        monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "sk-ant-oat-fake")
        status = get_claude_agent_sdk_auth_status()
        assert status["logged_in"] is True
        assert status["source"] == "CLAUDE_CODE_OAUTH_TOKEN"

    def test_credential_file_reports_logged_in(self, monkeypatch, tmp_path):
        from hermes_cli.auth import get_claude_agent_sdk_auth_status

        monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
        monkeypatch.setenv("HOME", str(tmp_path))
        cred_dir = tmp_path / ".claude"
        cred_dir.mkdir()
        (cred_dir / ".credentials.json").write_text("{}")
        status = get_claude_agent_sdk_auth_status()
        assert status["logged_in"] is True
        assert status["source"].endswith(".credentials.json")

    def test_no_credentials_reports_hint_not_certainty(self, monkeypatch, tmp_path):
        # Keychain-only logins are NOT probed (a security(1) lookup can raise
        # an interactive prompt from doctor) — the hint must say so instead
        # of pretending certainty.
        from hermes_cli.auth import get_claude_agent_sdk_auth_status

        monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
        monkeypatch.setenv("HOME", str(tmp_path))
        status = get_claude_agent_sdk_auth_status()
        assert status["logged_in"] is False
        assert "Keychain" in status["hint"]

    def test_get_auth_status_dispatches(self, monkeypatch, tmp_path):
        from hermes_cli.auth import get_auth_status

        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "sk-ant-oat-fake")
        status = get_auth_status("claude-agent-sdk")
        assert status["logged_in"] is True
        assert status["provider"] == "claude-agent-sdk"


# =============================================================================
# Setup / model flow (no login flow, no API key — the SDK owns auth)
# =============================================================================


class TestModelFlow:
    def test_flow_saves_model_and_provider_with_empty_base_url(self, monkeypatch):
        import hermes_cli.auth as auth
        from hermes_cli.model_setup_flows import _model_flow_claude_agent_sdk

        saved = {}
        monkeypatch.setattr(
            auth, "get_claude_agent_sdk_auth_status",
            lambda: {"logged_in": True, "source": "CLAUDE_CODE_OAUTH_TOKEN"},
        )
        monkeypatch.setattr(
            auth, "_prompt_model_selection",
            lambda models, current_model="", **k: "claude-opus-4-8",
        )
        monkeypatch.setattr(
            auth, "_save_model_choice", lambda m: saved.__setitem__("model", m)
        )
        monkeypatch.setattr(
            auth, "_update_config_for_provider",
            lambda pid, base_url, default_model=None: saved.update(
                provider=pid, base_url=base_url
            ),
        )
        _model_flow_claude_agent_sdk({}, current_model="")
        assert saved["model"] == "claude-opus-4-8"
        assert saved["provider"] == "claude-agent-sdk"
        # The SDK owns the endpoint; "" also clears a stale base_url.
        assert saved["base_url"] == ""

    def test_flow_offers_claude_models(self, monkeypatch):
        import hermes_cli.auth as auth
        from hermes_cli.model_setup_flows import _model_flow_claude_agent_sdk

        offered = {}
        monkeypatch.setattr(
            auth, "get_claude_agent_sdk_auth_status",
            lambda: {"logged_in": False, "hint": "x"},
        )

        def _capture(models, current_model="", **k):
            offered["models"] = list(models)
            return None  # user cancels

        monkeypatch.setattr(auth, "_prompt_model_selection", _capture)
        _model_flow_claude_agent_sdk({}, current_model="")
        assert offered["models"], "flow offered no models"
        assert any(m.startswith("claude-") for m in offered["models"])

    def test_main_dispatch_surface_exists(self):
        # select_provider_and_model dispatches via the name re-imported into
        # hermes_cli.main (the seam existing flow tests monkeypatch).
        import hermes_cli.main as main

        assert callable(main._model_flow_claude_agent_sdk)
