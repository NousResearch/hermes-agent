"""Focused tests for Token Kiosk provider wiring and resolution."""

from __future__ import annotations


class TestTokenKioskResolver:
    """The providers.py resolver must recognise token-kiosk and its aliases."""

    def test_resolve_provider_full_recognizes_token_kiosk(self):
        from hermes_cli.providers import resolve_provider_full

        pdef = resolve_provider_full("token-kiosk", {}, [])
        assert pdef is not None, (
            "resolve_provider_full('token-kiosk') returned None — config "
            "`provider: token-kiosk` would be discarded and auto-detect would win"
        )
        assert pdef.id == "token-kiosk"
        assert pdef.base_url == "https://api-token-kiosk.gaib.ai/v1"
        assert "TOKEN_KIOSK_API_KEY" in pdef.api_key_env_vars

    def test_resolve_provider_full_recognizes_aliases(self):
        from hermes_cli.providers import resolve_provider_full

        for alias in ("tokenkiosk", "token_kiosk", "tokenrouter"):
            pdef = resolve_provider_full(alias, {}, [])
            assert pdef is not None, f"Failed to resolve alias '{alias}'"
            assert pdef.id == "token-kiosk"


class TestTokenKioskOverlay:
    def test_overlay_exists(self):
        from hermes_cli.providers import HERMES_OVERLAYS

        assert "token-kiosk" in HERMES_OVERLAYS
        overlay = HERMES_OVERLAYS["token-kiosk"]
        assert overlay.transport == "openai_chat"
        assert overlay.is_aggregator is True
        assert overlay.extra_env_vars == ("TOKEN_KIOSK_API_KEY",)
        assert overlay.base_url_override == "https://api-token-kiosk.gaib.ai/v1"
        assert overlay.base_url_env_var == "TOKEN_KIOSK_BASE_URL"

    def test_provider_label(self):
        from hermes_cli.providers import get_label

        assert get_label("token-kiosk") == "Token Kiosk"


class TestTokenKioskEnvCatalog:
    """The dashboard/desktop Providers page lists only OPTIONAL_ENV_VARS keys
    whose category is "provider".
    """

    def test_optional_env_vars_include_token_kiosk(self):
        from hermes_cli.config import OPTIONAL_ENV_VARS

        assert "TOKEN_KIOSK_API_KEY" in OPTIONAL_ENV_VARS
        assert OPTIONAL_ENV_VARS["TOKEN_KIOSK_API_KEY"]["category"] == "provider"
        assert OPTIONAL_ENV_VARS["TOKEN_KIOSK_API_KEY"]["password"] is True
        assert OPTIONAL_ENV_VARS["TOKEN_KIOSK_API_KEY"]["url"]

        assert "TOKEN_KIOSK_BASE_URL" in OPTIONAL_ENV_VARS
        assert OPTIONAL_ENV_VARS["TOKEN_KIOSK_BASE_URL"]["category"] == "provider"
        assert OPTIONAL_ENV_VARS["TOKEN_KIOSK_BASE_URL"]["password"] is False


class TestTokenKioskModelNormalize:
    def test_matching_prefix_stripped(self):
        from hermes_cli.model_normalize import (
            _MATCHING_PREFIX_STRIP_PROVIDERS,
            normalize_model_for_provider,
        )

        assert "token-kiosk" in _MATCHING_PREFIX_STRIP_PROVIDERS
        assert (
            normalize_model_for_provider("token-kiosk/claude-3-5-sonnet", target_provider="token-kiosk")
            == "claude-3-5-sonnet"
        )


class TestTokenKioskProfile:
    def test_profile_registered_and_discoverable(self):
        from providers import get_provider_profile

        profile = get_provider_profile("token-kiosk")
        assert profile is not None
        assert profile.name == "token-kiosk"
        assert profile.display_name == "Token Kiosk"
        assert profile.base_url == "https://api-token-kiosk.gaib.ai/v1"
        assert profile.signup_url == "https://token-kiosk.gaib.ai"
        assert profile.auth_type == "api_key"
        assert profile.supports_vision is True
        assert "claude-3-5-sonnet" in profile.fallback_models

    def test_profile_aliases(self):
        from providers import get_provider_profile

        profile = get_provider_profile("tokenkiosk")
        assert profile is not None
        assert profile.name == "token-kiosk"


class TestTokenKioskConfigProviderWins:
    """End-to-end: explicit config provider must beat env auto-detect."""

    def test_explicit_token_kiosk_beats_stray_deepseek_key(self, monkeypatch):
        from hermes_cli.providers import resolve_provider_full

        monkeypatch.setenv("DEEPSEEK_API_KEY", "junk")
        monkeypatch.setenv("TOKEN_KIOSK_API_KEY", "tk-test-key")

        config_provider = "token-kiosk"
        active = ""
        if config_provider and config_provider != "auto":
            adef = resolve_provider_full(config_provider, {}, [])
            active = adef.id if adef is not None else ""

        assert active == "token-kiosk", (
            "explicit config provider should resolve to token-kiosk, not fall "
            "through to deepseek auto-detect"
        )
