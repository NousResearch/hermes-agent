"""Focused tests for Upstage Solar first-class provider wiring.

Regression guard for the bug where `hermes model` saved `provider: upstage`
correctly but, on re-entry, showed a different provider as active. Root cause:
`hermes_cli/providers.py` (the resolver behind `resolve_provider_full`) had no
`upstage` overlay, so `resolve_provider_full("upstage")` returned None, the
config provider was discarded, and resolution fell through to env auto-detect.
"""

from __future__ import annotations

import sys
import types


if "dotenv" not in sys.modules:
    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: None
    sys.modules["dotenv"] = fake_dotenv


class TestUpstageResolver:
    """The providers.py resolver must recognise upstage (the actual bug)."""

    def test_resolve_provider_full_recognizes_upstage(self):
        from hermes_cli.providers import resolve_provider_full

        pdef = resolve_provider_full("upstage", {}, [])
        assert pdef is not None, (
            "resolve_provider_full('upstage') returned None — config "
            "`provider: upstage` would be discarded and auto-detect would win"
        )
        assert pdef.id == "upstage"
        assert pdef.base_url == "https://api.upstage.ai/v1"
        assert "UPSTAGE_API_KEY" in pdef.api_key_env_vars

    def test_get_provider_returns_upstage_def(self):
        from hermes_cli.providers import get_provider

        pdef = get_provider("upstage")
        assert pdef is not None and pdef.id == "upstage"
        assert pdef.transport == "openai_chat"

    def test_solar_alias_normalizes_to_upstage(self):
        from hermes_cli.providers import normalize_provider, resolve_provider_full

        assert normalize_provider("solar") == "upstage"
        pdef = resolve_provider_full("solar", {}, [])
        assert pdef is not None and pdef.id == "upstage"


class TestUpstageOverlay:
    def test_overlay_exists(self):
        from hermes_cli.providers import HERMES_OVERLAYS

        assert "upstage" in HERMES_OVERLAYS
        overlay = HERMES_OVERLAYS["upstage"]
        assert overlay.transport == "openai_chat"
        assert overlay.extra_env_vars == ("UPSTAGE_API_KEY",)
        assert overlay.base_url_override == "https://api.upstage.ai/v1"
        assert overlay.base_url_env_var == "UPSTAGE_BASE_URL"
        assert not overlay.is_aggregator

    def test_provider_label(self):
        from hermes_cli.providers import get_label

        assert get_label("upstage") == "Upstage Solar"


class TestUpstageEnvCatalog:
    """The dashboard/desktop Providers page lists only OPTIONAL_ENV_VARS keys
    whose category is "provider". Without these entries UPSTAGE_API_KEY /
    UPSTAGE_BASE_URL never reach the frontend and Upstage stays invisible even
    though EnvPage.tsx has a matching PROVIDER_GROUPS prefix.
    """

    def test_optional_env_vars_include_upstage(self):
        from hermes_cli.config import OPTIONAL_ENV_VARS

        assert "UPSTAGE_API_KEY" in OPTIONAL_ENV_VARS
        assert OPTIONAL_ENV_VARS["UPSTAGE_API_KEY"]["category"] == "provider"
        assert OPTIONAL_ENV_VARS["UPSTAGE_API_KEY"]["password"] is True
        assert OPTIONAL_ENV_VARS["UPSTAGE_API_KEY"]["url"]

        assert "UPSTAGE_BASE_URL" in OPTIONAL_ENV_VARS
        assert OPTIONAL_ENV_VARS["UPSTAGE_BASE_URL"]["category"] == "provider"
        assert OPTIONAL_ENV_VARS["UPSTAGE_BASE_URL"]["password"] is False


class TestUpstageConfigProviderWins:
    """End-to-end: an explicit config provider must beat env auto-detect.

    Mirrors the display logic in `hermes_cli/main.py` (cmd_model): read
    `model.provider`, resolve it, and only fall back to auto-detect when that
    resolution fails. With a stray DEEPSEEK_API_KEY present (the user's case),
    upstage must still win because it is configured explicitly.
    """

    def test_explicit_upstage_beats_stray_deepseek_key(self, monkeypatch):
        from hermes_cli.providers import resolve_provider_full

        monkeypatch.setenv("DEEPSEEK_API_KEY", "junk")
        monkeypatch.setenv("UPSTAGE_API_KEY", "up-test-key")

        config_provider = "upstage"  # from config model.provider
        active = ""
        if config_provider and config_provider != "auto":
            adef = resolve_provider_full(config_provider, {}, [])
            active = adef.id if adef is not None else ""

        assert active == "upstage", (
            "explicit config provider should resolve to upstage, not fall "
            "through to deepseek auto-detect"
        )
