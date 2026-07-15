"""Focused tests for SCX.ai first-class provider wiring.

Mirrors the Upstage provider tests: verifies the resolver overlay
(`hermes_cli/providers.py`), auth registry auto-extension, the env-var
catalog, `provider:model` parsing, and picker/setup visibility — the full
fast-path contract for a bundled OpenAI-compatible api-key provider.
"""

from __future__ import annotations

import sys
import types


if "dotenv" not in sys.modules:
    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: None
    sys.modules["dotenv"] = fake_dotenv


class TestScxResolver:
    """The providers.py resolver must recognise scx (the upstage-bug class)."""

    def test_resolve_provider_full_recognizes_scx(self):
        from hermes_cli.providers import resolve_provider_full

        pdef = resolve_provider_full("scx", {}, [])
        assert pdef is not None, (
            "resolve_provider_full('scx') returned None — config "
            "`provider: scx` would be discarded and auto-detect would win"
        )
        assert pdef.id == "scx"
        assert pdef.base_url == "https://api.scx.ai/v1"
        assert "SCX_API_KEY" in pdef.api_key_env_vars

    def test_get_provider_returns_scx_def(self):
        from hermes_cli.providers import get_provider

        pdef = get_provider("scx")
        assert pdef is not None and pdef.id == "scx"
        assert pdef.transport == "openai_chat"

    def test_alias_normalizes_to_scx(self):
        from hermes_cli.providers import normalize_provider, resolve_provider_full

        assert normalize_provider("scx-ai") == "scx"
        pdef = resolve_provider_full("scx-ai", {}, [])
        assert pdef is not None and pdef.id == "scx"


class TestScxOverlay:
    def test_overlay_exists(self):
        from hermes_cli.providers import HERMES_OVERLAYS

        assert "scx" in HERMES_OVERLAYS
        overlay = HERMES_OVERLAYS["scx"]
        assert overlay.transport == "openai_chat"
        assert overlay.extra_env_vars == ("SCX_API_KEY",)
        assert overlay.base_url_override == "https://api.scx.ai/v1"
        assert overlay.base_url_env_var == "SCX_BASE_URL"
        assert not overlay.is_aggregator

    def test_provider_label(self):
        from hermes_cli.providers import get_label

        assert get_label("scx") == "SCX.ai"


class TestScxAuthRegistry:
    """PROVIDER_REGISTRY auto-extends from the plugin profile."""

    def test_auth_registry_has_scx(self):
        from hermes_cli.auth import PROVIDER_REGISTRY

        cfg = PROVIDER_REGISTRY.get("scx")
        assert cfg is not None, "scx must be auto-registered from its profile"
        assert cfg.auth_type == "api_key"
        assert cfg.api_key_env_vars == ("SCX_API_KEY",)
        assert cfg.base_url_env_var == "SCX_BASE_URL"
        assert cfg.inference_base_url == "https://api.scx.ai/v1"

    def test_auth_registry_resolves_alias(self):
        from hermes_cli.auth import PROVIDER_REGISTRY

        assert PROVIDER_REGISTRY.get("scx-ai") is PROVIDER_REGISTRY["scx"]


class TestScxModelPicker:
    """`hermes model` / `hermes setup` picker + provider:model parsing."""

    def test_scx_in_canonical_providers(self):
        from hermes_cli.models import CANONICAL_PROVIDERS, _PROVIDER_LABELS

        assert any(p.slug == "scx" for p in CANONICAL_PROVIDERS), (
            "scx must auto-extend CANONICAL_PROVIDERS from its profile"
        )
        assert _PROVIDER_LABELS.get("scx") == "SCX.ai"

    def test_provider_model_syntax_parses(self):
        from hermes_cli.models import parse_model_input

        assert parse_model_input("scx:coder", "openrouter") == ("scx", "coder")
        assert parse_model_input("scx:MAGPiE", "openrouter") == ("scx", "MAGPiE")
        # The alias works in the colon syntax too (gmi-cloud precedent).
        assert parse_model_input("scx-ai:coder", "openrouter") == ("scx", "coder")

    def test_default_model_is_coder(self):
        from hermes_cli.models import get_default_model_for_provider

        # Silent fallback when `provider: scx` is configured with no model —
        # reads fallback_models[0] from the profile (no static
        # _PROVIDER_MODELS entry for plugin providers).
        assert get_default_model_for_provider("scx") == "coder"

    def test_picker_catalog_is_curated_flagships(self, monkeypatch):
        from hermes_cli.models import provider_model_ids

        # With or without a key the picker shows exactly the curated
        # flagship list — fetch_models is disabled on the profile, so the
        # live catalog's non-agentic entries never leak in.
        monkeypatch.setenv("SCX_API_KEY", "scx-test-key")
        assert provider_model_ids("scx") == ["coder", "MAGPiE", "MiniMax-M2.7"]
        monkeypatch.delenv("SCX_API_KEY", raising=False)
        assert provider_model_ids("scx") == ["coder", "MAGPiE", "MiniMax-M2.7"]


class TestScxRuntimeResolution:
    def test_runtime_provider_resolution(self, monkeypatch):
        from hermes_cli.runtime_provider import resolve_runtime_provider

        monkeypatch.setenv("SCX_API_KEY", "scx-test-key")
        rt = resolve_runtime_provider(requested="scx", target_model="coder")
        assert rt["provider"] == "scx"
        assert rt["base_url"] == "https://api.scx.ai/v1"
        assert rt["api_mode"] == "chat_completions"
        assert rt["api_key"] == "scx-test-key"

    def test_base_url_env_override(self, monkeypatch):
        from hermes_cli.runtime_provider import resolve_runtime_provider

        monkeypatch.setenv("SCX_API_KEY", "scx-test-key")
        monkeypatch.setenv("SCX_BASE_URL", "https://au.api.scx.ai/v1")
        rt = resolve_runtime_provider(requested="scx", target_model="coder")
        assert rt["base_url"] == "https://au.api.scx.ai/v1"


class TestScxEnvCatalog:
    """The dashboard/desktop Providers page lists only OPTIONAL_ENV_VARS keys
    whose category is "provider" — without these entries SCX_API_KEY /
    SCX_BASE_URL never reach the frontend.
    """

    def test_optional_env_vars_include_scx(self):
        from hermes_cli.config import OPTIONAL_ENV_VARS

        assert "SCX_API_KEY" in OPTIONAL_ENV_VARS
        assert OPTIONAL_ENV_VARS["SCX_API_KEY"]["category"] == "provider"
        assert OPTIONAL_ENV_VARS["SCX_API_KEY"]["password"] is True
        assert OPTIONAL_ENV_VARS["SCX_API_KEY"]["url"]

        assert "SCX_BASE_URL" in OPTIONAL_ENV_VARS
        assert OPTIONAL_ENV_VARS["SCX_BASE_URL"]["category"] == "provider"
        assert OPTIONAL_ENV_VARS["SCX_BASE_URL"]["password"] is False


class TestScxConfigProviderWins:
    """An explicit config provider must beat env auto-detect."""

    def test_explicit_scx_beats_stray_deepseek_key(self, monkeypatch):
        from hermes_cli.providers import resolve_provider_full

        monkeypatch.setenv("DEEPSEEK_API_KEY", "junk")
        monkeypatch.setenv("SCX_API_KEY", "scx-test-key")

        config_provider = "scx"  # from config model.provider
        active = ""
        if config_provider and config_provider != "auto":
            adef = resolve_provider_full(config_provider, {}, [])
            active = adef.id if adef is not None else ""

        assert active == "scx", (
            "explicit config provider should resolve to scx, not fall "
            "through to deepseek auto-detect"
        )
