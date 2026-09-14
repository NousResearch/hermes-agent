"""Tests for OpenCode Free provider — registration, keyless contract, aliases.

The provider is KEYLESS: OpenCode's free tier is served anonymously and
rejects any unrecognized Authorization bearer with 401, so the provider
declares no env vars and every request goes out with an empty Authorization
header (see hermes_cli.models.opencode_zen_free_runtime).
"""

import os
from unittest.mock import patch


class TestOpenCodeFreeProviderRegistration:
    """Verify the opencode-free provider registers correctly."""

    def test_provider_is_registered(self):
        from providers import get_provider_profile
        profile = get_provider_profile("opencode-free")
        assert profile is not None
        assert profile.name == "opencode-free"

    def test_provider_has_correct_base_url(self):
        from providers import get_provider_profile
        profile = get_provider_profile("opencode-free")
        assert profile.base_url == "https://opencode.ai/zen/v1"

    def test_provider_is_keyless(self):
        """No env vars declared — the free tier requires no credential."""
        from providers import get_provider_profile
        profile = get_provider_profile("opencode-free")
        assert profile.env_vars == ()

    def test_provider_headers_override_sdk_bearer(self):
        """The profile's default headers blank Authorization so the SDK's
        Bearer never reaches the wire (the free tier 401s unknown bearers)."""
        from providers import get_provider_profile
        profile = get_provider_profile("opencode-free")
        assert profile.default_headers.get("Authorization") == ""

    def test_provider_uses_chat_completions_mode(self):
        from providers import get_provider_profile
        profile = get_provider_profile("opencode-free")
        assert profile.api_mode == "chat_completions"


class TestOpenCodeFreeAliases:
    """Verify alias resolution for the opencode-free provider."""

    def test_alias_free(self):
        from providers import get_provider_profile
        profile = get_provider_profile("free")
        assert profile is not None
        assert profile.name == "opencode-free"

    def test_alias_opencode_free(self):
        from providers import get_provider_profile
        profile = get_provider_profile("opencode_free")
        assert profile is not None
        assert profile.name == "opencode-free"


class TestOpenCodeFreeAuthAlias:
    """Verify the hardcoded alias in auth.py resolve_provider()."""

    def test_resolve_provider_free_alias(self):
        from hermes_cli.auth import resolve_provider
        # "free" should resolve to "opencode-free" without any credential
        result = resolve_provider("free")
        assert result == "opencode-free"


class TestOpenCodeFreeModelLists:
    """Curated keyless model lists exist and stay in sync."""

    def test_delisted_ox_alpha_not_in_floor(self):
        """x-preview-f-free was delisted by the relay 2026-08-26 (401s keyless);
        the offline floor must not offer it (#95914)."""
        from hermes_cli.models import _PROVIDER_MODELS
        assert "x-preview-f-free" not in _PROVIDER_MODELS["opencode-free"]


class TestOpenCodeFreeRuntimeKeyless:
    """The runtime resolver pins every opencode-free model keyless."""

    def test_free_provider_any_model_routes_keyless(self):
        from hermes_cli.models import (
            OPENCODE_ZEN_FREE_KEYLESS_PLACEHOLDER,
            opencode_zen_free_runtime,
        )
        rt = opencode_zen_free_runtime("opencode-free", "big-pickle")
        assert rt is not None
        assert rt["api_key"] == OPENCODE_ZEN_FREE_KEYLESS_PLACEHOLDER
        assert rt["base_url"] == "https://opencode.ai/zen/v1"
        assert rt["default_headers"]["Authorization"] == ""

    def test_free_provider_muse_routes_responses(self):
        """opencode-free inherits Zen's per-model endpoint routing."""
        from hermes_cli.models import opencode_zen_free_runtime
        rt = opencode_zen_free_runtime(
            "opencode-free", "muse-spark-1.2-contributor-free"
        )
        assert rt is not None
        assert rt["api_mode"] == "codex_responses"
