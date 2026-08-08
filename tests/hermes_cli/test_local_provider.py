"""Tests for the 'local' provider overlay (vLLM, llama.cpp, etc.)."""

import pytest


# =============================================================================
# HERMES_OVERLAYS registration
# =============================================================================


class TestLocalOverlay:
    def test_local_in_hermes_overlays(self):
        from hermes_cli.providers import HERMES_OVERLAYS

        assert "local" in HERMES_OVERLAYS

    def test_local_overlay_transport(self):
        from hermes_cli.providers import HERMES_OVERLAYS

        overlay = HERMES_OVERLAYS["local"]
        assert overlay.transport == "openai_chat"

    def test_local_overlay_auth_type(self):
        from hermes_cli.providers import HERMES_OVERLAYS

        overlay = HERMES_OVERLAYS["local"]
        assert overlay.auth_type == "api_key"

    def test_local_overlay_base_url_env_var(self):
        from hermes_cli.providers import HERMES_OVERLAYS

        overlay = HERMES_OVERLAYS["local"]
        assert overlay.base_url_env_var == "LOCAL_BASE_URL"

    def test_local_overlay_no_base_url_override(self):
        """local should not have a hardcoded base_url_override —
        the user must provide the URL via config or env var."""
        from hermes_cli.providers import HERMES_OVERLAYS

        overlay = HERMES_OVERLAYS["local"]
        assert overlay.base_url_override == ""


# =============================================================================
# CANONICAL_PROVIDERS registration
# =============================================================================


class TestLocalCanonicalEntry:
    def test_local_in_canonical_providers(self):
        from hermes_cli.models import CANONICAL_PROVIDERS

        slugs = [p.slug for p in CANONICAL_PROVIDERS]
        assert "local" in slugs

    def test_local_entry_label(self):
        from hermes_cli.models import _PROVIDER_LABELS

        assert _PROVIDER_LABELS.get("local") == "Local Server"


# =============================================================================
# Alias mapping (vllm, llamacpp → local)
# =============================================================================


class TestLocalAliases:
    def test_vllm_aliases_to_local(self):
        from hermes_cli.providers import ALIASES

        assert ALIASES.get("vllm") == "local"

    def test_llamacpp_aliases_to_local(self):
        from hermes_cli.providers import ALIASES

        assert ALIASES.get("llamacpp") == "local"


# =============================================================================
# ENV VAR resolution
# =============================================================================


class TestLocalBaseUrlEnvVar:
    def test_local_base_url_from_env(self, monkeypatch):
        """LOCAL_BASE_URL env var should be picked up for base URL."""
        monkeypatch.setenv("LOCAL_BASE_URL", "http://192.168.1.100:8000/v1")
        from hermes_cli.providers import HERMES_OVERLAYS

        overlay = HERMES_OVERLAYS["local"]
        assert overlay.base_url_env_var == "LOCAL_BASE_URL"
