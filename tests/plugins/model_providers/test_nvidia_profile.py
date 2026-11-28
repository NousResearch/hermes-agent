"""Unit tests for the NVIDIA NIM provider profile and catalog curation (#156995)."""

from __future__ import annotations

import pytest


@pytest.fixture
def nvidia_profile():
    """Resolve the registered NVIDIA NIM provider profile."""
    import model_tools  # noqa: F401 - triggers plugin discovery
    import providers

    profile = providers.get_provider_profile("nvidia")
    assert profile is not None, "nvidia provider profile must be registered"
    return profile


class TestNvidiaProfileCatalog:
    """NVIDIA provider catalog and fallback models contract."""

    def test_profile_attributes(self, nvidia_profile):
        assert nvidia_profile.name == "nvidia"
        assert nvidia_profile.base_url == "https://integrate.api.nvidia.com/v1"
        assert "NVIDIA_API_KEY" in nvidia_profile.env_vars

    def test_fallback_models_are_nemotron_flagships(self, nvidia_profile):
        """Fallback models must be active Nemotron flagships, not retired models (#156995)."""
        assert isinstance(nvidia_profile.fallback_models, tuple)
        assert len(nvidia_profile.fallback_models) >= 2
        assert all(m.startswith("nvidia/nemotron-") for m in nvidia_profile.fallback_models)
        # Retired models must not appear in fallbacks
        retired = {
            "nvidia/llama-3.1-nemotron-70b-instruct",
            "nvidia/llama-3.3-70b-instruct",
            "nvidia/nemotron-3-super-120b-a12b",
            "z-ai/glm-5.3",
            "z-ai/glm-5.2",
            "moonshotai/kimi-k2.6",
            "minimaxai/minimax-m3",
        }
        for model in nvidia_profile.fallback_models:
            assert model not in retired

    def test_static_catalog_excludes_retired_models(self):
        """The curated static catalog for nvidia must exclude retired models (#156995)."""
        from hermes_cli.models_catalog_static import _PROVIDER_MODELS

        nvidia_catalog = _PROVIDER_MODELS.get("nvidia", [])
        assert nvidia_catalog, "nvidia catalog must not be empty"

        retired = {
            "nvidia/nemotron-3-super-120b-a12b",
            "z-ai/glm-5.3",
            "z-ai/glm-5.2",
            "moonshotai/kimi-k2.6",
            "minimaxai/minimax-m3",
        }
        for model in nvidia_catalog:
            assert model not in retired, f"retired model {model} should not be in nvidia static catalog"

        assert "nvidia/nemotron-3.5-lightning-30b-a3b" in nvidia_catalog
        assert "nvidia/nemotron-3-ultra-550b-a55b" in nvidia_catalog

    def test_nvidia_in_live_first_picker_providers(self):
        """NVIDIA NIM live catalog rotates; live models lead the picker (#156995)."""
        from hermes_cli.models_catalog_static import _LIVE_FIRST_PICKER_PROVIDERS

        assert "nvidia" in _LIVE_FIRST_PICKER_PROVIDERS


class TestNvidiaMessagePreparation:
    """NVIDIA NIM requires stripping 'name' and 'tool_name' from tool messages."""

    def test_prepare_messages_strips_name_from_tool_message(self, nvidia_profile):
        messages = [
            {"role": "system", "content": "You are a test agent."},
            {"role": "user", "content": "hello"},
            {"role": "tool", "content": "result", "name": "terminal", "tool_call_id": "call_1"},
        ]
        cleaned = nvidia_profile.prepare_messages(messages)
        assert cleaned[2] == {"role": "tool", "content": "result", "tool_call_id": "call_1"}
        assert "name" not in cleaned[2]

    def test_prepare_messages_untouched_returns_same_list(self, nvidia_profile):
        messages = [
            {"role": "system", "content": "You are a test agent."},
            {"role": "user", "content": "hello"},
        ]
        cleaned = nvidia_profile.prepare_messages(messages)
        assert cleaned is messages
