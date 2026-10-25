"""Unit tests for the Kosmik provider profile."""

from __future__ import annotations

import pytest


@pytest.fixture
def kosmik_profile():
    """Resolve the registered Kosmik profile via the provider registry."""
    import model_tools  # noqa: F401
    import providers

    profile = providers.get_provider_profile("kosmik")
    assert profile is not None, "kosmik provider profile must be registered"
    return profile


class TestKosmikProfileIdentity:
    """Verify Kosmik profile identity, aliases, and endpoint configuration."""

    def test_profile_attributes(self, kosmik_profile):
        assert kosmik_profile.name == "kosmik"
        assert kosmik_profile.display_name == "Kosmik"
        assert kosmik_profile.base_url == "https://api.koscompute.com/v1"
        assert kosmik_profile.auth_type == "api_key"
        assert "KOSMIK_API_KEY" in kosmik_profile.env_vars
        assert "KOSCOMPUTE_API_KEY" in kosmik_profile.env_vars
        assert "KOSMIK_BASE_URL" in kosmik_profile.env_vars

    @pytest.mark.parametrize("alias", ["koscompute", "kosmik-ai", "kos"])
    def test_alias_resolution(self, alias):
        import providers

        profile = providers.get_provider_profile(alias)
        assert profile is not None
        assert profile.name == "kosmik"

    def test_fallback_models(self, kosmik_profile):
        assert kosmik_profile.fallback_models == ("qwen/qwen3.8-27b",)


class TestKosmikReasoning:
    """build_api_kwargs_extras maps Hermes reasoning controls to reasoning_effort."""

    def test_reasoning_disabled_emits_none(self, kosmik_profile):
        extra_body, top_level = kosmik_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": False}
        )
        assert extra_body == {}
        assert top_level == {"reasoning_effort": "none"}

    @pytest.mark.parametrize("effort", ["low", "medium", "high"])
    def test_explicit_effort_passes_through(self, kosmik_profile, effort):
        extra_body, top_level = kosmik_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": effort}
        )
        assert extra_body == {}
        assert top_level == {"reasoning_effort": effort}


class TestKosmikModelFiltering:
    """fetch_models filters non-chat audio/TTS models from the catalog."""

    def test_fetch_models_filters_audio_and_tts(self, kosmik_profile, monkeypatch):
        raw_models = [
            "openai/whisper-large-v3",
            "openai/whisper-large-v3-turbo",
            "kosmik/tts-piper-fast",
            "kosmik/tts-kokoro-quality",
            "qwen/qwen3.8-27b",
            "kosmik/tts-moss-nano",
            "kosmik/tts-supertonic-3",
        ]

        from providers.base import ProviderProfile

        monkeypatch.setattr(
            ProviderProfile,
            "fetch_models",
            lambda self, **kw: raw_models,
        )

        filtered = kosmik_profile.fetch_models(api_key="test-key")
        assert filtered == ["qwen/qwen3.8-27b"]
