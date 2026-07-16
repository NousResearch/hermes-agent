"""Unit tests for the Mistral provider profile's reasoning-effort wiring.

Mistral's API accepts ``reasoning_effort`` as a top-level kwarg only for
thinking-enabled models (mistral-small-2603+, mistral-medium-2604+).
Non-thinking models (codestral, mistral-large, pixtral, ministral) reject
it with HTTP 400.

These tests pin the profile's model-gating contract so Mistral requests stay
correctly shaped without going live.
"""

from __future__ import annotations

import pytest


@pytest.fixture
def mistral_profile():
    """Resolve the registered Mistral profile.

    Going through ``providers.get_provider_profile`` keeps the test honest —
    if someone later replaces the registered class with a plain
    ``ProviderProfile``, every assertion below collapses.
    """
    # ``model_tools`` triggers plugin discovery on import, which is what
    # registers the Mistral profile in the global provider registry.
    import model_tools  # noqa: F401
    import providers

    profile = providers.get_provider_profile("mistral")
    assert profile is not None, "mistral provider profile must be registered"
    return profile


class TestMistralReasoningWireShape:
    """``build_api_kwargs_extras`` produces Mistral's expected wire format."""

    def test_no_reasoning_config_emits_nothing(self, mistral_profile):
        """No reasoning_config → empty body, no top-level kwargs."""
        extra_body, top_level = mistral_profile.build_api_kwargs_extras(
            reasoning_config=None, model="mistral-small-latest"
        )
        assert extra_body == {}
        assert top_level == {}

    def test_empty_reasoning_config_emits_nothing(self, mistral_profile):
        """Empty reasoning_config → empty body, no top-level kwargs."""
        extra_body, top_level = mistral_profile.build_api_kwargs_extras(
            reasoning_config={}, model="mistral-small-latest"
        )
        assert extra_body == {}
        assert top_level == {}

    def test_enabled_with_high_effort_on_small(self, mistral_profile):
        """reasoning_effort passes through for thinking-capable models."""
        _, top_level = mistral_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "high"},
            model="mistral-small-latest",
        )
        assert top_level == {"reasoning_effort": "high"}

    def test_enabled_with_high_effort_on_medium(self, mistral_profile):
        """reasoning_effort passes through for mistral-medium-latest."""
        _, top_level = mistral_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "high"},
            model="mistral-medium-latest",
        )
        assert top_level == {"reasoning_effort": "high"}

    def test_unknown_effort_still_passes_through(self, mistral_profile):
        """Mistral enforces valid effort values server-side — we pass through."""
        _, top_level = mistral_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "garbage"},
            model="mistral-small-latest",
        )
        # We don't validate effort contents — server does.
        assert top_level == {"reasoning_effort": "garbage"}

    def test_empty_effort_omits_top_level(self, mistral_profile):
        """Empty effort string → omit reasoning_effort entirely."""
        _, top_level = mistral_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": ""},
            model="mistral-small-latest",
        )
        assert top_level == {}


class TestMistralModelGating:
    """Thinking-capable models get reasoning_effort; others don't."""

    @pytest.mark.parametrize(
        "model",
        [
            "mistral-small-latest",
            "mistral-small-2603",
            "mistral-small-2603-beta",
            "mistral-medium-latest",
            "mistral-medium",
            "mistral-medium-2604",
            "mistral-medium-3-5",
            "mistral-medium-3.5",
            "MISTRAL-SMALL-LATEST",  # case-insensitive
            # Vibe CLI aliases — real API model IDs with reasoning support
            "mistral-vibe-cli-latest",
            "mistral-vibe-cli-fast",
            "mistral-vibe-cli-with-tools",
            # Labs / experimental models
            "labs-leanstral-1-5",
            "labs-leanstral-1-5-1",
            # Future versioned models — covered by version thresholds, not
            # explicit prefix lists
            "mistral-small-2701",     # future small with reasoning
            "mistral-medium-2609",    # future medium with reasoning
            "mistral-large-2601",     # first large with reasoning (speculative)
            "magistral-small-2510",   # future magistral small
            "magistral-medium-2510",  # future magistral medium
        ],
    )
    def test_thinking_capable_models_get_reasoning_effort(self, mistral_profile, model):
        _, top_level = mistral_profile.build_api_kwargs_extras(
            reasoning_config={"effort": "high"}, model=model
        )
        assert top_level == {"reasoning_effort": "high"}, (
            f"Expected reasoning_effort for {model}, got {top_level}"
        )

    @pytest.mark.parametrize(
        "model",
        [
            "codestral-latest",
            "codestral-2508",
            "mistral-large-latest",
            "mistral-large-2512",
            "pixtral-12b-latest",
            "ministral-3b-latest",
            "ministral-8b-latest",
            "ministral-14b-latest",
            "mistral-small-2506",   # old non-thinking small variant
            "mistral-medium-2505",  # old non-thinking medium variant
            "mistral-medium-2508",  # old non-thinking medium variant
            "mistral-large-2599",   # pre-reasoning threshold (if ever added)
            "",                       # bare/unknown
            None,                     # missing
            "mistral-unknown",       # unrecognized
        ],
    )
    def test_non_thinking_models_emit_nothing(self, mistral_profile, model):
        extra_body, top_level = mistral_profile.build_api_kwargs_extras(
            reasoning_config={"effort": "high"}, model=model
        )
        assert extra_body == {}
        assert top_level == {}


class TestMistralAuxModel:
    """Mistral aux model is set on the profile so users don't see the
    bogus 'No auxiliary LLM provider configured' warning (#26924).
    """

    def test_profile_advertises_mistral_small(self, mistral_profile):
        assert mistral_profile.default_aux_model == "mistral-small-latest"

    def test_consumer_api_returns_mistral_small(self):
        from agent.auxiliary_client import _get_aux_model_for_provider
        assert _get_aux_model_for_provider("mistral") == "mistral-small-latest"

    def test_consumer_api_returns_non_empty(self):
        from agent.auxiliary_client import _get_aux_model_for_provider
        assert _get_aux_model_for_provider("mistral") != ""


class TestMistralProfileMetadata:
    """Provider profile metadata is correct."""

    def test_display_name(self, mistral_profile):
        assert mistral_profile.display_name == "Mistral AI"

    def test_env_vars(self, mistral_profile):
        assert "MISTRAL_API_KEY" in mistral_profile.env_vars

    def test_base_url(self, mistral_profile):
        assert mistral_profile.base_url == "https://api.mistral.ai/v1"

    def test_supports_vision(self, mistral_profile):
        assert mistral_profile.supports_vision is True

    def test_aliases_dont_include_mixtral(self, mistral_profile):
        """mixtral is an open-weight model family, not a Mistral API alias."""
        assert "mixtral" not in mistral_profile.aliases
