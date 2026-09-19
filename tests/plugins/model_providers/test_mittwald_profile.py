"""Unit tests for the mittwald AI Hosting provider profile.

mittwald serves vLLM behind an OpenAI-compatible surface, so most of the contract is the
plain api-key one every downstream layer (auth, models, doctor, runtime_provider, transport)
reads. The two provider-specific behaviours covered here are the reasoning-effort wire and
the chat-model filter on the live catalog, which mixes speech/embedding/rerank/OCR models
into the same ``/v1/models`` array.
"""

from __future__ import annotations

import pytest


@pytest.fixture
def mittwald_profile():
    """Resolve the registered profile through the registry, exercising plugin discovery."""
    import model_tools  # noqa: F401
    import providers

    profile = providers.get_provider_profile("mittwald")
    assert profile is not None, "mittwald provider profile must be registered"
    return profile


# Live ``GET /v1/models`` payload shape (2026-09-10): every hosted model, chat or not.
_MODELS_PAYLOAD = {
    "object": "list",
    "data": [
        {"id": "Qwen3.6-35B-A3B-FP8", "object": "model", "max_input_tokens": 256000},
        {"id": "Qwen3.8-27B-NVFP4", "object": "model", "max_input_tokens": 256000},
        {"id": "Qwen3.5-122B-A10B-FP8", "object": "model", "max_input_tokens": 245760},
        {"id": "Ministral-3-14B-Instruct-2512", "object": "model", "max_input_tokens": 262144},
        {"id": "Qwen3.5-0.8B", "object": "model", "max_input_tokens": 262144},
        {"id": "gpt-oss-120b", "object": "model", "max_input_tokens": 131072},
        {"id": "GLM-OCR", "object": "model", "max_input_tokens": 131072},
        {"id": "Qwen3-Embedding-8B", "object": "model", "max_input_tokens": 32768},
        {"id": "Qwen3-VL-Reranker-2B", "object": "model", "max_input_tokens": 32768},
        {"id": "whisper-large-v3-turbo", "object": "model"},
        {"id": "Qwen3-TTS-12Hz-1.7B-CustomVoice", "object": "model"},
    ],
}
# Chat models in the order the payload above lists them: the filter preserves catalog order.
_CHAT_IDS = [
    "Qwen3.6-35B-A3B-FP8", "Qwen3.8-27B-NVFP4", "Qwen3.5-122B-A10B-FP8",
    "Ministral-3-14B-Instruct-2512", "Qwen3.5-0.8B", "gpt-oss-120b",
]
# The offline catalog leads with the flagship and ends with the small/cheap model.
_FALLBACK_IDS = [
    "Qwen3.6-35B-A3B-FP8", "Qwen3.8-27B-NVFP4", "Qwen3.5-122B-A10B-FP8",
    "Ministral-3-14B-Instruct-2512", "gpt-oss-120b", "Qwen3.5-0.8B",
]


class TestMittwaldProfile:
    def test_identity_and_endpoint(self, mittwald_profile):
        assert mittwald_profile.name == "mittwald"
        assert mittwald_profile.api_mode == "chat_completions"
        assert mittwald_profile.auth_type == "api_key"
        assert mittwald_profile.base_url == "https://llm.aihosting.mittwald.de/v1"
        assert mittwald_profile.get_hostname() == "llm.aihosting.mittwald.de"

    @pytest.mark.parametrize("alias", ["mittwald-ai", "mittwald-aihosting", "aihosting"])
    def test_aliases_resolve(self, alias, mittwald_profile):
        import providers

        assert providers.get_provider_profile(alias) is mittwald_profile

    def test_env_vars(self, mittwald_profile):
        # Key vars first (mittwald's own docs name MITTWALD_LLM_API_KEY), base-URL override last.
        assert mittwald_profile.env_vars == (
            "MITTWALD_LLM_API_KEY", "MITTWALD_AI_API_KEY", "MITTWALD_BASE_URL")

    def test_fallback_models_are_chat_only(self, mittwald_profile):
        assert list(mittwald_profile.fallback_models) == _FALLBACK_IDS

    def test_aux_model_is_the_small_one(self, mittwald_profile):
        # Aux resolution is synchronous and cannot wait on a catalog round-trip.
        assert mittwald_profile.default_aux_model == "Qwen3.5-0.8B"

    def test_vision_default(self, mittwald_profile):
        assert mittwald_profile.default_vision_model() == "Qwen3.6-35B-A3B-FP8"

    def test_prompt_cache_key_not_advertised(self, mittwald_profile):
        # vLLM accepts the field but does nothing with it; the flag stays opt-in.
        assert mittwald_profile.supports_prompt_cache_key is False


class TestMittwaldCatalogFilter:
    """``fetch_models`` narrows the mixed live catalog down to the chat models."""

    @staticmethod
    def _stub_base_fetch(monkeypatch, result):
        """Replace the shared HTTP catalog fetch the profile's ``super()`` call lands in."""
        from providers.base import ProviderProfile

        monkeypatch.setattr(ProviderProfile, "fetch_models", lambda self, **kwargs: result)

    def test_non_chat_models_are_filtered_out(self, mittwald_profile, monkeypatch):
        self._stub_base_fetch(monkeypatch, [entry["id"] for entry in _MODELS_PAYLOAD["data"]])
        assert mittwald_profile.fetch_models(api_key="sk-test") == _CHAT_IDS

    def test_unreachable_catalog_stays_none(self, mittwald_profile, monkeypatch):
        # None (not []) so callers fall back to ``fallback_models`` instead of showing an empty picker.
        self._stub_base_fetch(monkeypatch, None)
        assert mittwald_profile.fetch_models(api_key="sk-test") is None


class TestMittwaldReasoning:
    """``build_api_kwargs_extras`` wires the top-level ``reasoning_effort`` field."""

    def test_effort_is_passed_through(self, mittwald_profile):
        assert mittwald_profile.build_api_kwargs_extras(
            reasoning_config={"effort": "low"}) == ({}, {"reasoning_effort": "low"})

    def test_disabled_reasoning_sends_none(self, mittwald_profile):
        # Verified live: "none" actually turns Qwen's thinking off, so no
        # chat_template_kwargs.enable_thinking fallback is needed.
        assert mittwald_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": False}) == ({}, {"reasoning_effort": "none"})
        assert mittwald_profile.build_api_kwargs_extras(
            reasoning_config={"effort": "none"}) == ({}, {"reasoning_effort": "none"})

    def test_unset_reasoning_sends_nothing(self, mittwald_profile):
        # No field at all so the server default applies.
        assert mittwald_profile.build_api_kwargs_extras(reasoning_config=None) == ({}, {})
        assert mittwald_profile.build_api_kwargs_extras(reasoning_config={}) == ({}, {})

    def test_unknown_effort_is_clamped_to_the_wire(self, mittwald_profile):
        _, extras = mittwald_profile.build_api_kwargs_extras(reasoning_config={"effort": "ultra"})
        from agent.reasoning_effort import OPENAI_COMPAT_WIRE_EFFORTS

        assert extras["reasoning_effort"] in OPENAI_COMPAT_WIRE_EFFORTS


class TestMittwaldContextFallbacks:
    """Offline fallbacks for the models whose window the ``qwen`` catch-all would understate."""

    @pytest.mark.parametrize("model,expected", [
        ("Qwen3.6-35B-A3B-FP8", 256000), ("Qwen3.8-27B-NVFP4", 256000),
        ("Qwen3.5-122B-A10B-FP8", 245760), ("Ministral-3-14B-Instruct-2512", 262144),
        ("Qwen3.5-0.8B", 262144), ("gpt-oss-120b", 131072)])
    def test_default_context_length(self, model, expected):
        from agent.model_metadata import DEFAULT_CONTEXT_LENGTHS, _longest_key_match

        hit = _longest_key_match(DEFAULT_CONTEXT_LENGTHS, model.lower())
        assert hit is not None and hit[1] == expected

    def test_live_catalog_reports_the_same_windows(self):
        # max_input_tokens is already in _CONTEXT_LENGTH_KEYS, so the live path needs no extra code.
        from agent.model_metadata import _CONTEXT_LENGTH_KEYS, _context_length_from_model_payload

        assert "max_input_tokens" in _CONTEXT_LENGTH_KEYS
        assert _context_length_from_model_payload(
            {"id": "Qwen3.6-35B-A3B-FP8", "max_input_tokens": 256000}) == 256000
