"""Tests for the bundled Meta Model API (Muse Spark) provider plugin.

Adapted from the plugin's original suite at
https://github.com/albertodepaola/hermes-meta-provider — the plugin is now
bundled, so profiles resolve through normal registry discovery.
"""

import pytest

from providers import get_provider_profile


def _profile():
    p = get_provider_profile("meta-ai")
    assert p is not None
    return p


class TestMetaAIProfile:
    def test_profile_registered(self):
        p = _profile()
        assert p.name == "meta-ai"
        assert p.base_url == "https://api.meta.ai/v1"
        assert p.auth_type == "api_key"
        # Responses API engages Muse prompt caching (0% on chat/completions
        # vs 93-99% on /v1/responses with prompt_cache_retention).
        assert p.api_mode == "codex_responses"
        assert "MODEL_API_KEY" in p.env_vars
        assert p.supports_vision is True
        assert p.default_aux_model == "muse-spark-1.2-contributor"
        assert p.default_max_tokens == 16384
        assert "muse-spark-1.2-contributor" in p.fallback_models
        assert "muse-spark-1.2" in p.fallback_models

    @pytest.mark.parametrize("alias", ["meta", "muse", "muse-spark", "model-api", "msl"])
    def test_aliases_resolve(self, alias):
        assert get_provider_profile(alias) is _profile()


def _effort(cfg):
    _eb, top = _profile().build_api_kwargs_extras(reasoning_config=cfg)
    return top.get("reasoning_effort")


class TestReasoningEffort:
    def test_mapping(self):
        assert _effort(None) == "medium"  # unset -> medium
        assert _effort({"enabled": True, "effort": "high"}) == "high"
        assert _effort({"enabled": True, "effort": "low"}) == "low"
        assert _effort({"enabled": True, "effort": "max"}) == "xhigh"
        assert _effort({"enabled": False}) == "minimal"  # disabled -> minimal
        # Meta 400s on "none" — must map to "minimal"
        assert _effort({"effort": "none"}) == "minimal"

    def test_never_uses_extra_body(self):
        """The dial must be top-level, not extra_body.reasoning (core-gated path)."""
        eb, top = _profile().build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "high"}
        )
        assert eb == {}
        assert "reasoning_effort" in top
