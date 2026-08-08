"""Targeted tests for ``utils.model_forces_max_completion_tokens``.

This helper decides whether a given model name requires the newer
``max_completion_tokens`` kwarg (rather than the legacy ``max_tokens``) on
``/v1/chat/completions``. It protects against the 400 ``unsupported_parameter``
error seen when third-party OpenAI-compatible endpoints serve gpt-4o / 4.1 /
5.x / o-series models by name and the caller only checks the URL host.
"""

from __future__ import annotations

from utils import model_forces_max_completion_tokens


# ─── Positive cases: families that require max_completion_tokens ────────────


class TestPositiveCases:
    def test_gpt_5_bare(self):
        assert model_forces_max_completion_tokens("gpt-5") is True




    def test_gpt_4o(self):
        assert model_forces_max_completion_tokens("gpt-4o") is True




    def test_o1(self):
        assert model_forces_max_completion_tokens("o1") is True



    def test_o3(self):
        assert model_forces_max_completion_tokens("o3") is True




# ─── Negative cases: older or non-OpenAI families still use max_tokens ──────


class TestNegativeCases:
    def test_gpt_3_5_turbo(self):
        assert model_forces_max_completion_tokens("gpt-3.5-turbo") is False

    def test_gpt_4(self):
        # Classic gpt-4 (non-omni) still uses max_tokens on chat completions.
        assert model_forces_max_completion_tokens("gpt-4") is False


    def test_claude_family(self):
        assert model_forces_max_completion_tokens("claude-3-opus") is False
        assert model_forces_max_completion_tokens("claude-sonnet-4-6") is False






# ─── Edge cases ─────────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_empty_string(self):
        assert model_forces_max_completion_tokens("") is False

    def test_none(self):
        assert model_forces_max_completion_tokens(None) is False  # type: ignore[arg-type]




    def test_vendor_prefix_stripped(self):
        # OpenRouter-style "vendor/model" names should match the tail.
        assert model_forces_max_completion_tokens("openai/gpt-5.4") is True
        assert model_forces_max_completion_tokens("openai/gpt-4o-mini") is True
        assert model_forces_max_completion_tokens("openai/o3-mini") is True



    def test_gpt_5_substring_in_middle_not_matched(self):
        # Only a prefix should match — "local-gpt-5-clone" is a different model.
        assert model_forces_max_completion_tokens("local-gpt-5-clone") is False

    def test_family_prefix_requires_a_model_boundary(self):
        # Custom model names that merely begin with a known family must keep
        # using max_tokens. Only the bare family, dotted releases, and hyphen
        # variants are part of the documented OpenAI model families.
        for model in (
            "gpt-4ocean",
            "gpt-4.1custom",
            "gpt-5clone",
            "o123",
            "o3mega",
            "o4custom",
            "vendor/o1clone",
        ):
            assert model_forces_max_completion_tokens(model) is False
