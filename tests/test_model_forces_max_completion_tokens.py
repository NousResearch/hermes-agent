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

    def test_gpt_5_point_release(self):
        # The case the user actually hit — gpt-5.4 on a custom OpenAI-compatible
        # endpoint was being sent max_tokens and getting 400 back.
        assert model_forces_max_completion_tokens("gpt-5.4") is True

    def test_gpt_5_mini(self):
        assert model_forces_max_completion_tokens("gpt-5-mini") is True

    def test_gpt_5_nano(self):
        assert model_forces_max_completion_tokens("gpt-5-nano") is True

    def test_gpt_4o(self):
        assert model_forces_max_completion_tokens("gpt-4o") is True

    def test_gpt_4o_mini(self):
        assert model_forces_max_completion_tokens("gpt-4o-mini") is True

    def test_gpt_4_1(self):
        assert model_forces_max_completion_tokens("gpt-4.1") is True

    def test_gpt_4_1_mini(self):
        assert model_forces_max_completion_tokens("gpt-4.1-mini") is True

    def test_o1(self):
        assert model_forces_max_completion_tokens("o1") is True

    def test_o1_preview(self):
        assert model_forces_max_completion_tokens("o1-preview") is True

    def test_o1_mini(self):
        assert model_forces_max_completion_tokens("o1-mini") is True

    def test_o3(self):
        assert model_forces_max_completion_tokens("o3") is True

    def test_o3_mini(self):
        assert model_forces_max_completion_tokens("o3-mini") is True

    def test_o4_mini(self):
        # Future-proofing — o4 is already listed publicly.
        assert model_forces_max_completion_tokens("o4-mini") is True


# ─── Negative cases: older or non-OpenAI families still use max_tokens ──────


class TestNegativeCases:
    def test_gpt_3_5_turbo(self):
        assert model_forces_max_completion_tokens("gpt-3.5-turbo") is False

    def test_gpt_4(self):
        # Classic gpt-4 (non-omni) still uses max_tokens on chat completions.
        assert model_forces_max_completion_tokens("gpt-4") is False

    def test_gpt_4_turbo(self):
        assert model_forces_max_completion_tokens("gpt-4-turbo") is False

    def test_claude_family(self):
        assert model_forces_max_completion_tokens("claude-3-opus") is False
        assert model_forces_max_completion_tokens("claude-sonnet-4-6") is False

    def test_llama_family(self):
        assert model_forces_max_completion_tokens("llama3") is False
        assert model_forces_max_completion_tokens("llama-3-70b-instruct") is False

    def test_mistral_family(self):
        assert model_forces_max_completion_tokens("mistral-7b-instruct") is False

    def test_qwen_family(self):
        assert model_forces_max_completion_tokens("qwen2.5-72b") is False

    def test_deepseek_family(self):
        assert model_forces_max_completion_tokens("deepseek-v4-flash") is False


# ─── Edge cases ─────────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_empty_string(self):
        assert model_forces_max_completion_tokens("") is False

    def test_none(self):
        assert model_forces_max_completion_tokens(None) is False  # type: ignore[arg-type]

    def test_whitespace_only(self):
        assert model_forces_max_completion_tokens("   ") is False

    def test_case_insensitive(self):
        assert model_forces_max_completion_tokens("GPT-5.4") is True
        assert model_forces_max_completion_tokens("Gpt-4o-Mini") is True
        assert model_forces_max_completion_tokens("O3-MINI") is True

    def test_leading_trailing_whitespace(self):
        assert model_forces_max_completion_tokens("  gpt-5  ") is True

    def test_vendor_prefix_stripped(self):
        # OpenRouter-style "vendor/model" names should match the tail.
        assert model_forces_max_completion_tokens("openai/gpt-5.4") is True
        assert model_forces_max_completion_tokens("openai/gpt-4o-mini") is True
        assert model_forces_max_completion_tokens("openai/o3-mini") is True

    def test_vendor_prefix_with_non_matching_tail(self):
        assert model_forces_max_completion_tokens("openai/gpt-3.5-turbo") is False
        assert model_forces_max_completion_tokens("anthropic/claude-3-opus") is False

    def test_fake_prefix_not_matched(self):
        # "o-series-but-not-really" doesn't start with o1/o3/o4.
        assert model_forces_max_completion_tokens("omni-chat") is False
        # "ox" isn't an o-series model, and "olive" / "opus" shouldn't collide.
        assert model_forces_max_completion_tokens("ox-large") is False
        assert model_forces_max_completion_tokens("opus-3") is False

    def test_gpt_5_substring_in_middle_not_matched(self):
        # Only a prefix should match — "local-gpt-5-clone" is a different model.
        assert model_forces_max_completion_tokens("local-gpt-5-clone") is False
