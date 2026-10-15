"""Tests for agent.model_metadata.free_tier_context_note (#47247).

Relationship-based: every family window is read out of DEFAULT_CONTEXT_LENGTHS and the
inputs are derived from it, so the tests never freeze a catalog number.
"""

import pytest

from agent.model_metadata import (
    DEFAULT_CONTEXT_LENGTHS,
    _longest_key_match,
    free_tier_context_note,
)


def _humanized(n: int) -> str:
    """The session-info block's display form (gateway.run_turn._format_session_info)."""
    return f"{n / 1_000_000:.1f}M" if n >= 1_000_000 else f"{n // 1_000}K" if n >= 1_000 else str(n)


class TestFires:
    def test_dash_free_below_family_window(self):
        native = DEFAULT_CONTEXT_LENGTHS["deepseek-v4-flash"]
        note = free_tier_context_note("deepseek-v4-flash-free", native // 5)
        assert note is not None
        assert _humanized(native) in note  # names the family window the reader expects
        assert _humanized(native // 5) in note  # and the cap actually resolved
        assert "deepseek-v4-flash-free" in note
        assert "deepseek-v4-flash" in note

    def test_colon_free_below_family_window(self):
        family = DEFAULT_CONTEXT_LENGTHS["glm-5.2"]
        note = free_tier_context_note("glm-5.2:free", family // 4, provider="openrouter")
        assert note is not None
        assert _humanized(family) in note

    def test_aggregator_prefix_still_resolves_base(self):
        family = DEFAULT_CONTEXT_LENGTHS["glm-5.2"]
        note = free_tier_context_note("z-ai/glm-5.2:free", family // 4)
        assert note is not None
        assert _humanized(family) in note
        assert "z-ai/glm-5.2:free" in note  # the id the user actually runs is named

    def test_note_is_a_single_plain_sentence(self):
        native = DEFAULT_CONTEXT_LENGTHS["deepseek-v4-flash"]
        note = free_tier_context_note("deepseek-v4-flash-free", native // 5)
        assert note is not None
        assert "\n" not in note
        assert not note.endswith("\n")


class TestSilent:
    def test_equal_window_does_not_fire(self):
        native = DEFAULT_CONTEXT_LENGTHS["deepseek-v4-flash"]
        assert free_tier_context_note("deepseek-v4-flash-free", native) is None

    def test_above_family_window_does_not_fire(self):
        native = DEFAULT_CONTEXT_LENGTHS["deepseek-v4-flash"]
        assert free_tier_context_note("deepseek-v4-flash-free", native * 2) is None

    def test_unknown_base_id_does_not_fire(self):
        # Verified against the catalog rather than assumed: no key can match the base.
        base = "ling-3.0-flash"
        assert _longest_key_match(DEFAULT_CONTEXT_LENGTHS, base) is None
        assert free_tier_context_note(f"{base}-free", 200_000) is None

    def test_paid_id_does_not_fire(self):
        native = DEFAULT_CONTEXT_LENGTHS["deepseek-v4-flash"]
        assert free_tier_context_note("deepseek-v4-flash", native // 5) is None

    def test_routing_suffix_after_free_marker_does_not_fire(self):
        # Documented non-case: a routing variant AFTER the marker (``:free:nitro``) is not
        # stripped, so the note stays silent rather than guessing OpenRouter's variants.
        family = DEFAULT_CONTEXT_LENGTHS["glm-5.2"]
        assert free_tier_context_note("glm-5.2:free:nitro", family // 4, provider="openrouter") is None

    @pytest.mark.parametrize("bad", [0, -1, -200_000])
    def test_non_positive_context_length(self, bad):
        assert free_tier_context_note("deepseek-v4-flash-free", bad) is None

    @pytest.mark.parametrize("model", ["", "   ", None])
    def test_empty_model(self, model):
        assert free_tier_context_note(model, 200_000) is None


class TestOverridePath:
    def test_providerless_emits_placeholder(self):
        native = DEFAULT_CONTEXT_LENGTHS["deepseek-v4-flash"]
        note = free_tier_context_note("deepseek-v4-flash-free", native // 5)
        assert note is not None
        assert "model_overrides.<provider>.deepseek-v4-flash-free.context_window" in note

    def test_provider_names_the_real_path(self):
        native = DEFAULT_CONTEXT_LENGTHS["deepseek-v4-flash"]
        note = free_tier_context_note(
            "deepseek-v4-flash-free", native // 5, provider="opencode-zen")
        assert note is not None
        assert "model_overrides.opencode-zen.deepseek-v4-flash-free.context_window" in note
        assert "<provider>" not in note
