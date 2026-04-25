"""Tests for context compression threshold calculation (issue #14690).

When context_length equals MINIMUM_CONTEXT_LENGTH (64000), the max() floor
clamps threshold_tokens to 64000 (100% of context), making compression impossible.
The fix: if the floor pushes threshold to >= context_length, fall back to the
percentage-based value.
"""

import pytest
from unittest.mock import patch

from agent.context_compressor import ContextCompressor, MINIMUM_CONTEXT_LENGTH


class TestThresholdAtMinimumContextLength:
    """Regression: threshold_tokens must never equal or exceed context_length,
    or should_compress() can never fire."""

    def test_threshold_below_context_when_at_minimum(self):
        """When context_length == 64000 and threshold=0.7, threshold_tokens
        should be 70% of 64000 (44800), not 64000 (100%)."""
        with patch("agent.context_compressor.get_model_context_length", return_value=64000):
            c = ContextCompressor(model="test/model", threshold_percent=0.7, quiet_mode=True)
        assert c.threshold_tokens == int(64000 * 0.7)
        assert c.threshold_tokens < c.context_length

    def test_threshold_below_context_at_minimum_with_default_percent(self):
        """Default threshold_percent is 0.50. At context_length=64000,
        threshold should be 32000, not 64000."""
        with patch("agent.context_compressor.get_model_context_length", return_value=64000):
            c = ContextCompressor(model="test/model", quiet_mode=True)
        assert c.threshold_tokens == int(64000 * 0.50)
        assert c.threshold_tokens < c.context_length

    def test_threshold_at_85_percent_minimum_context(self):
        """Even at 85%, the floor shouldn't dominate at 64000 context."""
        with patch("agent.context_compressor.get_model_context_length", return_value=64000):
            c = ContextCompressor(model="test/model", threshold_percent=0.85, quiet_mode=True)
        assert c.threshold_tokens == int(64000 * 0.85)
        assert c.threshold_tokens < c.context_length

    def test_should_compress_fires_at_minimum_context(self):
        """Compression should trigger when prompt tokens exceed the percentage-based
        threshold, even when context_length == MINIMUM_CONTEXT_LENGTH."""
        with patch("agent.context_compressor.get_model_context_length", return_value=64000):
            c = ContextCompressor(model="test/model", threshold_percent=0.7, quiet_mode=True)
        # 64000 * 0.7 = 44800. At 45000 tokens, should_compress should be True.
        assert c.should_compress(prompt_tokens=45000) is True

    def test_should_compress_does_not_fire_below_threshold_at_minimum_context(self):
        """Below the threshold, should_compress should still return False."""
        with patch("agent.context_compressor.get_model_context_length", return_value=64000):
            c = ContextCompressor(model="test/model", threshold_percent=0.7, quiet_mode=True)
        assert c.should_compress(prompt_tokens=40000) is False


class TestLargeContextFloorStillWorks:
    """The MINIMUM_CONTEXT_LENGTH floor should still prevent premature compression
    on large-context models where threshold% would give an unreasonably small value."""

    def test_floor_clamps_tiny_percentage_on_large_context(self):
        """On a 200K context model with threshold=0.3, the floor at 64000
        prevents threshold_tokens from being 60K (too aggressive)."""
        with patch("agent.context_compressor.get_model_context_length", return_value=200000):
            c = ContextCompressor(model="test/model", threshold_percent=0.3, quiet_mode=True)
        # 200000 * 0.3 = 60000, but floor is 64000
        assert c.threshold_tokens == 64000

    def test_large_context_percentage_above_floor(self):
        """When the percentage gives a value above the floor, use the percentage."""
        with patch("agent.context_compressor.get_model_context_length", return_value=200000):
            c = ContextCompressor(model="test/model", threshold_percent=0.7, quiet_mode=True)
        assert c.threshold_tokens == int(200000 * 0.7)

    def test_threshold_never_equals_context_length_on_large_model(self):
        """Even when floor applies, threshold should still be < context_length."""
        with patch("agent.context_compressor.get_model_context_length", return_value=200000):
            c = ContextCompressor(model="test/model", threshold_percent=0.3, quiet_mode=True)
        assert c.threshold_tokens < c.context_length


class TestUpdateModelThreshold:
    """The update_model() method has the same max() logic and the same bug."""

    def test_update_model_threshold_below_context_at_minimum(self):
        """After update_model(), threshold should be below context_length
        even when context_length == MINIMUM_CONTEXT_LENGTH."""
        with patch("agent.context_compressor.get_model_context_length", return_value=128000):
            c = ContextCompressor(model="test/model", threshold_percent=0.7, quiet_mode=True)

        # Simulate a model switch to a smaller context (e.g., split slot)
        c.update_model(
            model="test/smaller",
            context_length=64000,
            base_url="",
            api_key="",
            provider="",
            api_mode="",
        )
        assert c.threshold_tokens == int(64000 * 0.7)
        assert c.threshold_tokens < c.context_length

    def test_update_model_preserves_floor_for_large_context(self):
        """update_model() should still apply the floor for large contexts."""
        with patch("agent.context_compressor.get_model_context_length", return_value=64000):
            c = ContextCompressor(model="test/model", threshold_percent=0.5, quiet_mode=True)

        c.update_model(
            model="test/large",
            context_length=200000,
            base_url="",
            api_key="",
            provider="",
            api_mode="",
        )
        # With context_length=200000 and threshold=0.5, 0.5*200000=100000 > 64000
        assert c.threshold_tokens == 100000


class TestEdgeCases:
    """Edge cases around the boundary between floor and context_length."""

    def test_context_length_slightly_above_minimum(self):
        """Just above MINIMUM_CONTEXT_LENGTH, the percentage should still work
        even though floor is almost equal to context_length."""
        # 60000 * 0.85 = 51000, floor is 64000 > 51000 *and* >= 60000
        # The fix should fall back to 51000 so compression can actually fire
        with patch("agent.context_compressor.get_model_context_length", return_value=60000):
            c = ContextCompressor(model="test/model", threshold_percent=0.85, quiet_mode=True)
        assert c.threshold_tokens < c.context_length

    def test_threshold_at_boundary(self):
        """When threshold_percent=1.0 at context_length=MINIMUM, the user has
        explicitly set 'compress at 100%', which means 'never compress'.
        threshold_tokens equals context_length — that's correct; should_compress
        only fires when prompt_tokens >= threshold, and the API errors before
        we reach 100% anyway."""
        with patch("agent.context_compressor.get_model_context_length", return_value=64000):
            c = ContextCompressor(model="test/model", threshold_percent=1.0, quiet_mode=True)
        # At 100% threshold, pct_based == context_length, so no compression
        assert c.threshold_tokens == c.context_length

    def test_context_length_below_minimum_at_zero(self):
        """context_length=0 is degenerate (unknown context size). The floor
        applies — threshold should be MINIMUM_CONTEXT_LENGTH so we don't
        prematurely compress when we don't know the model's real capacity."""
        with patch("agent.context_compressor.get_model_context_length", return_value=0):
            c = ContextCompressor(model="test/model", threshold_percent=0.7, quiet_mode=True)
        # pct_based = 0, max(0, 64000) = 64000, 64000 >= 0 → fallback to pct_based = 0
        # But 0 is degenerate — the floor should protect us.
        # Actually: our fix says if threshold >= context_length and context_length > 0,
        # fall back. context_length=0, so the condition is False, and floor wins.
        assert c.threshold_tokens == MINIMUM_CONTEXT_LENGTH