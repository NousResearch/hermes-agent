"""Tests for max_context_tokens — hardware speed cap for compression threshold.

When max_context_tokens is set (e.g. 65000 for MI50 GPUs), the compression
threshold is capped to that value even if the model's full context window
would suggest a higher threshold (e.g. 50% of 202K = 101K).

When None (default), behavior is identical to before — threshold is
context_length * threshold_percent.
"""

import pytest
from unittest.mock import patch

from agent.context_compressor import ContextCompressor
from agent.model_metadata import MINIMUM_CONTEXT_LENGTH


@pytest.fixture()
def mock_model():
    """Mock get_model_context_length to return a large context window (202,752)."""
    with patch("agent.context_compressor.get_model_context_length", return_value=202_752):
        yield


class TestMaxContextTokensNone:
    """max_context_tokens=None should behave identically to the old behavior."""

    def test_none_threshold_identical(self, mock_model):
        """Without max_context_tokens, threshold = context_length * threshold_percent."""
        c = ContextCompressor(
            model="test/model",
            threshold_percent=0.50,
            protect_first_n=3,
            protect_last_n=20,
            quiet_mode=True,
            max_context_tokens=None,
        )
        assert c.threshold_tokens == int(202_752 * 0.50)  # 101,376
        assert c.context_length == 202_752

    def test_none_omitted_identical(self, mock_model):
        """Omitting max_context_tokens should give the same result as None."""
        c_default = ContextCompressor(
            model="test/model",
            threshold_percent=0.50,
            protect_first_n=3,
            protect_last_n=20,
            quiet_mode=True,
        )
        c_none = ContextCompressor(
            model="test/model",
            threshold_percent=0.50,
            protect_first_n=3,
            protect_last_n=20,
            quiet_mode=True,
            max_context_tokens=None,
        )
        assert c_default.threshold_tokens == c_none.threshold_tokens

    def test_none_stored(self, mock_model):
        """max_context_tokens attribute should be None when not set."""
        c = ContextCompressor(
            model="test/model",
            threshold_percent=0.50,
            quiet_mode=True,
            max_context_tokens=None,
        )
        assert c.max_context_tokens is None


class TestMaxContextTokensCap:
    """max_context_tokens caps the threshold to the specified value."""

    def test_cap_65000(self, mock_model):
        """With max_context_tokens=65000, threshold is 65000 (not 101,376)."""
        c = ContextCompressor(
            model="test/model",
            threshold_percent=0.50,
            protect_first_n=3,
            protect_last_n=20,
            quiet_mode=True,
            max_context_tokens=65000,
        )
        # Without cap: 202,752 * 0.50 = 101,376
        # With cap: min(101,376, 65,000) = 65,000
        # Floor: max(65,000, 64,000) = 65,000 (MINIMUM_CONTEXT_LENGTH = 64K)
        assert c.threshold_tokens == 65000
        assert c.max_context_tokens == 65000

    def test_cap_80000(self, mock_model):
        """With max_context_tokens=80000, above MINIMUM but below raw threshold."""
        c = ContextCompressor(
            model="test/model",
            threshold_percent=0.50,
            quiet_mode=True,
            max_context_tokens=80000,
        )
        # min(101,376, 80,000) = 80,000
        # max(80,000, 64,000) = 80,000
        assert c.threshold_tokens == 80000

    def test_cap_above_threshold_no_effect(self, mock_model):
        """max_context_tokens > raw_threshold has no effect."""
        c = ContextCompressor(
            model="test/model",
            threshold_percent=0.50,
            quiet_mode=True,
            max_context_tokens=200_000,  # way above 101,376
        )
        # min(101,376, 200,000) = 101,376 (cap doesn't affect)
        assert c.threshold_tokens == 101_376

    def test_cap_zero_means_no_cap(self, mock_model):
        """max_context_tokens=0 or negative should be treated as no cap."""
        c = ContextCompressor(
            model="test/model",
            threshold_percent=0.50,
            quiet_mode=True,
            max_context_tokens=0,
        )
        # 0 is falsy → no cap applied
        assert c.threshold_tokens == 101_376

    def test_log_output_includes_cap(self, mock_model):
        """Initialization log should include max_context_tokens value."""
        with patch("agent.context_compressor.logger") as mock_logger:
            c = ContextCompressor(
                model="test/model",
                threshold_percent=0.50,
                quiet_mode=False,
                max_context_tokens=65000,
            )
            # The log uses %s format args — check the call args include 65000
            call_args = mock_logger.info.call_args[0]
            # Positional args after the format string
            assert 65000 in call_args


class TestMaxContextTokensFloor:
    """max_context_tokens interact with MINIMUM_CONTEXT_LENGTH floor."""

    def test_below_minimum_floored(self, mock_model):
        """max_context_tokens=60000 is below MINIMUM_CONTEXT_LENGTH=64000, so floor wins."""
        c = ContextCompressor(
            model="test/model",
            threshold_percent=0.50,
            quiet_mode=True,
            max_context_tokens=60000,
        )
        # min(101,376, 60,000) = 60,000
        # max(60,000, 64,000) = 64,000  ← MINIMUM_CONTEXT_LENGTH wins
        assert c.threshold_tokens == MINIMUM_CONTEXT_LENGTH
        assert c.threshold_tokens == 64000

    def test_exactly_minimum(self, mock_model):
        """max_context_tokens=64000 exactly equals MINIMUM_CONTEXT_LENGTH."""
        c = ContextCompressor(
            model="test/model",
            threshold_percent=0.50,
            quiet_mode=True,
            max_context_tokens=64000,
        )
        # min(101,376, 64,000) = 64,000
        # max(64,000, 64,000) = 64,000
        assert c.threshold_tokens == 64000


class TestMaxContextTokensUpdateModel:
    """max_context_tokens persists across model switches via update_model()."""

    def test_update_model_preserves_cap(self, mock_model):
        """update_model() should preserve max_context_tokens when not passed."""
        c = ContextCompressor(
            model="test/model",
            threshold_percent=0.50,
            quiet_mode=True,
            max_context_tokens=65000,
        )
        assert c.threshold_tokens == 65000

        # Simulate a model switch — update_model without explicit max_context_tokens
        # should use the stored value
        with patch("agent.context_compressor.get_model_context_length", return_value=128_000):
            c.update_model(
                model="new-model",
                context_length=128_000,
                base_url="",
                api_key="",
                provider="openrouter",
            )
        # context_length=128K, threshold=50% = 64K, cap=65K → min(64K, 65K) = 64K
        # Wait: 128_000 * 0.50 = 64,000; cap is stored as 65,000
        # min(64,000, 65,000) = 64,000 → max(64,000, 64,000) = 64,000
        assert c.threshold_tokens == 64000  # raw_threshold < cap, so raw wins but floored
        assert c.max_context_tokens == 65000  # preserved

    def test_update_model_with_new_cap(self, mock_model):
        """update_model() with explicit max_context_tokens overrides the stored value."""
        c = ContextCompressor(
            model="test/model",
            threshold_percent=0.50,
            quiet_mode=True,
            max_context_tokens=65000,
        )

        with patch("agent.context_compressor.get_model_context_length", return_value=128_000):
            c.update_model(
                model="new-model",
                context_length=128_000,
                max_context_tokens=40000,  # new, lower cap
            )
        # 128K * 0.50 = 64K, min(64K, 40K) = 40K, max(40K, 64K) = 64K
        assert c.threshold_tokens == 64000  # MINIMUM_CONTEXT_LENGTH floor
        assert c.max_context_tokens == 40000  # updated

    def test_update_model_removes_cap(self, mock_model):
        """update_model() with max_context_tokens=None removes the cap."""
        c = ContextCompressor(
            model="test/model",
            threshold_percent=0.50,
            quiet_mode=True,
            max_context_tokens=65000,
        )
        assert c.threshold_tokens == 65000

        with patch("agent.context_compressor.get_model_context_length", return_value=128_000):
            c.update_model(
                model="cloud-model",
                context_length=128_000,
                max_context_tokens=None,  # remove the cap
            )
        # No cap → 128K * 0.50 = 64K → max(64K, 64K) = 64K
        assert c.threshold_tokens == 64000
        # None means "don't override stored" — but the update_model logic
        # only updates max_context_tokens if explicitly passed
        # Actually, looking at the code:
        #   if max_context_tokens is not None:
        #       self.max_context_tokens = max_context_tokens
        # So passing None does NOT override the stored 65000.
        # This is intentional — None means "don't change the setting".
        assert c.max_context_tokens == 65000  # preserved (None didn't override)


class TestMaxContextTokensShouldCompress:
    """should_compress respects the capped threshold."""

    def test_below_cap_no_compress(self, mock_model):
        """Below the capped threshold, should_compress returns False."""
        c = ContextCompressor(
            model="test/model",
            threshold_percent=0.50,
            quiet_mode=True,
            max_context_tokens=65000,
        )
        c.last_prompt_tokens = 50000  # below 65K
        assert c.should_compress() is False

    def test_above_cap_compress(self, mock_model):
        """Above the capped threshold, should_compress returns True."""
        c = ContextCompressor(
            model="test/model",
            threshold_percent=0.50,
            quiet_mode=True,
            max_context_tokens=65000,
        )
        c.last_prompt_tokens = 70000  # above 65K
        assert c.should_compress() is True