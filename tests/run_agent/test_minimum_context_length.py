"""Tests for the minimum context_length validation during agent init.

Regression tests for #8430: when the user explicitly sets model.context_length
in config.yaml, the minimum-64K check should be skipped — the user accepts the
reduced window and the error message itself tells them to do this.
"""

import pytest
from run_agent import validate_minimum_context
from agent.model_metadata import MINIMUM_CONTEXT_LENGTH


class TestMinimumContextLengthCheck:
    def test_rejects_small_context_without_config_override(self):
        """Models below 64K should be rejected when no config override is set."""
        with pytest.raises(ValueError, match="below the minimum"):
            validate_minimum_context("test-model", 32_000, config_context_length=None)

    def test_allows_small_context_with_config_override(self):
        """Models below 64K should be accepted when user explicitly sets context_length."""
        validate_minimum_context("test-model", 32_768, config_context_length=32_768)

    def test_still_rejects_without_override_even_at_boundary(self):
        """Models at exactly MINIMUM_CONTEXT_LENGTH - 1 should still be rejected."""
        with pytest.raises(ValueError, match="below the minimum"):
            validate_minimum_context(
                "test-model", MINIMUM_CONTEXT_LENGTH - 1, config_context_length=None
            )

    def test_accepts_at_minimum_without_override(self):
        """Models at exactly MINIMUM_CONTEXT_LENGTH should be accepted without override."""
        validate_minimum_context(
            "test-model", MINIMUM_CONTEXT_LENGTH, config_context_length=None
        )

    def test_accepts_zero_context_length(self):
        """Zero context length (unknown model) should not trigger the check."""
        validate_minimum_context("test-model", 0, config_context_length=None)
