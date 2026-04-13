"""Tests for per-model summarization model routing.

Covers the ``summary_models`` config option that maps main models to their
preferred summarization model, with longest-prefix matching and fallback
to the global ``summary_model``.
"""

import pytest
from unittest.mock import patch

from agent.context_compressor import ContextCompressor


def _make(**kwargs):
    """Create a ContextCompressor with mocked get_model_context_length."""
    defaults = {
        "model": "test/model",
        "quiet_mode": True,
    }
    defaults.update(kwargs)
    with patch(
        "agent.context_compressor.get_model_context_length", return_value=100000
    ):
        return ContextCompressor(**defaults)


class TestResolveSummaryModelExactMatch:
    def test_exact_match_wins(self):
        c = _make(
            summary_model_override="global/summary-model",
            summary_models={"test/model": "test/summary-model"},
        )
        assert c.summary_model == "test/summary-model"

    def test_exact_match_case_sensitive(self):
        c = _make(
            summary_model_override="global/summary",
            summary_models={"Test/Model": "other/summary"},
        )
        # "test/model" != "Test/Model"
        assert c.summary_model == "global/summary"


class TestResolveSummaryModelPrefixMatch:
    def test_prefix_match(self):
        c = _make(
            model="gpt-5.4-0620",
            summary_model_override="global/summary",
            summary_models={"gpt-5.4": "gpt-5.3-codex-spark"},
        )
        assert c.summary_model == "gpt-5.3-codex-spark"

    def test_longest_prefix_wins(self):
        c = _make(
            model="gpt-5.4-0620-preview",
            summary_model_override="global/summary",
            summary_models={
                "gpt-5": "gpt-5-base-summary",
                "gpt-5.4": "gpt-5.4-summary",
                "gpt-5.4-0620": "gpt-5.4-0620-summary",
            },
        )
        assert c.summary_model == "gpt-5.4-0620-summary"

    def test_prefix_no_partial_word_match(self):
        """Prefix matching is literal — 'gpt-5' matches 'gpt-5.4'."""
        c = _make(
            model="gpt-5.4",
            summary_model_override="global/summary",
            summary_models={"gpt-5": "gpt-5-summary"},
        )
        assert c.summary_model == "gpt-5-summary"


class TestResolveSummaryModelFallback:
    def test_no_match_uses_global(self):
        c = _make(
            model="claude-sonnet-4",
            summary_model_override="global/summary",
            summary_models={"gpt-5.4": "gpt-5.3-codex-spark"},
        )
        assert c.summary_model == "global/summary"

    def test_no_global_no_match(self):
        c = _make(
            model="claude-sonnet-4",
            summary_models={"gpt-5.4": "gpt-5.3-codex-spark"},
        )
        assert c.summary_model == ""

    def test_empty_dict_uses_global(self):
        c = _make(
            model="test/model",
            summary_model_override="global/summary",
            summary_models={},
        )
        assert c.summary_model == "global/summary"

    def test_none_dict_uses_global(self):
        c = _make(
            model="test/model",
            summary_model_override="global/summary",
            summary_models=None,
        )
        assert c.summary_model == "global/summary"

    def test_no_global_no_dict(self):
        c = _make(model="test/model")
        assert c.summary_model == ""


class TestUpdateModelResolvesSummary:
    def test_update_model_re_resolves(self):
        c = _make(
            model="glm-5.1",
            summary_model_override="global/summary",
            summary_models={
                "glm": "glm/cheap-summary",
                "gpt-5.4": "gpt-5.3-codex-spark",
            },
        )
        assert c.summary_model == "glm/cheap-summary"

        with patch(
            "agent.context_compressor.get_model_context_length", return_value=200000
        ):
            c.update_model(
                model="gpt-5.4",
                context_length=200000,
                base_url="",
                api_key="",
                provider="openai-codex",
            )
        assert c.summary_model == "gpt-5.3-codex-spark"

    def test_update_model_falls_back_to_global(self):
        c = _make(
            model="glm-5.1",
            summary_model_override="global/summary",
            summary_models={"gpt-5.4": "gpt-5.3-codex-spark"},
        )

        with patch(
            "agent.context_compressor.get_model_context_length", return_value=200000
        ):
            c.update_model(
                model="claude-sonnet-4",
                context_length=200000,
            )
        assert c.summary_model == "global/summary"

    def test_update_model_preserves_summary_models_dict(self):
        """The summary_models dict should survive model switches."""
        sm = {"gpt-5.4": "gpt-5.3-codex-spark", "glm": "glm/summary"}
        c = _make(
            model="gpt-5.4",
            summary_model_override="global/summary",
            summary_models=sm,
        )
        assert c.summary_model == "gpt-5.3-codex-spark"

        # Switch to a model with no per-model override
        with patch(
            "agent.context_compressor.get_model_context_length", return_value=200000
        ):
            c.update_model(model="unknown-model", context_length=200000)
        assert c.summary_model == "global/summary"

        # Switch back — should still work
        with patch(
            "agent.context_compressor.get_model_context_length", return_value=200000
        ):
            c.update_model(model="gpt-5.4", context_length=200000)
        assert c.summary_model == "gpt-5.3-codex-spark"


class TestBackwardCompatibility:
    """Existing configs without summary_models should work identically."""

    def test_no_summary_models_uses_override(self):
        c = _make(
            model="test/model",
            summary_model_override="google/gemini-3-flash-preview",
        )
        assert c.summary_model == "google/gemini-3-flash-preview"

    def test_no_summary_models_no_override(self):
        c = _make(model="test/model")
        assert c.summary_model == ""

    def test_existing_fixture_pattern(self):
        """The default test fixture pattern should still work."""
        with patch(
            "agent.context_compressor.get_model_context_length", return_value=100000
        ):
            c = ContextCompressor(
                model="test/model",
                threshold_percent=0.85,
                protect_first_n=2,
                protect_last_n=2,
                quiet_mode=True,
            )
        assert c.summary_model == ""
