"""Tests for per-model compression threshold overrides.

Users who swap between models with very different context windows (e.g. a
256K model and a 1M model) need different compaction trigger points.
``compression.model_thresholds`` in config.yaml lets them set per-model
overrides that are resolved by longest substring match.
"""

from unittest.mock import patch

from agent.context_compressor import ContextCompressor, resolve_model_threshold
from agent.context_engine import ContextEngine


# ---------------------------------------------------------------------------
# resolve_model_threshold helper
# ---------------------------------------------------------------------------

class TestResolveModelThreshold:
    def test_no_overrides_returns_default(self):
        assert resolve_model_threshold("glm-5.2", None, 0.50) == 0.50
        assert resolve_model_threshold("glm-5.2", {}, 0.50) == 0.50

    def test_empty_model_returns_default(self):
        assert resolve_model_threshold("", {"glm": 0.70}, 0.50) == 0.50

    def test_exact_match(self):
        overrides = {"glm-5.2": 0.70}
        assert resolve_model_threshold("glm-5.2", overrides, 0.50) == 0.70

    def test_substring_match(self):
        overrides = {"glm-5.2": 0.70}
        assert resolve_model_threshold("openai/glm-5.2", overrides, 0.50) == 0.70

    def test_longest_match_wins(self):
        overrides = {"glm-5.2": 0.70, "glm-5.2-1M": 0.25}
        # "glm-5.2-1M" is a longer match than "glm-5.2"
        assert resolve_model_threshold("glm-5.2-1M", overrides, 0.50) == 0.25
        # "glm-5.2" alone still matches at 0.70
        assert resolve_model_threshold("glm-5.2", overrides, 0.50) == 0.70

    def test_no_match_returns_default(self):
        overrides = {"claude-sonnet-4": 0.60}
        assert resolve_model_threshold("glm-5.2", overrides, 0.50) == 0.50

    def test_override_can_lower_threshold(self):
        """Per-model overrides work in both directions (raise and lower)."""
        overrides = {"small-model": 0.30}
        assert resolve_model_threshold("small-model", overrides, 0.50) == 0.30

    def test_override_can_raise_threshold(self):
        overrides = {"big-model": 0.85}
        assert resolve_model_threshold("big-model", overrides, 0.50) == 0.85


# ---------------------------------------------------------------------------
# ContextCompressor with model_thresholds
# ---------------------------------------------------------------------------

class TestContextCompressorModelThresholds:
    @patch("agent.context_compressor.get_model_context_length", return_value=256_000)
    def test_init_applies_per_model_override(self, _mock):
        """When model_thresholds contains a match, __init__ uses the override."""
        cc = ContextCompressor(
            model="glm-5.2",
            threshold_percent=0.50,
            model_thresholds={"glm-5.2": 0.70},
            quiet_mode=True,
        )
        assert cc.threshold_percent == 0.70
        assert cc._base_threshold_percent == 0.50
        # threshold_tokens should be 70% of 256K = 179,200
        assert cc.threshold_tokens == int(256_000 * 0.70)

    @patch("agent.context_compressor.get_model_context_length", return_value=1_000_000)
    def test_init_no_override_uses_global(self, _mock):
        """When no model_thresholds match, __init__ uses the global threshold."""
        cc = ContextCompressor(
            model="glm-5.2",
            threshold_percent=0.50,
            model_thresholds={"claude-sonnet-4": 0.60},
            quiet_mode=True,
        )
        assert cc.threshold_percent == 0.50
        assert cc.threshold_tokens == int(1_000_000 * 0.50)

    @patch("agent.context_compressor.get_model_context_length", return_value=256_000)
    def test_init_no_model_thresholds_dict(self, _mock):
        """Empty model_thresholds dict = backward compatible."""
        cc = ContextCompressor(
            model="glm-5.2",
            threshold_percent=0.50,
            quiet_mode=True,
        )
        assert cc.threshold_percent == 0.50
        assert cc.model_thresholds == {}

    @patch("agent.context_compressor.get_model_context_length", return_value=256_000)
    def test_init_none_model_thresholds(self, _mock):
        """Passing None for model_thresholds is safe."""
        cc = ContextCompressor(
            model="glm-5.2",
            threshold_percent=0.50,
            model_thresholds=None,
            quiet_mode=True,
        )
        assert cc.threshold_percent == 0.50
        assert cc.model_thresholds == {}

    @patch("agent.context_compressor.get_model_context_length")
    def test_update_model_re_resolves_threshold(self, mock_ctx):
        """Switching models re-resolves the per-model threshold."""
        mock_ctx.return_value = 256_000
        cc = ContextCompressor(
            model="glm-5.2",
            threshold_percent=0.50,
            model_thresholds={"glm-5.2": 0.70, "glm-5.2-1M": 0.25},
            quiet_mode=True,
        )
        assert cc.threshold_percent == 0.70

        # Switch to the 1M model
        mock_ctx.return_value = 1_000_000
        cc.update_model(
            model="glm-5.2-1M",
            context_length=1_000_000,
        )
        assert cc.threshold_percent == 0.25
        assert cc.threshold_tokens == int(1_000_000 * 0.25)

    @patch("agent.context_compressor.get_model_context_length")
    def test_update_model_falls_back_to_global(self, mock_ctx):
        """Switching to a model with no override uses the global threshold."""
        mock_ctx.return_value = 256_000
        cc = ContextCompressor(
            model="glm-5.2",
            threshold_percent=0.50,
            model_thresholds={"glm-5.2": 0.70},
            quiet_mode=True,
        )
        assert cc.threshold_percent == 0.70

        # Switch to a model with no override
        mock_ctx.return_value = 128_000
        cc.update_model(
            model="some-other-model",
            context_length=128_000,
        )
        assert cc.threshold_percent == 0.50
        assert cc.threshold_tokens == int(128_000 * 0.50)


# ---------------------------------------------------------------------------
# ContextEngine base class
# ---------------------------------------------------------------------------

class TestContextEngineModelThresholds:
    def test_base_class_update_model_applies_overrides(self):
        """The base-class update_model() applies model_thresholds if set."""
        # Create a minimal concrete engine for testing
        class TestEngine(ContextEngine):
            @property
            def name(self):
                return "test"

            def update_from_response(self, usage):
                pass

            def should_compress(self, prompt_tokens=None):
                return False

            def compress(self, messages, current_tokens=None, focus_topic=None):
                return messages

        engine = TestEngine()
        engine.threshold_percent = 0.50
        engine.context_length = 0
        engine.model_thresholds = {"glm-5.2-1M": 0.25}

        engine.update_model(model="glm-5.2-1M", context_length=1_000_000)
        assert engine.threshold_percent == 0.25
        assert engine.threshold_tokens == int(1_000_000 * 0.25)

    def test_base_class_update_model_no_overrides(self):
        """Without model_thresholds, the base class behaves as before."""
        class TestEngine(ContextEngine):
            @property
            def name(self):
                return "test"

            def update_from_response(self, usage):
                pass

            def should_compress(self, prompt_tokens=None):
                return False

            def compress(self, messages, current_tokens=None, focus_topic=None):
                return messages

        engine = TestEngine()
        engine.threshold_percent = 0.50
        engine._base_threshold_percent = 0.50
        engine.context_length = 0
        engine.model_thresholds = {}

        engine.update_model(model="glm-5.2", context_length=256_000)
        assert engine.threshold_percent == 0.50
        assert engine.threshold_tokens == int(256_000 * 0.50)