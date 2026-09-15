"""Tests for per-model compression threshold overrides.

Users who swap between models with very different context windows (e.g. a
256K model and a 1M model) need different compaction trigger points.
``compression.model_thresholds`` in config.yaml lets them set per-model
overrides that are resolved by longest substring match. The small-context
floor (75% for <512K models) still applies on top of per-model overrides.
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


    def test_exact_match(self):
        overrides = {"glm-5.2": 0.70}
        assert resolve_model_threshold("glm-5.2", overrides, 0.50) == 0.70






# ---------------------------------------------------------------------------
# ContextCompressor integration
# ---------------------------------------------------------------------------

class TestContextCompressorModelThresholds:
    @patch("agent.context_compressor.get_model_context_length", return_value=1_000_000)
    def test_init_large_context_with_override(self, _mock):
        """Large context (>=512K) + per-model override: override applies directly."""
        cc = ContextCompressor(
            model="glm-5.2",
            threshold_percent=0.50,
            model_thresholds={"glm-5.2": 0.40},
            quiet_mode=True,
        )
        # 1M context >= 512K, so no small-context floor — override wins
        assert cc.threshold_percent == 0.40
        assert cc.threshold_tokens == int(1_000_000 * 0.40)




    @patch("agent.context_compressor.get_model_context_length", return_value=256_000)
    def test_init_no_model_thresholds_dict(self, _mock):
        """Empty model_thresholds dict = backward compatible."""
        cc = ContextCompressor(
            model="glm-5.2",
            threshold_percent=0.50,
            quiet_mode=True,
        )
        # Resolve while mock is active (lazy init defers floor past __init__).
        _ = cc.context_length
        # 256K < 512K → floored at 0.75
        assert cc.threshold_percent == 0.75
        assert cc.model_thresholds == {}


    @patch("agent.context_compressor.get_model_context_length")
    def test_update_model_re_resolves_threshold(self, mock_ctx):
        """Switching models re-resolves the per-model threshold + re-applies floor."""
        mock_ctx.return_value = 256_000
        cc = ContextCompressor(
            model="glm-5.2",
            threshold_percent=0.50,
            model_thresholds={"glm-5.2": 0.80, "glm-5.2-1M": 0.25},
            quiet_mode=True,
        )
        # 256K < 512K → floor at 0.75; override 0.80 > 0.75, so 0.80 wins
        assert cc.threshold_percent == 0.80

        # Switch to the 1M model (large context, no floor)
        mock_ctx.return_value = 1_000_000
        cc.update_model(
            model="glm-5.2-1M",
            context_length=1_000_000,
        )
        # 1M >= 512K → no floor; override 0.25 applies directly
        assert cc.threshold_percent == 0.25
        assert cc.threshold_tokens == int(1_000_000 * 0.25)



# ---------------------------------------------------------------------------
# ContextEngine base class
# ---------------------------------------------------------------------------

class TestContextEngineModelThresholds:
    def test_base_class_update_model_applies_overrides(self):
        """The base-class update_model() applies model_thresholds if set."""
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
        engine._config_threshold_percent = 0.50
        engine.context_length = 0
        engine.model_thresholds = {"glm-5.2-1M": 0.25}

        engine.update_model(model="glm-5.2-1M", context_length=1_000_000)
        assert engine.threshold_percent == 0.25
        assert engine.threshold_tokens == int(1_000_000 * 0.25)

