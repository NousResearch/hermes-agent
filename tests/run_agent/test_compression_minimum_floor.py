"""Configurable minimum context compression floor (#31600)."""

from unittest.mock import patch

from agent.model_metadata import (
    MINIMUM_CONTEXT_LENGTH,
    _MINIMUM_CONTEXT_FLOOR_HARD_LIMIT,
    get_configurable_minimum_context,
)


class TestConfigurableMinimumContext:
    """Unit tests for ``get_configurable_minimum_context``."""

    def test_default_returns_minimum_context_length(self):
        assert get_configurable_minimum_context(None) == MINIMUM_CONTEXT_LENGTH
        assert get_configurable_minimum_context() == MINIMUM_CONTEXT_LENGTH

    def test_config_floor_respected(self):
        assert get_configurable_minimum_context(32_000) == 32_000
        assert get_configurable_minimum_context(65_536) == 65_536

    def test_config_floor_clamped_to_hard_limit(self):
        assert get_configurable_minimum_context(1_000) == _MINIMUM_CONTEXT_FLOOR_HARD_LIMIT
        assert get_configurable_minimum_context(0) == _MINIMUM_CONTEXT_FLOOR_HARD_LIMIT
        assert get_configurable_minimum_context(-1) == _MINIMUM_CONTEXT_FLOOR_HARD_LIMIT
        assert get_configurable_minimum_context(15_999) == _MINIMUM_CONTEXT_FLOOR_HARD_LIMIT


class TestContextCompressorFloor:
    """Verify ContextCompressor uses the configurable floor in real paths."""

    def test_default_floor_in_threshold(self):
        from agent.context_compressor import ContextCompressor

        with patch("agent.context_compressor.get_model_context_length", return_value=100_000):
            cc = ContextCompressor(model="test", quiet_mode=True)

        assert cc._minimum_context_floor == MINIMUM_CONTEXT_LENGTH
        assert cc.threshold_tokens == MINIMUM_CONTEXT_LENGTH

    def test_small_model_with_lowered_floor(self):
        from agent.context_compressor import ContextCompressor

        with patch("agent.context_compressor.get_model_context_length", return_value=48_000):
            cc = ContextCompressor(
                model="test", quiet_mode=True, minimum_context_floor=24_000
            )

        assert cc._minimum_context_floor == 24_000
        assert cc.threshold_tokens == 24_000

    def test_floor_dominates_on_large_models_too(self):
        from agent.context_compressor import ContextCompressor

        with patch("agent.context_compressor.get_model_context_length", return_value=1_000_000):
            cc = ContextCompressor(
                model="test",
                quiet_mode=True,
                threshold_percent=0.02,
                minimum_context_floor=32_000,
            )

        assert cc.threshold_tokens == 32_000

    def test_update_model_uses_floor(self):
        from agent.context_compressor import ContextCompressor

        with patch("agent.context_compressor.get_model_context_length", return_value=200_000):
            cc = ContextCompressor(
                model="test", quiet_mode=True, minimum_context_floor=40_000
            )

        cc.update_model("switched", context_length=64_000)
        assert cc.threshold_tokens == 40_000
