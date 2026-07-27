"""Follow-up regression tests for per-model compression threshold overrides.

Covers the gaps flagged in the review of PR #63020:

1. Plugin-engine init ordering — ``compression.model_thresholds`` must be
   assigned to a selected plugin context engine BEFORE the initial
   ``update_model()`` call in agent init, so the initial model already gets
   its override (previously the override only took effect after the first
   ``/model`` switch).
2. ``compression.model_thresholds`` is a public key in ``DEFAULT_CONFIG``.
3. Floor interaction on the ``update_model()`` (model-switch) path:
   an override below the small-context floor is raised to the floor
   (raise-only); an override above the floor wins.
4. Base-class ``update_model()`` snapshots the pre-override percent once,
   so repeated switches fall back to the engine's configured threshold
   rather than a previous model's override.
"""

from unittest.mock import patch

from agent.context_compressor import ContextCompressor
from agent.context_engine import ContextEngine


class _StubEngine(ContextEngine):
    """Minimal concrete context engine for init-ordering tests."""

    @property
    def name(self) -> str:
        return "stub"

    def update_from_response(self, usage):
        pass

    def should_compress(self, prompt_tokens=None):
        return False

    def compress(self, messages, current_tokens=None):
        return messages


def test_plugin_engine_gets_model_thresholds_before_initial_update_model():
    """The initial model's override must apply during AIAgent init.

    Regression test for the PR #63020 review finding: the plugin engine was
    initialized through update_model() before model_thresholds was assigned,
    so the initial model kept the global threshold until a /model switch.
    """
    engine = _StubEngine()
    engine.threshold_percent = 0.50

    cfg = {
        "context": {"engine": "stub"},
        "agent": {},
        "compression": {
            "threshold": 0.50,
            "model_thresholds": {"glm-5.2": 0.25},
        },
    }

    with (
        patch("hermes_cli.config.load_config", return_value=cfg),
        patch("plugins.context_engine.load_context_engine", return_value=engine),
        patch("agent.model_metadata.get_model_context_length", return_value=1_000_000),
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        from run_agent import AIAgent

        agent = AIAgent(
            model="glm-5.2",
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )

    assert agent.context_compressor is engine
    # The override map arrived before the initial update_model() call, so the
    # very first resolution already used it.
    assert engine.model_thresholds == {"glm-5.2": 0.25}
    assert engine.threshold_percent == 0.25
    assert engine.threshold_tokens == int(1_000_000 * 0.25)


def test_plugin_engine_without_overrides_keeps_global_threshold():
    """Empty model_thresholds leaves plugin-engine init behavior unchanged."""
    engine = _StubEngine()
    engine.threshold_percent = 0.50

    cfg = {
        "context": {"engine": "stub"},
        "agent": {},
        "compression": {"threshold": 0.50},
    }

    with (
        patch("hermes_cli.config.load_config", return_value=cfg),
        patch("plugins.context_engine.load_context_engine", return_value=engine),
        patch("agent.model_metadata.get_model_context_length", return_value=1_000_000),
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        from run_agent import AIAgent

        agent = AIAgent(
            model="glm-5.2",
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )

    assert agent.context_compressor is engine
    assert getattr(engine, "model_thresholds", {}) == {}
    assert engine.threshold_percent == 0.50
    assert engine.threshold_tokens == int(1_000_000 * 0.50)


def test_model_thresholds_key_in_default_config():
    """compression.model_thresholds is a public DEFAULT_CONFIG key."""
    from hermes_cli.config import DEFAULT_CONFIG

    assert "model_thresholds" in DEFAULT_CONFIG["compression"]
    assert DEFAULT_CONFIG["compression"]["model_thresholds"] == {}


class TestFloorInteractionOnModelSwitch:
    """The small-context floor stacks on per-model overrides at switch time."""

    @patch("agent.context_compressor.get_model_context_length")
    def test_switch_override_below_floor_is_raised_to_floor(self, mock_ctx):
        """Switching to a small-context model with a sub-floor override → floor."""
        mock_ctx.return_value = 1_000_000
        cc = ContextCompressor(
            model="glm-5.2-1M",
            threshold_percent=0.50,
            model_thresholds={"glm-5.2-1M": 0.25, "small-model": 0.40},
            quiet_mode=True,
        )
        assert cc.threshold_percent == 0.25  # large context: override direct

        # Switch to a <512K model whose override (0.40) is below the 0.75 floor.
        mock_ctx.return_value = 128_000
        cc.update_model(model="small-model", context_length=128_000)
        assert cc.threshold_percent == 0.75  # raise-only floor wins
        assert cc.threshold_tokens == int(128_000 * 0.75)

    @patch("agent.context_compressor.get_model_context_length")
    def test_switch_override_above_floor_wins(self, mock_ctx):
        """Switching to a small-context model with an above-floor override → override."""
        mock_ctx.return_value = 1_000_000
        cc = ContextCompressor(
            model="glm-5.2-1M",
            threshold_percent=0.50,
            model_thresholds={"glm-5.2-1M": 0.25, "small-model": 0.85},
            quiet_mode=True,
        )
        mock_ctx.return_value = 128_000
        cc.update_model(model="small-model", context_length=128_000)
        assert cc.threshold_percent == 0.85  # above the 0.75 floor: override wins
        assert cc.threshold_tokens == int(128_000 * 0.85)


class TestBaseEngineConfigSnapshot:
    """Base-class update_model() must not compound a previous override."""

    def test_repeated_switches_fall_back_to_original_threshold(self):
        engine = _StubEngine()
        engine.threshold_percent = 0.50
        engine.context_length = 0
        engine.model_thresholds = {"glm-5.2-1M": 0.25}
        # NOTE: _config_threshold_percent deliberately NOT pre-set — the base
        # class must snapshot the original 0.50 on the first call, so the
        # second switch (no matching override) falls back to 0.50, not 0.25.

        engine.update_model(model="glm-5.2-1M", context_length=1_000_000)
        assert engine.threshold_percent == 0.25

        engine.update_model(model="some-other-model", context_length=1_000_000)
        assert engine.threshold_percent == 0.50
        assert engine.threshold_tokens == int(1_000_000 * 0.50)
