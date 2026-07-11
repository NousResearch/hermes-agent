import unittest

from agent.block_logic_threshold import (
    SessionMetrics,
    ThresholdConfig,
    ThresholdLevel,
    evaluate,
)


class BlockLogicThresholdTests(unittest.TestCase):
    def test_recommends_block_logic_at_default_82_percent_boundary(self):
        decision = evaluate(SessionMetrics(context_used=82_000, context_limit=100_000))

        self.assertIs(decision.level, ThresholdLevel.RECOMMEND)
        self.assertTrue(decision.should_offer_run_now)
        self.assertFalse(decision.should_create_temporary_rescue)
        self.assertFalse(decision.canonical_write_allowed)

    def test_power_sprint_recommends_without_authorizing_rescue(self):
        decision = evaluate(
            SessionMetrics(
                context_used=78_000,
                context_limit=100_000,
                elapsed_hours=36,
                message_count=190,
                decision_count=24,
                changed_files=18,
                model_handoffs=3,
            )
        )

        self.assertIs(decision.level, ThresholdLevel.RECOMMEND)
        self.assertIn("power-sprint continuity override triggered", decision.reasons)
        self.assertFalse(decision.should_create_temporary_rescue)

    def test_prior_compression_can_trigger_rescue_at_prominent_pressure(self):
        decision = evaluate(
            SessionMetrics(
                context_used=91_000,
                context_limit=100_000,
                compression_events=1,
            )
        )

        self.assertIs(decision.level, ThresholdLevel.RESCUE)
        self.assertTrue(decision.should_create_temporary_rescue)
        self.assertFalse(decision.canonical_write_allowed)

    def test_config_mapping_overrides_thresholds_without_enabling_writes(self):
        config = ThresholdConfig.from_mapping(
            {
                "quiet_threshold": 0.60,
                "recommend_threshold": 0.70,
                "prominent_threshold": 0.78,
                "rescue_threshold": 0.82,
            }
        )
        decision = evaluate(
            SessionMetrics(context_used=70_000, context_limit=100_000),
            config,
        )

        self.assertIs(decision.level, ThresholdLevel.RECOMMEND)
        self.assertFalse(decision.canonical_write_allowed)


if __name__ == "__main__":
    unittest.main()
