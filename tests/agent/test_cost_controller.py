"""Unit tests for agent/cost_controller.py."""

import threading
import unittest
from agent.cost_controller import (
    CostAlert,
    CostBudget,
    CostController,
    CostSnapshot,
    build_default_budget,
)


class TestCostBudget(unittest.TestCase):
    def test_default_values(self):
        b = CostBudget()
        self.assertEqual(b.max_cost_usd, 0.0)
        self.assertEqual(b.alert_thresholds, (0.5, 0.8, 1.0))
        self.assertTrue(b.enabled)

    def test_has_limit_true_when_positive(self):
        b = CostBudget(max_cost_usd=5.0)
        self.assertTrue(b.has_limit)

    def test_has_limit_false_when_zero(self):
        b = CostBudget(max_cost_usd=0.0)
        self.assertFalse(b.has_limit)

    def test_has_limit_false_when_negative(self):
        b = CostBudget(max_cost_usd=-1.0)
        self.assertFalse(b.has_limit)


class TestCostSnapshot(unittest.TestCase):
    def test_budget_label_no_limit(self):
        snap = CostSnapshot(
            cumulative_cost_usd=1.5,
            budget_usd=0.0,
            spent_ratio=0.0,
            is_over_budget=False,
            alert_threshold_crossed=None,
            thresholds_triggered=(),
            output_tokens_per_dollar=0.0,
            tool_call_count=0,
        )
        self.assertEqual(snap.budget_label, "no-limit")

    def test_budget_label_with_limit(self):
        snap = CostSnapshot(
            cumulative_cost_usd=1.5,
            budget_usd=5.0,
            spent_ratio=0.3,
            is_over_budget=False,
            alert_threshold_crossed=None,
            thresholds_triggered=(),
            output_tokens_per_dollar=0.0,
            tool_call_count=0,
        )
        self.assertEqual(snap.budget_label, "$1.5000 / $5.00")

    def test_spent_percent_no_limit(self):
        snap = CostSnapshot(
            cumulative_cost_usd=2.0,
            budget_usd=0.0,
            spent_ratio=0.0,
            is_over_budget=False,
            alert_threshold_crossed=None,
            thresholds_triggered=(),
            output_tokens_per_dollar=0.0,
            tool_call_count=0,
        )
        self.assertEqual(snap.spent_percent, "$2.0000")

    def test_spent_percent_with_limit(self):
        snap = CostSnapshot(
            cumulative_cost_usd=2.5,
            budget_usd=5.0,
            spent_ratio=0.5,
            is_over_budget=False,
            alert_threshold_crossed=None,
            thresholds_triggered=(),
            output_tokens_per_dollar=0.0,
            tool_call_count=0,
        )
        self.assertEqual(snap.spent_percent, "50.0%")


class TestCostControllerBasic(unittest.TestCase):
    def test_default_constructor(self):
        cc = CostController()
        self.assertIsInstance(cc.budget, CostBudget)
        self.assertEqual(cc.budget.max_cost_usd, 0.0)

    def test_custom_budget_via_constructor(self):
        budget = CostBudget(max_cost_usd=10.0, alert_thresholds=(0.25, 0.5, 0.75, 1.0))
        cc = CostController(budget=budget)
        self.assertEqual(cc.budget.max_cost_usd, 10.0)
        self.assertEqual(cc.budget.alert_thresholds, (0.25, 0.5, 0.75, 1.0))

    def test_budget_setter(self):
        cc = CostController()
        new_budget = CostBudget(max_cost_usd=3.0)
        cc.budget = new_budget
        self.assertEqual(cc.budget, new_budget)

    def test_reset_clears_counters(self):
        cc = CostController(budget=CostBudget(max_cost_usd=5.0))
        cc.add_cost(1.0, output_tokens=100, tool_calls=2)
        cc.reset()
        snap = cc.snapshot()
        self.assertEqual(snap.cumulative_cost_usd, 0.0)
        self.assertEqual(snap.output_tokens_per_dollar, 0.0)
        self.assertEqual(snap.tool_call_count, 0)
        self.assertIsNone(snap.alert_threshold_crossed)

    def test_snapshot_no_budget(self):
        cc = CostController(budget=CostBudget(max_cost_usd=0.0))
        cc.add_cost(2.0, output_tokens=500)
        snap = cc.snapshot()
        self.assertEqual(snap.cumulative_cost_usd, 2.0)
        self.assertEqual(snap.budget_usd, 0.0)
        self.assertEqual(snap.spent_ratio, 0.0)
        self.assertFalse(snap.is_over_budget)

    def test_snapshot_with_budget_under_limit(self):
        cc = CostController(budget=CostBudget(max_cost_usd=10.0))
        cc.add_cost(3.0, output_tokens=600)
        snap = cc.snapshot()
        self.assertEqual(snap.cumulative_cost_usd, 3.0)
        self.assertEqual(snap.budget_usd, 10.0)
        self.assertAlmostEqual(snap.spent_ratio, 0.3)
        self.assertFalse(snap.is_over_budget)

    def test_snapshot_with_budget_at_limit(self):
        cc = CostController(budget=CostBudget(max_cost_usd=5.0))
        cc.add_cost(5.0, output_tokens=1000)
        snap = cc.snapshot()
        self.assertEqual(snap.cumulative_cost_usd, 5.0)
        self.assertEqual(snap.spent_ratio, 1.0)
        self.assertTrue(snap.is_over_budget)

    def test_snapshot_with_budget_over_limit(self):
        cc = CostController(budget=CostBudget(max_cost_usd=5.0))
        cc.add_cost(7.0, output_tokens=2000)
        snap = cc.snapshot()
        self.assertTrue(snap.is_over_budget)
        self.assertGreater(snap.spent_ratio, 1.0)

    def test_cost_effectiveness_ratio(self):
        cc = CostController(budget=CostBudget(max_cost_usd=10.0))
        cc.add_cost(2.0, output_tokens=1000)
        snap = cc.snapshot()
        self.assertEqual(snap.output_tokens_per_dollar, 500.0)

    def test_cost_effectiveness_zero_cost(self):
        cc = CostController(budget=CostBudget(max_cost_usd=10.0))
        cc.add_cost(0.0, output_tokens=100)
        snap = cc.snapshot()
        self.assertEqual(snap.output_tokens_per_dollar, 0.0)


class TestCostControllerAlertThresholds(unittest.TestCase):
    def test_no_alert_without_budget(self):
        cc = CostController(budget=CostBudget(max_cost_usd=0.0))
        alert = cc.add_cost(1.0)
        self.assertIsNone(alert)

    def test_no_alert_when_disabled(self):
        cc = CostController(budget=CostBudget(max_cost_usd=5.0, enabled=False))
        alert = cc.add_cost(4.0)
        self.assertIsNone(alert)

    def test_no_alert_on_zero_cost(self):
        cc = CostController(budget=CostBudget(max_cost_usd=5.0))
        alert = cc.add_cost(0.0)
        self.assertIsNone(alert)

    def test_no_alert_on_negative_cost(self):
        cc = CostController(budget=CostBudget(max_cost_usd=5.0))
        alert = cc.add_cost(-1.0)
        self.assertIsNone(alert)

    def test_first_threshold_50_percent(self):
        cc = CostController(budget=CostBudget(max_cost_usd=10.0, alert_thresholds=(0.5, 0.8, 1.0)))
        alert = cc.add_cost(5.0)
        self.assertIsNotNone(alert)
        self.assertEqual(alert.threshold, 0.5)
        self.assertEqual(alert.threshold_label, "50%")
        self.assertFalse(alert.is_hard_limit)

    def test_second_threshold_80_percent(self):
        cc = CostController(budget=CostBudget(max_cost_usd=10.0, alert_thresholds=(0.5, 0.8, 1.0)))
        cc.add_cost(5.0)  # 50% - triggers first alert
        alert = cc.add_cost(3.0)  # 80% - triggers second
        self.assertIsNotNone(alert)
        self.assertEqual(alert.threshold, 0.8)
        self.assertEqual(alert.threshold_label, "80%")

    def test_hard_limit_at_100_percent(self):
        cc = CostController(budget=CostBudget(max_cost_usd=10.0, alert_thresholds=(0.5, 0.8, 1.0)))
        cc.add_cost(5.0)
        cc.add_cost(3.0)
        alert = cc.add_cost(2.0)  # exactly 100%
        self.assertIsNotNone(alert)
        self.assertEqual(alert.threshold, 1.0)
        self.assertEqual(alert.threshold_label, "100% (budget exceeded)")
        self.assertTrue(alert.is_hard_limit)

    def test_alert_only_fires_on_new_threshold(self):
        """Once a threshold is triggered, subsequent add_cost still returns an alert
        because spent_ratio is still >= the triggered threshold. Only when a new,
        higher threshold is crossed does a different alert fire."""
        cc = CostController(budget=CostBudget(max_cost_usd=10.0, alert_thresholds=(0.5, 0.8, 1.0)))
        alert1 = cc.add_cost(5.0)  # 50% - fires
        alert2 = cc.add_cost(1.0)  # 60% - still >= 50%, alert re-fires
        self.assertIsNotNone(alert1)
        self.assertIsNotNone(alert2)
        self.assertEqual(alert2.threshold, 0.5)  # Still at 50% threshold
        # Crossing 80% fires the new threshold
        alert3 = cc.add_cost(3.0)  # 80%
        self.assertIsNotNone(alert3)
        self.assertEqual(alert3.threshold, 0.8)

    def test_alert_accumulates_thresholds(self):
        cc = CostController(budget=CostBudget(max_cost_usd=10.0, alert_thresholds=(0.5, 0.8, 1.0)))
        cc.add_cost(5.0)   # 50%
        cc.add_cost(3.0)   # 80%
        cc.add_cost(2.0)   # 100%
        snap = cc.snapshot()
        self.assertEqual(snap.thresholds_triggered, (0.5, 0.8, 1.0))
        self.assertEqual(snap.alert_threshold_crossed, 1.0)

    def test_check_alert_no_cost_change(self):
        cc = CostController(budget=CostBudget(max_cost_usd=10.0))
        cc.add_cost(5.0)  # 50%
        # check_alert should re-check without adding cost
        alert = cc.check_alert()
        self.assertIsNotNone(alert)
        self.assertEqual(alert.threshold, 0.5)

    def test_callback_invoked_on_alert(self):
        received = []

        def on_alert(alert: CostAlert):
            received.append(alert)

        cc = CostController(
            budget=CostBudget(max_cost_usd=10.0),
            on_alert=on_alert,
        )
        cc.add_cost(5.0)
        self.assertEqual(len(received), 1)
        self.assertEqual(received[0].threshold, 0.5)

    def test_callback_exception_does_not_crash(self):
        def bad_callback(alert):
            raise RuntimeError("boom")

        cc = CostController(
            budget=CostBudget(max_cost_usd=10.0),
            on_alert=bad_callback,
        )
        # Should not raise
        alert = cc.add_cost(5.0)
        self.assertIsNotNone(alert)


class TestCostControllerThreadSafety(unittest.TestCase):
    def test_concurrent_add_cost(self):
        cc = CostController(budget=CostBudget(max_cost_usd=1000.0))
        errors = []

        def worker(n):
            try:
                for _ in range(50):
                    cc.add_cost(0.1, output_tokens=10, tool_calls=1)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(errors, [])
        snap = cc.snapshot()
        self.assertAlmostEqual(snap.cumulative_cost_usd, 25.0, places=5)
        self.assertAlmostEqual(snap.output_tokens_per_dollar, 100.0, places=3)  # 2500 tokens / 25.0 USD
        self.assertEqual(snap.tool_call_count, 250)


class TestBuildDefaultBudget(unittest.TestCase):
    def test_default_values(self):
        b = build_default_budget()
        self.assertEqual(b.max_cost_usd, 0.0)
        self.assertEqual(b.alert_thresholds, (0.5, 0.8, 1.0))
        self.assertTrue(b.enabled)

    def test_custom_max_cost(self):
        b = build_default_budget(max_cost_usd=5.0)
        self.assertEqual(b.max_cost_usd, 5.0)

    def test_custom_thresholds(self):
        b = build_default_budget(alert_thresholds=(0.3, 0.7, 1.0))
        self.assertEqual(b.alert_thresholds, (0.3, 0.7, 1.0))

    def test_disabled_budget(self):
        b = build_default_budget(enabled=False)
        self.assertFalse(b.enabled)


class TestFormatAlertMessage(unittest.TestCase):
    def test_hard_limit_message(self):
        cc = CostController()
        alert = CostAlert(
            threshold=1.0,
            threshold_label="100% (budget exceeded)",
            cumulative_cost_usd=5.0,
            budget_usd=5.0,
            spent_ratio=1.0,
            is_hard_limit=True,
        )
        msg = cc.format_alert_message(alert)
        self.assertIn("HARD LIMIT REACHED", msg)
        self.assertIn("5.0000", msg)
        self.assertIn("5.00", msg)

    def test_soft_limit_message(self):
        cc = CostController()
        alert = CostAlert(
            threshold=0.5,
            threshold_label="50%",
            cumulative_cost_usd=2.5,
            budget_usd=5.0,
            spent_ratio=0.5,
            is_hard_limit=False,
        )
        msg = cc.format_alert_message(alert)
        self.assertIn("50%", msg)
        self.assertIn("2.5000", msg)
        self.assertIn("5.00", msg)


if __name__ == "__main__":
    unittest.main()
