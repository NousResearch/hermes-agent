"""Tests for agent/hermes/cost_dashboard.py."""
import threading
import time
import unittest
from unittest.mock import MagicMock

from agent.hermes.cost_dashboard import (
    COST_DASHBOARD_UPDATED,
    CostDashboard,
    CostDashboardSnapshot,
)
from agent.hermes.analytics import EventBus


class TestCostDashboardSnapshot(unittest.TestCase):
    def test_default_values(self):
        snap = CostDashboardSnapshot()
        self.assertEqual(snap.total_cost_usd, 0.0)
        self.assertEqual(snap.cost_per_minute, 0.0)
        self.assertEqual(snap.cost_per_call, 0.0)
        self.assertEqual(snap.budget_used_pct, 0.0)
        self.assertEqual(snap.is_over_budget, False)
        self.assertEqual(snap.alert_count, 0)
        self.assertEqual(snap.per_source, {})
        self.assertEqual(snap.per_tool, {})

    def test_to_dict_roundtrip(self):
        snap = CostDashboardSnapshot(
            total_cost_usd=1.234567,
            cost_per_minute=0.5,
            cost_per_call=0.1,
            budget_used_pct=0.25,
            per_source={"claude": 1.0, "openai": 0.5},
            per_tool={"terminal": 0.01},
            total_tokens=1000,
            input_tokens=800,
            output_tokens=200,
            llm_call_count=5,
            tool_call_count=10,
            session_elapsed_seconds=120.0,
        )
        d = snap.to_dict()
        self.assertAlmostEqual(d["total_cost_usd"], 1.234567, places=5)
        self.assertAlmostEqual(d["cost_per_minute"], 0.5, places=5)
        self.assertAlmostEqual(d["cost_per_call"], 0.1, places=5)
        self.assertAlmostEqual(d["budget_used_pct"], 0.25, places=2)
        self.assertEqual(d["per_source"], {"claude": 1.0, "openai": 0.5})
        self.assertEqual(d["per_tool"], {"terminal": 0.01})
        self.assertEqual(d["total_tokens"], 1000)
        self.assertEqual(d["llm_call_count"], 5)
        self.assertEqual(d["tool_call_count"], 10)
        self.assertAlmostEqual(d["session_elapsed_seconds"], 120.0, places=1)

    def test_immutable(self):
        snap = CostDashboardSnapshot(total_cost_usd=5.0)
        with self.assertRaises(AttributeError):
            snap.total_cost_usd = 10.0


class TestCostDashboardFormatTerminal(unittest.TestCase):
    def test_empty_at_startup(self):
        """No rendering when nothing has happened."""
        snap = CostDashboardSnapshot()
        result = CostDashboard.format_terminal(snap)
        self.assertEqual(result, "")

    def test_shows_cost_without_budget(self):
        snap = CostDashboardSnapshot(
            total_cost_usd=0.48,
            has_budget=False,
            session_elapsed_seconds=60.0,
        )
        result = CostDashboard.format_terminal(snap)
        self.assertIn("0.4800", result)
        self.assertIn("💰", result)

    def test_shows_budget_bar(self):
        snap = CostDashboardSnapshot(
            total_cost_usd=0.48,
            budget_remaining_usd=4.52,
            budget_used_pct=0.096,
            has_budget=True,
            is_over_budget=False,
            session_elapsed_seconds=60.0,
        )
        result = CostDashboard.format_terminal(snap)
        self.assertIn("$0.4800", result)
        # 9.6% rounds to 9%
        self.assertIn("9%", result)
        self.assertIn("⏱", result)

    def test_shows_alert_indicator(self):
        snap = CostDashboardSnapshot(
            total_cost_usd=5.0,
            has_budget=True,
            budget_used_pct=1.0,
            is_over_budget=True,
            is_over_hard_limit=True,
            alert_count=1,
            active_alerts=[{"severity": "HARD_LIMIT", "message": "budget exceeded"}],
            session_elapsed_seconds=60.0,
        )
        result = CostDashboard.format_terminal(snap)
        # Hard limit shows 🔴🔔 indicator
        self.assertIn("🔴🔔1", result)


class TestCostDashboardFormatDetailed(unittest.TestCase):
    def test_format_detailed_complete(self):
        snap = CostDashboardSnapshot(
            total_cost_usd=1.5,
            cost_per_minute=0.25,
            cost_per_call=0.15,
            budget_remaining_usd=3.5,
            budget_used_pct=0.3,
            has_budget=True,
            per_source={"claude": 1.2, "openai": 0.3},
            per_tool={"terminal": 0.05},
            total_tokens=5000,
            input_tokens=4000,
            output_tokens=1000,
            cache_read_tokens=500,
            cache_write_tokens=100,
            llm_call_count=10,
            tool_call_count=20,
            output_tokens_per_dollar=666.67,
            session_elapsed_seconds=360.0,
        )
        result = CostDashboard.format_detailed(snap)
        self.assertIn("$1.500000", result)
        self.assertIn("$0.250000/min", result)
        self.assertIn("claude: $1.200000", result)
        self.assertIn("terminal: $0.050000", result)
        self.assertIn("tok/$", result)
        self.assertIn("10 calls", result)
        self.assertIn("20 calls", result)


class TestCostDashboardBasic(unittest.TestCase):
    def test_default_constructor(self):
        dashboard = CostDashboard()
        snap = dashboard.get_snapshot()
        self.assertIsInstance(snap, CostDashboardSnapshot)
        self.assertEqual(snap.total_cost_usd, 0.0)

    def test_with_mock_attributor(self):
        """Dashboard aggregates data from CostAttributor."""
        mock_attributor = MagicMock()
        mock_attributor.get_cost_breakdown.return_value = {
            "total_cost_usd": 2.5,
            "per_source": {"claude": 2.0, "openai": 0.5},
            "per_tool": {"terminal": 0.01},
            "per_source_usage": {
                "claude": {"input_tokens": 1000, "output_tokens": 500,
                           "cache_read_tokens": 100, "cache_write_tokens": 10,
                           "total_tokens": 1610},
            },
        }

        dashboard = CostDashboard(cost_attributor=mock_attributor)
        snap = dashboard.get_snapshot()

        self.assertAlmostEqual(snap.total_cost_usd, 2.5)
        self.assertEqual(snap.per_source, {"claude": 2.0, "openai": 0.5})
        self.assertEqual(snap.per_tool, {"terminal": 0.01})
        self.assertEqual(snap.total_tokens, 1610)
        self.assertEqual(snap.input_tokens, 1000)
        self.assertEqual(snap.output_tokens, 500)

    def test_with_mock_controller(self):
        """Dashboard reads budget from CostController."""
        mock_controller = MagicMock()
        mock_snap = MagicMock()
        mock_snap.spent_ratio = 0.4
        mock_snap.budget_remaining_usd = 3.0
        mock_snap.is_over_budget = False
        mock_snap.budget_label = "$2.00 / $5.00"
        mock_snap.budget_usd = 5.0
        mock_snap.output_tokens_per_dollar = 250.0
        mock_snap.thresholds_triggered = []
        mock_controller.snapshot.return_value = mock_snap

        dashboard = CostDashboard(cost_controller=mock_controller)
        snap = dashboard.get_snapshot()

        self.assertAlmostEqual(snap.budget_used_pct, 0.4)
        self.assertAlmostEqual(snap.budget_remaining_usd, 3.0)
        self.assertFalse(snap.is_over_budget)
        self.assertTrue(snap.has_budget)
        self.assertEqual(snap.output_tokens_per_dollar, 250.0)

    def test_with_mock_alert_manager(self):
        """Dashboard reads active alerts from AlertManager."""
        mock_alert_mgr = MagicMock()
        mock_alert = MagicMock()
        mock_alert.to_dict.return_value = {
            "alert_id": "alert-1",
            "severity": "WARNING",
            "category": "cost",
            "message": "50% budget used",
        }
        mock_alert_mgr.get_active_alerts.return_value = [mock_alert]

        dashboard = CostDashboard(alert_manager=mock_alert_mgr)
        snap = dashboard.get_snapshot()

        self.assertEqual(snap.alert_count, 1)
        self.assertEqual(snap.active_alerts[0]["severity"], "WARNING")

    def test_cost_per_minute_derived(self):
        """Cost per minute is derived from elapsed time."""
        elapsed = 120.0
        dashboard = CostDashboard(session_elapsed_seconds=lambda: elapsed)

        mock_ca = MagicMock()
        mock_ca.get_cost_breakdown.return_value = {
            "total_cost_usd": 1.0,
            "per_source": {},
            "per_tool": {},
            "per_source_usage": {},
        }
        dashboard._cost_attributor = mock_ca

        snap = dashboard.get_snapshot()
        # $1.00 over 120s = $0.50/min
        self.assertAlmostEqual(snap.cost_per_minute, 0.5, places=4)

    def test_format_terminal_empty(self):
        """format_terminal returns empty string at startup."""
        dashboard = CostDashboard()
        snap = dashboard.get_snapshot()
        result = CostDashboard.format_terminal(snap)
        self.assertEqual(result, "")


class TestCostDashboardCallbacks(unittest.TestCase):
    def test_on_update_callback(self):
        received = []

        def cb(snap):
            received.append(snap)

        dashboard = CostDashboard()
        dashboard.on_update(cb)
        dashboard.emit_update()

        self.assertEqual(len(received), 1)
        self.assertIsInstance(received[0], CostDashboardSnapshot)

    def test_multiple_callbacks(self):
        results = [[], []]

        def cb1(snap):
            results[0].append(snap)

        def cb2(snap):
            results[1].append(snap)

        dashboard = CostDashboard()
        dashboard.on_update(cb1)
        dashboard.on_update(cb2)
        dashboard.emit_update()

        self.assertEqual(len(results[0]), 1)
        self.assertEqual(len(results[1]), 1)

    def test_callback_exception_swallowed(self):
        def bad_cb(snap):
            raise RuntimeError("boom")

        dashboard = CostDashboard()
        dashboard.on_update(bad_cb)
        # Should not raise
        dashboard.emit_update()

    def test_clear_callbacks(self):
        called = []

        def cb(snap):
            called.append(snap)

        dashboard = CostDashboard()
        dashboard.on_update(cb)
        dashboard.clear_callbacks()
        dashboard.emit_update()
        self.assertEqual(len(called), 0)


class TestCostDashboardEventBus(unittest.TestCase):
    def test_registers_with_event_bus(self):
        event_bus = EventBus()
        dashboard = CostDashboard(event_bus=event_bus)
        # Verify subscriptions were registered (using EventType constants)
        self.assertGreater(event_bus.get_handler_count("llm.response"), 0)
        self.assertGreater(event_bus.get_handler_count("tool.result"), 0)

    def test_emits_cost_dashboard_updated(self):
        received = []

        def handler(event):
            received.append(event)

        event_bus = EventBus()
        event_bus.subscribe(COST_DASHBOARD_UPDATED, handler)

        dashboard = CostDashboard(event_bus=event_bus)
        dashboard.emit_update()

        self.assertEqual(len(received), 1)
        self.assertEqual(received[0].type, COST_DASHBOARD_UPDATED)

    def test_llm_call_count_tracked(self):
        event_bus = EventBus()
        dashboard = CostDashboard(event_bus=event_bus)

        for _ in range(5):
            event_bus.emit_event("llm.response", {}, session_id="test")

        snap = dashboard.get_snapshot()
        self.assertEqual(snap.llm_call_count, 5)

    def test_tool_call_count_tracked(self):
        event_bus = EventBus()
        dashboard = CostDashboard(event_bus=event_bus)

        for _ in range(3):
            event_bus.emit_event("tool.result", {}, session_id="test")

        snap = dashboard.get_snapshot()
        self.assertEqual(snap.tool_call_count, 3)


class TestCostDashboardThreadSafety(unittest.TestCase):
    def test_concurrent_callbacks(self):
        errors = []
        received = []

        def cb(snap):
            try:
                received.append(snap.total_cost_usd)
            except Exception as e:
                errors.append(e)

        dashboard = CostDashboard()

        mock_ca = MagicMock()
        mock_ca.get_cost_breakdown.return_value = {
            "total_cost_usd": 1.0,
            "per_source": {},
            "per_tool": {},
            "per_source_usage": {},
        }
        mock_ca._llm_event_count = 10  # Must be int for comparison
        dashboard._cost_attributor = mock_ca

        dashboard.on_update(cb)

        def worker():
            for _ in range(50):
                dashboard.emit_update()

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(errors, [])
        self.assertEqual(len(received), 200)


if __name__ == "__main__":
    unittest.main()
