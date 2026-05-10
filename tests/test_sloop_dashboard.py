"""Tests for the Sloop dashboard (tools/sloop_dashboard.py)."""

import json
from unittest.mock import MagicMock, patch

import pytest


class TestAsciiBarChart:
    def test_basic_chart(self):
        from tools.sloop_dashboard import _ascii_bar_chart

        chart = _ascii_bar_chart([1, 2, 3, 4, 5], height=5)
        assert isinstance(chart, str)
        assert "│" in chart

    def test_empty_values(self):
        from tools.sloop_dashboard import _ascii_bar_chart

        chart = _ascii_bar_chart([])
        assert chart == "(no data)"

    def test_all_zeros(self):
        from tools.sloop_dashboard import _ascii_bar_chart

        chart = _ascii_bar_chart([0, 0, 0])
        assert isinstance(chart, str)


class TestSparkline:
    def test_sparkline_basic(self):
        from tools.sloop_dashboard import _sparkline

        result = _sparkline([0, 1, 2, 3, 4, 5])
        assert isinstance(result, str)
        assert len(result) == 6

    def test_sparkline_empty(self):
        from tools.sloop_dashboard import _sparkline

        result = _sparkline([])
        assert result == "·"

    def test_sparkline_all_same(self):
        from tools.sloop_dashboard import _sparkline

        result = _sparkline([5, 5, 5])
        assert isinstance(result, str)


class TestStatusIcons:
    def test_healthy(self):
        from tools.sloop_dashboard import _status_icon, _health_word

        assert _status_icon(98) == "🟢"
        assert _health_word(98) == "HEALTHY"

    def test_degraded(self):
        from tools.sloop_dashboard import _status_icon, _health_word

        assert _status_icon(85) == "🟡"
        assert _health_word(85) == "DEGRADED"

    def test_unhealthy(self):
        from tools.sloop_dashboard import _status_icon, _health_word

        assert _status_icon(60) == "🟠"
        assert _health_word(60) == "UNHEALTHY"

    def test_critical(self):
        from tools.sloop_dashboard import _status_icon, _health_word

        assert _status_icon(30) == "🔴"
        assert _health_word(30) == "CRITICAL"


class TestRenderSloopStatus:
    def test_render_with_data(self):
        from tools.sloop_dashboard import render_sloop_status

        display = render_sloop_status(
            success_rate={
                "total": 50,
                "successes": 45,
                "failures": 5,
                "rate_percent": 90.0,
                "hours": 24,
            },
            avg_duration_ms=2500.0,
            failure_trend=[
                {"total": 5, "failures": 1, "successes": 4},
                {"total": 3, "failures": 0, "successes": 3},
                {"total": 4, "failures": 2, "successes": 2},
            ],
            recent_entries=[
                {
                    "timestamp": "2026-05-06T10:00:00",
                    "success": 1,
                    "duration_ms": 1200,
                    "job_name": "test_job",
                    "error_type": None,
                },
            ],
            error_types=[
                {"error_type": "timeout", "count": 3},
            ],
            consecutive_failures={"job_a": 4},
        )

        assert "Sloop Loop Status" in display
        assert "DEGRADED" in display  # 90% rate
        assert "90.0%" in display
        assert "2.5s" in display  # avg duration
        assert "CONSECUTIVE FAILURES" in display
        assert "test_job" in display
        assert "timeout" in display

    def test_render_empty(self):
        from tools.sloop_dashboard import render_sloop_status

        display = render_sloop_status(
            success_rate={"total": 0, "successes": 0, "failures": 0, "rate_percent": 0, "hours": 24},
            avg_duration_ms=None,
            failure_trend=[],
            recent_entries=[],
            error_types=[],
            consecutive_failures=None,
        )

        assert "Sloop Loop Status" in display
        assert "CRITICAL" in display  # 0% rate
        assert "(no data)" in display


class TestGetDashboardData:
    def test_gather_data(self):
        """Test that get_sloop_dashboard_data assembles data from feedback_collector."""
        mock_fc = MagicMock()
        mock_fc.get_success_rate.return_value = {
            "total": 10, "successes": 8, "failures": 2,
            "rate_percent": 80.0, "hours": 24,
        }
        mock_fc.get_avg_duration.return_value = 1500.0
        mock_fc.get_failure_trend.return_value = []
        mock_fc.get_recent_feedback.return_value = [
            {"job_id": "j1", "job_name": "Job 1", "success": 1},
        ]
        mock_fc.get_error_types.return_value = []
        mock_fc.get_consecutive_failures.return_value = 0

        from tools.sloop_dashboard import get_sloop_dashboard_data
        data = get_sloop_dashboard_data(feedback_collector=mock_fc)

        assert data["success_rate"]["rate_percent"] == 80.0
        assert data["avg_duration_ms"] == 1500.0
        assert len(data["recent_entries"]) == 1


class TestSloopDashboardTool:
    def test_status_action(self):
        mock_fc = MagicMock()
        mock_fc.get_success_rate.return_value = {
            "total": 5, "successes": 5, "failures": 0,
            "rate_percent": 100.0, "hours": 24,
        }
        mock_fc.get_avg_duration.return_value = 500.0
        mock_fc.get_failure_trend.return_value = [
            {"total": 2, "failures": 0, "successes": 2},
        ]
        mock_fc.get_recent_feedback.return_value = []
        mock_fc.get_error_types.return_value = []
        mock_fc.get_consecutive_failures.return_value = 0

        with patch("tools.sloop_dashboard.get_sloop_dashboard_data") as mock_gdd:
            mock_gdd.return_value = {
                "success_rate": mock_fc.get_success_rate(),
                "avg_duration_ms": 500.0,
                "failure_trend": mock_fc.get_failure_trend(),
                "recent_entries": [],
                "error_types": [],
                "consecutive_failures": {},
            }
            from tools.sloop_dashboard import sloop_dashboard_tool
            result = json.loads(sloop_dashboard_tool(action="status"))

        assert result["success"] is True
        assert "display" in result

    def test_data_action(self):
        with patch("tools.sloop_dashboard.get_sloop_dashboard_data") as mock_gdd:
            mock_gdd.return_value = {
                "success_rate": {"total": 0, "successes": 0, "failures": 0, "rate_percent": 0, "hours": 24},
                "avg_duration_ms": None,
                "failure_trend": [],
                "recent_entries": [],
                "error_types": [],
                "consecutive_failures": {},
            }
            from tools.sloop_dashboard import sloop_dashboard_tool
            result = json.loads(sloop_dashboard_tool(action="data"))

        assert result["success"] is True

    def test_unknown_action(self):
        from tools.sloop_dashboard import sloop_dashboard_tool
        result = json.loads(sloop_dashboard_tool(action="bogus"))
        assert result["success"] is False


class TestRegistry:
    def test_sloop_dashboard_registered(self):
        try:
            import tools.sloop_dashboard  # noqa: F401
        except ImportError:
            pytest.skip("sloop_dashboard not importable")
        from tools.registry import registry
        entry = registry.get_entry("sloop_dashboard")
        assert entry is not None
        assert entry.toolset == "sloop"
        assert entry.emoji == "📈"
