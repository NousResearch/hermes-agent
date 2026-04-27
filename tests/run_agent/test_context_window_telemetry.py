"""Tests for context window usage telemetry."""
import pytest


class TestContextWindowTelemetry:
    """Agent should report context window usage to the user."""

    def test_usage_report_format(self):
        """Usage report should show percentage and absolute numbers."""
        from run_agent import ContextUsageReport

        report = ContextUsageReport(
            used_tokens=50000,
            total_tokens=128000,
            message_count=20,
        )

        assert report.percentage == pytest.approx(39.06, abs=0.1)
        assert report.used_tokens == 50000
        assert report.total_tokens == 128000
        assert "39%" in report.summary() or "39.1%" in report.summary()

    def test_usage_report_below_threshold_no_warning(self):
        """Below the warning threshold, is_warning should be False."""
        from run_agent import ContextUsageReport

        report = ContextUsageReport(
            used_tokens=50000,
            total_tokens=128000,
            message_count=20,
        )

        assert report.is_warning(threshold_percent=85) is False

    def test_usage_report_above_threshold_warns(self):
        """Above the warning threshold, is_warning should be True."""
        from run_agent import ContextUsageReport

        report = ContextUsageReport(
            used_tokens=110000,
            total_tokens=128000,
            message_count=40,
        )

        assert report.is_warning(threshold_percent=85) is True

    def test_summary_includes_message_count(self):
        """Summary should include the message count for context."""
        from run_agent import ContextUsageReport

        report = ContextUsageReport(
            used_tokens=50000,
            total_tokens=128000,
            message_count=42,
        )

        summary = report.summary()
        assert "42" in summary
        assert "50,000" in summary or "50000" in summary

    def test_zero_total_tokens_returns_zero_percent(self):
        """A zero context limit must not raise ZeroDivisionError."""
        from run_agent import ContextUsageReport

        report = ContextUsageReport(
            used_tokens=1000,
            total_tokens=0,
            message_count=5,
        )

        assert report.percentage == 0.0
        assert report.is_warning() is False

    def test_over_100_percent_usage(self):
        """Used tokens exceeding the limit should report > 100% and warn."""
        from run_agent import ContextUsageReport

        report = ContextUsageReport(
            used_tokens=130000,
            total_tokens=128000,
            message_count=60,
        )

        assert report.percentage > 100.0
        assert report.is_warning() is True

    def test_is_warning_at_exact_threshold(self):
        """is_warning should be True when usage equals the threshold exactly."""
        from run_agent import ContextUsageReport

        # 108,800 / 128,000 = exactly 85.0%
        report = ContextUsageReport(
            used_tokens=108800,
            total_tokens=128000,
            message_count=30,
        )

        assert report.percentage == pytest.approx(85.0)
        assert report.is_warning(threshold_percent=85.0) is True

    def test_is_warning_just_below_threshold(self):
        """is_warning should be False for usage just below the threshold."""
        from run_agent import ContextUsageReport

        report = ContextUsageReport(
            used_tokens=108799,
            total_tokens=128000,
            message_count=30,
        )

        assert report.is_warning(threshold_percent=85.0) is False
