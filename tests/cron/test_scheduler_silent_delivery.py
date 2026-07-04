"""Regression tests for cron [SILENT] delivery suppression.

A cron job should suppress delivery only when the final response is exactly the
silent sentinel. Reports that mention the sentinel as prose must still deliver.
"""

from cron import scheduler


def test_silent_response_requires_exact_marker():
    assert scheduler._is_silent_response("[SILENT]") is True
    assert scheduler._is_silent_response("  [silent]\n") is True


def test_report_mentioning_silent_marker_still_delivers():
    response = """# HQ Self-improvement Review

There is useful content here.
Future unchanged ticks may return `[SILENT]`, but this report should deliver.
"""
    assert scheduler._is_silent_response(response) is False
