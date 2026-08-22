"""Tests for WARN-only unified report template assertions (TKT-0033 Phase A).

``check_report_markers(body)`` returns a list of warning strings. An empty
list means the report passed every marker check. Violations are returned as
human-readable warning strings — the delivery pipeline logs them as WARNINGS
and NEVER blocks delivery on them.
"""

from __future__ import annotations

from cron.template_asserts import check_report_markers


_FULL_REPORT = """\
Unified Report — 2026-08-17
Verdict: PASS — all systems nominal

## Gateway
✅ telegram adapter healthy

## Cron
✅ 12 jobs ran, 0 failures
"""


def test_full_report_returns_no_warnings():
    assert check_report_markers(_FULL_REPORT) == []


def test_missing_verdict_line_warns_about_verdict():
    body = "## Gateway\n✅ all good\n"
    warnings = check_report_markers(body)
    assert any("verdict" in w.lower() for w in warnings), warnings


def test_missing_health_icon_warns_about_icon():
    body = "Verdict: PASS\n\n## Gateway\nall good\n"
    warnings = check_report_markers(body)
    assert any("icon" in w.lower() for w in warnings), warnings


def test_missing_section_structure_warns_about_section():
    body = "Verdict: PASS\n✅ all good, no headers at all\n"
    warnings = check_report_markers(body)
    assert any("section" in w.lower() for w in warnings), warnings
