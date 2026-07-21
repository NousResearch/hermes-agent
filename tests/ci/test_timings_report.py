"""Tests for scripts/ci/timings_report.py — generate_review_status().

The review status is a JSON array in the unified nested format consumed
by the review comment assembler. It classifies the CI timings result as
info/warning (never error — timings is an observability job, not a gate)
and provides a one-line summary plus optional per-job delta detail.
"""

from __future__ import annotations

import importlib.util
from datetime import datetime, timezone
from pathlib import Path

_PATH = Path(__file__).resolve().parents[2] / "scripts" / "ci" / "timings_report.py"
_spec = importlib.util.spec_from_file_location("timings_report", _PATH)
if _spec is None or _spec.loader is None:
    raise ImportError("Failed to load timings_report.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

_T0 = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)


def _ts(seconds: float) -> str:
    """ISO timestamp `seconds` after T0."""
    dt = _T0.timestamp() + seconds
    return datetime.fromtimestamp(dt, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _job(name: str, dur_s: float, start_s: float = 0.0, conclusion: str = "success") -> dict:
    """Build a normalized job dict with realistic timestamps for wall-time math."""
    return {
        "name": name,
        "duration_s": dur_s,
        "conclusion": conclusion,
        "started_at": _ts(start_s),
        "completed_at": _ts(start_s + dur_s),
        "wait_s": 0.0,
    }


def _timings(jobs: list[dict]) -> dict:
    return {"run_id": "123", "head_sha": "abc", "created_at": "", "jobs": jobs}


def _result(statuses: list[dict]) -> dict:
    """Extract the single result dict from the nested format."""
    assert len(statuses) == 1
    assert statuses[0]["source"] == "ci timing"
    results = statuses[0]["results"]
    assert len(results) == 1
    return results[0]


def test_no_baseline_is_info():
    t = _timings([_job("tests", 60.0)])
    result = _result(_mod.generate_review_status(t, None))
    assert result["kind"] == "info"
    assert "no baseline" in result["summary"].lower()
    assert "link" not in result  # no report_url → no link field


def test_no_regression_is_info():
    cur = _timings([_job("tests", 60.0)])
    bl = _timings([_job("tests", 60.0)])
    result = _result(_mod.generate_review_status(cur, bl))
    assert result["kind"] == "info"
    assert "+0.0%" in result["summary"]


def test_small_regression_is_info():
    cur = _timings([_job("tests", 65.0)])
    bl = _timings([_job("tests", 60.0)])
    result = _result(_mod.generate_review_status(cur, bl))
    # +8.3% — well under the 25% warning threshold
    assert result["kind"] == "info"


def test_large_regression_is_warning():
    cur = _timings([_job("tests", 80.0)])
    bl = _timings([_job("tests", 60.0)])
    result = _result(_mod.generate_review_status(cur, bl))
    # +33% — above the 25% threshold
    assert result["kind"] == "warning"
    assert "+33" in result["summary"]


def test_improvement_is_info():
    cur = _timings([_job("tests", 40.0)])
    bl = _timings([_job("tests", 60.0)])
    result = _result(_mod.generate_review_status(cur, bl))
    assert result["kind"] == "info"
    assert "-33" in result["summary"]


def test_detail_shows_top_deltas():
    cur = _timings([_job("slow-job", 120.0), _job("fast-job", 30.0, start_s=120.0)])
    bl = _timings([_job("slow-job", 60.0), _job("fast-job", 60.0, start_s=60.0)])
    result = _result(_mod.generate_review_status(cur, bl))
    assert "slow-job" in result["detail"]
    assert "fast-job" in result["detail"]
    # Sorted by abs delta — slow-job (+60) before fast-job (-30)
    assert result["detail"].index("slow-job") < result["detail"].index("fast-job")


def test_skipped_jobs_excluded_from_detail():
    cur = _timings([_job("skipped-job", 0.0, conclusion="skipped"), _job("tests", 60.0)])
    bl = _timings([_job("skipped-job", 0.0, conclusion="skipped"), _job("tests", 60.0)])
    result = _result(_mod.generate_review_status(cur, bl))
    assert "skipped-job" not in result["detail"]


def test_report_url_passed_through():
    t = _timings([_job("tests", 60.0)])
    result = _result(_mod.generate_review_status(t, None, report_url="https://artifact/123"))
    assert result["link"] == "https://artifact/123"
    assert result["link_label"] == "View report"


def test_never_error_severity():
    """Timings is observability — even huge regressions are warnings, not errors."""
    cur = _timings([_job("tests", 600.0)])
    bl = _timings([_job("tests", 60.0)])
    result = _result(_mod.generate_review_status(cur, bl))
    assert result["kind"] == "warning"
    assert result["kind"] != "error"


def test_nested_format_structure():
    """The return value is a list with one {source, results: [...]} entry."""
    t = _timings([_job("tests", 60.0)])
    statuses = _mod.generate_review_status(t, None)
    assert isinstance(statuses, list)
    assert len(statuses) == 1
    assert statuses[0]["source"] == "ci timing"
    assert isinstance(statuses[0]["results"], list)
    assert len(statuses[0]["results"]) == 1
    r = statuses[0]["results"][0]
    assert r["kind"] == "info"
    assert r["title"] == "CI timings"
    assert "summary" in r
    assert "detail" in r
