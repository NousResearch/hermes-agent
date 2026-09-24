"""Unit tests for the shared session activity observation contract."""

import sys

import pytest

from agent.session_activity import (
    ACTIVITY_DESCRIPTION_MAX,
    ActivityProvenance,
    bound_activity_description,
    build_activity_snapshot,
    format_iteration_progress,
    normalize_activity_provenance,
)


@pytest.mark.parametrize(
    "max_iterations, expected",
    [
        (sys.maxsize, "iteration 3"),  # AIAgent's default: unbounded, so no ceiling is shown
        (None, "iteration 3"),
        (250, "iteration 3/250"),  # a real budget (e.g. delegation.max_iterations) keeps N/M
    ],
)
def test_format_iteration_progress_hides_unbounded_ceiling(max_iterations, expected):
    out = format_iteration_progress(3, max_iterations)
    assert out == expected
    assert str(sys.maxsize) not in out


def test_bound_activity_description_truncates():
    long = "x" * (ACTIVITY_DESCRIPTION_MAX + 80)
    out = bound_activity_description(long)
    assert len(out) == ACTIVITY_DESCRIPTION_MAX
    assert out.endswith("…")






def test_normalize_activity_provenance_defaults_to_unknown():
    assert normalize_activity_provenance(None) is ActivityProvenance.UNKNOWN
    assert normalize_activity_provenance("") is ActivityProvenance.UNKNOWN
    assert normalize_activity_provenance("not-a-real-source") is ActivityProvenance.UNKNOWN
    assert normalize_activity_provenance("agent.activity") is ActivityProvenance.UNKNOWN
    assert (
        normalize_activity_provenance(ActivityProvenance.AGENT_COMPRESSION)
        is ActivityProvenance.AGENT_COMPRESSION
    )
    assert (
        normalize_activity_provenance("agent.compression_timeout")
        is ActivityProvenance.AGENT_COMPRESSION_TIMEOUT
    )


def test_build_activity_snapshot_includes_compat_aliases():
    snap = build_activity_snapshot(
        last_activity_at=100.0,
        last_activity_description="starting API call #1",
        last_activity_provenance=ActivityProvenance.UNKNOWN,
        now=110.0,
        extra={"api_call_count": 1},
    )
    assert snap["last_activity_at"] == 100.0
    assert snap["last_activity_description"] == "starting API call #1"
    assert snap["last_activity_provenance"] == "unknown"
    assert snap["seconds_since_activity"] == 10.0
    assert snap["last_activity_ts"] == 100.0
    assert snap["last_activity_desc"] == "starting API call #1"
    assert snap["description"] == "starting API call #1"
    assert snap["api_call_count"] == 1
    assert "phase" not in snap
    assert "last_progress_at" not in snap




