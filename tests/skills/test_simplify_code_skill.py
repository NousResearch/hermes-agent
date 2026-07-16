"""Tests for skills/software-development/simplify-code/SKILL.md — Phase 2.5 filter & judge.

simplify-code ships as SKILL.md only (no importable module yet), so this file
pins the acceptance-filter contract described in SKILL.md lines 147-195:
drop non-completed / empty-summary results, treat tool_trace errors as
advisory, and build the survivor list preserving each result's original
task_index. The filter logic is replicated inline from the spec so the test
runs without a separate Python module.
"""

import pytest

from pathlib import Path

SKILL_PATH = (
    Path(__file__).resolve().parents[2]
    / "skills"
    / "software-development"
    / "simplify-code"
    / "SKILL.md"
)


def _filter_survivors(results):
    """Replicate SKILL.md Phase 2.5 acceptance filter (147-162).

    Returns (survivors, discarded_count, survivor_count) where survivors keep
    their original task_index.
    """
    survivors = []
    discarded = 0
    for r in results:
        if r.get("status") != "completed":
            discarded += 1
            continue
        summary = (r.get("summary") or "").strip()
        if not summary:
            discarded += 1
            continue
        # tool_trace errors are advisory only (SKILL.md 156-158): a recovered
        # failure does not invalidate a result with a non-empty, useful summary.
        survivors.append(r)
    return survivors, discarded, len(survivors)


def _make_result(task_index, *, status="completed", summary="done", tool_trace=None):
    r = {"task_index": task_index, "status": status, "summary": summary}
    if tool_trace is not None:
        r["tool_trace"] = tool_trace
    return r


class TestAcceptanceFilter:
    def test_drops_empty_summaries(self):
        results = [
            _make_result(0, summary=""),
            _make_result(1, summary="real review here"),
        ]
        survivors, discarded, survivor_count = _filter_survivors(results)
        assert survivor_count == 1
        assert discarded == 1
        assert survivors[0]["task_index"] == 1

    def test_tool_trace_error_keeps_nonempty_summary(self):
        results = [
            _make_result(
                0,
                summary="valid simplification found",
                tool_trace=[{"status": "error", "tool": "grep"}],
            ),
        ]
        survivors, discarded, survivor_count = _filter_survivors(results)
        # Advisory: error in tool_trace must NOT drop a useful result.
        assert survivor_count == 1
        assert discarded == 0
        assert survivors[0]["task_index"] == 0

    def test_survivor_list_preserves_original_task_index(self):
        # Middle reviewer (task_index 1) returns an empty summary and is dropped;
        # survivors must keep their original indices [0, 2], not be renumbered.
        results = [
            _make_result(0, summary="review A"),
            _make_result(1, summary="   "),
            _make_result(2, summary="review C"),
        ]
        survivors, discarded, survivor_count = _filter_survivors(results)
        assert survivor_count == 2
        assert discarded == 1
        assert [s["task_index"] for s in survivors] == [0, 2]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
