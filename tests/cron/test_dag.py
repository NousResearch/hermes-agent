"""Tests for cron/dag.py - cron job dependency resolution."""

import pytest

from cron.dag import (
    DagCycleError,
    explain_blocked,
    resolve_ready,
    SUCCESS,
    topological_order,
    validate_no_cycles,
)


# --- Job helpers --------------------------------------------------------


def _j(jid, depends_on=None):
    return {"job_id": jid, "depends_on": depends_on or []}


def _empty_status():
    return {}


# --- resolve_ready ------------------------------------------------------


def test_jobs_without_deps_are_always_ready():
    jobs = [_j("a"), _j("b"), _j("c")]
    assert resolve_ready(jobs, _empty_status()) == ["a", "b", "c"]


def test_dep_unlocks_after_success():
    jobs = [_j("a"), _j("b", ["a"])]
    # No status yet -> b blocked
    assert resolve_ready(jobs, _empty_status()) == ["a"]
    # a succeeds -> b unblocks
    assert resolve_ready(jobs, {"a": SUCCESS}) == ["a", "b"]


def test_failed_dep_keeps_job_blocked():
    jobs = [_j("a"), _j("b", ["a"])]
    assert resolve_ready(jobs, {"a": "failure"}) == ["a"]


def test_missing_dep_blocks():
    jobs = [_j("b", ["ghost"])]
    assert resolve_ready(jobs, _empty_status()) == []


def test_multi_dep_requires_all_to_succeed():
    jobs = [_j("a"), _j("b"), _j("c", ["a", "b"])]
    # Only a succeeded -> c still blocked
    assert resolve_ready(jobs, {"a": SUCCESS}) == ["a", "b"]
    # Both succeeded -> c unblocks
    assert resolve_ready(jobs, {"a": SUCCESS, "b": SUCCESS}) == ["a", "b", "c"]


def test_results_are_sorted_for_determinism():
    jobs = [_j("z"), _j("y"), _j("a")]
    assert resolve_ready(jobs, _empty_status()) == ["a", "y", "z"]


def test_jobs_with_blank_id_are_skipped():
    assert resolve_ready([{"depends_on": []}, _j("ok")], {}) == ["ok"]


def test_depends_on_accepts_strings_only():
    # Non-string entries are coerced to strings (defensive); blank/whitespace
    # entries are dropped. So [None, "", 42, "b"] becomes ["42", "b"].
    jobs = [_j("a", [None, "", 42, "b"]), _j("b")]
    assert resolve_ready(jobs, {"42": SUCCESS, "b": SUCCESS}) == ["a", "b"]
    # Without the deps succeeding, 'a' is blocked.
    assert resolve_ready(jobs, {}) == ["b"]


def test_id_field_falls_back_when_job_id_missing():
    jobs = [{"id": "alpha"}, {"job_id": "beta"}]
    assert resolve_ready(jobs, {}) == ["alpha", "beta"]


def test_dep_on_unknown_field_is_no_op():
    """Sanity: a job dict without depends_on is treated as no deps."""
    jobs = [{"job_id": "naked"}]
    assert resolve_ready(jobs, {}) == ["naked"]


# --- explain_blocked ----------------------------------------------------


def test_explain_blocked_lists_unmet_deps():
    jobs = [_j("a"), _j("b", ["a", "ghost"]), _j("c", ["a"])]
    blocked = explain_blocked(jobs, {"a": "failure"})
    assert len(blocked) == 2
    by_id = {b["job_id"]: b for b in blocked}
    assert by_id["b"]["blocked_by"] == [
        {"job_id": "a", "status": "failure"},
        {"job_id": "ghost", "status": "missing"},
    ]
    assert by_id["c"]["blocked_by"] == [{"job_id": "a", "status": "failure"}]


def test_explain_blocked_empty_when_all_unlocked():
    jobs = [_j("a"), _j("b", ["a"])]
    assert explain_blocked(jobs, {"a": SUCCESS}) == []


def test_explain_blocked_dedupes_dep_list():
    jobs = [_j("a", ["x", "x", "x"])]
    assert explain_blocked(jobs, {}) == [
        {"job_id": "a", "blocked_by": [{"job_id": "x", "status": "missing"}]}
    ]


# --- Cycle detection ----------------------------------------------------


def test_validate_no_cycles_passes_on_dag():
    jobs = [_j("a"), _j("b", ["a"]), _j("c", ["a", "b"])]
    validate_no_cycles(jobs)  # no exception


def test_validate_no_cycles_detects_simple_cycle():
    jobs = [_j("a", ["b"]), _j("b", ["a"])]
    with pytest.raises(DagCycleError) as ei:
        validate_no_cycles(jobs)
    assert "a" in ei.value.cycle_path
    assert "b" in ei.value.cycle_path


def test_validate_no_cycles_detects_self_loop():
    jobs = [_j("a", ["a"])]
    with pytest.raises(DagCycleError):
        validate_no_cycles(jobs)


def test_validate_no_cycles_detects_three_node_cycle():
    jobs = [_j("a", ["b"]), _j("b", ["c"]), _j("c", ["a"])]
    with pytest.raises(DagCycleError) as ei:
        validate_no_cycles(jobs)
    # Path should contain all three
    p = set(ei.value.cycle_path)
    assert {"a", "b", "c"}.issubset(p)


def test_validate_no_cycles_passes_with_unrelated_components():
    jobs = [_j("a"), _j("b"), _j("c", ["b"])]
    validate_no_cycles(jobs)


def test_validate_no_cycles_ignores_blank_job_ids():
    jobs = [{"depends_on": []}, {"job_id": "a", "depends_on": []}]
    validate_no_cycles(jobs)  # no exception


# --- topological_order --------------------------------------------------


def test_topological_orders_deps_first():
    jobs = [_j("c", ["a", "b"]), _j("a"), _j("b", ["a"])]
    order = topological_order(jobs)
    assert order.index("a") < order.index("b")
    assert order.index("b") < order.index("c")


def test_topological_handles_disconnected_graph():
    jobs = [_j("z"), _j("a", ["b"]), _j("b")]
    order = topological_order(jobs)
    assert set(order) == {"a", "b", "z"}
    assert order.index("b") < order.index("a")


def test_topological_raises_on_cycle():
    jobs = [_j("a", ["b"]), _j("b", ["a"])]
    with pytest.raises(DagCycleError):
        topological_order(jobs)


def test_topological_returns_only_dag_nodes():
    """Dangling deps (a depends on a missing job) still surface in the order."""
    jobs = [_j("a", ["ghost"]), _j("b")]
    order = topological_order(jobs)
    assert "ghost" in order  # topological includes dep targets
    assert "a" in order and "b" in order
