"""M4 T20-T22: Observer V4 — exit total order, exact multiset, overlap."""

import pytest
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from hermes_cli.kanban_orch_observer import (
    EXIT_CLEAN,
    EXIT_HARD_VIOLATION,
    EXIT_PRECEDENCE,
    EXIT_SCHEMA_UNSUPPORTED,
    EXIT_WARNING_ONLY,
    ExpectedNode,
    Finding,
    ObservedRun,
    check_delivery_satisfied,
    check_exact_multiset,
    check_legacy_untyped,
    check_parent_terminal_before_synthesis,
    check_required_lane_overlap,
    check_scope_mismatch,
    compute_exit,
    run_observer,
)


def test_exit_total_order():
    """T22: exit precedence plus busy/schema/corruption/internal catch-all.

    EXIT_PRECEDENCE = (64, 5, 4, 2, 3, 0) — first matching wins, not max.
    """
    # Single findings
    assert compute_exit([Finding(64, "cli_err")]) == 64
    assert compute_exit([Finding(5, "path_err")]) == 5
    assert compute_exit([Finding(4, "schema_err")]) == 4
    assert compute_exit([Finding(2, "hard_err")]) == 2
    assert compute_exit([Finding(3, "warn")]) == 3
    assert compute_exit([Finding(0, "clean")]) == 0

    # Mixed: 2 beats 3 beats 0
    assert compute_exit([Finding(3, "warn"), Finding(2, "hard")]) == 2
    assert compute_exit([Finding(0, "clean"), Finding(3, "warn")]) == 3
    assert compute_exit([Finding(2, "hard"), Finding(3, "warn"), Finding(0, "clean")]) == 2

    # 4 beats 2
    assert compute_exit([Finding(4, "schema"), Finding(2, "hard")]) == 4

    # 5 beats 4
    assert compute_exit([Finding(5, "path"), Finding(4, "schema")]) == 5

    # 64 beats everything
    assert compute_exit([Finding(64, "cli"), Finding(5, "path"), Finding(2, "hard")]) == 64

    # No findings = clean
    assert compute_exit([]) == 0

    # Precedence order is NOT numeric max
    assert EXIT_PRECEDENCE == (64, 5, 4, 2, 3, 0)


def test_exact_multiset_and_overlap():
    """T21: observer exact universe/digests/accepted overlap.

    Missing/extra/duplicate nodes are hard violations.
    Same-lane retry does not count as parallel.
    Overlap requires max(start) < min(end).
    """
    # Exact match — no findings
    expected = [
        ExpectedNode("lane-1", "lane", "A", True, 1),
        ExpectedNode("lane-2", "lane", "B", True, 2),
        ExpectedNode("parent", "parent", "P", False, 0),
    ]
    observed = ["lane-1", "lane-2", "parent"]
    assert check_exact_multiset(expected, observed) is None

    # Missing node
    assert check_exact_multiset(expected, ["lane-1", "parent"]) is not None
    assert check_exact_multiset(expected, ["lane-1", "parent"]).code == "missing_observed_nodes"

    # Extra node
    assert check_exact_multiset(expected, ["lane-1", "lane-2", "parent", "extra"]).code == "extra_observed_nodes"

    # Duplicate
    assert check_exact_multiset(expected, ["lane-1", "lane-1", "lane-2", "parent"]).code == "duplicate_observed_nodes"

    # Overlap: 2 required lanes, one run each, overlapping intervals
    lanes = [
        ExpectedNode("lane-1", "lane", "A", True, 1),
        ExpectedNode("lane-2", "lane", "B", True, 2),
    ]
    runs = [
        ObservedRun("lane-1", "run-1", started_at=100, ended_at=200),
        ObservedRun("lane-2", "run-2", started_at=150, ended_at=250),
    ]
    assert check_required_lane_overlap(lanes, runs) is None  # max(100,150)=150 < min(200,250)=200

    # No overlap: sequential runs
    sequential = [
        ObservedRun("lane-1", "run-1", started_at=100, ended_at=200),
        ObservedRun("lane-2", "run-2", started_at=200, ended_at=300),
    ]
    result = check_required_lane_overlap(lanes, sequential)
    assert result is not None
    assert result.code == "no_required_lane_overlap"

    # Same lane duplicate run
    dup_run = [
        ObservedRun("lane-1", "run-1", started_at=100, ended_at=200),
        ObservedRun("lane-1", "run-2", started_at=150, ended_at=250),
    ]
    result = check_required_lane_overlap(lanes, dup_run)
    assert result is not None
    assert result.code == "duplicate_lane_run"

    # Missing required acceptance
    missing_run = [ObservedRun("lane-1", "run-1", 100, 200)]
    result = check_required_lane_overlap(lanes, missing_run)
    assert result is not None
    assert result.code == "missing_required_acceptance"

    # Fewer than 2 required lanes
    single = [ExpectedNode("lane-1", "lane", "A", True, 1)]
    result = check_required_lane_overlap(single, [ObservedRun("lane-1", "r", 100, 200)])
    assert result is not None
    assert result.code == "insufficient_required_lanes"


def test_snapshot_and_path_truth():
    """T20: observer single snapshot/commit churn/path replacement.

    Scope mismatch, legacy untyped, parent terminal, delivery gaps.
    """
    # Clean: exact match, overlap, scope match, delivery satisfied
    expected = [
        ExpectedNode("lane-1", "lane", "A", True, 1),
        ExpectedNode("lane-2", "lane", "B", True, 2),
    ]
    runs = [
        ObservedRun("lane-1", "run-1", 100, 200),
        ObservedRun("lane-2", "run-2", 150, 250),
    ]
    exit_code = run_observer(
        has_v4_schema=True,
        expected_nodes=expected,
        observed_node_keys=["lane-1", "lane-2"],
        accepted_runs=runs,
        parent_state="completed",
        has_result=True,
        origin_kind="messaging",
        required_acks=1,
        acked_count=1,
    )
    assert exit_code == EXIT_CLEAN

    # Scope mismatch
    exit_code = run_observer(
        has_v4_schema=True,
        expected_nodes=expected,
        observed_node_keys=["lane-1", "lane-2"],
        accepted_runs=runs,
        expected_board="board-A",
        observed_board="board-B",
    )
    assert exit_code == EXIT_HARD_VIOLATION

    # Legacy untyped: schema missing, legacy title present → exit 3
    exit_code = run_observer(
        has_v4_schema=False,
        expected_nodes=[],
        observed_node_keys=[],
        accepted_runs=[],
        has_legacy_title=True,
    )
    assert exit_code == EXIT_WARNING_ONLY

    # Schema missing, no legacy → exit 4
    exit_code = run_observer(
        has_v4_schema=False,
        expected_nodes=[],
        observed_node_keys=[],
        accepted_runs=[],
        has_legacy_title=False,
    )
    assert exit_code == EXIT_SCHEMA_UNSUPPORTED

    # Parent terminal before synthesis
    exit_code = run_observer(
        has_v4_schema=True,
        expected_nodes=expected,
        observed_node_keys=["lane-1", "lane-2"],
        accepted_runs=runs,
        parent_state="completed",
        has_result=False,  # no result → terminal before synthesis
    )
    assert exit_code == EXIT_HARD_VIOLATION

    # Delivery not satisfied
    exit_code = run_observer(
        has_v4_schema=True,
        expected_nodes=expected,
        observed_node_keys=["lane-1", "lane-2"],
        accepted_runs=runs,
        parent_state="completed",
        has_result=True,
        origin_kind="messaging",
        required_acks=2,
        acked_count=1,  # missing one ACK
    )
    assert exit_code == EXIT_HARD_VIOLATION

    # Board-only with zero required → clean
    exit_code = run_observer(
        has_v4_schema=True,
        expected_nodes=[ExpectedNode("parent", "parent", "P", False, 0)],
        observed_node_keys=["parent"],
        accepted_runs=[],
        parent_state="completed",
        has_result=True,
        origin_kind="board_only",
        required_acks=0,
        acked_count=0,
    )
    assert exit_code == EXIT_CLEAN

    # Board-only with required obligations → hard violation
    exit_code = run_observer(
        has_v4_schema=True,
        expected_nodes=[ExpectedNode("parent", "parent", "P", False, 0)],
        observed_node_keys=["parent"],
        accepted_runs=[],
        parent_state="completed",
        has_result=True,
        origin_kind="board_only",
        required_acks=1,  # board_only should have 0
    )
    assert exit_code == EXIT_HARD_VIOLATION

    # Mixed: hard violation + warning → 2 beats 3
    exit_code = run_observer(
        has_v4_schema=True,
        expected_nodes=expected,
        observed_node_keys=["lane-1", "lane-2"],
        accepted_runs=runs,
        parent_state="completed",
        has_result=True,
        origin_kind="messaging",
        required_acks=2,
        acked_count=0,  # delivery not satisfied (exit 2)
    )
    assert exit_code == EXIT_HARD_VIOLATION  # 2 beats any warning
