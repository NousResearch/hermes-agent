from __future__ import annotations

from agent import continuous_completion as cc


def _node(capability_id: str, **overrides):
    node = {
        "CAPABILITY_ID": capability_id,
        "STATE": "READY",
        "DEPS": [],
        "RISK": 1,
        "WEIGHT": 1,
        "FIRST_UNPROVEN_BOUNDARY": "IMPLEMENT",
        "BASE_HEAD": "head-a",
        "BASE_TREE": "tree-a",
        "CHANGED_PATHS": ["agent/demo.py"],
        "SEMANTIC_OWNER": capability_id,
        "RUNTIME_IMPACT": "low",
        "RELEASE_IMPACT": "low",
        "BLOCK_LIVE": False,
        "MERGE_PRIORITY": 1,
    }
    node.update(overrides)
    return node


def test_dag_validation_rejects_missing_dependency_and_cycles():
    report = cc.validate_inventory(
        [
            _node("a", DEPS=["missing"]),
            _node("b", STATE="RUNNING", DEPS=["c"]),
            _node("c", STATE="BLOCKED", DEPS=["b"]),
        ]
    )

    assert report["DAG_VALIDATION"] is False
    assert any("unknown DEPS" in error for error in report["ERRORS"])
    assert any("dependency cycle detected" in error for error in report["ERRORS"])


def test_ready_frontier_respects_done_dependencies_and_live_priority():
    nodes = [
        _node("prep", STATE="DONE"),
        _node("non_live", DEPS=["prep"], WEIGHT=20, MERGE_PRIORITY=1),
        _node("live", DEPS=["prep"], BLOCK_LIVE=True, WEIGHT=2, MERGE_PRIORITY=5),
        _node("blocked", DEPS=["prep"], STATE="BLOCKED"),
    ]

    assert cc.ready_frontier(nodes) == ["live", "non_live"]


def test_critical_path_prioritizes_live_chain_over_non_blocking_debt():
    nodes = [
        _node("live-root", BLOCK_LIVE=True, MERGE_PRIORITY=10, WEIGHT=3),
        _node("live-child", BLOCK_LIVE=True, MERGE_PRIORITY=8, WEIGHT=2, DEPS=["live-root"]),
        _node("debt-root", MERGE_PRIORITY=50, WEIGHT=20),
        _node("debt-child", MERGE_PRIORITY=40, WEIGHT=15, DEPS=["debt-root"]),
    ]

    assert cc.critical_path(nodes) == ["live-root", "live-child"]


def test_parity_classification_distinguishes_exact_head_moved_and_stale():
    assert cc.classify_parity("head", "tree", "head", "tree") == cc.PARITY_EXACT
    assert (
        cc.classify_parity("head-a", "tree", "head-b", "tree")
        == cc.PARITY_HEAD_MOVED_TREE_STABLE
    )
    assert (
        cc.classify_parity("head-a", "tree-a", "head-b", "tree-b")
        == cc.PARITY_STALE_BASELINE
    )


def test_parallel_safe_frontier_enforces_one_writer_per_semantic_owner():
    nodes = [
        _node("running-owner-a", STATE="RUNNING", SEMANTIC_OWNER="owner-a"),
        _node("ready-owner-a", SEMANTIC_OWNER="owner-a"),
        _node("ready-owner-b-1", SEMANTIC_OWNER="owner-b", MERGE_PRIORITY=5),
        _node("ready-owner-b-2", SEMANTIC_OWNER="owner-b", MERGE_PRIORITY=1),
        _node("ready-owner-c", SEMANTIC_OWNER="owner-c", MERGE_PRIORITY=2),
    ]

    assert cc.parallel_safe_frontier(nodes) == ["ready-owner-b-1", "ready-owner-c"]


def test_durable_pass_gate_reports_failed_checks():
    gate = cc.durable_pass_gate(
        {
            "DAG_VALIDATION": True,
            "READY_FRONTIER_PASS": True,
            "CRITICAL_PATH_PASS": False,
            "PARITY_CLASSIFICATION_PASS": True,
            "ONE_WRITER_PASS": True,
        }
    )

    assert gate["DURABLE_PASS_GATE"] is False
    assert gate["FAILED_CHECKS"] == ["CRITICAL_PATH_PASS"]
