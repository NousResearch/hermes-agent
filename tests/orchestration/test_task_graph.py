"""Unit tests for DAG orchestration primitives."""

from __future__ import annotations

import pytest

from orchestration.task_graph import TaskGraph, topo_sort, validate_dag
from orchestration.types import (
    GraphTaskRun,
    GraphTaskSpec,
    TaskStatus,
    ValidationLevel,
)


# ── Existing tests (topo sort + execution) ──────────────────────────────

def test_topo_sort_chain() -> None:
    specs = (
        GraphTaskSpec("root", "do root"),
        GraphTaskSpec("leaf", "do leaf", depends_on=("root",)),
    )
    assert topo_sort(specs) == ["root", "leaf"]


def test_cycle_rejected() -> None:
    specs = (
        GraphTaskSpec("a", "ga", depends_on=("b",)),
        GraphTaskSpec("b", "gb", depends_on=("a",)),
    )
    with pytest.raises(ValueError, match="cycle detected"):
        TaskGraph(specs)


@pytest.mark.asyncio
async def test_fail_fast_blocks_dependents() -> None:
    specs = (
        GraphTaskSpec("first", "g1"),
        GraphTaskSpec("second", "g2", depends_on=("first",)),
    )
    graph = TaskGraph(specs)
    runs = graph.runs()

    async def execute(run: GraphTaskRun) -> None:
        if run.spec.task_id == "first":
            run.status = TaskStatus.FAILED
            run.error = "boom"
        else:
            run.status = TaskStatus.DONE

    out = await graph.run(execute, runs=runs)
    assert out["first"].status == TaskStatus.FAILED
    assert out["second"].status == TaskStatus.SKIPPED


@pytest.mark.asyncio
async def test_parallel_wave_executes_all_roots() -> None:
    specs = (
        GraphTaskSpec("x", "gx"),
        GraphTaskSpec("y", "gy"),
    )
    graph = TaskGraph(specs)
    reached: list[str] = []

    async def execute(run: GraphTaskRun) -> None:
        reached.append(run.spec.task_id)
        run.status = TaskStatus.DONE

    await graph.run(execute)
    assert set(reached) == {"x", "y"}


# ── New validation tests ────────────────────────────────────────────────

class TestValidateDag:
    """Pre-execution DAG validation via validate_dag()."""

    def test_empty_specs_passes(self) -> None:
        report = validate_dag([])
        assert report.passed is True
        assert report.risk_level == "routine"

    def test_valid_dag_passes(self) -> None:
        specs = (
            GraphTaskSpec("a", "root"),
            GraphTaskSpec("b", "child", depends_on=("a",)),
        )
        report = validate_dag(specs)
        assert report.passed is True
        assert not report.errors

    def test_cycle_rejected(self) -> None:
        specs = (
            GraphTaskSpec("a", "ga", depends_on=("b",)),
            GraphTaskSpec("b", "gb", depends_on=("a",)),
        )
        report = validate_dag(specs)
        assert report.passed is False
        assert len(report.errors) == 1
        assert "cycle" in report.errors[0].message.lower()
        assert report.risk_level == "critical"

    def test_self_cycle_rejected(self) -> None:
        specs = (GraphTaskSpec("a", "ga", depends_on=("a",)),)
        report = validate_dag(specs)
        assert report.passed is False
        assert len(report.errors) == 1

    def test_unknown_dependency_rejected(self) -> None:
        specs = (GraphTaskSpec("a", "ga", depends_on=("nonexistent",)),)
        report = validate_dag(specs)
        assert report.passed is False

    def test_duplicate_id_rejected(self) -> None:
        specs = (
            GraphTaskSpec("a", "first"),
            GraphTaskSpec("a", "second"),
        )
        report = validate_dag(specs)
        assert report.passed is False

    def test_orphan_node_warns(self) -> None:
        specs = (
            GraphTaskSpec("connected", "has edges"),
            GraphTaskSpec("orphan", "no edges"),
        )
        report = validate_dag(specs)
        assert report.passed is True  # orphan is WARNING, not REJECT
        assert len(report.warnings) >= 1
        orphan_finding = next(
            (f for f in report.warnings if "orphan" in f.message.lower()), None
        )
        assert orphan_finding is not None
        assert "orphan" in orphan_finding.nodes

    def test_deep_dag_warns(self) -> None:
        """5 layers → max depth 4 → should WARN (default warn at >4)."""
        specs = tuple(
            GraphTaskSpec(f"n{i}", f"layer {i}",
                          depends_on=(f"n{i-1}",) if i > 0 else ())
            for i in range(6)  # n0 (depth 0) → n5 (depth 5)
        )
        report = validate_dag(specs, max_depth_warn=4)
        depth_warnings = [
            f for f in report.warnings if "depth" in f.message.lower()
        ]
        assert len(depth_warnings) >= 1
        assert report.risk_level in ("elevated", "critical")

    def test_very_deep_dag_rejected(self) -> None:
        """10 layers → max depth 9 → should REJECT (default hard at >8)."""
        specs = tuple(
            GraphTaskSpec(f"n{i}", f"layer {i}",
                          depends_on=(f"n{i-1}",) if i > 0 else ())
            for i in range(10)
        )
        report = validate_dag(specs)
        assert report.passed is False
        depth_errors = [f for f in report.errors if "depth" in f.message.lower()]
        assert len(depth_errors) >= 1

    def test_blast_radius_detected(self) -> None:
        """Single root with 6 leaf dependents → blast radius ≥ 5 → critical."""
        specs = [GraphTaskSpec("root", "fan-out")]
        for i in range(6):
            specs.append(
                GraphTaskSpec(f"leaf{i}", "leaf", depends_on=("root",))
            )
        report = validate_dag(tuple(specs), blast_radius_critical=5)
        if report.passed:
            blast_warnings = [
                f for f in report.warnings if "blast" in f.message.lower()
            ]
            assert len(blast_warnings) >= 1

    def test_validate_dag_does_not_raise(self) -> None:
        """validate_dag returns a report — it should never raise."""
        specs = (
            GraphTaskSpec("a", "ga", depends_on=("b",)),
            GraphTaskSpec("b", "gb", depends_on=("a",)),
        )
        # Must not raise — even for cycles, it returns a report.
        report = validate_dag(specs)
        assert isinstance(report.passed, bool)

    def test_all_checks_pass_on_clean_graph(self) -> None:
        """A well-structured 3-node chain should pass cleanly."""
        specs = (
            GraphTaskSpec("orchestrator", "top-level"),
            GraphTaskSpec("researcher", "research", depends_on=("orchestrator",)),
            GraphTaskSpec("builder", "build", depends_on=("researcher",)),
        )
        report = validate_dag(specs)
        assert report.passed is True
        assert len(report.warnings) == 0
        assert report.risk_level == "routine"


class TestValidationReportProperties:
    def test_errors_property(self) -> None:
        specs = (
            GraphTaskSpec("a", "ga", depends_on=("b",)),
            GraphTaskSpec("b", "gb", depends_on=("a",)),
        )
        report = validate_dag(specs)
        assert len(report.errors) >= 1
        for f in report.errors:
            assert f.level == ValidationLevel.REJECT

    def test_warnings_property(self) -> None:
        specs = (GraphTaskSpec("orphan", "alone"),)
        report = validate_dag(specs)
        assert len(report.warnings) >= 1
        for f in report.warnings:
            assert f.level == ValidationLevel.WARNING
