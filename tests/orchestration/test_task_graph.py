"""Unit tests for DAG orchestration primitives."""

from __future__ import annotations

import pytest

from orchestration.task_graph import TaskGraph, topo_sort
from orchestration.types import GraphTaskRun, GraphTaskSpec, TaskStatus


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
