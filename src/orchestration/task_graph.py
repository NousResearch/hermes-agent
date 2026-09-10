"""DAG task graph with async parallel execution."""

from __future__ import annotations

import asyncio
from collections import defaultdict, deque
from typing import Awaitable, Callable, Dict, Iterable, List, Mapping, Sequence, Tuple

from orchestration.types import GraphTaskRun, GraphTaskSpec, TaskStatus


def _validate_specs(specs: Sequence[GraphTaskSpec]) -> Dict[str, GraphTaskSpec]:
    by_id: Dict[str, GraphTaskSpec] = {}
    for spec in specs:
        if spec.task_id in by_id:
            raise ValueError(f"duplicate task_id {spec.task_id!r}")
        by_id[spec.task_id] = spec
    for spec in specs:
        for dep in spec.depends_on:
            if dep not in by_id:
                raise ValueError(f"unknown dependency {dep!r} for task {spec.task_id!r}")
        if spec.task_id in spec.depends_on:
            raise ValueError(f"self-cycle on task {spec.task_id!r}")
    # Cycle detection (DFS)
    WHITE, GREY, BLACK = 0, 1, 2
    color: Dict[str, int] = {k: WHITE for k in by_id}

    def visit(node: str) -> None:
        color[node] = GREY
        for dep in by_id[node].depends_on:
            if color[dep] == GREY:
                raise ValueError(f"cycle detected involving {node!r} -> {dep!r}")
            if color[dep] == WHITE:
                visit(dep)
        color[node] = BLACK

    for tid in by_id:
        if color[tid] == WHITE:
            visit(tid)
    return by_id


class TaskGraph:
    """Directed acyclic batch executor."""

    def __init__(self, specs: Sequence[GraphTaskSpec]) -> None:
        self._spec_by_id = _validate_specs(tuple(specs))
        self._dependents: Dict[str, List[str]] = defaultdict(list)
        self._pending_deps: Dict[str, int] = {}
        for tid, spec in self._spec_by_id.items():
            self._pending_deps[tid] = len(spec.depends_on)
            for dep in spec.depends_on:
                self._dependents[dep].append(tid)

    @property
    def task_ids(self) -> Tuple[str, ...]:
        return tuple(self._spec_by_id.keys())

    def runs(self) -> Dict[str, GraphTaskRun]:
        return {tid: GraphTaskRun(spec=self._spec_by_id[tid]) for tid in self._spec_by_id}

    async def run(
        self,
        execute: Callable[[GraphTaskRun], Awaitable[None]],
        *,
        runs: Mapping[str, GraphTaskRun] | None = None,
    ) -> Dict[str, GraphTaskRun]:
        """Execute nodes in parallel waves respecting dependencies."""

        local: Dict[str, GraphTaskRun] = dict(runs) if runs else self.runs()
        pending = dict(self._pending_deps)
        queue: deque[str] = deque(tid for tid, c in pending.items() if c == 0)

        while queue:
            wave = list(queue)
            queue.clear()
            await asyncio.gather(*(execute(local[w]) for w in wave))
            for tid in wave:
                if local[tid].status == TaskStatus.FAILED:
                    continue
                for dst in self._dependents[tid]:
                    pending[dst] -= 1
                    if pending[dst] == 0:
                        queue.append(dst)

        unfinished = [tid for tid, r in local.items() if r.status == TaskStatus.PENDING]
        if unfinished:
            for tid in unfinished:
                local[tid].status = TaskStatus.SKIPPED
                local[tid].error = "skipped due to upstream failure or deadlock"
        return local


def topo_sort(specs: Iterable[GraphTaskSpec]) -> List[str]:
    """Return topological order (stable tie-break by task_id)."""

    specs_t = tuple(specs)
    _validate_specs(specs_t)
    by_id = {s.task_id: s for s in specs_t}
    indegree: Dict[str, int] = {t: len(by_id[t].depends_on) for t in by_id}
    ready = sorted(t for t, d in indegree.items() if d == 0)
    order: List[str] = []
    while ready:
        tid = ready.pop(0)
        order.append(tid)
        for oid, spec in sorted(by_id.items()):
            if tid in spec.depends_on:
                indegree[oid] -= 1
                if indegree[oid] == 0:
                    ready.append(oid)
                    ready.sort()
    if len(order) != len(by_id):
        raise ValueError("task graph has a cycle or missing nodes")
    return order
