"""DAG task graph with async parallel execution and pre-flight validation."""

from __future__ import annotations

import asyncio
from collections import defaultdict, deque
from typing import Awaitable, Callable, Dict, Iterable, List, Mapping, Sequence, Tuple

from orchestration.types import (
    Finding,
    GraphTaskRun,
    GraphTaskSpec,
    TaskStatus,
    ValidationLevel,
    ValidationReport,
)

# ── Configurable thresholds ──────────────────────────────────────────────
_DEFAULT_MAX_DEPTH_WARN = 4   # depth > this → WARNING
_DEFAULT_MAX_DEPTH_HARD = 8   # depth > this → REJECT
_BLAST_RADIUS_WARN = 3        # affected ≥ this → elevated risk
_BLAST_RADIUS_CRITICAL = 5    # affected ≥ this → critical risk


# ── Internal validation (unchanged contract) ─────────────────────────────

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


# ── Public validation API ────────────────────────────────────────────────

def validate_dag(
    specs: Sequence[GraphTaskSpec],
    *,
    max_depth_warn: int = _DEFAULT_MAX_DEPTH_WARN,
    max_depth_hard: int = _DEFAULT_MAX_DEPTH_HARD,
    blast_radius_warn: int = _BLAST_RADIUS_WARN,
    blast_radius_critical: int = _BLAST_RADIUS_CRITICAL,
) -> ValidationReport:
    """Run full pre-execution DAG validation.

    Returns a structured ``ValidationReport`` with per-step findings,
    risk level, and a final ``passed`` verdict.  Does NOT raise — all
    findings are collected and returned in the report.

    Checks performed (in order):
      1. Cycle detection (DFS) — REJECT
      2. Orphan detection — WARNING for nodes with zero in-degree + out-degree
      3. Depth limit — WARNING if max depth > ``max_depth_warn``,
         REJECT if > ``max_depth_hard``
      4. Blast radius — BFS from every node to compute worst-case
         impact; risk level is elevated/critical based on thresholds
    """

    if not specs:
        return ValidationReport(passed=True, risk_level="routine")

    findings: list[Finding] = []
    risk_level = "routine"

    # ── 1. Cycle check ───────────────────────────────────────────────
    try:
        by_id = _validate_specs(tuple(specs))
    except ValueError as exc:
        findings.append(Finding(ValidationLevel.REJECT, str(exc)))
        return ValidationReport(
            passed=False,
            findings=findings,
            risk_level="critical",
        )

    # ── 2. Orphan detection ──────────────────────────────────────────
    in_degree: Dict[str, int] = defaultdict(int)
    out_degree: Dict[str, int] = defaultdict(int)
    for spec in specs:
        out_degree[spec.task_id] = len(spec.depends_on)
        for dep in spec.depends_on:
            in_degree[dep] += 1
            _ = in_degree[spec.task_id]  # ensure key exists

    orphans = [
        tid for tid in by_id
        if in_degree.get(tid, 0) == 0 and out_degree.get(tid, 0) == 0
    ]
    if orphans:
        findings.append(Finding(
            ValidationLevel.WARNING,
            f"Orphan node(s) detected: {', '.join(sorted(orphans))}. "
            "These have no dependencies and nothing depends on them.",
            nodes=sorted(orphans),
        ))

    # ── 3. Depth limit ───────────────────────────────────────────────
    # Compute depth = longest path from any root (no predecessors) to node.
    roots = [tid for tid, spec in by_id.items() if len(spec.depends_on) == 0]
    # Build forward adjacency: who directly depends on me?
    children: Dict[str, list[str]] = defaultdict(list)
    for spec in specs:
        for dep in spec.depends_on:
            children[dep].append(spec.task_id)

    depth: Dict[str, int] = {}
    q: deque[tuple[str, int]] = deque((r, 0) for r in roots)
    while q:
        node, d = q.popleft()
        if d > depth.get(node, -1):
            depth[node] = d
        for child in children.get(node, []):
            q.append((child, d + 1))

    max_depth = max(depth.values()) if depth else 0
    deep_nodes = [tid for tid, d in depth.items() if d > max_depth_warn]

    if max_depth > max_depth_hard:
        findings.append(Finding(
            ValidationLevel.REJECT,
            f"DAG max depth {max_depth} exceeds hard limit {max_depth_hard}. "
            f"Deepest nodes: {', '.join(sorted(deep_nodes, key=lambda n: -depth[n])[:5])}.",
            nodes=sorted(deep_nodes),
        ))
        risk_level = "critical"
    elif max_depth > max_depth_warn:
        findings.append(Finding(
            ValidationLevel.WARNING,
            f"DAG max depth {max_depth} exceeds recommended limit {max_depth_warn}. "
            f"Deepest nodes: {', '.join(sorted(deep_nodes, key=lambda n: -depth[n])[:5])}.",
            nodes=sorted(deep_nodes),
        ))
        if risk_level == "routine":
            risk_level = "elevated"

    # ── 4. Blast radius ──────────────────────────────────────────────
    # Reuse children map from depth computation above.

    max_affected = 0
    worst_node = ""
    for tid in by_id:
        visited: set[str] = {tid}
        bfs_q: deque[str] = deque([tid])
        while bfs_q:
            cur = bfs_q.popleft()
            for child in children.get(cur, []):
                if child not in visited:
                    visited.add(child)
                    bfs_q.append(child)
        affected = len(visited) - 1  # exclude self
        if affected > max_affected:
            max_affected = affected
            worst_node = tid

    if max_affected >= blast_radius_critical:
        findings.append(Finding(
            ValidationLevel.WARNING,
            f"Critical blast radius: failure of '{worst_node}' would affect "
            f"{max_affected} downstream node(s).",
            nodes=[worst_node],
        ))
        if risk_level == "routine":
            risk_level = "critical"
    elif max_affected >= blast_radius_warn:
        findings.append(Finding(
            ValidationLevel.WARNING,
            f"Elevated blast radius: failure of '{worst_node}' would affect "
            f"{max_affected} downstream node(s).",
            nodes=[worst_node],
        ))
        if risk_level == "routine":
            risk_level = "elevated"

    # ── Final verdict ─────────────────────────────────────────────────
    has_reject = any(f.level == ValidationLevel.REJECT for f in findings)
    return ValidationReport(
        passed=not has_reject,
        findings=findings,
        risk_level=risk_level,
    )


# ── TaskGraph (runtime executor) ─────────────────────────────────────────

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
