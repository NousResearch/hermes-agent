"""Cron job dependency DAG: resolve which jobs are ready to fire.

A pure, framework-free module. Given the current set of jobs and the last
known status of each (success/failure/skipped/never-run), compute:

  - which jobs are ready to fire RIGHT NOW (deps satisfied)
  - which jobs are blocked by missing/failed/pending deps
  - whether the dep graph contains a cycle (must be reported, never silently
    skipped)

The module does NOT touch the scheduler tick loop. The scheduler calls
:meth:`resolve_ready` each tick and filters its "due" set to the intersection.
That keeps this change additive: existing schedules with no ``depends_on``
field behave identically.

Dependency model
----------------

Each job MAY carry a ``depends_on`` key (list of job_id strings). A job is
ready iff every dependency's last status is "success". Any failure, missing,
or unknown dependency blocks the job (recorded in the resolver's blocked
list with a reason; never raises).

Cycles are detected up front using Kahn's algorithm so a misconfigured
graph fails loud at the first tick instead of producing a slow trickle of
timeouts.

Job shape
---------

Only a tiny subset of the cron job dict is read:

  * ``job_id`` (or ``id``)
  * ``depends_on`` (optional list of job_id strings)

Everything else (schedule, prompt, skills) is ignored — this module only
resolves deps.

Example
-------

    from cron.dag import resolve_ready, validate_no_cycles

    jobs = load_jobs()
    validate_no_cycles(jobs)              # raise on misconfiguration
    last = last_status_by_id(jobs)        # your own state lookup
    ready_ids = resolve_ready(jobs, last) # filter "due" by these
"""

from __future__ import annotations

import logging
from collections import defaultdict, deque
from typing import Any, Dict, Iterable, List, Set, Tuple

logger = logging.getLogger(__name__)


# Status values used by this module. The cron subsystem records these as
# ``effective_job_state``; we accept any of them but only "success" unlocks.
SUCCESS = "success"
FAILURE = "failure"
PENDING = "pending"
SKIPPED = "skipped"
NEVER_RUN = "never_run"

_UNLOCKING_STATUSES = frozenset({SUCCESS})


class DagCycleError(ValueError):
    """Raised when the dependency graph contains a cycle.

    Includes the cycle path so callers can point the operator at the
    misconfigured job(s).
    """

    def __init__(self, cycle_path: List[str]):
        self.cycle_path = list(cycle_path)
        super().__init__(
            "cron dependency cycle detected: " + " -> ".join(cycle_path)
        )


def _job_id(job: Dict[str, Any]) -> str:
    """Pull the canonical job_id out of a job dict (job_id OR id)."""
    jid = job.get("job_id")
    if not jid:
        jid = job.get("id")
    return str(jid or "").strip()


def _job_deps(job: Dict[str, Any]) -> List[str]:
    """Normalise the depends_on field to a list of strings (deduped, ordered)."""
    raw = job.get("depends_on") or []
    if not isinstance(raw, (list, tuple)):
        return []
    seen: Set[str] = set()
    out: List[str] = []
    for item in raw:
        s = str(item or "").strip()
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def _build_graph(
    jobs: Iterable[Dict[str, Any]],
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]], Set[str]]:
    """Build adjacency lists.

    Returns:
        forward: job_id -> list of job_ids that depend on it (children)
        reverse: job_id -> list of job_ids it depends on (parents)
        all_nodes: every node in the graph (jobs + their deps)
    """
    forward: Dict[str, List[str]] = defaultdict(list)
    reverse: Dict[str, List[str]] = defaultdict(list)
    all_nodes: Set[str] = set()

    for job in jobs:
        jid = _job_id(job)
        if not jid:
            continue
        all_nodes.add(jid)
        deps = _job_deps(job)
        reverse[jid].extend(deps)
        all_nodes.update(deps)
        for dep in deps:
            forward[dep].append(jid)
    return forward, reverse, all_nodes


def validate_no_cycles(jobs: Iterable[Dict[str, Any]]) -> None:
    """Raise :class:`DagCycleError` if the dep graph has a cycle.

    Uses Kahn's algorithm (BFS over in-degree) so a cycle is detected
    without recursion; the cycle path is reconstructed from leftover
    nodes when Kahn finishes with a non-empty remainder.
    """
    forward, reverse, all_nodes = _build_graph(jobs)
    in_degree: Dict[str, int] = {n: len(reverse.get(n, [])) for n in all_nodes}
    queue: deque = deque(sorted(n for n, d in in_degree.items() if d == 0))
    removed = 0
    while queue:
        n = queue.popleft()
        removed += 1
        for child in forward.get(n, []):
            in_degree[child] -= 1
            if in_degree[child] == 0:
                queue.append(child)
    leftover = [n for n, d in in_degree.items() if d > 0]
    if not leftover:
        return
    # Reconstruct a concrete cycle path: pick any leftover, walk parents.
    cycle = _extract_cycle(leftover[0], reverse)
    raise DagCycleError(cycle)


def _extract_cycle(start: str, reverse: Dict[str, List[str]]) -> List[str]:
    """Return a cycle path starting and ending at ``start``."""
    # Walk until we revisit a node. Use a deterministic order.
    path: List[str] = [start]
    visited: Set[str] = {start}
    current = start
    while True:
        parents = [p for p in reverse.get(current, []) if p in visited or True]
        # Prefer a parent that has been seen before (closing the cycle).
        seen_parent = next((p for p in parents if p in visited), None)
        next_node = seen_parent or (parents[0] if parents else start)
        if next_node in visited:
            path.append(next_node)
            # Trim the prefix that does not belong to the cycle.
            idx = path.index(next_node)
            return path[idx:]
        path.append(next_node)
        visited.add(next_node)
        current = next_node


def resolve_ready(
    jobs: Iterable[Dict[str, Any]],
    last_status_by_id: Dict[str, str],
) -> List[str]:
    """Return job_ids whose dependencies are all satisfied.

    A job is ready iff:

      * it has no ``depends_on`` field, OR
      * every dep's value in ``last_status_by_id`` is in
        :data:`_UNLOCKING_STATUSES` (``success``).

    Jobs with unknown / failed / pending deps are NOT returned. They will
    be re-checked on the next tick after their deps succeed.

    Cycles are NOT detected here — call :func:`validate_no_cycles` once
    per scheduler reload to surface misconfigurations. A cyclic dep chain
    will simply never become ready (all of its members blocked), so the
    scheduler won't crash, but the cycle won't surface via this function.

    Order: deterministic by job_id, ascending. This makes the function
    idempotent and easy to test.
    """
    forward, reverse, all_nodes = _build_graph(jobs)
    ready: List[str] = []
    for job in jobs:
        jid = _job_id(job)
        if not jid:
            continue
        deps = _job_deps(job)
        if not deps:
            ready.append(jid)
            continue
        if all(last_status_by_id.get(d) in _UNLOCKING_STATUSES for d in deps):
            ready.append(jid)
    return sorted(ready)


def explain_blocked(
    jobs: Iterable[Dict[str, Any]],
    last_status_by_id: Dict[str, str],
) -> List[Dict[str, Any]]:
    """Return one record per blocked job describing why it's not ready.

    Useful for surfacing diagnostics to operators without revealing internal
    state. Each record::

        {
          "job_id": "...",
          "blocked_by": [
            {"job_id": "...", "status": "failure" | "pending" | "missing"}
          ]
        }
    """
    out: List[Dict[str, Any]] = []
    for job in jobs:
        jid = _job_id(job)
        if not jid:
            continue
        deps = _job_deps(job)
        if not deps:
            continue
        blocked_by: List[Dict[str, str]] = []
        for d in deps:
            status = last_status_by_id.get(d)
            if status in _UNLOCKING_STATUSES:
                continue
            blocked_by.append({"job_id": d, "status": str(status or "missing")})
        if blocked_by:
            out.append({"job_id": jid, "blocked_by": blocked_by})
    out.sort(key=lambda r: r["job_id"])
    return out


def topological_order(jobs: Iterable[Dict[str, Any]]) -> List[str]:
    """Return job_ids in a valid topological order (deps first).

    Raises :class:`DagCycleError` if the graph is cyclic. Useful for the
    "replay" / one-shot-drain use case where the caller wants to fire
    every job once in dep-respecting order.
    """
    forward, reverse, all_nodes = _build_graph(jobs)
    in_degree: Dict[str, int] = {n: len(reverse.get(n, [])) for n in all_nodes}
    queue: deque = deque(sorted(n for n, d in in_degree.items() if d == 0))
    order: List[str] = []
    while queue:
        n = queue.popleft()
        order.append(n)
        for child in forward.get(n, []):
            in_degree[child] -= 1
            if in_degree[child] == 0:
                queue.append(child)
    if len(order) != len(all_nodes):
        leftover = [n for n, d in in_degree.items() if d > 0]
        cycle = _extract_cycle(leftover[0], reverse)
        raise DagCycleError(cycle)
    return order
