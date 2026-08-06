"""Deterministic failure-diagnosis engine for the loop-diagnostics subsystem.

Implements the diagnosis half of ``docs/loop-diagnostics-design.md`` against
the machine-readable contract in
``hermes_cli/observability/schemas/hermes.loop_diagnostics.v1.schema.json``.

The engine is deliberately **independent from the graph recorder**: it shares
only the schema and the storage layout, never writes a trace, and never reads
a diagnosis. The recorder (``loop_diagnostics_recorder.py``) and this engine
can therefore be developed and shipped independently.

Input
-----
A run's trace file (JSONL) at ``<loop-traces>/<task_id>/<run_id>.jsonl``.
Every line conforms to one of the six record kinds:
``run_header`` / ``action_start`` / ``action_end`` / ``edge`` /
``run_footer`` / ``diagnosis_result``.

Output
------
A ``DiagnosisResult`` (see schema ``#/$defs/DiagnosisResult``) with status one
of ``root_cause_found`` / ``candidates`` / ``unknown`` / ``malformed_trace``.

Guarantees (contract §8, §13)
-----------------------------
* Deterministic: same trace file -> byte-identical result. Iteration order is
  insertion order; ties break lexicographically on ``(action_id, ts)``.
* Safe on cyclic input: traversal keeps a visited set and a depth cap; corrupt
  edges (self-loop, dangling action_id) are ignored with a ``malformed_lines``
  note. Worst case is O(V + E) with V,E capped by the run's event count.
* Graceful degradation: missing/empty trace -> ``unknown``; >=50% invalid
  lines -> ``malformed_trace``; truncated run (no footer / missing action_ends)
  degrades toward ``unknown`` / ``timeout_propagation`` /
  ``cancellation_propagation`` as evidence dictates. Never raises.
* Stdlib-only: this repo pins exact deps and ``jsonschema`` is not a declared
  dependency, so validation is performed with a small hand-rolled validator
  that mirrors the schema surface (union branch, required fields, enums,
  additionalProperties:false, field types).

No LLM is used; the engine is deterministic and bounded.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "hermes.loop_diagnostics.v1"
DIAGNOSIS_VERSION = "hermes.loop_diagnostics.diagnosis.v1"

# Failure categories, in contract §7 priority order (highest first).
CATEGORY_PRIORITY: List[str] = [
    "malformed_trace",
    "retry_exhausted",
    "loop_repeated",
    "input_invalid",
    "timeout_propagation",
    "cancellation_propagation",
    "action_error",
    "unknown",
]

# Edge kinds that carry a failure forward from a producer to a consumer.
PROPAGATION_EDGE_KINDS = ("data", "causal")

# Statuses that make an action a failure.
FAILURE_STATUSES = ("error", "blocked", "timed_out", "cancelled")

# Statuses that indicate an interruption (rather than a plain error).
INTERRUPTION_STATUSES = ("timed_out", "cancelled")

# A retry loop uses this stable loop_id prefix.
RETRY_LOOP_PREFIX = "retry"

# Bounds (contract §8.3).
DEFAULT_MAX_DEPTH = 1000
DEFAULT_MALFORMED_RATIO = 0.5

# Default storage root when no board / env override is given.
DEFAULT_BOARD = "default"


# ---------------------------------------------------------------------------
# Lightweight schema validation (stdlib-only; mirrors the JSON Schema surface)
# ---------------------------------------------------------------------------


_KIND_TO_BRANCH = {
    "run_header": "RunHeader",
    "action_start": "ActionStart",
    "action_end": "ActionEnd",
    "edge": "DependencyEdge",
    "run_footer": "RunFooter",
    "diagnosis_result": "DiagnosisResult",
}

_BRANCH_ENUMS = {
    "ActionStatus": {"ok", "error", "blocked", "cancelled", "unknown"},
    "ActionKind": {
        "tool_call",
        "llm_call",
        "loop_start",
        "loop_iteration",
        "checkpoint",
        "subagent",
        "other",
    },
    "InterventionKind": {
        "retry_from_checkpoint",
        "alternative_tool",
        "trajectory_repair",
        "escalate",
        "none",
    },
    "FailureCategory": {
        "action_error",
        "input_invalid",
        "loop_repeated",
        "retry_exhausted",
        "timeout_propagation",
        "cancellation_propagation",
        "malformed_trace",
        "unknown",
    },
    "DiagnosisStatus": {
        "root_cause_found",
        "candidates",
        "unknown",
        "malformed_trace",
    },
}

# Per-branch required fields (from the schema).
_BRANCH_REQUIRED: Dict[str, List[str]] = {
    "RunHeader": ["schema_version", "kind", "task_id", "run_id", "attempt", "ts"],
    "RunFooter": ["schema_version", "kind", "task_id", "run_id", "ts", "outcome"],
    "ActionStart": [
        "schema_version",
        "kind",
        "task_id",
        "run_id",
        "action_id",
        "ts",
        "action_kind",
    ],
    "ActionEnd": [
        "schema_version",
        "kind",
        "task_id",
        "run_id",
        "action_id",
        "ts",
        "status",
    ],
    "DependencyEdge": [
        "schema_version",
        "kind",
        "task_id",
        "run_id",
        "from_action_id",
        "to_action_id",
        "edge_kind",
    ],
    "DiagnosisResult": [
        "schema_version",
        "kind",
        "diagnosis_version",
        "task_id",
        "run_id",
        "status",
        "category",
        "explanation",
        "interventions",
    ],
}

# Per-branch allowed keys (additionalProperties:false).
_BRANCH_KEYS: Dict[str, set] = {
    "RunHeader": {
        "schema_version", "kind", "task_id", "run_id", "attempt",
        "profile", "goal_mode", "ts",
    },
    "RunFooter": {
        "schema_version", "kind", "task_id", "run_id", "ts",
        "outcome", "error", "event_count",
    },
    "ActionStart": {
        "schema_version", "kind", "task_id", "run_id", "action_id",
        "parent_action_id", "loop_id", "iteration", "ts", "action_kind",
        "tool_name", "summary", "model", "turn_id", "api_request_id",
        "tool_call_id",
    },
    "ActionEnd": {
        "schema_version", "kind", "task_id", "run_id", "action_id", "ts",
        "status", "duration_ms", "summary", "error_type", "error_message",
        "result_hash",
    },
    "DependencyEdge": {
        "schema_version", "kind", "task_id", "run_id",
        "from_action_id", "to_action_id", "edge_kind", "ts",
    },
    "DiagnosisResult": {
        "schema_version", "kind", "diagnosis_version", "task_id", "run_id",
        "status", "root_cause_action_ids", "confidence", "category",
        "propagation_path", "explanation", "interventions", "evidence",
    },
}

# Which enum each field must respect (only checked for known fields).
_FIELD_ENUM: Dict[Tuple[str, str], set] = {}
for _branch, _fields in (
    ("ActionStart", {"action_kind": "ActionKind"}),
    ("ActionEnd", {"status": "ActionStatus"}),
    ("DependencyEdge", {"edge_kind": None}),  # validated separately below
    ("DiagnosisResult", {"status": "DiagnosisStatus", "category": "FailureCategory"}),
):
    for _field, _enum_name in _fields.items():
        if _enum_name:
            _FIELD_ENUM[(_branch, _field)] = _BRANCH_ENUMS[_enum_name]

_EDGE_KINDS = {"data", "causal", "loop", "retry"}


def _is_valid_record(record: Any) -> bool:
    """Return True when ``record`` conforms to the loop-diagnostics schema."""
    if not isinstance(record, dict):
        return False
    kind = record.get("kind")
    if not isinstance(kind, str):
        return False
    branch = _KIND_TO_BRANCH.get(kind)
    if branch is None:
        return False
    # schema_version const (every branch except DiagnosisResult carries it;
    # DiagnosisResult also carries it).
    if record.get("schema_version") != SCHEMA_VERSION:
        return False
    # required fields
    for field in _BRANCH_REQUIRED[branch]:
        if field not in record:
            return False
    # additionalProperties: false
    allowed = _BRANCH_KEYS[branch]
    stray = set(record.keys()) - allowed
    if stray:
        return False
    # field enums
    for (b, f), enum in _FIELD_ENUM.items():
        if b == branch and f in record and record[f] is not None:
            if record[f] not in enum:
                return False
    if branch == "DependencyEdge" and record.get("edge_kind") not in _EDGE_KINDS:
        return False
    # cheap type checks (nullable fields skip when None)
    checks = {
        "RunHeader": {"task_id": str, "run_id": int, "attempt": int, "ts": int},
        "RunFooter": {"task_id": str, "run_id": int, "ts": int, "outcome": str},
        "ActionStart": {
            "task_id": str, "run_id": int, "action_id": str, "ts": int,
        },
        "ActionEnd": {
            "task_id": str, "run_id": int, "action_id": str, "ts": int,
            "status": str,
        },
        "DependencyEdge": {
            "task_id": str, "run_id": int,
            "from_action_id": str, "to_action_id": str, "edge_kind": str,
        },
        "DiagnosisResult": {
            "task_id": str, "run_id": int, "status": str, "category": str,
        },
    }
    for field, typ in checks.get(branch, {}).items():
        val = record.get(field)
        if val is None:
            return False
        if not isinstance(val, typ):
            return False
    # action_id format: "<run_id>:<seq>" (loose check)
    if branch == "ActionStart" and "action_id" in record:
        aid = record["action_id"]
        if not isinstance(aid, str) or ":" not in aid:
            return False
    return True


# ---------------------------------------------------------------------------
# Storage layout (shared with the recorder)
# ---------------------------------------------------------------------------


def loop_traces_dir(board: Optional[str] = None) -> Path:
    """Return the per-board loop-traces root directory.

    Mirrors the recorder's layout (and ``worker_logs_dir`` in kanban_db.py):
    the ``default`` board keeps the legacy root under ``<kanban-home>/kanban/``,
    all other boards live under ``<kanban-home>/kanban/boards/<slug>/``.
    """
    from hermes_cli.kanban_db import board_dir, DEFAULT_BOARD, get_current_board

    slug = board or os.environ.get("HERMES_KANBAN_BOARD", "") or get_current_board()
    if slug == DEFAULT_BOARD:
        from hermes_cli.kanban_db import kanban_home
        return kanban_home() / "kanban" / "loop-traces"
    return board_dir(slug) / "loop-traces"


def trace_path_for(task_id: str, run_id: int, board: Optional[str] = None) -> Path:
    """Return the trace file path for ``task_id`` / ``run_id``."""
    return loop_traces_dir(board) / task_id / f"{run_id}.jsonl"


# ---------------------------------------------------------------------------
# Trace loading + validation
# ---------------------------------------------------------------------------


class LoadedTrace:
    """Parsed and validated trace contents."""

    def __init__(
        self,
        *,
        records: List[Dict[str, Any]],
        malformed_line_numbers: List[int],
        has_footer: bool,
        truncated: bool,
        run_id: int,
        task_id: str,
    ) -> None:
        self.records = records
        self.malformed_line_numbers = malformed_line_numbers
        self.has_footer = has_footer
        self.truncated = truncated
        self.run_id = run_id
        self.task_id = task_id


def load_trace(
    path: Path,
    *,
    max_depth: int = DEFAULT_MAX_DEPTH,
    malformed_ratio: float = DEFAULT_MALFORMED_RATIO,
) -> LoadedTrace:
    """Load and validate a trace file.

    Invalid JSON lines and schema-invalid records are skipped and recorded by
    line number. A missing/empty file yields an empty trace (caller decides
    ``unknown`` vs ``malformed_trace``). Never raises.
    """
    records: List[Dict[str, Any]] = []
    malformed: List[int] = []
    has_footer = False
    truncated = False
    run_id: int = 0
    task_id: str = ""

    if path is None or not path.exists():
        return LoadedTrace(
            records=records,
            malformed_line_numbers=malformed,
            has_footer=has_footer,
            truncated=truncated,
            run_id=run_id,
            task_id=task_id,
        )

    try:
        with open(path, "r", encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except (ValueError, TypeError):
                    malformed.append(lineno)
                    continue
                if not _is_valid_record(record):
                    malformed.append(lineno)
                    continue
                records.append(record)
                kind = record.get("kind")
                if kind == "run_footer":
                    has_footer = True
                    truncated = bool(record.get("truncated", False))
                elif kind == "run_header":
                    run_id = record.get("run_id", run_id)
                    task_id = record.get("task_id", task_id)
    except OSError as exc:  # pragma: no cover — defensive
        logger.debug("loop-diagnostics: trace read failed: %s", exc)
        return LoadedTrace(
            records=[],
            malformed_line_numbers=[],
            has_footer=False,
            truncated=False,
            run_id=run_id,
            task_id=task_id,
        )

    return LoadedTrace(
        records=records,
        malformed_line_numbers=malformed,
        has_footer=has_footer,
        truncated=truncated,
        run_id=run_id,
        task_id=task_id,
    )


# ---------------------------------------------------------------------------
# Graph model
# ---------------------------------------------------------------------------


class _ActionNode:
    """One action (node) in the dependency graph."""

    __slots__ = (
        "action_id", "loop_id", "iteration", "tool_name", "status",
        "error_type", "error_message", "result_hash", "action_kind",
        "incoming",  # list of (from_action_id, edge_kind)
        "ts",
    )

    def __init__(self, action_id: str) -> None:
        self.action_id = action_id
        self.loop_id: Optional[str] = None
        self.iteration: Optional[int] = None
        self.tool_name: Optional[str] = None
        self.status: Optional[str] = None
        self.error_type: Optional[str] = None
        self.error_message: Optional[str] = None
        self.result_hash: Optional[str] = None
        self.action_kind: Optional[str] = None
        self.incoming: List[Tuple[str, str]] = []
        self.ts: int = 0


class _TraceGraph:
    """Validated action graph built from trace records."""

    def __init__(self) -> None:
        self.nodes: Dict[str, _ActionNode] = {}
        # action_id -> list of outgoing (to_action_id, edge_kind)
        self.outgoing: Dict[str, List[Tuple[str, str]]] = {}
        # action_id -> list of incoming (from_action_id, edge_kind)
        self.incoming: Dict[str, List[Tuple[str, str]]] = {}
        self.missing_action_ends: List[str] = []
        self.duplicate_hashes: List[str] = []
        self.malformed_edge_lines: List[int] = []

    def node(self, action_id: str) -> _ActionNode:
        if action_id not in self.nodes:
            self.nodes[action_id] = _ActionNode(action_id)
        return self.nodes[action_id]

    def add_edge(self, from_id: str, to_id: str, kind: str) -> None:
        self.incoming.setdefault(to_id, []).append((from_id, kind))
        self.outgoing.setdefault(from_id, []).append((to_id, kind))
        # Also populate the consumer node's incoming list so walkers can
        # read predecessor edges directly off the node.
        self.node(to_id).incoming.append((from_id, kind))


def _build_graph(records: Sequence[Dict[str, Any]]) -> _TraceGraph:
    """Build an action graph from validated trace records (deterministic).

    Two passes: first collect all action starts/ends, then process edges.
    The contract's sample traces emit an edge right after the producer's
    ``action_end`` and BEFORE the consumer's ``action_start`` (§14), so a
    single-pass builder would drop edges to not-yet-started consumers.
    Two passes make the engine robust to either edge ordering.
    """
    graph = _TraceGraph()
    starts: Dict[str, Dict[str, Any]] = {}
    ends: Dict[str, Dict[str, Any]] = {}
    edges: List[Dict[str, Any]] = []

    for record in records:
        kind = record.get("kind")
        if kind == "action_start":
            aid = record["action_id"]
            starts[aid] = record
            node = graph.node(aid)
            node.loop_id = record.get("loop_id")
            node.iteration = record.get("iteration")
            node.tool_name = record.get("tool_name")
            node.action_kind = record.get("action_kind")
            node.ts = record.get("ts", 0)
        elif kind == "action_end":
            aid = record["action_id"]
            ends[aid] = record
            node = graph.node(aid)
            node.status = record.get("status")
            node.error_type = record.get("error_type")
            node.error_message = record.get("error_message")
            node.result_hash = record.get("result_hash")
            node.ts = record.get("ts", node.ts or 0)
        elif kind == "edge":
            edges.append(record)

    # Second pass: edges (order within edges is preserved; deterministic).
    for record in edges:
        from_id = record.get("from_action_id")
        to_id = record.get("to_action_id")
        edge_kind = record.get("edge_kind")
        if not from_id or not to_id or not edge_kind:
            # corrupt edge (shouldn't happen post-validation; defensive)
            continue
        # Ignore corrupt edges: self-loop or dangling action_id.
        if from_id == to_id:
            continue
        if from_id not in starts and from_id not in ends:
            continue
        if to_id not in starts and to_id not in ends:
            continue
        graph.add_edge(from_id, to_id, edge_kind)

    # missing action ends
    for aid in sorted(starts.keys()):
        if aid not in ends:
            graph.missing_action_ends.append(aid)

    # duplicate hashes on 2+ failed actions in the same loop
    hash_to_loop: Dict[Tuple[Optional[str], Optional[str]], int] = {}
    for aid in sorted(ends.keys()):
        end = ends[aid]
        status = end.get("status")
        if status not in FAILURE_STATUSES:
            continue
        rh = end.get("result_hash")
        if not rh:
            continue
        if not isinstance(rh, str):
            continue
        start = starts.get(aid, {})
        key = (start.get("loop_id"), rh)
        hash_to_loop[key] = hash_to_loop.get(key, 0) + 1
    for (loop_id, rh), count in sorted(hash_to_loop.items()):
        if count >= 2:
            if rh not in graph.duplicate_hashes:
                graph.duplicate_hashes.append(rh)

    return graph


# ---------------------------------------------------------------------------
# Diagnosis
# ---------------------------------------------------------------------------


def _failure_rank(graph: _TraceGraph, action_id: str) -> int:
    """Return the failure preference rank for an action.

    0 = error, 1 = blocked, 2 = timed_out, 3 = cancelled, 4 = missing-end.
    Lower is better (more actionable as the failure site).
    """
    node = graph.nodes.get(action_id)
    if node is None or node.status is None:
        return 4
    status_order = {"error": 0, "blocked": 1, "timed_out": 2, "cancelled": 3}
    return status_order.get(node.status, 4)


def _failed_action_ids(graph: _TraceGraph) -> List[str]:
    """Return failed action ids in deterministic order.

    Preference: error > blocked > timed_out > cancelled (contract §8.1.2).
    Actions with a missing ``action_end`` (status None) in a truncated run
    are treated as interrupted (timeout/cancellation) candidates.
    """
    status_order = {"error": 0, "blocked": 1, "timed_out": 2, "cancelled": 3}
    failed = []
    for aid, node in graph.nodes.items():
        if node.status in FAILURE_STATUSES:
            failed.append((status_order.get(node.status, 99), aid, node.ts))
        elif node.status is None and aid in graph.missing_action_ends:
            # interrupted action (no end): rank after cancelled
            failed.append((4, aid, node.ts))
    failed.sort(key=lambda t: (t[0], t[1], t[2]))
    return [aid for _, aid, _ in failed]


def _propagated_failure(
    graph: _TraceGraph, node: _ActionNode
) -> Optional[Tuple[str, str]]:
    """Return the first failing producer along a data/causal edge, or None.

    Only ``data`` and ``causal`` edges propagate a producer failure to a
    consumer. ``loop``/``retry`` edges are handled by the loop/retry logic.
    A producer with a missing ``action_end`` (status None) in a truncated
    run is treated as a failing producer — the interruption is the cause.
    Returns ``(producer_action_id, edge_kind)``.
    """
    for from_id, kind in node.incoming:
        if kind not in PROPAGATION_EDGE_KINDS:
            continue
        producer = graph.nodes.get(from_id)
        if producer is None:
            continue
        if producer.status in FAILURE_STATUSES:
            return (from_id, kind)
        # missing end = interrupted action = failure for propagation purposes
        if producer.status is None and from_id in graph.missing_action_ends:
            return (from_id, kind)
    return None


def _walk_to_root(
    graph: _TraceGraph,
    start_id: str,
    *,
    max_depth: int = DEFAULT_MAX_DEPTH,
) -> Tuple[List[str], Optional[str]]:
    """Walk causal predecessors from ``start_id`` to the earliest failure.

    Returns ``(path, root_id)`` where ``path`` is the ordered chain
    ``[start_id, ..., root_id]`` and ``root_id`` is the earliest actionable
    node (or None when no root is found).

    Termination: keeps a visited set and a depth cap; a cycle terminates by
    the visited set, and the first failing node on the walk wins.
    """
    path: List[str] = [start_id]
    visited = {start_id}
    current = graph.nodes.get(start_id)
    if current is None:
        return path, None

    depth = 0
    while depth < max_depth:
        depth += 1
        # A failing producer along data/causal edge is the propagation target.
        propagated = _propagated_failure(graph, current)
        if propagated is None:
            # No failing predecessor: this is the root.
            return path, current.action_id
        from_id, _kind = propagated
        if from_id in visited:
            # Cycle: stop at the current node; the cycle itself is the root.
            return path, current.action_id
        visited.add(from_id)
        path.append(from_id)
        current = graph.nodes[from_id]
        if current.status not in FAILURE_STATUSES:
            # producer isn't actually failed (edge said it was, defensive)
            return path, current.action_id
    # Depth cap reached: report the deepest node as root (best effort).
    return path, current.action_id


def _classify(
    graph: _TraceGraph,
    root_id: Optional[str],
    root_node: Optional[_ActionNode],
    *,
    path: List[str],
    failed_id: str,
    truncated: bool,
    duplicate_hashes: List[str],
    retry_loop_failed: bool,
    footer_outcome: Optional[str] = None,
) -> str:
    """Classify the failure per contract §7 priority order."""
    # malformed_trace is handled by the caller (needs trace-level info).

    # retry_exhausted: retry loop hit its budget with all attempts failed.
    if retry_loop_failed:
        return "retry_exhausted"

    # loop_repeated: >=2 consecutive iterations in the same loop failed with
    # identical result_hash.
    if duplicate_hashes:
        return "loop_repeated"

    if root_id is None:
        # No root found. If truncated, degrade toward interruption categories.
        if truncated:
            return "timeout_propagation" if _has_timeout(graph, footer_outcome) else (
                "cancellation_propagation" if _has_cancellation(graph) else "unknown"
            )
        return "unknown"

    # Timeout/cancellation/input propagation: walk the propagation path and
    # find the first data-edge producer failure. The failed action's DIRECT
    # data producer may be absent (e.g. a causal final hop); the root cause
    # is still the earliest failed data producer along the path.
    for aid in path:
        producer = _failing_data_producer(graph, aid)
        if producer is None:
            continue
        if producer.status == "cancelled":
            return "cancellation_propagation"
        if producer.status in FAILURE_STATUSES:
            return "input_invalid"
        # producer has no end (status None) but footer says timed_out.
        if producer.status is None and _footer_says_timeout(footer_outcome):
            return "timeout_propagation"
        if producer.status is None and _footer_says_cancelled(footer_outcome):
            return "cancellation_propagation"
        return "input_invalid"

    # The failed action itself was interrupted (no failing producer).
    if root_node is not None and root_node.status == "cancelled":
        return "cancellation_propagation"
    if root_node is not None and root_node.status is None:
        if _footer_says_timeout(footer_outcome):
            return "timeout_propagation"
        if _footer_says_cancelled(footer_outcome):
            return "cancellation_propagation"

    # Plain action failure with no failing data predecessor.
    if root_node is not None and root_node.status in FAILURE_STATUSES:
        return "action_error"

    return "unknown"


def _footer_says_timeout(outcome: Optional[str]) -> bool:
    return outcome in ("timed_out",)


def _footer_says_cancelled(outcome: Optional[str]) -> bool:
    return outcome in ("cancelled", "reclaimed", "gave_up")


def _failing_data_producer(
    graph: _TraceGraph, action_id: str
) -> Optional[_ActionNode]:
    """Return the first failing producer along an incoming data edge.

    Deterministic: iterates ``node.incoming`` in insertion order (the order
    edges were recorded in the trace). A producer with a missing
    ``action_end`` (status None) counts as a failure cause — an interrupted
    action leaving downstream state partial.
    """
    node = graph.nodes.get(action_id)
    if node is None:
        return None
    for from_id, kind in node.incoming:
        if kind != "data":
            continue
        producer = graph.nodes.get(from_id)
        if producer is None:
            continue
        if producer.status in FAILURE_STATUSES:
            return producer
        if producer.status is None and from_id in graph.missing_action_ends:
            return producer
    return None


def _has_data_propagation(graph: _TraceGraph, action_id: str) -> bool:
    """True when any incoming data edge's producer also failed."""
    return _failing_data_producer(graph, action_id) is not None


def _has_timeout(graph: _TraceGraph, footer_outcome: Optional[str] = None) -> bool:
    if footer_outcome == "timed_out":
        return True
    # a missing action end in a timed-out run is also timeout evidence
    return bool(graph.missing_action_ends)


def _has_cancellation(graph: _TraceGraph) -> bool:
    return any(n.status == "cancelled" for n in graph.nodes.values())


def _retry_loop_failed(graph: _TraceGraph) -> bool:
    """True when a retry loop hit its budget with all attempts failed.

    A retry loop is a set of >=2 actions sharing ``loop_id`` starting with
    ``retry``, all failed. The contract models retries as a retry loop; the
    engine treats 'budget hit' as: >=2 consecutive failures joined by
    ``retry``/``loop`` edges within the same retry loop_id.
    """
    retry_groups: Dict[str, List[_ActionNode]] = {}
    for node in graph.nodes.values():
        lid = node.loop_id or ""
        if lid.startswith(RETRY_LOOP_PREFIX) and node.status in FAILURE_STATUSES:
            retry_groups.setdefault(lid, []).append(node)
    for lid, nodes in retry_groups.items():
        # sort by iteration for deterministic 'consecutive' check
        nodes.sort(key=lambda n: (n.iteration if n.iteration is not None else -1, n.action_id))
        if len(nodes) >= 2:
            return True
    return False


def _explanation_for(
    category: str,
    failed_id: str,
    root_id: Optional[str],
    path: List[str],
    graph: _TraceGraph,
) -> str:
    """Build a concise deterministic explanation (<=2048 chars)."""
    failed_node = graph.nodes.get(failed_id)
    failed_tool = failed_node.tool_name if failed_node else None
    failed_status = failed_node.status if failed_node else None
    failed_desc = failed_id
    if failed_tool:
        failed_desc = f"{failed_id} ({failed_tool})"
    if failed_status:
        failed_desc += f" [{failed_status}]"

    if category == "unknown":
        return f"No failure evidence found for action {failed_id}."
    if category == "malformed_trace":
        return "Trace is too malformed to diagnose reliably."
    if category == "retry_exhausted":
        return (
            f"Action {failed_desc} failed after exhausting its retry budget; "
            "all attempts in the retry loop failed."
        )
    if category == "loop_repeated":
        hashes = ", ".join(graph.duplicate_hashes[:3])
        return (
            f"Action {failed_desc} failed repeatedly in the same loop with "
            f"identical results (hash(es): {hashes})."
        )
    if category == "input_invalid":
        root_node = graph.nodes.get(root_id) if root_id else None
        root_tool = root_node.tool_name if root_node else None
        root_desc = root_id or "?"
        if root_tool:
            root_desc = f"{root_id} ({root_tool})"
        return (
            f"Action {failed_desc} failed because its data predecessor "
            f"{root_desc} failed, producing invalid or missing input; the "
            "consumer failure is downstream propagation."
        )
    if category == "timeout_propagation":
        return (
            f"Action {failed_desc} failed because a preceding action timed out, "
            "leaving downstream actions on interrupted state."
        )
    if category == "cancellation_propagation":
        return (
            f"Action {failed_desc} failed because a preceding action was "
            "cancelled, leaving downstream actions on partial state."
        )
    # action_error
    root_node = graph.nodes.get(root_id) if root_id else None
    root_tool = root_node.tool_name if root_node else None
    root_desc = root_id or failed_id
    if root_tool:
        root_desc = f"{root_id} ({root_tool})"
    err = root_node.error_message if root_node and root_node.error_message else ""
    if err:
        return f"Action {root_desc} failed with an action error: {err[:500]}."
    return f"Action {root_desc} failed with an action error."


def _first_loop_failure(graph: _TraceGraph, path: List[str]) -> Optional[str]:
    """Return the earliest failed action in the repeated loop.

    For ``loop_repeated``, the root cause is the first iteration that
    failed, not the last (the failure site). The repeated loop is the
    ``loop_id`` shared by the failed actions in the graph (all failed
    actions with a loop_id that appears on the path); pick the action
    with the smallest ``iteration`` (ties broken by action_id).
    """
    # Find the loop_id(s) of the failed path actions.
    path_nodes = [graph.nodes.get(aid) for aid in path]
    loop_ids = {
        n.loop_id for n in path_nodes
        if n is not None and n.loop_id and n.status in FAILURE_STATUSES
    }
    if not loop_ids:
        return None
    # All failed actions in those loops, across the whole graph.
    candidates: List[_ActionNode] = []
    for node in graph.nodes.values():
        if (
            node.loop_id in loop_ids
            and node.status in FAILURE_STATUSES
        ):
            candidates.append(node)
    if not candidates:
        return None
    candidates.sort(
        key=lambda n: (n.iteration if n.iteration is not None else -1, n.action_id)
    )
    return candidates[0].action_id


def _connected_to_path(graph: _TraceGraph, action_id: str, path: List[str]) -> bool:
    """True when ``action_id`` is graph-connected to any node in ``path``.

    Two failures are NOT independent when any edge path connects them — a
    loop/retry lineage or a causal chain means the second failure is not a
    separate root cause. BUT the traversal only flows *through failed
    actions*: a healthy fan-out parent (e.g. delegate_task) is an
    orchestration node, not a causal bridge between its independently-
    failing children. Sibling failures under a healthy parent are separate
    candidates; only a chain of failed actions propagates causation.
    Undirected BFS over ``incoming`` + ``outgoing`` with a visited set
    (cycle-safe).
    """
    if action_id in path:
        return True
    path_nodes = set(path)
    visited = {action_id}
    queue = [action_id]
    while queue:
        cur = queue.pop(0)
        if cur in path_nodes:
            return True
        for neighbor, _kind in _graph_neighbors(graph, cur):
            if neighbor in visited:
                continue
            # Only traverse through actions that actually failed. A healthy
            # node (ok / no end) is not a causal bridge between failures.
            nnode = graph.nodes.get(neighbor)
            if nnode is None or nnode.status not in FAILURE_STATUSES:
                continue
            visited.add(neighbor)
            queue.append(neighbor)
    return False


def _graph_neighbors(graph: _TraceGraph, action_id: str) -> List[Tuple[str, str]]:
    """Undirected neighbor (id, kind) pairs from incoming + outgoing edges."""
    out: List[Tuple[str, str]] = []
    for from_id, kind in graph.incoming.get(action_id, []):
        out.append((from_id, kind))
    for to_id, kind in graph.outgoing.get(action_id, []):
        out.append((to_id, kind))
    return out


def _verified_checkpoint(graph: _TraceGraph, root_id: Optional[str]) -> Optional[str]:
    """Return the nearest verified (status=ok) predecessor of ``root_id``.

    A checkpoint is an action the worker can resume from — it must have
    completed successfully (``status=ok``) AND be a direct data/causal
    producer of the root (or the earliest such action in a chain of
    successful producers). The propagation path is the *failure* chain, so
    ``path[idx-1]`` is the failed consumer — never a checkpoint.

    Returns the action_id, or None when no verified producer exists.
    """
    if root_id is None:
        return None
    node = graph.nodes.get(root_id)
    if node is None:
        return None
    # Walk incoming data/causal edges to the nearest verified producer.
    # Stop at the first non-ok action (a failed/unknown producer is not a
    # safe resume point — it is the root cause itself).
    seen: set = set()
    frontier = [(root_id, 0)]
    while frontier:
        frontier.sort(key=lambda t: t[1])  # nearest first (deterministic)
        cur, depth = frontier.pop(0)
        if cur in seen:
            continue
        seen.add(cur)
        cur_node = graph.nodes.get(cur)
        if cur_node is None:
            continue
        if cur == root_id:
            pass  # expand the root's producers below
        elif cur_node.status == "ok" and cur_node.action_kind != "llm_call":
            return cur
        elif cur_node.status == "ok":
            # A completed LLM turn is not a resumable checkpoint — it is a
            # synthetic goal-loop anchor with no artifact. Expand through it
            # to find a real tool checkpoint.
            pass
        elif cur_node.status in FAILURE_STATUSES or (
            cur_node.status is None and cur in graph.missing_action_ends
        ):
            continue  # failed/interrupted — not a checkpoint, stop expanding
        for from_id, kind in cur_node.incoming:
            if kind not in PROPAGATION_EDGE_KINDS:
                continue
            frontier.append((from_id, depth + 1))
    return None


def _recommend_interventions(
    category: str,
    root_id: Optional[str],
    failed_id: str,
    graph: _TraceGraph,
    path: List[str],
) -> List[Dict[str, Any]]:
    """Return ranked intervention recommendations (never executed)."""
    root_node = graph.nodes.get(root_id) if root_id else None
    root_tool = root_node.tool_name if root_node else None

    # Verified checkpoint before the root — only when a completed producer
    # actually exists. Never fabricate a checkpoint from the failure path.
    checkpoint_id = _verified_checkpoint(graph, root_id)

    interventions: List[Dict[str, Any]] = []

    if category == "retry_exhausted":
        interventions.append({
            "kind": "escalate",
            "action_id": root_id or failed_id,
            "rationale": (
                "Blind retry already exhausted the retry budget; escalate "
                "instead of recommending more retries."
            ),
            "payload": {},
        })
        return interventions

    if category == "loop_repeated":
        if root_tool:
            interventions.append({
                "kind": "alternative_tool",
                "action_id": root_id or failed_id,
                "rationale": (
                    f"{root_tool} failed repeatedly with identical results; "
                    "try an alternative tool that can meet the same goal."
                ),
                "payload": {"tool_name": root_tool},
            })
        else:
            interventions.append({
                "kind": "escalate",
                "action_id": root_id or failed_id,
                "rationale": (
                    "Repeated identical loop failures; escalate for manual "
                    "review."
                ),
                "payload": {},
            })
        return interventions

    if category == "input_invalid":
        if checkpoint_id:
            interventions.append({
                "kind": "retry_from_checkpoint",
                "action_id": root_id or failed_id,
                "rationale": (
                    "A failed producer left invalid input; retry from the "
                    "last verified checkpoint after fixing the producer."
                ),
                "payload": {"checkpoint_action_id": checkpoint_id},
            })
        else:
            interventions.append({
                "kind": "trajectory_repair",
                "action_id": root_id or failed_id,
                "rationale": (
                    "No verified checkpoint precedes the failed producer; "
                    "patch the bad state before retrying."
                ),
                "payload": {"patch_hint": "repair the failed producer state"},
            })
        return interventions

    if category in ("timeout_propagation", "cancellation_propagation"):
        if checkpoint_id:
            interventions.append({
                "kind": "retry_from_checkpoint",
                "action_id": root_id or failed_id,
                "rationale": (
                    "An interrupted action left downstream state partial; "
                    "retry from the last verified checkpoint."
                ),
                "payload": {"checkpoint_action_id": checkpoint_id},
            })
        else:
            interventions.append({
                "kind": "trajectory_repair",
                "action_id": root_id or failed_id,
                "rationale": (
                    "Interrupted action with no verified checkpoint; repair "
                    "the trajectory before retrying."
                ),
                "payload": {"patch_hint": "repair interrupted action state"},
            })
        return interventions

    if category == "action_error":
        if root_tool:
            interventions.append({
                "kind": "alternative_tool",
                "action_id": root_id or failed_id,
                "rationale": (
                    f"{root_tool} failed with an action error; try an "
                    "alternative tool for the same goal."
                ),
                "payload": {"tool_name": root_tool},
            })
        else:
            interventions.append({
                "kind": "escalate",
                "action_id": root_id or failed_id,
                "rationale": "Action error with no obvious alternative; escalate.",
                "payload": {},
            })
        return interventions

    # unknown / malformed_trace
    interventions.append({
        "kind": "escalate",
        "action_id": failed_id,
        "rationale": "Diagnosis inconclusive; escalate for manual review.",
        "payload": {},
    })
    return interventions


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def diagnose(
    task_id: str,
    run_id: int,
    *,
    failed_action_id: Optional[str] = None,
    board: Optional[str] = None,
    trace_path: Optional[Path] = None,
    max_depth: int = DEFAULT_MAX_DEPTH,
    malformed_ratio: float = DEFAULT_MALFORMED_RATIO,
) -> Dict[str, Any]:
    """Run the deterministic diagnosis engine on a run's trace.

    Args:
        task_id: kanban task id (``t_...``).
        run_id: the attempt id (``task_runs.id``).
        failed_action_id: optional explicit failure site. When omitted, the
            engine picks the last non-ok action (error > blocked > timed_out
            > cancelled).
        board: board slug; defaults to the current board (storage layout).
        trace_path: optional explicit trace file path (tests use this).
        max_depth: traversal depth cap (cycles terminate safely).
        malformed_ratio: fraction of invalid lines above which the trace is
            considered ``malformed_trace``.

    Returns:
        A ``DiagnosisResult`` dict conforming to the schema (never raises).
    """
    path = trace_path if trace_path is not None else trace_path_for(task_id, run_id, board)

    loaded = load_trace(path, max_depth=max_depth, malformed_ratio=malformed_ratio)

    total_lines = len(loaded.records) + len(loaded.malformed_line_numbers)
    malformed_frac = (
        len(loaded.malformed_line_numbers) / total_lines
        if total_lines > 0
        else 0.0
    )

    result: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "kind": "diagnosis_result",
        "diagnosis_version": DIAGNOSIS_VERSION,
        "task_id": task_id,
        "run_id": run_id,
        "status": "unknown",
        "category": "unknown",
        "explanation": f"No trace recorded for run {run_id}.",
        "interventions": [],
        "evidence": {
            "failed_action_id": failed_action_id or "",
            "trace_event_count": len(loaded.records),
            "missing_action_ends": [],
            "malformed_lines": list(loaded.malformed_line_numbers),
            "duplicate_hashes": [],
        },
    }

    # No trace file / no records.
    if not loaded.records:
        result["status"] = "unknown"
        result["category"] = "unknown"
        result["confidence"] = 0.0
        result["explanation"] = f"No trace recorded for run {run_id}."
        result["interventions"] = [{
            "kind": "escalate",
            "action_id": failed_action_id or "",
            "rationale": "Diagnosis inconclusive; escalate for manual review.",
            "payload": {},
        }]
        return result

    # >=50% malformed -> malformed_trace.
    if malformed_frac >= malformed_ratio:
        result["status"] = "malformed_trace"
        result["category"] = "malformed_trace"
        result["confidence"] = 0.0
        result["evidence"]["malformed_lines"] = list(loaded.malformed_line_numbers)
        result["explanation"] = (
            f"Trace is too malformed to diagnose reliably "
            f"({len(loaded.malformed_line_numbers)}/{total_lines} invalid lines)."
        )
        result["interventions"] = [{
            "kind": "escalate",
            "action_id": failed_action_id or "",
            "rationale": "Diagnosis inconclusive; escalate for manual review.",
            "payload": {},
        }]
        return result

    graph = _build_graph(loaded.records)

    result["evidence"]["missing_action_ends"] = list(graph.missing_action_ends)
    result["evidence"]["duplicate_hashes"] = list(graph.duplicate_hashes)

    # Locate the failure site.
    failed_id = failed_action_id
    if failed_id is None:
        failed_ids = _failed_action_ids(graph)
        if failed_ids:
            # _failed_action_ids ranks by preference: error > blocked >
            # timed_out > cancelled > missing-end. Within the highest-
            # preference tier, prefer the last (chronologically latest)
            # failure — the action whose error ended the run. NEVER pick a
            # lower-preference entry (e.g. a missing-end in-flight span)
            # over a real error.
            best = failed_ids[0]
            for aid in failed_ids[1:]:
                if _failure_rank(graph, aid) != _failure_rank(graph, best):
                    break
                best = aid
            failed_id = best
    if failed_id is None:
        # No failed actions at all: not a failure trace.
        result["status"] = "unknown"
        result["category"] = "unknown"
        result["confidence"] = 0.0
        result["explanation"] = f"No failed actions found in run {run_id}."
        result["interventions"] = [{
            "kind": "none",
            "action_id": "",
            "rationale": "No failure to diagnose.",
            "payload": {},
        }]
        return result

    result["evidence"]["failed_action_id"] = failed_id

    # Walk to root.
    path, root_id = _walk_to_root(graph, failed_id, max_depth=max_depth)
    root_node = graph.nodes.get(root_id) if root_id else None

    # Truncated detection.
    truncated = (not loaded.has_footer) or loaded.truncated
    truncated = truncated or bool(graph.missing_action_ends)

    # Footer outcome (drives timeout/cancellation classification).
    footer_outcome = None
    for record in loaded.records:
        if record.get("kind") == "run_footer":
            footer_outcome = record.get("outcome")
            break

    # Retry loop exhausted?
    retry_failed = _retry_loop_failed(graph)

    category = _classify(
        graph,
        root_id,
        root_node,
        path=path,
        failed_id=failed_id,
        truncated=truncated,
        duplicate_hashes=graph.duplicate_hashes,
        retry_loop_failed=retry_failed,
        footer_outcome=footer_outcome,
    )

    result["category"] = category
    result["propagation_path"] = list(path)

    # Ambiguity: multiple independent failing branches with no causal link.
    all_failed = _failed_action_ids(graph)
    independent_failures = [
        aid for aid in all_failed
        if aid != failed_id and aid not in path
        and not _connected_to_path(graph, aid, path)
    ]
    if category in ("malformed_trace", "unknown"):
        result["status"] = "malformed_trace" if category == "malformed_trace" else "unknown"
        result["confidence"] = 0.0
        result["explanation"] = _explanation_for(
            category, failed_id, root_id, path, graph
        )
        result["interventions"] = [{
            "kind": "escalate",
            "action_id": failed_id,
            "rationale": "Diagnosis inconclusive; escalate for manual review.",
            "payload": {},
        }]
        return result

    if independent_failures:
        # Ranked candidates (contract §8.2).
        result["status"] = "candidates"
        result["confidence"] = 0.5
        result["root_cause_action_ids"] = [failed_id] + independent_failures
        result["explanation"] = (
            _explanation_for(category, failed_id, root_id, path, graph)
            + " Additional independent failures were found: "
            + ", ".join(independent_failures[:5])
            + "."
        )
        result["interventions"] = _recommend_interventions(
            category, root_id, failed_id, graph, path
        )
        return result

    # Single deterministic root.
    result["status"] = "root_cause_found"
    result["confidence"] = 0.9
    if category == "loop_repeated":
        # The root is the FIRST failed iteration of the repeated loop, not
        # the last (which is the failure site the engine walked to).
        first_loop_failure = _first_loop_failure(graph, path)
        result["root_cause_action_ids"] = [first_loop_failure] if first_loop_failure else ([root_id] if root_id else [failed_id])
        if first_loop_failure:
            # point the propagation path at the repeated loop origin
            result["propagation_path"] = list(path)
    else:
        result["root_cause_action_ids"] = [root_id] if root_id else [failed_id]
    result["explanation"] = _explanation_for(category, failed_id, root_id, path, graph)
    result["interventions"] = _recommend_interventions(
        category, root_id, failed_id, graph, path
    )
    return result
