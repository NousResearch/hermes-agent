"""ORCH V4 Observer — expected vs observed truth + exit total order.

Source: 拆卡機制四大缺陷-詳細實作計畫.md §9 (Observer V4) + §I.8 (exit order).

This module is deliberately runtime-independent.  It models the observer's
expected/observed separation, exact multiset equality, overlap proof, and
exit precedence so the focused tests can exercise every invariant without
opening the live Kanban database.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable


# §9.3 Exit precedence — first matching class wins, NOT numeric max.
EXIT_PRECEDENCE: tuple[int, ...] = (64, 5, 4, 2, 3, 0)

EXIT_INVALID_CLI = 64
EXIT_PATH_UNAVAILABLE = 5
EXIT_SCHEMA_UNSUPPORTED = 4
EXIT_HARD_VIOLATION = 2
EXIT_WARNING_ONLY = 3
EXIT_CLEAN = 0

EXIT_LABELS: dict[int, str] = {
    64: "invalid_cli_syntax",
    5: "path_unavailable",
    4: "schema_unsupported",
    2: "hard_violation",
    3: "warning_only",
    0: "clean",
}


class ObserverError(Exception):
    """Hard invariant violation found by observer."""
    def __init__(self, code: str, exit_code: int = EXIT_HARD_VIOLATION):
        super().__init__(code)
        self.code = code
        self.exit_code = exit_code


class ObserverPathError(Exception):
    """Path/open/read/replacement unavailable."""
    def __init__(self, code: str = "path_unavailable"):
        super().__init__(code)
        self.code = code
        self.exit_code = EXIT_PATH_UNAVAILABLE


class ObserverSchemaError(Exception):
    """Schema/migration/version unsupported."""
    def __init__(self, code: str = "schema_unsupported"):
        super().__init__(code)
        self.code = code
        self.exit_code = EXIT_SCHEMA_UNSUPPORTED


@dataclass(frozen=True)
class ExpectedNode:
    """Expected plan node from immutable plan header."""
    node_key: str
    role: str
    lane_label: str
    required: bool
    ordinal: int


@dataclass(frozen=True)
class ExpectedEdge:
    parent_node_key: str
    child_node_key: str
    edge_kind: str


@dataclass(frozen=True)
class ObservedRun:
    """Observed accepted run from materialization."""
    node_key: str
    run_id: str
    started_at: int
    ended_at: int


@dataclass(frozen=True)
class Finding:
    """One observer finding with its exit class."""
    exit_code: int
    code: str
    detail: str = ""


def compute_exit(findings: list[Finding]) -> int:
    """§9.3: First matching class in EXIT_PRECEDENCE wins.

    Collects all findings, then picks the first exit code by precedence.
    NOT numeric max — 2 beats 3 beats 0.
    """
    if not findings:
        return EXIT_CLEAN
    exit_set = {f.exit_code for f in findings}
    for code in EXIT_PRECEDENCE:
        if code in exit_set:
            return code
    return EXIT_CLEAN


def check_exact_multiset(
    expected_nodes: list[ExpectedNode],
    observed_node_keys: list[str],
) -> Finding | None:
    """§9.4: Expected/observed node multiset must be exact.

    No missing, no extra, no duplicate on either side.
    """
    from collections import Counter

    expected_keys = [n.node_key for n in expected_nodes]
    expected_counts = Counter(expected_keys)
    observed_counts = Counter(observed_node_keys)

    if any(v > 1 for v in expected_counts.values()):
        dups = sorted(k for k, v in expected_counts.items() if v > 1)
        return Finding(EXIT_HARD_VIOLATION, "duplicate_expected_nodes", f"dups: {dups}")
    if any(v > 1 for v in observed_counts.values()):
        dups = sorted(k for k, v in observed_counts.items() if v > 1)
        return Finding(EXIT_HARD_VIOLATION, "duplicate_observed_nodes", f"dups: {dups}")

    expected_set = set(expected_counts)
    observed_set = set(observed_counts)
    missing = expected_set - observed_set
    extra = observed_set - expected_set
    if missing:
        return Finding(EXIT_HARD_VIOLATION, "missing_observed_nodes", f"missing: {missing}")
    if extra:
        return Finding(EXIT_HARD_VIOLATION, "extra_observed_nodes", f"extra: {extra}")
    # counts must match exactly (defensive; dups already rejected)
    if expected_counts != observed_counts:
        return Finding(EXIT_HARD_VIOLATION, "multiset_count_mismatch")
    return None


def check_required_lane_overlap(
    expected_lanes: list[ExpectedNode],
    accepted_runs: list[ObservedRun],
) -> Finding | None:
    """§8.2 + §9.4: Required lanes must have N-way overlap.

    Parallel PASS only if:
    - distinct lane IDs
    - one run each
    - max(started_at) < min(ended_at)
    """
    required = [n for n in expected_lanes if n.required]
    if not required:
        return None  # board-only or no required lanes — no overlap to prove
    if len(required) < 2:
        return Finding(EXIT_HARD_VIOLATION, "insufficient_required_lanes")

    # Map node_key to accepted run
    lane_runs: dict[str, ObservedRun] = {}
    for run in accepted_runs:
        if run.node_key in {n.node_key for n in required}:
            if run.node_key in lane_runs:
                return Finding(EXIT_HARD_VIOLATION, "duplicate_lane_run", f"lane={run.node_key}")
            lane_runs[run.node_key] = run

    # All required lanes must have accepted runs
    missing_lanes = {n.node_key for n in required} - set(lane_runs.keys())
    if missing_lanes:
        return Finding(EXIT_HARD_VIOLATION, "missing_required_acceptance", f"lanes: {missing_lanes}")

    # Overlap check: max(started_at) < min(ended_at)
    starts = [r.started_at for r in lane_runs.values()]
    ends = [r.ended_at for r in lane_runs.values()]

    if max(starts) >= min(ends):
        return Finding(EXIT_HARD_VIOLATION, "no_required_lane_overlap",
                       f"max_start={max(starts)} >= min_end={min(ends)}")

    return None


def check_legacy_untyped(
    has_v4_schema: bool,
    has_legacy_title: bool,
) -> Finding | None:
    """Legacy untyped ORCH can never be V4 CLEAN. Best case exit 3."""
    if not has_v4_schema and has_legacy_title:
        return Finding(EXIT_WARNING_ONLY, "legacy_untyped_only")
    if not has_v4_schema:
        return Finding(EXIT_SCHEMA_UNSUPPORTED, "schema_unsupported")
    return None


def check_scope_mismatch(
    expected_board: str,
    observed_board: str,
    expected_tenant: str,
    observed_tenant: str,
) -> Finding | None:
    """§9.4: Stable board/tenant scope must match."""
    if expected_board != observed_board:
        return Finding(EXIT_HARD_VIOLATION, "board_scope_mismatch")
    if expected_tenant != observed_tenant:
        return Finding(EXIT_HARD_VIOLATION, "tenant_scope_mismatch")
    return None


def check_parent_terminal_before_synthesis(
    parent_state: str,
    has_result: bool,
) -> Finding | None:
    """Parent must not be terminal before synthesis."""
    terminal = {"completed", "failed", "cancelled"}
    if parent_state in terminal and not has_result:
        return Finding(EXIT_HARD_VIOLATION, "parent_terminal_before_synthesis")
    return None


def check_delivery_satisfied(
    origin_kind: str,
    required_count: int | None = None,
    acked_count: int | None = None,
    *,
    required_obligation_keys: list[str] | None = None,
    acked_obligation_keys: list[str] | None = None,
) -> Finding | None:
    """§9.4: Delivery manifest must be satisfied for parent completion.

    Prefer exact obligation-key multisets. Count-only inputs remain as a
    compatibility path but cannot pass when keys are also supplied and disagree.
    """
    from collections import Counter

    if required_obligation_keys is not None or acked_obligation_keys is not None:
        req_keys = list(required_obligation_keys or [])
        ack_keys = list(acked_obligation_keys or [])
        if origin_kind == "board_only":
            if req_keys:
                return Finding(EXIT_HARD_VIOLATION, "board_only_has_required_obligations")
            return None
        if not req_keys:
            return Finding(EXIT_HARD_VIOLATION, "delivery_required_set_empty")
        req_counts = Counter(req_keys)
        ack_counts = Counter(ack_keys)
        if any(v != 1 for v in req_counts.values()):
            return Finding(EXIT_HARD_VIOLATION, "duplicate_required_obligation_keys")
        if any(v != 1 for v in ack_counts.values()):
            return Finding(EXIT_HARD_VIOLATION, "duplicate_acked_obligation_keys")
        missing = sorted(set(req_counts) - set(ack_counts))
        extra = sorted(set(ack_counts) - set(req_counts))
        if missing or extra:
            return Finding(
                EXIT_HARD_VIOLATION,
                "delivery_required_acks_missing",
                f"missing={missing} extra={extra}",
            )
        return None

    # Compatibility count path (weaker; still fail-closed on gaps)
    rc = 0 if required_count is None else int(required_count)
    ac = 0 if acked_count is None else int(acked_count)
    if origin_kind == "board_only":
        if rc != 0:
            return Finding(EXIT_HARD_VIOLATION, "board_only_has_required_obligations")
        return None
    if rc > 0 and ac < rc:
        return Finding(
            EXIT_HARD_VIOLATION,
            "delivery_required_acks_missing",
            f"required={rc} acked={ac}",
        )
    if rc > 0 and ac != rc:
        # exact count equality required when using count path
        return Finding(
            EXIT_HARD_VIOLATION,
            "delivery_ack_count_mismatch",
            f"required={rc} acked={ac}",
        )
    if rc == 0 and origin_kind != "board_only":
        return Finding(EXIT_HARD_VIOLATION, "delivery_required_set_empty")
    return None


def run_observer(
    *,
    has_v4_schema: bool,
    expected_nodes: list[ExpectedNode],
    observed_node_keys: list[str],
    accepted_runs: list[ObservedRun],
    expected_board: str = "board",
    observed_board: str = "board",
    expected_tenant: str = "",
    observed_tenant: str = "",
    parent_state: str = "waiting_lanes",
    has_result: bool = False,
    origin_kind: str = "messaging",
    required_acks: int = 0,
    acked_count: int = 0,
    required_obligation_keys: list[str] | None = None,
    acked_obligation_keys: list[str] | None = None,
    has_legacy_title: bool = False,
) -> int:
    """Run all observer checks and return the exit code by precedence.

    Collects all findings, then applies EXIT_PRECEDENCE.
    Never short-circuits on first error.
    """
    findings: list[Finding] = []

    # Schema check first
    if not has_v4_schema:
        legacy = check_legacy_untyped(has_v4_schema, has_legacy_title)
        if legacy:
            findings.append(legacy)
        return compute_exit(findings)

    # Hard checks
    f = check_scope_mismatch(expected_board, observed_board, expected_tenant, observed_tenant)
    if f:
        findings.append(f)

    f = check_exact_multiset(expected_nodes, observed_node_keys)
    if f:
        findings.append(f)

    f = check_required_lane_overlap(expected_nodes, accepted_runs)
    if f:
        findings.append(f)

    f = check_parent_terminal_before_synthesis(parent_state, has_result)
    if f:
        findings.append(f)

    f = check_delivery_satisfied(
        origin_kind,
        required_acks,
        acked_count,
        required_obligation_keys=required_obligation_keys,
        acked_obligation_keys=acked_obligation_keys,
    )
    if f:
        findings.append(f)

    return compute_exit(findings)


__all__ = [
    "EXIT_PRECEDENCE", "EXIT_INVALID_CLI", "EXIT_PATH_UNAVAILABLE",
    "EXIT_SCHEMA_UNSUPPORTED", "EXIT_HARD_VIOLATION", "EXIT_WARNING_ONLY",
    "EXIT_CLEAN", "EXIT_LABELS", "ObserverError", "ObserverPathError",
    "ObserverSchemaError", "ExpectedNode", "ExpectedEdge", "ObservedRun",
    "Finding", "compute_exit", "check_exact_multiset",
    "check_required_lane_overlap", "check_legacy_untyped",
    "check_scope_mismatch", "check_parent_terminal_before_synthesis",
    "check_delivery_satisfied", "run_observer",
]
