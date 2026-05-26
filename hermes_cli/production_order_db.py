"""Production Workflow v1 — runtime bridge module.

Provides the deterministic runtime layer that transforms an approved brief
into a governed, stateful production order with Kanban graph, event log,
and handoff packets.

Ownership boundary (from spec section 18):

    LLM decides judgment.
    Runtime validates state.
    Kanban records work.
    Profiles own stages.

This module validates deterministic state ONLY. It does NOT contain profile
reasoning, routing decisions, or approval judgments.

Usage (programmatic)::

    conn = kb.connect()
    po = create_production_order(
        conn,
        title="Implement Slice 4",
        source_brief="...",
        approved_by="Jarren",
        priority_lane="Hermes OS",
        repo_or_workspace="hermes-agent",
    )

    child_ids = create_production_kanban_graph(conn, po)
    handoff_packet = create_orchestrator_handoff(po)
    freeze_handoff_on_card(conn, child_ids[0], handoff_packet)
    transition_state(conn, po, "ORCHESTRATOR_TRIAGE", "orchestrator_os")

Usage (CLI)::

    hermes production-order create --brief <path> [--board <slug>]

Current bridge coverage:

    approved brief → production order → six-card Kanban graph
    → ORCHESTRATOR_TRIAGE → ARCHITECT_SPEC → ARCHITECT_READY_FOR_DEV
    → DEV_IMPLEMENTING → DEV_COMPLETE → AUDIT_REVIEW → AUDIT_PASSED
    → ARCHITECT_RECONCILE → ARCHITECT_ACCEPTED → DEFAULT_FINAL_REVIEW
    → DONE

    First implemented failure-loop slice:

    AUDIT_REVIEW → AUDIT_REJECTED → DEV_REWORK → DEV_COMPLETE → AUDIT_REVIEW

Out of scope (do NOT implement):
    - OrchestratorOS agent spawn
    - ArchitectOS/DevOS/AuditOS dispatch
    - Dashboard UI
    - Full rework loop automation beyond the first audited rejection slice
    - Full event log dashboard
    - Profile SOUL.md / orchestration contract modifications
"""

from __future__ import annotations

import json
import os
import re
import secrets
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from hermes_cli import kanban_db as kb


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PRODUCTION_ORDER_FIELD = "production_order_id"
STATE_FIELD = "current_state"
WORKFLOW_TEMPLATE_ID = "hermes-production-workflow-v1"
WORKFLOW_SPEC_SOURCE = (
    "/Users/jarren/Projects/hermes-workspace/specs/architecture/workflows/"
    "hermes-production-workflow-v1.md"
)

# Only implemented runtime bridge transitions are wired. Remaining
# failure-loop transitions become live when each downstream profile uses the
# full state machine.
VALID_TRANSITIONS: dict[str, set[str]] = {
    "BRIEF_DRAFTED":                {"ACTION_APPROVED"},
    "ACTION_APPROVED":              {"PRODUCTION_ORDER_CREATED"},
    "PRODUCTION_ORDER_CREATED":     {"ORCHESTRATOR_TRIAGE"},
    "ORCHESTRATOR_TRIAGE":          {"ARCHITECT_SPEC", "DEV_REWORK", "SPEC_REWORK"},
    "ARCHITECT_SPEC":               {"ARCHITECT_READY_FOR_DEV"},
    "ARCHITECT_READY_FOR_DEV":      {"DEV_IMPLEMENTING"},
    "DEV_IMPLEMENTING":             {"DEV_COMPLETE"},
    "DEV_COMPLETE":                 {"AUDIT_REVIEW"},
    "AUDIT_REVIEW":                 {"AUDIT_PASSED", "AUDIT_REJECTED"},
    "AUDIT_REJECTED":               {"DEV_REWORK", "SPEC_REWORK", "BLOCKED_NEEDS_JARREN"},
    "AUDIT_PASSED":                 {"ARCHITECT_RECONCILE"},
    "ARCHITECT_RECONCILE":          {"ARCHITECT_ACCEPTED"},
    "ARCHITECT_ACCEPTED":           {"DEFAULT_FINAL_REVIEW"},
    "DEFAULT_FINAL_REVIEW":         {"DONE", "DEFAULT_REJECTED"},
    "DEFAULT_REJECTED":             {"ORCHESTRATOR_TRIAGE", "DEV_REWORK", "SPEC_REWORK", "BLOCKED_NEEDS_JARREN"},
    "BLOCKED_NEEDS_JARREN":         set(),
    "DEV_REWORK":                   {"DEV_COMPLETE"},
}

ORCHESTRATOR_CLASSIFICATION_ROUTE_MAP: dict[str, dict[str, str]] = {
    "implementation_mismatch": {
        "route_target": "DEV_REWORK",
        "next_handoff_target": "dev_os",
        "child_card_index": "2",
        "route_reason": "Default rejection indicates implementation mismatch that DevOS must correct.",
    },
    "spec_or_design_mismatch": {
        "route_target": "SPEC_REWORK",
        "next_handoff_target": "architect_os",
        "child_card_index": "1",
        "route_reason": "Default rejection indicates a spec or design mismatch that ArchitectOS must correct.",
    },
}

# STATE_OWNERS maps each state to the profile permitted to advance the
# order out of that state. It is transition-validation ownership for the
# current state, not a record of which profile originally produced it.
STATE_OWNERS: dict[str, str] = {
    "BRIEF_DRAFTED":                "hermes",
    "ACTION_APPROVED":              "hermes",
    "PRODUCTION_ORDER_CREATED":     "orchestrator_os",
    "ORCHESTRATOR_TRIAGE":          "orchestrator_os",
    "ARCHITECT_SPEC":               "architect_os",
    "ARCHITECT_READY_FOR_DEV":      "dev_os",
    "DEV_IMPLEMENTING":             "dev_os",
    "DEV_COMPLETE":                 "audit_os",
    "AUDIT_REVIEW":                 "audit_os",
    "AUDIT_REJECTED":               "orchestrator_os",
    "AUDIT_PASSED":                 "architect_os",
    "ARCHITECT_RECONCILE":          "architect_os",
    "ARCHITECT_ACCEPTED":           "default",
    "DEFAULT_FINAL_REVIEW":         "default",
    "DEFAULT_REJECTED":             "orchestrator_os",
    "BLOCKED_NEEDS_JARREN":         "default",
    "DONE":                         "default",
    "DEV_REWORK":                   "dev_os",
    "SPEC_REWORK":                  "architect_os",
}

WORKFLOW_INITIAL_STATE = "PRODUCTION_ORDER_CREATED"

# Handoff from spec section 9. The bridge creates only the first handoff
# (from Default Hermes to OrchestratorOS). All subsequent handoffs are
# agent-level (spec section 17: "LLM decides judgment").
ORCHESTRATOR_HANDOFF_TEMPLATE: dict[str, str] = {
    "from_profile": "hermes",
    "to_profile": "orchestrator_os",
    "current_state": WORKFLOW_INITIAL_STATE,
    "requested_next_state": "ORCHESTRATOR_TRIAGE",
    "objective": (
        "Triage the approved brief, verify the Kanban graph, "
        "and create the first downstream handoff"
    ),
    "expected_output": (
        "Triage classification + ArchitectOS handoff packet"
    ),
    "acceptance_criteria": (
        "Correct workflow path chosen, ArchitectOS card assigned `ready`"
    ),
    "source_truth": WORKFLOW_SPEC_SOURCE,
    "stop_conditions": (
        "From spec section 12: missing source truth, unclear owner, "
        "ambiguous scope, unapproved external action, approval boundary hit, "
        "or task belongs to a lighter layer."
    ),
    "approval_required_before": (
        "From spec section 11: spending, publishing, destructive changes, "
        "new profiles, permission widening, secret access, external services, "
        "scheduled automation, or scope expansion."
    ),
    "evidence_required": "Triage classification result",
}

ARCHITECT_HANDOFF_TEMPLATE: dict[str, str] = {
    "from_profile": "orchestrator_os",
    "to_profile": "architect_os",
    "current_state": "ORCHESTRATOR_TRIAGE",
    "requested_next_state": "ARCHITECT_SPEC",
    "objective": (
        "Produce the bounded ArchitectOS spec and DevOS handoff packet "
        "from the frozen production-order brief"
    ),
    "expected_output": (
        "ArchitectOS spec, contracts, quality gates, and DevOS handoff packet"
    ),
    "acceptance_criteria": (
        "Spec is bounded to the approved brief, preserves non-goals, and "
        "names verification requirements before DevOS implementation"
    ),
    "source_truth": WORKFLOW_SPEC_SOURCE,
    "stop_conditions": (
        "Missing source truth, ambiguous product behavior, contract/schema "
        "decision not present in the frozen brief, approval boundary hit, "
        "or scope expansion beyond the production-order metadata."
    ),
    "approval_required_before": (
        "Spending, publishing, destructive changes, new profiles, permission "
        "widening, secret access, external services, scheduled automation, "
        "or scope expansion."
    ),
    "evidence_required": "ArchitectOS spec artifact and DevOS handoff packet",
}

# Approval phrases from spec section 3
APPROVAL_PHRASES = [
    "action approved for this brief",
    "approved. execute this brief",
    "send this into the production workflow",
    "proceed with this approved scope",
]

# Required brief fields from the handoff skill Step 2 checklist
REQUIRED_BRIEF_FIELDS = {
    "objective",
    "target repo or workspace",
    "scope",
    "out of scope",
    "acceptance criteria",
    "stop conditions",
    "approval boundaries",
    "constraints",
    "expected output",
}

BRIEF_FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "target repo or workspace": ("target_repo_or_workspace",),
    "out of scope": ("out_of_scope",),
    "acceptance criteria": ("acceptance_criteria",),
    "stop conditions": ("stop_conditions",),
    "approval boundaries": ("approval_boundaries",),
    "expected output": ("expected_output",),
}

DEVOS_HANDOFF_TEMPLATE: dict[str, Any] = {
    "from_profile": "architect_os",
    "to_profile": "dev_os",
    "current_state": "ARCHITECT_SPEC",
    "requested_next_state": "ARCHITECT_READY_FOR_DEV",
    "source_truth": [WORKFLOW_SPEC_SOURCE],
    "approval_boundaries": [],
    "artifact_references": [],
}

REQUIRED_ARCHITECT_SPEC_PACKET_FIELDS = {
    "production_order_id",
    "stage",
    "owner_profile",
    "objective",
    "source_truth",
    "scope",
    "out_of_scope",
    "acceptance_criteria",
    "devos_task",
    "files_or_areas_allowed",
    "stop_conditions",
    "next_state",
}

DEVOS_BUILD_RESULT_TEMPLATE: dict[str, Any] = {
    "from_profile": "dev_os",
    "to_profile": "audit_os",
    "from_state": "DEV_IMPLEMENTING",
    "to_state": "DEV_COMPLETE",
    "reference_workflow_spec": WORKFLOW_SPEC_SOURCE,
}

REQUIRED_DEVOS_BUILD_PACKET_FIELDS = {
    "production_order_id",
    "owner_profile",
    "source_state",
    "summary",
    "tests_run",
    "test_status",
    "limitations_or_notes",
    "next_handoff_target",
}

ARCHITECT_RECONCILE_HANDOFF_TEMPLATE: dict[str, Any] = {
    "from_profile": "audit_os",
    "to_profile": "architect_os",
    "current_state": "AUDIT_PASSED",
    "requested_next_state": "ARCHITECT_RECONCILE",
    "reference_workflow_spec": WORKFLOW_SPEC_SOURCE,
}

REQUIRED_AUDITOS_REVIEW_PACKET_FIELDS = {
    "production_order_id",
    "owner_profile",
    "source_state",
    "review_result",
    "summary",
    "evidence",
    "tests_reviewed",
    "verdict",
    "risks_or_notes",
    "next_handoff_target",
}

REQUIRED_AUDITOS_REJECTION_PACKET_FIELDS = REQUIRED_AUDITOS_REVIEW_PACKET_FIELDS | {
    "correction_request",
}

DEFAULT_FINAL_REVIEW_HANDOFF_TEMPLATE: dict[str, Any] = {
    "from_profile": "architect_os",
    "to_profile": "default",
    "current_state": "ARCHITECT_ACCEPTED",
    "requested_next_state": "DEFAULT_FINAL_REVIEW",
    "reference_workflow_spec": WORKFLOW_SPEC_SOURCE,
}

REQUIRED_ARCHITECT_RECONCILE_PACKET_FIELDS = {
    "production_order_id",
    "owner_profile",
    "source_state",
    "reconcile_result",
    "summary",
    "architecture_alignment",
    "drift_assessment",
    "spec_patch_needed",
    "risks_or_notes",
    "next_handoff_target",
}

REQUIRED_DEFAULT_FINAL_REVIEW_PACKET_FIELDS = {
    "production_order_id",
    "owner_profile",
    "source_state",
    "final_review_result",
    "summary",
    "original_brief_alignment",
    "artifacts_reviewed",
    "evidence_summary",
    "final_status",
    "next_action",
}

REQUIRED_DEFAULT_REJECTION_PACKET_FIELDS = {
    "production_order_id",
    "owner_profile",
    "source_state",
    "review_result",
    "summary",
    "original_brief_mismatch",
    "rejection_reason",
    "evidence",
    "recommended_route",
    "next_handoff_target",
}

REQUIRED_ORCHESTRATOR_CLASSIFICATION_PACKET_FIELDS = {
    "production_order_id",
    "owner_profile",
    "source_state",
    "default_rejection_reason",
    "classification",
    "route_target",
    "route_reason",
    "next_handoff_target",
    "correction_request",
}

DEFAULT_REJECTION_HANDOFF_TEMPLATE: dict[str, Any] = {
    "from_profile": "default",
    "to_profile": "orchestrator_os",
    "current_state": "DEFAULT_REJECTED",
    "requested_next_state": "ORCHESTRATOR_TRIAGE",
    "reference_workflow_spec": WORKFLOW_SPEC_SOURCE,
}

ORCHESTRATOR_CLASSIFICATION_HANDOFF_TEMPLATE: dict[str, Any] = {
    "from_profile": "orchestrator_os",
    "current_state": "ORCHESTRATOR_TRIAGE",
    "reference_workflow_spec": WORKFLOW_SPEC_SOURCE,
}

# Child card definitions for production Kanban graph (from spec section 7 + feature brief section 8)
CHILD_CARD_DEFS = [
    (1, "OrchestratorOS - Triage + Graph",       "orchestrator_os", "ready"),
    (2, "ArchitectOS - Spec + Architecture",      "architect_os",   "todo"),
    (3, "DevOS - Build + Tests",                  "dev_os",         "todo"),
    (4, "AuditOS - Review + Evidence",            "audit_os",       "todo"),
    (5, "ArchitectOS - Reconcile",                "architect_os",   "todo"),
    (6, "Default Hermes - Final Review",          "hermes",         "todo"),
]


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class StageEntry:
    """A single entry in the production order's state transition history."""
    from_state: str
    to_state: str
    owner_profile: str
    timestamp: int


@dataclass
class ProductionOrder:
    """In-memory runtime view of a production order.

    NOT stored in a separate SQLite table in Slice 4 (avoids dual-storage
    drift). Reconstructable from:

    - The parent Kanban card body (source_brief, metadata)
    - The child card assignments via ``task_links``
    - The ``production_order_events`` table for state history
    - The ``current_state`` field on the parent card
    """
    production_order_id: str
    title: str
    source_brief: str
    approved_by: str
    approved_at: int
    priority_lane: str
    repo_or_workspace: str
    current_state: str
    current_owner_profile: str
    stage_history: list[StageEntry] = field(default_factory=list)
    parent_kanban_card_id: str = ""
    child_kanban_card_ids: list[str] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)
    approval_boundaries: list[str] = field(default_factory=list)
    final_status: Optional[str] = None


# ---------------------------------------------------------------------------
# ID Generation
# ---------------------------------------------------------------------------


def generate_production_order_id() -> str:
    """Generate a unique production order ID.

    Format: ``PO-YYYYMMDD-XXXX`` where XXXX is 4 base36 random characters
    (36^4 ≈ 1.7M possibilities per day — collision probability at 1k/day
    is ~3e-7).
    """
    date_part = datetime.now(timezone.utc).strftime("%Y%m%d")
    # 4 base36 chars = 36^4 ≈ 1.7M permutations
    random_part = _base36_random(4)
    return f"PO-{date_part}-{random_part}"


def _base36_random(length: int) -> str:
    """Return a random base36 string of ``length`` characters."""
    chars = "0123456789abcdefghijklmnopqrstuvwxyz"
    return "".join(secrets.choice(chars) for _ in range(length))


# ---------------------------------------------------------------------------
# Approval Detection
# ---------------------------------------------------------------------------


def detect_approval_phrase(text: str) -> Optional[str]:
    """Return ``'approved'`` if ``text`` contains an explicit approval phrase.

    Returns ``None`` if no phrase matches. Only the 4 phrases from spec
    section 3 are accepted. Bare ``approved``, ``looks good``, ``go ahead``,
    and ``sounds right`` are deliberately excluded.
    """
    text_lower = text.strip().lower()
    for phrase in APPROVAL_PHRASES:
        if phrase in text_lower:
            return "approved"
    return None


def validate_brief(brief: dict[str, Any]) -> list[str]:
    """Validate a brief dict has all required fields.

    Returns a list of missing field names (empty list = valid). Supports both
    canonical human-label keys and backward-compatible snake_case aliases.
    """
    missing: list[str] = []
    for field_name in REQUIRED_BRIEF_FIELDS:
        value = get_brief_value(brief, field_name)
        if value is None or (isinstance(value, str) and not value.strip()):
            missing.append(field_name)
    return missing


def get_brief_value(
    brief: dict[str, Any],
    canonical_key: str,
    default: Any = "",
) -> Any:
    """Return a brief value by canonical human label or snake_case alias."""
    keys = (canonical_key, *BRIEF_FIELD_ALIASES.get(canonical_key, ()))
    for key in keys:
        if key in brief:
            return brief[key]
    return default


def _parse_source_brief(source_brief: str) -> dict[str, Any]:
    """Best-effort JSON parse for frozen brief metadata."""
    if not source_brief:
        return {}
    try:
        parsed = json.loads(source_brief)
    except (TypeError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _reconstruct_stage_history(
    conn: sqlite3.Connection,
    production_order_id: str,
) -> list[StageEntry]:
    """Derive stage history from production_order_events."""
    rows = conn.execute(
        "SELECT timestamp, from_state, to_state, owner_profile "
        "FROM production_order_events "
        "WHERE production_order_id = ? "
        "AND event_type NOT IN ("
        "'dispatch_planned','dispatch_started','dispatch_handoff_created',"
        "'dispatch_completed','dispatch_failed','dispatch_blocked',"
        "'packet_validated','packet_rejected'"
        ") "
        "AND (to_state IS NOT NULL OR from_state IS NOT NULL) "
        "ORDER BY id",
        (production_order_id,),
    ).fetchall()
    return [
        StageEntry(
            row["from_state"] or "",
            row["to_state"] or "",
            row["owner_profile"] or "",
            row["timestamp"] or 0,
        )
        for row in rows
        if row["from_state"] or row["to_state"]
    ]


def _reconstruct_child_card_ids(
    conn: sqlite3.Connection,
    parent_kanban_card_id: str,
) -> list[str]:
    """Read child production-order card IDs from task_links in stage order."""
    stage_order_cases = []
    for idx, (_, title, _, _) in enumerate(CHILD_CARD_DEFS, 1):
        escaped_title = title.replace("'", "''")
        stage_order_cases.append(f"WHEN '{escaped_title}' THEN {idx}")
    stage_order_sql = " ".join(stage_order_cases)
    rows = conn.execute(
        f"""
        SELECT l.child_id
        FROM task_links l
        LEFT JOIN tasks t ON t.id = l.child_id
        WHERE l.parent_id = ?
        ORDER BY CASE t.title {stage_order_sql} ELSE 999 END, t.created_at, l.child_id
        """,
        (parent_kanban_card_id,),
    ).fetchall()
    return [row["child_id"] for row in rows]


def _reconstruct_production_order(
    conn: sqlite3.Connection,
    row: sqlite3.Row,
) -> ProductionOrder:
    """Reconstruct a trustworthy runtime view from Kanban + event rows."""
    po_id = row["production_order_id"]
    state = row["current_state"] or WORKFLOW_INITIAL_STATE
    source_brief = row["body"] or ""
    brief = _parse_source_brief(source_brief)
    repo_or_workspace = get_brief_value(brief, "target repo or workspace", "")
    return ProductionOrder(
        production_order_id=po_id,
        title=row["title"].replace("Production Order: ", "", 1),
        source_brief=source_brief,
        approved_by=row["created_by"] or "",
        approved_at=row["created_at"] or 0,
        priority_lane=brief.get("priority_lane", ""),
        repo_or_workspace=repo_or_workspace or "",
        current_state=state,
        current_owner_profile=STATE_OWNERS.get(state, ""),
        stage_history=_reconstruct_stage_history(conn, po_id),
        parent_kanban_card_id=row["id"],
        child_kanban_card_ids=_reconstruct_child_card_ids(conn, row["id"]),
    )


# ---------------------------------------------------------------------------
# State Machine
# ---------------------------------------------------------------------------


class StateTransitionError(ValueError):
    """Raised when a state transition is invalid."""


def validate_state_transition(
    from_state: str,
    to_state: str,
    calling_profile: str,
) -> bool:
    """Validate a state transition.

    Checks:
    1. ``from_state`` exists in ``VALID_TRANSITIONS``
    2. ``to_state`` is in the allowed set for ``from_state``
    3. ``calling_profile`` matches the profile permitted to advance from
       ``from_state``

    Returns ``True`` or raises :class:`StateTransitionError`.
    """
    if not from_state:
        raise ValueError("from_state is required")
    if not to_state:
        raise ValueError("to_state is required")
    if not calling_profile:
        raise ValueError("calling_profile is required")

    allowed = VALID_TRANSITIONS.get(from_state)
    if allowed is None:
        raise StateTransitionError(
            f"Unknown from_state {from_state!r}: "
            f"no transitions defined for this state"
        )
    if to_state not in allowed:
        raise StateTransitionError(
            f"Transition {from_state!r} → {to_state!r} is not valid. "
            f"Allowed targets from {from_state!r}: {sorted(allowed)}"
        )
    owner = STATE_OWNERS.get(from_state)
    if owner is None:
        raise StateTransitionError(
            f"No owner defined for state {from_state!r}"
        )
    if calling_profile != owner:
        raise StateTransitionError(
            f"Profile {calling_profile!r} does not own state {from_state!r}. "
            f"Owner is {owner!r}"
        )

    return True


# ---------------------------------------------------------------------------
# Event Logging
# ---------------------------------------------------------------------------


def log_workflow_event(
    conn: sqlite3.Connection,
    production_order_id: str,
    event_type: str,
    *,
    from_state: Optional[str] = None,
    to_state: Optional[str] = None,
    owner_profile: Optional[str] = None,
    target_profile: Optional[str] = None,
    kanban_card_id: Optional[str] = None,
    packet_id: Optional[str] = None,
    result: Optional[str] = None,
    error: Optional[str] = None,
    next_action: Optional[str] = None,
) -> int:
    """Write an event to the ``production_order_events`` table.

    Returns the new event row id. Raises :class:`ValueError` if
    ``production_order_id`` or ``event_type`` is empty.
    """
    if not production_order_id:
        raise ValueError("production_order_id is required for event logging")
    if not event_type:
        raise ValueError("event_type is required for event logging")

    now = int(time.time())
    cur = conn.execute(
        """\
INSERT INTO production_order_events (
    production_order_id, timestamp, event_type,
    from_state, to_state, owner_profile, target_profile,
    kanban_card_id, packet_id, result, error, next_action
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            production_order_id,
            now,
            event_type,
            from_state,
            to_state,
            owner_profile,
            target_profile,
            kanban_card_id,
            packet_id,
            result,
            error,
            next_action,
        ),
    )
    return cur.lastrowid


# ---------------------------------------------------------------------------
# Production Order Creation
# ---------------------------------------------------------------------------


def create_production_order(
    conn: sqlite3.Connection,
    *,
    title: str,
    source_brief: str,
    approved_by: str = "Jarren",
    priority_lane: str = "Hermes OS",
    repo_or_workspace: str = "",
    idempotency_key: Optional[str] = None,
    board: Optional[str] = None,
) -> ProductionOrder:
    """Create a production order and its parent Kanban card.

    Steps:
    1. Generate unique ``production_order_id``
    2. Create the **parent Kanban card** with:
       - ``body`` = full frozen ``source_brief``
       - ``production_order_id`` set
       - ``current_state`` = ``PRODUCTION_ORDER_CREATED`` (via transition below)
       - ``workflow_template_id`` = ``hermes-production-workflow-v1``
    3. Log ``production_order_created`` event
    4. Transition state from ``BRIEF_DRAFTED`` → ``ACTION_APPROVED``
       and then to ``PRODUCTION_ORDER_CREATED``

    Returns the :class:`ProductionOrder` instance.
    """
    po_id = generate_production_order_id()
    approved_at = int(time.time())

    # Idempotency: if the caller provides an idempotency_key and a task
    # with that key already exists, return the existing order.
    existing = _find_existing_order(conn, idempotency_key) if idempotency_key else None
    if existing is not None:
        return existing

    # Create the parent Kanban card with the frozen brief in the body.
    parent_id = kb.create_task(
        conn,
        title=f"Production Order: {title}",
        body=source_brief,
        created_by=approved_by,
        initial_status="running",
        idempotency_key=idempotency_key,
        board=board,
    )

    # Set production-order fields on parent card.
    conn.execute(
        "UPDATE tasks SET production_order_id = ?, current_state = ?, "
        "workflow_template_id = ? WHERE id = ?",
        (po_id, WORKFLOW_INITIAL_STATE, WORKFLOW_TEMPLATE_ID, parent_id),
    )

    # Build the ProductionOrder object.
    po = ProductionOrder(
        production_order_id=po_id,
        title=title,
        source_brief=source_brief,
        approved_by=approved_by,
        approved_at=approved_at,
        priority_lane=priority_lane,
        repo_or_workspace=repo_or_workspace,
        current_state=WORKFLOW_INITIAL_STATE,
        current_owner_profile="orchestrator_os",
        stage_history=[
            StageEntry("BRIEF_DRAFTED", "ACTION_APPROVED", "hermes", approved_at),
            StageEntry("ACTION_APPROVED", WORKFLOW_INITIAL_STATE, "hermes", approved_at),
        ],
        parent_kanban_card_id=parent_id,
    )

    # Log events.
    log_workflow_event(
        conn, po_id, "brief_approved",
        to_state="ACTION_APPROVED",
        owner_profile="hermes",
        kanban_card_id=parent_id,
        next_action="create_production_order",
    )
    log_workflow_event(
        conn, po_id, "production_order_created",
        from_state="ACTION_APPROVED",
        to_state=WORKFLOW_INITIAL_STATE,
        owner_profile="hermes",
        kanban_card_id=parent_id,
        next_action="create_kanban_graph",
    )

    return po


def _find_existing_order(
    conn: sqlite3.Connection,
    idempotency_key: str,
) -> Optional[ProductionOrder]:
    """If a non-archived task with this idempotency key exists and has a
    ``production_order_id`` set, reconstruct and return the
    :class:`ProductionOrder`.  Returns ``None`` otherwise.
    """
    row = conn.execute(
        "SELECT id, title, body, production_order_id, current_state, "
        "created_by, created_at "
        "FROM tasks WHERE idempotency_key = ? "
        "AND status != 'archived' AND production_order_id IS NOT NULL "
        "ORDER BY created_at DESC LIMIT 1",
        (idempotency_key,),
    ).fetchone()
    if row is None:
        return None
    return _reconstruct_production_order(conn, row)


# ---------------------------------------------------------------------------
# Kanban Graph Creation
# ---------------------------------------------------------------------------


def create_production_kanban_graph(
    conn: sqlite3.Connection,
    po: ProductionOrder,
    *,
    board: Optional[str] = None,
) -> list[str]:
    """Create the 6 child Kanban cards for a production order.

    Each child card:
    - Has ``production_order_id`` and ``current_state`` set
    - Has ``workflow_template_id = hermes-production-workflow-v1``
    - Is linked to the parent via ``link_tasks()``
    - Has the correct ``assignee`` (owner profile)

    Status handling:
    - OrchestratorOS (card 1) starts as ``ready``
    - All other cards start as ``todo`` (demoted by parent link)
    - ``create_task()`` only accepts ``running``/``blocked`` as initial_status,
      so we call with ``initial_status='running'`` and rely on auto-status
      resolution: no pre-existing parents ⇒ status becomes ``ready``. After
      linking to the parent (which is not ``done``), ``link_tasks()`` demotes
      to ``todo``. We then restore card 1 (OrchestratorOS) to ``ready``.

    Returns the list of child card IDs in stage order.
    """
    child_ids: list[str] = []
    for order, title, owner, required_status in CHILD_CARD_DEFS:
        child_id = kb.create_task(
            conn,
            title=title,
            assignee=owner,
            body=_child_card_body_json(po, title, owner),
            created_by=po.approved_by,
            initial_status="running",
            board=board,
        )
        # Set production-order fields on child card.
        conn.execute(
            "UPDATE tasks SET production_order_id = ?, current_state = ?, "
            "workflow_template_id = ? WHERE id = ?",
            (po.production_order_id, po.current_state, WORKFLOW_TEMPLATE_ID, child_id),
        )
        # Link child to parent — this demotes to 'todo' since parent isn't done.
        kb.link_tasks(conn, po.parent_kanban_card_id, child_id)
        child_ids.append(child_id)

    # Restore card 1 (OrchestratorOS) to 'ready' if the brief says so.
    # CHILD_CARD_DEFS[0][3] is 'ready' — promote back after link_tasks demotion.
    if child_ids and CHILD_CARD_DEFS[0][3] == "ready":
        conn.execute(
            "UPDATE tasks SET status = 'ready' WHERE id = ?",
            (child_ids[0],),
        )

    po.child_kanban_card_ids = child_ids

    # Log kanban_graph_created event.
    log_workflow_event(
        conn, po.production_order_id, "kanban_graph_created",
        to_state=po.current_state,
        owner_profile="hermes",
        kanban_card_id=po.parent_kanban_card_id,
        result=f"Created {len(child_ids)} child cards",
        next_action="create_orchestrator_handoff",
    )

    return child_ids


def _child_card_body_json(
    po: ProductionOrder,
    card_title: str,
    owner_profile: str,
) -> str:
    """Build the structured body JSON for a child Kanban card.

    Format from feature brief section 8.
    """
    payload: dict[str, Any] = {
        "production_order_id": po.production_order_id,
        "parent_brief": po.title,
        "owner_profile": owner_profile,
        "current_state": po.current_state,
        "input_packet": f"Parent card: {po.parent_kanban_card_id}",
        "expected_output": f"Stage: {card_title}",
        "acceptance_criteria": "See spec section 8 stage contract",
        "stop_conditions": "See spec section 12",
        "artifact_links": [],
        "evidence_links": [],
        "next_transition": _next_transition_for_card(card_title),
    }
    return json.dumps(payload, indent=2)


def _next_transition_for_card(card_title: str) -> str:
    """Return the expected next state transition for a child card."""
    mapping: dict[str, str] = {
        "OrchestratorOS": "ORCHESTRATOR_TRIAGE",
        "ArchitectOS - Spec": "ARCHITECT_SPEC",
        "ArchitectOS - Reconcile": "ARCHITECT_RECONCILE",
        "DevOS": "DEV_IMPLEMENTING",
        "AuditOS": "AUDIT_REVIEW",
        "Default Hermes": "DEFAULT_FINAL_REVIEW",
    }
    for key, transition in mapping.items():
        if key in card_title:
            return transition
    return "PRODUCTION_ORDER_CREATED"


# ---------------------------------------------------------------------------
# State Transition
# ---------------------------------------------------------------------------


def transition_state(
    conn: sqlite3.Connection,
    po: ProductionOrder,
    to_state: str,
    calling_profile: str,
    *,
    result: str = "done",
    error: Optional[str] = None,
    next_action: Optional[str] = None,
    card_id: Optional[str] = None,
    event_type: str = "state_transitioned",
) -> str:
    """Validate and execute a state transition on ``po``.

    Steps:
    1. Call ``validate_state_transition()``
    2. Log ``state_transitioned`` event
    3. Update ``po.current_state``
    4. Update ``po.current_owner_profile``
    5. Append to ``po.stage_history``
    6. Write the new ``current_state`` on the parent Kanban card
    7. Return the new state

    Note: this mutates ``po`` in place.
    """
    from_state = po.current_state
    validate_state_transition(from_state, to_state, calling_profile)
    now = int(time.time())

    # Update parent card in Kanban.
    target_card_id = card_id or po.parent_kanban_card_id
    conn.execute(
        "UPDATE tasks SET current_state = ? WHERE id = ?",
        (to_state, target_card_id),
    )

    # Log event.
    log_workflow_event(
        conn, po.production_order_id, event_type,
        from_state=from_state,
        to_state=to_state,
        owner_profile=calling_profile,
        kanban_card_id=target_card_id,
        result=result,
        error=error,
        next_action=next_action or "",
    )

    # Update PO object.
    po.current_state = to_state
    po.current_owner_profile = STATE_OWNERS.get(to_state, calling_profile)
    po.stage_history.append(StageEntry(from_state, to_state, calling_profile, now))

    return to_state


# ---------------------------------------------------------------------------
# Handoff Packet
# ---------------------------------------------------------------------------


def create_orchestrator_handoff(
    po: ProductionOrder,
    *,
    context: Optional[str] = None,
    scope: str = "",
    out_of_scope: str = "",
    inputs: str = "",
) -> dict[str, str]:
    """Create the first handoff packet from Default Hermes to OrchestratorOS.

    Returns the handoff dict matching the spec section 9 schema.

    ``context`` defaults to the frozen brief title. ``scope`` and
    ``out_of_scope`` are meant to be extracted from the frozen brief.
    """
    packet = dict(ORCHESTRATOR_HANDOFF_TEMPLATE)
    packet["production_order_id"] = po.production_order_id
    packet["context"] = context or f"Frozen brief: {po.title}"
    packet["scope"] = scope or "From the frozen brief"
    packet["out_of_scope"] = out_of_scope or "From the frozen brief"
    packet["inputs"] = (
        inputs
        or f"Parent card ID: {po.parent_kanban_card_id}, "
        f"Child card IDs: {', '.join(po.child_kanban_card_ids)}, "
        f"Frozen brief: {po.title}"
    )
    return packet


def create_architect_handoff(po: ProductionOrder) -> dict[str, str]:
    """Create the Slice 5 handoff packet from OrchestratorOS to ArchitectOS.

    The packet is deterministic and is derived only from the reconstructed
    production order plus the frozen brief metadata stored on the parent card.
    """
    brief = _parse_source_brief(po.source_brief)
    packet: dict[str, Any] = dict(ARCHITECT_HANDOFF_TEMPLATE)
    packet["production_order_id"] = po.production_order_id
    packet["context"] = (
        f"Frozen production-order brief: {po.title}; "
        f"parent card: {po.parent_kanban_card_id}"
    )
    packet["scope"] = str(get_brief_value(brief, "scope", "From the frozen brief"))
    packet["out_of_scope"] = str(
        get_brief_value(brief, "out of scope", "From the frozen brief")
    )
    packet["inputs"] = (
        f"Parent card ID: {po.parent_kanban_card_id}; "
        f"Child card IDs: {', '.join(po.child_kanban_card_ids)}; "
        f"Repo/workspace: {po.repo_or_workspace or get_brief_value(brief, 'target repo or workspace', '')}; "
        "Frozen brief metadata is stored on the parent card body."
    )
    return packet


def validate_architect_spec_packet(
    packet: dict[str, Any],
    *,
    expected_production_order_id: str,
) -> dict[str, Any]:
    """Validate the deterministic ArchitectOS spec-completion packet.

    The runtime does not invent ArchitectOS reasoning. The packet must be
    provided by the caller and contain the minimum fields required to prepare
    the DevOS handoff without changing workflow contracts.
    """
    if not isinstance(packet, dict):
        raise ValueError("architect spec packet must be a JSON object")

    missing = sorted(
        field for field in REQUIRED_ARCHITECT_SPEC_PACKET_FIELDS
        if field not in packet or packet[field] in (None, "", [], {})
    )
    if missing:
        raise ValueError(
            "architect spec packet missing required field(s): "
            + ", ".join(missing)
        )

    if packet["production_order_id"] != expected_production_order_id:
        raise ValueError(
            "architect spec packet production_order_id does not match the "
            "requested production order"
        )
    if str(packet["stage"]).lower() != "architect_spec":
        raise ValueError("architect spec packet stage must be 'architect_spec'")
    if packet["owner_profile"] != "architect_os":
        raise ValueError("architect spec packet owner_profile must be 'architect_os'")
    if packet["next_state"] != "ARCHITECT_READY_FOR_DEV":
        raise ValueError(
            "architect spec packet next_state must be 'ARCHITECT_READY_FOR_DEV'"
        )

    source_truth = packet["source_truth"]
    if isinstance(source_truth, str):
        source_truth_values = {source_truth}
    else:
        try:
            source_truth_values = {str(value) for value in source_truth}
        except TypeError as exc:
            raise ValueError(
                "architect spec packet source_truth must be a string or list"
            ) from exc
    if WORKFLOW_SPEC_SOURCE not in source_truth_values:
        raise ValueError(
            "architect spec packet source_truth must include the canonical workflow spec: "
            f"{WORKFLOW_SPEC_SOURCE}"
        )

    return packet


def create_devos_handoff(
    po: ProductionOrder,
    architect_packet: dict[str, Any],
) -> dict[str, Any]:
    """Create the deterministic DevOS handoff packet from ArchitectOS output."""
    packet: dict[str, Any] = dict(DEVOS_HANDOFF_TEMPLATE)
    packet["production_order_id"] = po.production_order_id
    packet["objective"] = architect_packet["objective"]
    packet["source_truth"] = architect_packet["source_truth"]
    packet["scope"] = architect_packet["scope"]
    packet["out_of_scope"] = architect_packet["out_of_scope"]
    packet["acceptance_criteria"] = architect_packet["acceptance_criteria"]
    packet["devos_task"] = architect_packet["devos_task"]
    packet["allowed_files_or_areas"] = architect_packet["files_or_areas_allowed"]
    packet["stop_conditions"] = architect_packet["stop_conditions"]
    packet["approval_boundaries"] = architect_packet.get("approval_boundaries", [])
    packet["artifact_references"] = architect_packet.get("artifact_references", [])
    return packet


def _require_non_empty_field(
    packet: dict[str, Any],
    field_name: str,
    *,
    aliases: tuple[str, ...] = (),
    label: str = "DevOS build packet",
) -> Any:
    for key in (field_name, *aliases):
        if key in packet and packet[key] not in (None, "", [], {}):
            return packet[key]
    alias_text = f" (or {', '.join(aliases)})" if aliases else ""
    raise ValueError(
        f"{label} missing required field: {field_name}{alias_text}"
    )


def _missing_required_fields(
    packet: dict[str, Any],
    required_fields: set[str],
) -> list[str]:
    return sorted(
        field for field in required_fields
        if field not in packet or packet[field] in (None, "", [], {})
    )


def _coerce_packet_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip().lower()
    return json.dumps(value, sort_keys=True, default=str).lower()


def _require_positive_result(
    value: Any,
    *,
    field_name: str,
    label: str,
    positive_tokens: tuple[str, ...],
) -> None:
    text = _coerce_packet_text(value)
    if not text:
        raise ValueError(f"{label} {field_name} must be non-empty")
    negative_tokens = (
        "fail",
        "failed",
        "reject",
        "rejected",
        "blocked",
        "partial",
        "needs_approval",
        "needs approval",
        "cancelled",
        "canceled",
        "rework",
    )
    if any(token in text for token in negative_tokens):
        raise ValueError(f"{label} {field_name} is not a happy-path result")
    if not any(token in text for token in positive_tokens):
        raise ValueError(
            f"{label} {field_name} must clearly indicate a happy-path result"
        )


def _require_negative_result(
    value: Any,
    *,
    field_name: str,
    label: str,
    negative_tokens: tuple[str, ...],
) -> None:
    text = _coerce_packet_text(value)
    if not text:
        raise ValueError(f"{label} {field_name} must be non-empty")
    if not any(token in text for token in negative_tokens):
        raise ValueError(
            f"{label} {field_name} must clearly indicate a reject/rework result"
        )


def _normalize_test_status(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("DevOS build packet test_status must be a non-empty string")
    return value.strip().lower()


def validate_devos_build_packet(
    packet: dict[str, Any],
    *,
    expected_production_order_id: str,
    expected_source_state: str = "ARCHITECT_READY_FOR_DEV",
    expected_next_handoff_target: str = "audit_os",
) -> dict[str, Any]:
    """Validate the explicit DevOS build/result packet for Slice 7."""
    if not isinstance(packet, dict):
        raise ValueError("DevOS build packet must be a JSON object")

    missing = sorted(
        field for field in REQUIRED_DEVOS_BUILD_PACKET_FIELDS
        if field not in packet or packet[field] in (None, "", [], {})
    )
    if missing:
        raise ValueError(
            "DevOS build packet missing required field(s): " + ", ".join(missing)
        )

    if "result_type" not in packet and "stage_result" not in packet:
        raise ValueError("DevOS build packet requires result_type or stage_result")
    if "files_changed" not in packet and "implementation_artifacts" not in packet:
        raise ValueError(
            "DevOS build packet requires files_changed or implementation_artifacts"
        )

    if packet["production_order_id"] != expected_production_order_id:
        raise ValueError(
            "DevOS build packet production_order_id does not match the requested production order"
        )
    if packet["owner_profile"] != "dev_os":
        raise ValueError("DevOS build packet owner_profile must be 'dev_os'")
    if packet["source_state"] != expected_source_state:
        raise ValueError(
            f"DevOS build packet source_state must be {expected_source_state!r}"
        )
    if packet["next_handoff_target"] != expected_next_handoff_target:
        raise ValueError(
            f"DevOS build packet next_handoff_target must be {expected_next_handoff_target!r}"
        )

    test_status = _normalize_test_status(packet["test_status"])
    if not any(token in test_status for token in ("pass", "green", "success")):
        raise ValueError(
            "DevOS build packet test_status must clearly indicate pass/green/success"
        )
    if any(token in test_status for token in ("fail", "failed", "red", "error")):
        raise ValueError(
            "DevOS build packet test_status must not indicate failure/red/error"
        )

    _require_non_empty_field(packet, "result_type", aliases=("stage_result",))
    _require_non_empty_field(
        packet,
        "files_changed",
        aliases=("implementation_artifacts",),
    )

    return packet


def create_auditos_handoff(
    po: ProductionOrder,
    devos_packet: dict[str, Any],
    *,
    from_state: str = "DEV_IMPLEMENTING",
    to_state: str = "DEV_COMPLETE",
) -> dict[str, Any]:
    """Create the deterministic AuditOS handoff packet from DevOS output."""
    packet = dict(DEVOS_BUILD_RESULT_TEMPLATE)
    packet["from_state"] = from_state
    packet["to_state"] = to_state
    packet["production_order_id"] = po.production_order_id
    packet["devos_summary"] = devos_packet["summary"]
    packet["implementation_artifacts"] = devos_packet.get(
        "files_changed",
        devos_packet.get("implementation_artifacts", []),
    )
    packet["tests_and_evidence"] = {
        "tests_run": devos_packet["tests_run"],
        "test_status": devos_packet["test_status"],
    }
    packet["acceptance_criteria"] = [
        "Verify the implementation matches the approved brief and ArchitectOS spec.",
        "Verify the supplied test evidence is credible and green.",
        "Reject if scope widened beyond the approved production-order brief.",
    ]
    packet["stop_conditions"] = [
        "Evidence is missing or not credible.",
        "Implementation exceeds approved scope.",
        "Audit requires unavailable environment access.",
    ]
    packet["limitations_or_notes"] = devos_packet["limitations_or_notes"]
    return packet


def create_devos_rework_handoff(
    po: ProductionOrder,
    rejection_packet: dict[str, Any],
    *,
    source_state: str = "AUDIT_REJECTED",
    route_reason: str | None = None,
    route_target: str = "DEV_REWORK",
    next_handoff_target: str = "dev_os",
) -> dict[str, Any]:
    """Create a deterministic DevOS rework handoff from a rejection packet."""
    packet: dict[str, Any] = dict(DEVOS_HANDOFF_TEMPLATE)
    packet["from_profile"] = "orchestrator_os"
    packet["to_profile"] = "dev_os"
    packet["current_state"] = source_state
    packet["requested_next_state"] = route_target
    packet["production_order_id"] = po.production_order_id
    packet["context"] = (
        f"Rework route selected by OrchestratorOS for production order {po.production_order_id}"
    )
    packet["objective"] = "Apply the correction request and rework the implementation"
    packet["expected_output"] = "Reworked implementation report and updated evidence"
    packet["scope"] = str(rejection_packet.get("correction_request", rejection_packet.get("summary", "")))
    packet["out_of_scope"] = "Any new scope beyond the approved brief and correction request"
    packet["inputs"] = (
        f"Parent card ID: {po.parent_kanban_card_id}; "
        f"Child card IDs: {', '.join(po.child_kanban_card_ids)}; "
        "Rejection or classification packet frozen on the originating card."
    )
    packet["acceptance_criteria"] = [
        "Address the selected correction request.",
        "Preserve approved scope and repo/workspace boundaries.",
        f"Return to DEV_COMPLETE with credible evidence for {next_handoff_target}.",
    ]
    packet["stop_conditions"] = [
        "Correction request is ambiguous or missing.",
        "The rework would expand scope beyond the approved brief.",
        "The repo/workspace target is unavailable.",
    ]
    packet["approval_boundaries"] = ["No scope expansion or destructive changes."]
    packet["artifact_references"] = ["Origin rejection packet"]
    packet["default_rejection_reason"] = rejection_packet.get(
        "default_rejection_reason",
        rejection_packet.get("rejection_reason", ""),
    )
    packet["correction_request"] = rejection_packet.get("correction_request", [])
    packet["classification"] = rejection_packet.get("classification", "audit_rework")
    packet["route_target"] = route_target
    packet["route_reason"] = route_reason or "Rework route selected by OrchestratorOS"
    packet["next_handoff_target"] = next_handoff_target
    return packet


def create_architect_rework_handoff(
    po: ProductionOrder,
    classification_packet: dict[str, Any],
) -> dict[str, Any]:
    """Create the deterministic ArchitectOS spec-rework handoff."""
    packet: dict[str, Any] = dict(ARCHITECT_HANDOFF_TEMPLATE)
    packet["from_profile"] = "orchestrator_os"
    packet["to_profile"] = "architect_os"
    packet["current_state"] = "ORCHESTRATOR_TRIAGE"
    packet["requested_next_state"] = "SPEC_REWORK"
    packet["production_order_id"] = po.production_order_id
    packet["objective"] = "Correct the spec or design mismatch identified by OrchestratorOS"
    packet["expected_output"] = "Spec-rework correction packet and updated architecture guidance"
    packet["scope"] = str(classification_packet.get("correction_request", classification_packet.get("default_rejection_reason", "")))
    packet["out_of_scope"] = "Any implementation work or workflow completion beyond the spec correction"
    packet["inputs"] = (
        f"Parent card ID: {po.parent_kanban_card_id}; "
        f"Child card IDs: {', '.join(po.child_kanban_card_ids)}; "
        "OrchestratorOS classification packet frozen on the originating card."
    )
    packet["acceptance_criteria"] = [
        "Address the spec or design mismatch.",
        "Preserve the approved brief and downstream implementation contract.",
        "Return ArchitectOS-ready correction guidance.",
    ]
    packet["stop_conditions"] = [
        "The classification packet is incomplete or inconsistent.",
        "The correction would expand scope beyond the approved brief.",
        "The repo/workspace target is unavailable.",
    ]
    packet["approval_boundaries"] = ["No scope expansion or destructive changes."]
    packet["artifact_references"] = ["OrchestratorOS classification packet"]
    packet["classification"] = classification_packet.get("classification", "spec_or_design_mismatch")
    packet["default_rejection_reason"] = classification_packet.get("default_rejection_reason", "")
    packet["correction_request"] = classification_packet.get("correction_request", [])
    packet["route_target"] = "SPEC_REWORK"
    packet["route_reason"] = classification_packet.get("route_reason", "Spec or design mismatch requires ArchitectOS correction")
    packet["next_handoff_target"] = "architect_os"
    return packet


def validate_auditos_review_packet(
    packet: dict[str, Any],
    *,
    expected_production_order_id: str,
    expected_source_state: str,
) -> dict[str, Any]:
    """Validate the explicit AuditOS review packet for the happy path."""
    label = "AuditOS review packet"
    if not isinstance(packet, dict):
        raise ValueError(f"{label} must be a JSON object")

    missing = _missing_required_fields(packet, REQUIRED_AUDITOS_REVIEW_PACKET_FIELDS)
    if missing:
        raise ValueError(
            f"{label} missing required field(s): " + ", ".join(missing)
        )

    if packet["production_order_id"] != expected_production_order_id:
        raise ValueError(
            f"{label} production_order_id does not match the requested production order"
        )
    if packet["owner_profile"] != "audit_os":
        raise ValueError(f"{label} owner_profile must be 'audit_os'")
    if packet["source_state"] not in {"DEV_COMPLETE", "AUDIT_REVIEW"}:
        raise ValueError(
            f"{label} source_state must be 'DEV_COMPLETE' or 'AUDIT_REVIEW'"
        )
    if packet["source_state"] != expected_source_state:
        raise ValueError(
            f"{label} source_state must match current production order state "
            f"{expected_source_state!r}"
        )
    if packet["next_handoff_target"] != "architect_os":
        raise ValueError(f"{label} next_handoff_target must be 'architect_os'")

    _require_positive_result(
        packet["verdict"],
        field_name="verdict",
        label=label,
        positive_tokens=("pass", "passed", "accept", "accepted", "approve", "approved"),
    )
    _require_positive_result(
        packet["review_result"],
        field_name="review_result",
        label=label,
        positive_tokens=("pass", "passed", "accept", "accepted", "approve", "approved"),
    )

    return packet


def validate_auditos_rejection_packet(
    packet: dict[str, Any],
    *,
    expected_production_order_id: str,
    expected_source_state: str,
) -> dict[str, Any]:
    """Validate the explicit AuditOS rejection packet for the rejection loop."""
    label = "AuditOS rejection packet"
    if not isinstance(packet, dict):
        raise ValueError(f"{label} must be a JSON object")

    missing = _missing_required_fields(packet, REQUIRED_AUDITOS_REJECTION_PACKET_FIELDS)
    if missing:
        raise ValueError(
            f"{label} missing required field(s): " + ", ".join(missing)
        )

    if packet["production_order_id"] != expected_production_order_id:
        raise ValueError(
            f"{label} production_order_id does not match the requested production order"
        )
    if packet["owner_profile"] != "audit_os":
        raise ValueError(f"{label} owner_profile must be 'audit_os'")
    if packet["source_state"] not in {"DEV_COMPLETE", "AUDIT_REVIEW"}:
        raise ValueError(
            f"{label} source_state must be 'DEV_COMPLETE' or 'AUDIT_REVIEW'"
        )
    if packet["source_state"] != expected_source_state:
        raise ValueError(
            f"{label} source_state must match current production order state "
            f"{expected_source_state!r}"
        )
    if packet["next_handoff_target"] != "orchestrator_os":
        raise ValueError(f"{label} next_handoff_target must be 'orchestrator_os'")

    _require_negative_result(
        packet["verdict"],
        field_name="verdict",
        label=label,
        negative_tokens=("fail", "failed", "reject", "rejected", "block", "blocked", "rework"),
    )
    _require_negative_result(
        packet["review_result"],
        field_name="review_result",
        label=label,
        negative_tokens=("fail", "failed", "reject", "rejected", "block", "blocked", "rework"),
    )

    _require_non_empty_field(packet, "correction_request", label=label)

    return packet


def create_architect_reconcile_handoff(
    po: ProductionOrder,
    audit_packet: dict[str, Any],
) -> dict[str, Any]:
    """Create the ArchitectOS reconcile handoff from AuditOS review output."""
    packet = dict(ARCHITECT_RECONCILE_HANDOFF_TEMPLATE)
    packet["production_order_id"] = po.production_order_id
    packet["audit_summary"] = audit_packet["summary"]
    packet["audit_evidence"] = audit_packet["evidence"]
    packet["tests_reviewed"] = audit_packet["tests_reviewed"]
    packet["risks_or_notes"] = audit_packet["risks_or_notes"]
    packet["acceptance_criteria"] = [
        "Confirm the accepted implementation preserves the ArchitectOS spec.",
        "Confirm no source-truth patch is required before final review.",
    ]
    packet["stop_conditions"] = [
        "Implementation drift is found.",
        "Spec patch or rework is required.",
        "Rework owner is unclear.",
    ]
    return packet


def validate_architect_reconcile_packet(
    packet: dict[str, Any],
    *,
    expected_production_order_id: str,
    expected_source_state: str,
) -> dict[str, Any]:
    """Validate the explicit ArchitectOS reconciliation packet."""
    label = "ArchitectOS reconcile packet"
    if not isinstance(packet, dict):
        raise ValueError(f"{label} must be a JSON object")

    missing = _missing_required_fields(
        packet,
        REQUIRED_ARCHITECT_RECONCILE_PACKET_FIELDS,
    )
    if missing:
        raise ValueError(
            f"{label} missing required field(s): " + ", ".join(missing)
        )

    if packet["production_order_id"] != expected_production_order_id:
        raise ValueError(
            f"{label} production_order_id does not match the requested production order"
        )
    if packet["owner_profile"] != "architect_os":
        raise ValueError(f"{label} owner_profile must be 'architect_os'")
    if packet["source_state"] not in {"AUDIT_PASSED", "ARCHITECT_RECONCILE"}:
        raise ValueError(
            f"{label} source_state must be 'AUDIT_PASSED' or 'ARCHITECT_RECONCILE'"
        )
    if packet["source_state"] != expected_source_state:
        raise ValueError(
            f"{label} source_state must match current production order state "
            f"{expected_source_state!r}"
        )
    if packet["next_handoff_target"] != "default":
        raise ValueError(f"{label} next_handoff_target must be 'default'")

    _require_positive_result(
        packet["reconcile_result"],
        field_name="reconcile_result",
        label=label,
        positive_tokens=("accept", "accepted", "align", "aligned", "pass", "passed"),
    )
    _require_positive_result(
        packet["architecture_alignment"],
        field_name="architecture_alignment",
        label=label,
        positive_tokens=("align", "aligned", "accept", "accepted", "pass", "passed"),
    )
    spec_patch_text = _coerce_packet_text(packet["spec_patch_needed"])
    if spec_patch_text not in {"false", "no", "none", "not needed", "not_required", "not required"}:
        raise ValueError(
            f"{label} spec_patch_needed must clearly indicate no patch is required"
        )

    return packet


def create_default_final_review_handoff(
    po: ProductionOrder,
    reconcile_packet: dict[str, Any],
) -> dict[str, Any]:
    """Create the Default Hermes final-review handoff from reconciliation."""
    packet = dict(DEFAULT_FINAL_REVIEW_HANDOFF_TEMPLATE)
    packet["production_order_id"] = po.production_order_id
    packet["reconcile_summary"] = reconcile_packet["summary"]
    packet["architecture_alignment"] = reconcile_packet["architecture_alignment"]
    packet["drift_assessment"] = reconcile_packet["drift_assessment"]
    packet["risks_or_notes"] = reconcile_packet["risks_or_notes"]
    packet["inputs"] = (
        f"Parent card ID: {po.parent_kanban_card_id}; "
        f"Child card IDs: {', '.join(po.child_kanban_card_ids)}; "
        "Review frozen brief, DevOS result, AuditOS result, and ArchitectOS reconciliation."
    )
    packet["expected_output"] = "Final done report for Jarren"
    packet["acceptance_criteria"] = [
        "Final output matches the approved brief.",
        "All stage gates are complete and accepted.",
    ]
    packet["stop_conditions"] = [
        "Final output does not match the approved brief.",
        "Required audit or reconciliation evidence is missing.",
        "Scope changed without approval.",
    ]
    return packet


def validate_default_final_review_packet(
    packet: dict[str, Any],
    *,
    expected_production_order_id: str,
    expected_source_state: str,
) -> dict[str, Any]:
    """Validate the explicit Default Hermes final-review packet."""
    label = "Default final review packet"
    if not isinstance(packet, dict):
        raise ValueError(f"{label} must be a JSON object")

    missing = _missing_required_fields(
        packet,
        REQUIRED_DEFAULT_FINAL_REVIEW_PACKET_FIELDS,
    )
    if missing:
        raise ValueError(
            f"{label} missing required field(s): " + ", ".join(missing)
        )

    if packet["production_order_id"] != expected_production_order_id:
        raise ValueError(
            f"{label} production_order_id does not match the requested production order"
        )
    if packet["owner_profile"] not in {"default", "default_hermes", "hermes"}:
        raise ValueError(
            f"{label} owner_profile must be 'default', 'default_hermes', or 'hermes'"
        )
    if packet["source_state"] not in {"ARCHITECT_ACCEPTED", "DEFAULT_FINAL_REVIEW"}:
        raise ValueError(
            f"{label} source_state must be 'ARCHITECT_ACCEPTED' or "
            "'DEFAULT_FINAL_REVIEW'"
        )
    if packet["source_state"] != expected_source_state:
        raise ValueError(
            f"{label} source_state must match current production order state "
            f"{expected_source_state!r}"
        )

    _require_positive_result(
        packet["final_review_result"],
        field_name="final_review_result",
        label=label,
        positive_tokens=("done", "complete", "completed", "accept", "accepted", "pass", "passed"),
    )
    _require_positive_result(
        packet["final_status"],
        field_name="final_status",
        label=label,
        positive_tokens=("done", "complete", "completed", "accept", "accepted", "pass", "passed"),
    )

    return packet


def create_default_rejection_handoff(
    po: ProductionOrder,
    rejection_packet: dict[str, Any],
) -> dict[str, Any]:
    """Create the OrchestratorOS triage handoff from a Default rejection."""
    packet = dict(DEFAULT_REJECTION_HANDOFF_TEMPLATE)
    packet["production_order_id"] = po.production_order_id
    packet["rejection_summary"] = rejection_packet["summary"]
    packet["original_brief_mismatch"] = rejection_packet["original_brief_mismatch"]
    packet["rejection_reason"] = rejection_packet["rejection_reason"]
    packet["evidence"] = rejection_packet["evidence"]
    packet["recommended_route"] = rejection_packet["recommended_route"]
    packet["next_handoff_target"] = rejection_packet["next_handoff_target"]
    packet["input_packet"] = (
        f"Default rejection packet for production order {po.production_order_id}"
    )
    packet["expected_output"] = "OrchestratorOS triage packet"
    packet["acceptance_criteria"] = [
        "Route the rejected final review into OrchestratorOS triage.",
        "Preserve the final-review rejection evidence and mismatch summary.",
    ]
    packet["stop_conditions"] = [
        "The rejection packet is incomplete or inconsistent.",
        "Routing would bypass OrchestratorOS triage.",
        "The production order identity does not match.",
    ]
    return packet


def validate_default_rejection_packet(
    packet: dict[str, Any],
    *,
    expected_production_order_id: str,
    expected_source_state: str,
) -> dict[str, Any]:
    """Validate the explicit Default Hermes rejection packet."""
    label = "Default rejection packet"
    if not isinstance(packet, dict):
        raise ValueError(f"{label} must be a JSON object")

    missing = _missing_required_fields(
        packet,
        REQUIRED_DEFAULT_REJECTION_PACKET_FIELDS,
    )
    if missing:
        raise ValueError(
            f"{label} missing required field(s): " + ", ".join(missing)
        )

    if packet["production_order_id"] != expected_production_order_id:
        raise ValueError(
            f"{label} production_order_id does not match the requested production order"
        )
    if packet["owner_profile"] != "default":
        raise ValueError(f"{label} owner_profile must be 'default'")
    if packet["source_state"] != "DEFAULT_FINAL_REVIEW":
        raise ValueError(
            f"{label} source_state must be 'DEFAULT_FINAL_REVIEW'"
        )
    if packet["source_state"] != expected_source_state:
        raise ValueError(
            f"{label} source_state must match current production order state "
            f"{expected_source_state!r}"
        )
    if packet["recommended_route"] != "orchestrator_triage":
        raise ValueError(
            f"{label} recommended_route must be 'orchestrator_triage'"
        )
    if packet["next_handoff_target"] != "orchestrator_os":
        raise ValueError(
            f"{label} next_handoff_target must be 'orchestrator_os'"
        )

    _require_negative_result(
        packet["review_result"],
        field_name="review_result",
        label=label,
        negative_tokens=("fail", "failed", "reject", "rejected", "block", "blocked", "rework"),
    )
    _require_non_empty_field(packet, "summary", label=label)
    _require_non_empty_field(packet, "original_brief_mismatch", label=label)
    _require_non_empty_field(packet, "rejection_reason", label=label)
    _require_non_empty_field(packet, "evidence", label=label)

    return packet


def validate_orchestrator_classification_packet(
    packet: dict[str, Any],
    *,
    expected_production_order_id: str,
    expected_source_state: str,
) -> dict[str, Any]:
    """Validate the OrchestratorOS default-rejection classification packet."""
    label = "OrchestratorOS classification packet"
    if not isinstance(packet, dict):
        raise ValueError(f"{label} must be a JSON object")

    missing = _missing_required_fields(
        packet,
        REQUIRED_ORCHESTRATOR_CLASSIFICATION_PACKET_FIELDS,
    )
    if missing:
        raise ValueError(
            f"{label} missing required field(s): " + ", ".join(missing)
        )

    if packet["production_order_id"] != expected_production_order_id:
        raise ValueError(
            f"{label} production_order_id does not match the requested production order"
        )
    if packet["owner_profile"] != "orchestrator_os":
        raise ValueError(f"{label} owner_profile must be 'orchestrator_os'")
    if packet["source_state"] != "ORCHESTRATOR_TRIAGE":
        raise ValueError(f"{label} source_state must be 'ORCHESTRATOR_TRIAGE'")
    if packet["source_state"] != expected_source_state:
        raise ValueError(
            f"{label} source_state must match current production order state "
            f"{expected_source_state!r}"
        )

    classification = str(packet["classification"]).strip().lower()
    route_spec = ORCHESTRATOR_CLASSIFICATION_ROUTE_MAP.get(classification)
    if route_spec is None:
        allowed = ", ".join(sorted(ORCHESTRATOR_CLASSIFICATION_ROUTE_MAP))
        raise ValueError(
            f"{label} classification must be one of: {allowed}"
        )

    if packet["route_target"] != route_spec["route_target"]:
        raise ValueError(
            f"{label} route_target must be {route_spec['route_target']!r} "
            f"for classification {classification!r}"
        )
    if packet["next_handoff_target"] != route_spec["next_handoff_target"]:
        raise ValueError(
            f"{label} next_handoff_target must be {route_spec['next_handoff_target']!r} "
            f"for classification {classification!r}"
        )

    _require_non_empty_field(packet, "default_rejection_reason", label=label)
    _require_non_empty_field(packet, "route_reason", label=label)
    _require_non_empty_field(packet, "correction_request", label=label)

    packet["classification"] = classification
    return packet


def _append_json_packet_to_card(
    conn: sqlite3.Connection,
    card_id: str,
    packet: dict[str, Any],
    *,
    marker: str,
) -> None:
    current_body = conn.execute(
        "SELECT body FROM tasks WHERE id = ?", (card_id,)
    ).fetchone()
    if current_body is None:
        raise ValueError(f"Card {card_id} not found")
    existing = current_body["body"] or ""
    section = (
        f"\n\n--- {marker} ---\n"
        f"{json.dumps(packet, indent=2)}\n"
        f"--- END {marker} ---"
    )
    conn.execute(
        "UPDATE tasks SET body = ? WHERE id = ?",
        (existing + section, card_id),
    )


def freeze_handoff_on_card(
    conn: sqlite3.Connection,
    card_id: str,
    handoff_packet: dict[str, Any],
) -> None:
    """Store/serialize the handoff packet onto a Kanban card's body.

    Appends the serialized handoff packet to the card body as a JSON
    comment section so the receiving profile can read it without
    requiring chat memory.  The existing body is preserved.
    """
    _append_json_packet_to_card(
        conn,
        card_id,
        handoff_packet,
        marker="HANDOFF PACKET",
    )


def freeze_result_on_card(
    conn: sqlite3.Connection,
    card_id: str,
    result_packet: dict[str, Any],
) -> None:
    """Store/serialize the DevOS result packet onto the DevOS card body."""
    _append_json_packet_to_card(
        conn,
        card_id,
        result_packet,
        marker="RESULT PACKET",
    )


# ---------------------------------------------------------------------------
# Full Bridge
# ---------------------------------------------------------------------------


def run_full_bridge(
    conn: sqlite3.Connection,
    *,
    title: str,
    source_brief: str,
    approved_by: str = "Jarren",
    priority_lane: str = "Hermes OS",
    repo_or_workspace: str = "",
    idempotency_key: Optional[str] = None,
    board: Optional[str] = None,
) -> ProductionOrder:
    """Run the full Slice 4 runtime bridge end-to-end.

    1. Create production order + parent card
    2. Create Kanban graph (6 child cards)
    3. Create OrchestratorOS handoff packet
    4. Freeze handoff on OrchestratorOS card
    5. Transition state to ORCHESTRATOR_TRIAGE
    6. Log handoff_created event

    Returns the :class:`ProductionOrder` instance.
    """
    po = create_production_order(
        conn,
        title=title,
        source_brief=source_brief,
        approved_by=approved_by,
        priority_lane=priority_lane,
        repo_or_workspace=repo_or_workspace,
        idempotency_key=idempotency_key,
        board=board,
    )

    child_ids = create_production_kanban_graph(conn, po, board=board)

    handoff = create_orchestrator_handoff(
        po,
        scope="From the frozen brief",
        out_of_scope="From the frozen brief",
    )
    freeze_handoff_on_card(conn, child_ids[0], handoff)

    transition_state(
        conn, po, "ORCHESTRATOR_TRIAGE", "orchestrator_os",
        next_action="dispatch_orchestrator_os",
        card_id=po.parent_kanban_card_id,
    )

    log_workflow_event(
        conn, po.production_order_id, "handoff_created",
        from_state=WORKFLOW_INITIAL_STATE,
        to_state="ORCHESTRATOR_TRIAGE",
        owner_profile="hermes",
        kanban_card_id=child_ids[0],
        result="done",
        next_action="dispatch_orchestrator_os",
    )

    # Update all child cards to reflect the new state.
    for cid in po.child_kanban_card_ids:
        conn.execute(
            "UPDATE tasks SET current_state = ? WHERE id = ?",
            (po.current_state, cid),
        )

    return po


def run_orchestrator_triage_bridge(
    conn: sqlite3.Connection,
    *,
    production_order_id: str,
) -> ProductionOrder:
    """Run deterministic Slice 5 triage for an existing production order.

    Preconditions are intentionally strict: the order must already be in
    ORCHESTRATOR_TRIAGE, be owned by OrchestratorOS, and have the existing
    6-card Kanban graph. This function does not create a new production order
    and does not advance beyond ARCHITECT_SPEC.
    """
    matches = [
        order for order in list_production_orders(conn)
        if order.production_order_id == production_order_id
    ]
    if not matches:
        raise ValueError(f"production order {production_order_id!r} not found")
    po = matches[0]

    if po.current_state != "ORCHESTRATOR_TRIAGE":
        raise StateTransitionError(
            f"Production order {production_order_id} is in {po.current_state!r}; "
            "expected 'ORCHESTRATOR_TRIAGE'"
        )
    if po.current_owner_profile != "orchestrator_os":
        raise StateTransitionError(
            f"Production order {production_order_id} is owned by "
            f"{po.current_owner_profile!r}; expected 'orchestrator_os'"
        )
    if not po.parent_kanban_card_id:
        raise ValueError("production order parent Kanban card is missing")
    if len(po.child_kanban_card_ids) != 6:
        raise ValueError(
            f"production order Kanban graph must have 6 child cards; "
            f"found {len(po.child_kanban_card_ids)}"
        )
    if len(set(po.child_kanban_card_ids)) != 6:
        raise ValueError("production order Kanban graph contains duplicate child card IDs")

    placeholders = ",".join(["?"] * len(po.child_kanban_card_ids))
    existing_child_count = conn.execute(
        f"SELECT COUNT(*) AS n FROM tasks WHERE id IN ({placeholders})",
        tuple(po.child_kanban_card_ids),
    ).fetchone()["n"]
    if existing_child_count != 6:
        raise ValueError(
            f"production order Kanban graph references {6 - existing_child_count} "
            "missing child card(s)"
        )

    architect_card_id = po.child_kanban_card_ids[1]
    handoff = create_architect_handoff(po)
    freeze_handoff_on_card(conn, architect_card_id, handoff)

    transition_state(
        conn,
        po,
        "ARCHITECT_SPEC",
        "orchestrator_os",
        result="triage complete; ArchitectOS handoff attached",
        next_action="dispatch_architect_os",
        card_id=po.parent_kanban_card_id,
        event_type="orchestrator_triage_completed",
    )

    log_workflow_event(
        conn,
        po.production_order_id,
        "handoff_created",
        from_state="ORCHESTRATOR_TRIAGE",
        to_state="ARCHITECT_SPEC",
        owner_profile="orchestrator_os",
        kanban_card_id=architect_card_id,
        result="ArchitectOS handoff packet attached",
        next_action="dispatch_architect_os",
    )

    for cid in po.child_kanban_card_ids:
        conn.execute(
            "UPDATE tasks SET current_state = ? WHERE id = ?",
            (po.current_state, cid),
        )
    conn.execute(
        "UPDATE tasks SET status = 'ready' WHERE id = ?",
        (architect_card_id,),
    )

    return po


def run_architect_spec_bridge(
    conn: sqlite3.Connection,
    *,
    production_order_id: str,
    architect_packet: dict[str, Any],
) -> ProductionOrder:
    """Advance ARCHITECT_SPEC → ARCHITECT_READY_FOR_DEV for an existing order.

    Preconditions are deterministic and strict: the existing production order
    must already be in ARCHITECT_SPEC, be owned by ArchitectOS, retain the
    existing 6-card Kanban graph, and receive an explicit ArchitectOS packet.
    The runtime then prepares the DevOS handoff packet and advances ownership
    to DevOS without creating or duplicating any cards.
    """
    matches = [
        order for order in list_production_orders(conn)
        if order.production_order_id == production_order_id
    ]
    if not matches:
        raise ValueError(f"production order {production_order_id!r} not found")
    po = matches[0]

    if po.current_state != "ARCHITECT_SPEC":
        raise StateTransitionError(
            f"Production order {production_order_id} is in {po.current_state!r}; "
            "expected 'ARCHITECT_SPEC'"
        )
    if po.current_owner_profile != "architect_os":
        raise StateTransitionError(
            f"Production order {production_order_id} is owned by "
            f"{po.current_owner_profile!r}; expected 'architect_os'"
        )
    if not po.parent_kanban_card_id:
        raise ValueError("production order parent Kanban card is missing")
    if len(po.child_kanban_card_ids) != 6:
        raise ValueError(
            f"production order Kanban graph must have 6 child cards; "
            f"found {len(po.child_kanban_card_ids)}"
        )
    if len(set(po.child_kanban_card_ids)) != 6:
        raise ValueError("production order Kanban graph contains duplicate child card IDs")

    placeholders = ",".join(["?"] * len(po.child_kanban_card_ids))
    existing_child_count = conn.execute(
        f"SELECT COUNT(*) AS n FROM tasks WHERE id IN ({placeholders})",
        tuple(po.child_kanban_card_ids),
    ).fetchone()["n"]
    if existing_child_count != 6:
        raise ValueError(
            f"production order Kanban graph references {6 - existing_child_count} "
            "missing child card(s)"
        )

    packet = validate_architect_spec_packet(
        architect_packet,
        expected_production_order_id=production_order_id,
    )
    devos_card_id = po.child_kanban_card_ids[2]
    devos_handoff = create_devos_handoff(po, packet)
    freeze_handoff_on_card(conn, devos_card_id, devos_handoff)

    transition_state(
        conn,
        po,
        "ARCHITECT_READY_FOR_DEV",
        "architect_os",
        result="architect spec completed; DevOS handoff attached",
        next_action="dispatch_dev_os",
        card_id=po.parent_kanban_card_id,
        event_type="architect_spec_completed",
    )

    log_workflow_event(
        conn,
        po.production_order_id,
        "handoff_created",
        from_state="ARCHITECT_SPEC",
        to_state="ARCHITECT_READY_FOR_DEV",
        owner_profile="architect_os",
        kanban_card_id=devos_card_id,
        result="DevOS handoff packet attached",
        next_action="dispatch_dev_os",
    )

    for cid in po.child_kanban_card_ids:
        conn.execute(
            "UPDATE tasks SET current_state = ? WHERE id = ?",
            (po.current_state, cid),
        )
    conn.execute(
        "UPDATE tasks SET status = 'ready' WHERE id = ?",
        (devos_card_id,),
    )

    return po


def run_devos_complete_bridge(
    conn: sqlite3.Connection,
    *,
    production_order_id: str,
    devos_packet: dict[str, Any],
) -> ProductionOrder:
    """Advance ARCHITECT_READY_FOR_DEV → DEV_IMPLEMENTING → DEV_COMPLETE."""
    matches = [
        order for order in list_production_orders(conn)
        if order.production_order_id == production_order_id
    ]
    if not matches:
        raise ValueError(f"production order {production_order_id!r} not found")
    po = matches[0]

    if po.current_state != "ARCHITECT_READY_FOR_DEV":
        raise StateTransitionError(
            f"Production order {production_order_id} is in {po.current_state!r}; "
            "expected 'ARCHITECT_READY_FOR_DEV'"
        )
    if po.current_owner_profile != "dev_os":
        raise StateTransitionError(
            f"Production order {production_order_id} is owned by "
            f"{po.current_owner_profile!r}; expected 'dev_os'"
        )
    if not po.parent_kanban_card_id:
        raise ValueError("production order parent Kanban card is missing")
    if len(po.child_kanban_card_ids) != 6:
        raise ValueError(
            f"production order Kanban graph must have 6 child cards; "
            f"found {len(po.child_kanban_card_ids)}"
        )
    if len(set(po.child_kanban_card_ids)) != 6:
        raise ValueError("production order Kanban graph contains duplicate child card IDs")

    placeholders = ",".join(["?"] * len(po.child_kanban_card_ids))
    existing_child_count = conn.execute(
        f"SELECT COUNT(*) AS n FROM tasks WHERE id IN ({placeholders})",
        tuple(po.child_kanban_card_ids),
    ).fetchone()["n"]
    if existing_child_count != 6:
        raise ValueError(
            f"production order Kanban graph references {6 - existing_child_count} "
            "missing child card(s)"
        )

    packet = validate_devos_build_packet(
        devos_packet,
        expected_production_order_id=production_order_id,
    )
    devos_card_id = po.child_kanban_card_ids[2]
    auditos_card_id = po.child_kanban_card_ids[3]
    audit_handoff = create_auditos_handoff(po, packet)

    freeze_result_on_card(conn, devos_card_id, packet)

    transition_state(
        conn,
        po,
        "DEV_IMPLEMENTING",
        "dev_os",
        result="dev implementation started",
        next_action="complete_dev_build",
        card_id=po.parent_kanban_card_id,
        event_type="dev_build_started",
    )

    freeze_handoff_on_card(conn, auditos_card_id, audit_handoff)

    transition_state(
        conn,
        po,
        "DEV_COMPLETE",
        "dev_os",
        result="dev build completed; AuditOS handoff attached",
        next_action="dispatch_audit_os",
        card_id=po.parent_kanban_card_id,
        event_type="dev_build_completed",
    )

    log_workflow_event(
        conn,
        po.production_order_id,
        "handoff_created",
        from_state="DEV_IMPLEMENTING",
        to_state="DEV_COMPLETE",
        owner_profile="dev_os",
        kanban_card_id=auditos_card_id,
        result="AuditOS handoff packet attached",
        next_action="dispatch_audit_os",
    )

    for cid in po.child_kanban_card_ids:
        conn.execute(
            "UPDATE tasks SET current_state = ? WHERE id = ?",
            (po.current_state, cid),
        )
    conn.execute(
        "UPDATE tasks SET status = 'ready' WHERE id = ?",
        (auditos_card_id,),
    )

    return po


def run_auditos_review_reject_bridge(
    conn: sqlite3.Connection,
    *,
    production_order_id: str,
    rejection_packet: dict[str, Any],
) -> ProductionOrder:
    """Advance DEV_COMPLETE/AUDIT_REVIEW to AUDIT_REJECTED for an existing order."""
    po = _load_existing_production_order(conn, production_order_id)
    _assert_bridge_preconditions(
        conn,
        po,
        expected_states={"DEV_COMPLETE", "AUDIT_REVIEW"},
        expected_owner="audit_os",
    )
    packet = validate_auditos_rejection_packet(
        rejection_packet,
        expected_production_order_id=production_order_id,
        expected_source_state=po.current_state,
    )

    audit_card_id = po.child_kanban_card_ids[3]
    freeze_result_on_card(conn, audit_card_id, packet)

    if po.current_state == "DEV_COMPLETE":
        transition_state(
            conn,
            po,
            "AUDIT_REVIEW",
            "audit_os",
            result="audit review started",
            next_action="complete_audit_review",
            card_id=po.parent_kanban_card_id,
            event_type="audit_review_started",
        )

    transition_state(
        conn,
        po,
        "AUDIT_REJECTED",
        "audit_os",
        result="audit review rejected; correction packet attached",
        next_action="route_orchestrator_rework",
        card_id=po.parent_kanban_card_id,
        event_type="state_transitioned",
    )

    log_workflow_event(
        conn,
        po.production_order_id,
        "stage_rejected",
        from_state="AUDIT_REVIEW",
        to_state="AUDIT_REJECTED",
        owner_profile="audit_os",
        kanban_card_id=audit_card_id,
        result="AuditOS rejection packet attached",
        next_action="route_orchestrator_rework",
    )

    _sync_child_current_state(conn, po)
    return po


def run_orchestrator_rework_bridge(
    conn: sqlite3.Connection,
    *,
    production_order_id: str,
    rejection_packet: dict[str, Any],
) -> ProductionOrder:
    """Advance AUDIT_REJECTED to DEV_REWORK for an existing order."""
    po = _load_existing_production_order(conn, production_order_id)
    _assert_bridge_preconditions(
        conn,
        po,
        expected_states={"AUDIT_REJECTED"},
        expected_owner="orchestrator_os",
    )

    audit_card_id = po.child_kanban_card_ids[3]
    devos_card_id = po.child_kanban_card_ids[2]
    correction_handoff = create_devos_rework_handoff(po, rejection_packet)
    freeze_handoff_on_card(conn, devos_card_id, correction_handoff)

    transition_state(
        conn,
        po,
        "DEV_REWORK",
        "orchestrator_os",
        result="orchestrator routed AuditOS correction to DevOS",
        next_action="dispatch_dev_rework",
        card_id=po.parent_kanban_card_id,
        event_type="retry_started",
    )

    log_workflow_event(
        conn,
        po.production_order_id,
        "handoff_created",
        from_state="AUDIT_REJECTED",
        to_state="DEV_REWORK",
        owner_profile="orchestrator_os",
        kanban_card_id=devos_card_id,
        result="DevOS rework handoff packet attached",
        next_action="dispatch_dev_rework",
    )

    _sync_child_current_state(conn, po)
    conn.execute(
        "UPDATE tasks SET status = 'ready' WHERE id = ?",
        (devos_card_id,),
    )
    return po


def run_orchestrator_classification_bridge(
    conn: sqlite3.Connection,
    *,
    production_order_id: str,
    classification_packet: dict[str, Any],
) -> ProductionOrder:
    """Route DEFAULT_REJECTED/ORCHESTRATOR_TRIAGE into a rework lane."""
    po = _load_existing_production_order(conn, production_order_id)
    _assert_bridge_preconditions(
        conn,
        po,
        expected_states={"ORCHESTRATOR_TRIAGE"},
        expected_owner="orchestrator_os",
    )
    if not any(
        entry.from_state == "DEFAULT_REJECTED" and entry.to_state == "ORCHESTRATOR_TRIAGE"
        for entry in po.stage_history
    ):
        raise ValueError(
            "Orchestrator classification routing requires prior DEFAULT_REJECTED triage"
        )
    packet = validate_orchestrator_classification_packet(
        classification_packet,
        expected_production_order_id=production_order_id,
        expected_source_state=po.current_state,
    )

    route_target = packet["route_target"]
    if route_target == "DEV_REWORK":
        target_card_id = po.child_kanban_card_ids[2]
        rework_handoff = create_devos_rework_handoff(
            po,
            packet,
            source_state="ORCHESTRATOR_TRIAGE",
            route_reason=packet["route_reason"],
            route_target=route_target,
            next_handoff_target="dev_os",
        )
        next_action = "dispatch_dev_rework"
        handoff_result = "DevOS rework handoff packet attached"
    elif route_target == "SPEC_REWORK":
        target_card_id = po.child_kanban_card_ids[1]
        rework_handoff = create_architect_rework_handoff(po, packet)
        next_action = "dispatch_spec_rework"
        handoff_result = "ArchitectOS spec-rework handoff packet attached"
    else:  # pragma: no cover - validate_orchestrator_classification_packet prevents this
        raise ValueError(f"Unsupported route_target {route_target!r}")

    freeze_handoff_on_card(conn, target_card_id, rework_handoff)

    log_workflow_event(
        conn,
        po.production_order_id,
        "retry_started",
        from_state="ORCHESTRATOR_TRIAGE",
        to_state=route_target,
        owner_profile="orchestrator_os",
        kanban_card_id=po.parent_kanban_card_id,
        result=f"OrchestratorOS started {route_target.lower()} routing",
        next_action=next_action,
    )

    transition_state(
        conn,
        po,
        route_target,
        "orchestrator_os",
        result=f"orchestrator classified default rejection; {route_target.lower()} attached",
        next_action=next_action,
        card_id=po.parent_kanban_card_id,
        event_type="state_transitioned",
    )

    log_workflow_event(
        conn,
        po.production_order_id,
        "handoff_created",
        from_state="ORCHESTRATOR_TRIAGE",
        to_state=route_target,
        owner_profile="orchestrator_os",
        kanban_card_id=target_card_id,
        result=handoff_result,
        next_action=next_action,
    )

    _sync_child_current_state(conn, po)
    conn.execute(
        "UPDATE tasks SET status = 'ready' WHERE id = ?",
        (target_card_id,),
    )
    return po


def run_devos_rework_complete_bridge(
    conn: sqlite3.Connection,
    *,
    production_order_id: str,
    devos_packet: dict[str, Any],
) -> ProductionOrder:
    """Advance DEV_REWORK to DEV_COMPLETE for an existing order."""
    po = _load_existing_production_order(conn, production_order_id)
    _assert_bridge_preconditions(
        conn,
        po,
        expected_states={"DEV_REWORK"},
        expected_owner="dev_os",
    )
    packet = validate_devos_build_packet(
        devos_packet,
        expected_production_order_id=production_order_id,
        expected_source_state="DEV_REWORK",
    )

    devos_card_id = po.child_kanban_card_ids[2]
    auditos_card_id = po.child_kanban_card_ids[3]
    audit_handoff = create_auditos_handoff(
        po,
        packet,
        from_state="DEV_REWORK",
        to_state="DEV_COMPLETE",
    )

    freeze_result_on_card(conn, devos_card_id, packet)
    freeze_handoff_on_card(conn, auditos_card_id, audit_handoff)

    transition_state(
        conn,
        po,
        "DEV_COMPLETE",
        "dev_os",
        result="dev rework completed; AuditOS handoff attached",
        next_action="dispatch_audit_os",
        card_id=po.parent_kanban_card_id,
        event_type="stage_completed",
    )

    log_workflow_event(
        conn,
        po.production_order_id,
        "handoff_created",
        from_state="DEV_REWORK",
        to_state="DEV_COMPLETE",
        owner_profile="dev_os",
        kanban_card_id=auditos_card_id,
        result="AuditOS handoff packet attached",
        next_action="dispatch_audit_os",
    )

    _sync_child_current_state(conn, po)
    conn.execute(
        "UPDATE tasks SET status = 'ready' WHERE id = ?",
        (auditos_card_id,),
    )
    return po


def _load_existing_production_order(
    conn: sqlite3.Connection,
    production_order_id: str,
) -> ProductionOrder:
    matches = [
        order for order in list_production_orders(conn)
        if order.production_order_id == production_order_id
    ]
    if not matches:
        raise ValueError(f"production order {production_order_id!r} not found")
    return matches[0]


def _assert_bridge_preconditions(
    conn: sqlite3.Connection,
    po: ProductionOrder,
    *,
    expected_states: set[str],
    expected_owner: str,
) -> None:
    if po.current_state not in expected_states:
        expected_text = ", ".join(sorted(expected_states))
        raise StateTransitionError(
            f"Production order {po.production_order_id} is in {po.current_state!r}; "
            f"expected one of {expected_text!r}"
        )
    if po.current_owner_profile != expected_owner:
        raise StateTransitionError(
            f"Production order {po.production_order_id} is owned by "
            f"{po.current_owner_profile!r}; expected {expected_owner!r}"
        )
    if not po.parent_kanban_card_id:
        raise ValueError("production order parent Kanban card is missing")
    if len(po.child_kanban_card_ids) != 6:
        raise ValueError(
            f"production order Kanban graph must have 6 child cards; "
            f"found {len(po.child_kanban_card_ids)}"
        )
    if len(set(po.child_kanban_card_ids)) != 6:
        raise ValueError("production order Kanban graph contains duplicate child card IDs")

    placeholders = ",".join(["?"] * len(po.child_kanban_card_ids))
    existing_child_count = conn.execute(
        f"SELECT COUNT(*) AS n FROM tasks WHERE id IN ({placeholders})",
        tuple(po.child_kanban_card_ids),
    ).fetchone()["n"]
    if existing_child_count != 6:
        raise ValueError(
            f"production order Kanban graph references {6 - existing_child_count} "
            "missing child card(s)"
        )


def _sync_child_current_state(conn: sqlite3.Connection, po: ProductionOrder) -> None:
    for cid in po.child_kanban_card_ids:
        conn.execute(
            "UPDATE tasks SET current_state = ? WHERE id = ?",
            (po.current_state, cid),
        )


def run_auditos_review_complete_bridge(
    conn: sqlite3.Connection,
    *,
    production_order_id: str,
    review_packet: dict[str, Any],
) -> ProductionOrder:
    """Advance DEV_COMPLETE/AUDIT_REVIEW to AUDIT_PASSED for an existing order."""
    po = _load_existing_production_order(conn, production_order_id)
    _assert_bridge_preconditions(
        conn,
        po,
        expected_states={"DEV_COMPLETE", "AUDIT_REVIEW"},
        expected_owner="audit_os",
    )
    packet = validate_auditos_review_packet(
        review_packet,
        expected_production_order_id=production_order_id,
        expected_source_state=po.current_state,
    )

    audit_card_id = po.child_kanban_card_ids[3]
    reconcile_card_id = po.child_kanban_card_ids[4]
    reconcile_handoff = create_architect_reconcile_handoff(po, packet)

    freeze_result_on_card(conn, audit_card_id, packet)
    freeze_handoff_on_card(conn, reconcile_card_id, reconcile_handoff)

    if po.current_state == "DEV_COMPLETE":
        transition_state(
            conn,
            po,
            "AUDIT_REVIEW",
            "audit_os",
            result="audit review started",
            next_action="complete_audit_review",
            card_id=po.parent_kanban_card_id,
            event_type="audit_review_started",
        )

    transition_state(
        conn,
        po,
        "AUDIT_PASSED",
        "audit_os",
        result="audit review passed; ArchitectOS reconcile handoff attached",
        next_action="dispatch_architect_reconcile",
        card_id=po.parent_kanban_card_id,
        event_type="audit_review_completed",
    )

    log_workflow_event(
        conn,
        po.production_order_id,
        "handoff_created",
        from_state="AUDIT_REVIEW",
        to_state="AUDIT_PASSED",
        owner_profile="audit_os",
        kanban_card_id=reconcile_card_id,
        result="ArchitectOS reconcile handoff packet attached",
        next_action="dispatch_architect_reconcile",
    )

    _sync_child_current_state(conn, po)
    conn.execute(
        "UPDATE tasks SET status = 'ready' WHERE id = ?",
        (reconcile_card_id,),
    )
    return po


def run_architect_reconcile_bridge(
    conn: sqlite3.Connection,
    *,
    production_order_id: str,
    reconcile_packet: dict[str, Any],
) -> ProductionOrder:
    """Advance AUDIT_PASSED/ARCHITECT_RECONCILE to ARCHITECT_ACCEPTED."""
    po = _load_existing_production_order(conn, production_order_id)
    _assert_bridge_preconditions(
        conn,
        po,
        expected_states={"AUDIT_PASSED", "ARCHITECT_RECONCILE"},
        expected_owner="architect_os",
    )
    packet = validate_architect_reconcile_packet(
        reconcile_packet,
        expected_production_order_id=production_order_id,
        expected_source_state=po.current_state,
    )

    reconcile_card_id = po.child_kanban_card_ids[4]
    final_card_id = po.child_kanban_card_ids[5]
    final_handoff = create_default_final_review_handoff(po, packet)

    freeze_result_on_card(conn, reconcile_card_id, packet)
    freeze_handoff_on_card(conn, final_card_id, final_handoff)

    if po.current_state == "AUDIT_PASSED":
        transition_state(
            conn,
            po,
            "ARCHITECT_RECONCILE",
            "architect_os",
            result="architect reconciliation started",
            next_action="complete_architect_reconcile",
            card_id=po.parent_kanban_card_id,
            event_type="architect_reconcile_started",
        )

    transition_state(
        conn,
        po,
        "ARCHITECT_ACCEPTED",
        "architect_os",
        result="architecture reconciled; final review handoff attached",
        next_action="dispatch_default_final_review",
        card_id=po.parent_kanban_card_id,
        event_type="architect_reconcile_completed",
    )

    log_workflow_event(
        conn,
        po.production_order_id,
        "handoff_created",
        from_state="ARCHITECT_RECONCILE",
        to_state="ARCHITECT_ACCEPTED",
        owner_profile="architect_os",
        kanban_card_id=final_card_id,
        result="Default final review handoff packet attached",
        next_action="dispatch_default_final_review",
    )

    _sync_child_current_state(conn, po)
    conn.execute(
        "UPDATE tasks SET status = 'ready' WHERE id = ?",
        (final_card_id,),
    )
    return po


def run_default_final_review_bridge(
    conn: sqlite3.Connection,
    *,
    production_order_id: str,
    final_packet: dict[str, Any],
) -> ProductionOrder:
    """Advance ARCHITECT_ACCEPTED/DEFAULT_FINAL_REVIEW to DONE."""
    po = _load_existing_production_order(conn, production_order_id)
    _assert_bridge_preconditions(
        conn,
        po,
        expected_states={"ARCHITECT_ACCEPTED", "DEFAULT_FINAL_REVIEW"},
        expected_owner="default",
    )
    packet = validate_default_final_review_packet(
        final_packet,
        expected_production_order_id=production_order_id,
        expected_source_state=po.current_state,
    )

    final_card_id = po.child_kanban_card_ids[5]
    freeze_result_on_card(conn, final_card_id, packet)

    if po.current_state == "ARCHITECT_ACCEPTED":
        transition_state(
            conn,
            po,
            "DEFAULT_FINAL_REVIEW",
            "default",
            result="default final review started",
            next_action="complete_default_final_review",
            card_id=po.parent_kanban_card_id,
            event_type="default_final_review_started",
        )

    transition_state(
        conn,
        po,
        "DONE",
        "default",
        result="workflow completed",
        next_action=packet["next_action"],
        card_id=po.parent_kanban_card_id,
        event_type="default_final_review_completed",
    )

    log_workflow_event(
        conn,
        po.production_order_id,
        "workflow_completed",
        owner_profile="default",
        kanban_card_id=po.parent_kanban_card_id,
        result=str(packet["final_status"]),
        next_action=packet["next_action"],
    )

    po.final_status = str(packet["final_status"])
    _sync_child_current_state(conn, po)
    conn.execute(
        "UPDATE tasks SET status = 'done' WHERE id = ?",
        (final_card_id,),
    )
    return po


def run_default_final_review_reject_bridge(
    conn: sqlite3.Connection,
    *,
    production_order_id: str,
    rejection_packet: dict[str, Any],
) -> ProductionOrder:
    """Advance DEFAULT_FINAL_REVIEW to DEFAULT_REJECTED for an existing order."""
    po = _load_existing_production_order(conn, production_order_id)
    _assert_bridge_preconditions(
        conn,
        po,
        expected_states={"DEFAULT_FINAL_REVIEW"},
        expected_owner="default",
    )
    packet = validate_default_rejection_packet(
        rejection_packet,
        expected_production_order_id=production_order_id,
        expected_source_state=po.current_state,
    )

    final_card_id = po.child_kanban_card_ids[5]
    freeze_result_on_card(conn, final_card_id, packet)

    transition_state(
        conn,
        po,
        "DEFAULT_REJECTED",
        "default",
        result="default final review rejected; triage handoff pending",
        next_action="route_orchestrator_triage",
        card_id=po.parent_kanban_card_id,
        event_type="state_transitioned",
    )

    log_workflow_event(
        conn,
        po.production_order_id,
        "stage_rejected",
        from_state="DEFAULT_FINAL_REVIEW",
        to_state="DEFAULT_REJECTED",
        owner_profile="default",
        kanban_card_id=final_card_id,
        result="Default rejection packet attached",
        next_action="route_orchestrator_triage",
    )

    _sync_child_current_state(conn, po)
    return po


def run_orchestrator_default_rejection_triage_bridge(
    conn: sqlite3.Connection,
    *,
    production_order_id: str,
    rejection_packet: dict[str, Any],
) -> ProductionOrder:
    """Advance DEFAULT_REJECTED to ORCHESTRATOR_TRIAGE for an existing order."""
    po = _load_existing_production_order(conn, production_order_id)
    _assert_bridge_preconditions(
        conn,
        po,
        expected_states={"DEFAULT_REJECTED"},
        expected_owner="orchestrator_os",
    )

    orchestrator_card_id = po.child_kanban_card_ids[0]
    triage_handoff = create_default_rejection_handoff(po, rejection_packet)
    freeze_handoff_on_card(conn, orchestrator_card_id, triage_handoff)

    transition_state(
        conn,
        po,
        "ORCHESTRATOR_TRIAGE",
        "orchestrator_os",
        result="default rejection routed to OrchestratorOS triage",
        next_action="dispatch_orchestrator_triage",
        card_id=po.parent_kanban_card_id,
        event_type="state_transitioned",
    )

    log_workflow_event(
        conn,
        po.production_order_id,
        "handoff_created",
        from_state="DEFAULT_REJECTED",
        to_state="ORCHESTRATOR_TRIAGE",
        owner_profile="orchestrator_os",
        kanban_card_id=orchestrator_card_id,
        result="OrchestratorOS triage handoff packet attached",
        next_action="dispatch_orchestrator_triage",
    )

    _sync_child_current_state(conn, po)
    conn.execute(
        "UPDATE tasks SET status = 'ready' WHERE id = ?",
        (orchestrator_card_id,),
    )
    return po


# ---------------------------------------------------------------------------
# Status / List Helpers
# ---------------------------------------------------------------------------


def format_production_order_status(po: ProductionOrder) -> str:
    """Return a human-readable status summary for a production order."""
    lines = [
        f"Production Order: {po.production_order_id}",
        f"Title: {po.title}",
        f"State: {po.current_state}",
        f"Owner: {po.current_owner_profile}",
        f"Approved by: {po.approved_by}",
        f"Approved at: {datetime.fromtimestamp(po.approved_at, tz=timezone.utc).strftime('%Y-%m-%d %H:%M UTC') if po.approved_at else 'N/A'}",
        f"Priority lane: {po.priority_lane}",
        f"Repo: {po.repo_or_workspace or 'N/A'}",
        f"Parent card: {po.parent_kanban_card_id or 'N/A'}",
        f"Child cards: {len(po.child_kanban_card_ids)}",
        f"Stage history: {len(po.stage_history)} transitions",
        f"Blockers: {len(po.blockers)}",
    ]
    return "\n".join(lines)


def list_production_orders(
    conn: sqlite3.Connection,
    *,
    board: Optional[str] = None,
) -> list[ProductionOrder]:
    """List all active production orders from the Kanban board.

    Scans tasks with ``production_order_id`` set (parent cards) and
    reconstructs :class:`ProductionOrder` objects.
    """
    rows = conn.execute(
        "SELECT id, title, body, production_order_id, current_state, "
        "created_by, created_at "
        "FROM tasks "
        "WHERE production_order_id IS NOT NULL "
        "AND status != 'archived' "
        "AND title LIKE 'Production Order:%' "
        "ORDER BY created_at DESC"
    ).fetchall()

    orders: list[ProductionOrder] = []
    for row in rows:
        orders.append(_reconstruct_production_order(conn, row))
    return orders
