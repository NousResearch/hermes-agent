"""Production Workflow v1 — runtime bridge module (Slice 4).

Provides the deterministic runtime layer that transforms an approved brief
into a governed, stateful production order with Kanban graph, event log,
and first handoff packet.

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

Slice 4 implements only:

    - production_order_id generation (PO-YYYYMMDD-XXXX)
    - ProductionOrder dataclass
    - create_production_order()
    - create_production_kanban_graph()  (parent + 6 child cards)
    - validate_state_transition()
    - transition_state()
    - log_workflow_event()
    - detect_approval_phrase()
    - validate_brief()
    - create_orchestrator_handoff()
    - Slice 4 valid transitions: BRIEF_DRAFTED → ACTION_APPROVED
                                  → PRODUCTION_ORDER_CREATED
                                  → ORCHESTRATOR_TRIAGE

Out of scope (do NOT implement):
    - OrchestratorOS agent spawn
    - ArchitectOS/DevOS/AuditOS dispatch
    - Dashboard UI
    - Rework loop automation
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

# Only Slice 4 + Slice 5 transitions are wired. Remaining transitions become
# live when each downstream profile uses the state machine.
VALID_TRANSITIONS: dict[str, set[str]] = {
    "BRIEF_DRAFTED":                {"ACTION_APPROVED"},
    "ACTION_APPROVED":              {"PRODUCTION_ORDER_CREATED"},
    "PRODUCTION_ORDER_CREATED":     {"ORCHESTRATOR_TRIAGE"},
    "ORCHESTRATOR_TRIAGE":          {"ARCHITECT_SPEC"},
}

STATE_OWNERS: dict[str, str] = {
    "BRIEF_DRAFTED":                "hermes",
    "ACTION_APPROVED":              "hermes",
    "PRODUCTION_ORDER_CREATED":     "orchestrator_os",
    "ORCHESTRATOR_TRIAGE":          "orchestrator_os",
    "ARCHITECT_SPEC":               "architect_os",
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
    "source_truth": "specs/architecture/workflows/hermes-production-workflow-v1.md",
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
    "source_truth": "specs/architecture/workflows/hermes-production-workflow-v1.md",
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
    3. ``calling_profile`` matches the owner of ``from_state``

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
    kanban_card_id: Optional[str] = None,
    result: Optional[str] = None,
    error: Optional[str] = None,
    next_action: Optional[str] = None,
) -> int:
    """Write an event to the ``production_order_events`` table.

    Returns the new event row id.  Raises :class:`ValueError` if
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
    from_state, to_state, owner_profile,
    kanban_card_id, result, error, next_action
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            production_order_id,
            now,
            event_type,
            from_state,
            to_state,
            owner_profile,
            kanban_card_id,
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
    packet = dict(ARCHITECT_HANDOFF_TEMPLATE)
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


def freeze_handoff_on_card(
    conn: sqlite3.Connection,
    card_id: str,
    handoff_packet: dict[str, str],
) -> None:
    """Store/serialize the handoff packet onto a Kanban card's body.

    Appends the serialized handoff packet to the card body as a JSON
    comment section so the receiving profile can read it without
    requiring chat memory.  The existing body is preserved.
    """
    current_body = conn.execute(
        "SELECT body FROM tasks WHERE id = ?", (card_id,)
    ).fetchone()
    if current_body is None:
        raise ValueError(f"Card {card_id} not found")
    existing = current_body["body"] or ""
    handoff_section = (
        f"\n\n--- HANDOFF PACKET ---\n"
        f"{json.dumps(handoff_packet, indent=2)}\n"
        f"--- END HANDOFF PACKET ---"
    )
    conn.execute(
        "UPDATE tasks SET body = ? WHERE id = ?",
        (existing + handoff_section, card_id),
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
