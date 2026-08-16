"""Idempotent Kanban successor / card-creation policy.

Gating logic for watchdog and other automated card creators so that the same
root-cause never spawns an unbounded pile of Kanban cards.

Public API
----------
evaluate_successor(conn, req, *, state, policy) -> SuccessorDecision
apply_successor(conn, req, decision, *, dry_run) -> str | None

The two-step design keeps DB writes out of the decision phase, making unit
tests cheap (no real DB needed for evaluate_successor if you supply a stub).
apply_successor handles the actual create_task / audit-event write.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Reason-class allow-list
# ---------------------------------------------------------------------------

#: Reason classes that are allowed to produce new cards.
ACTIONABLE_REASON_CLASSES: frozenset[str] = frozenset(
    {
        "review-required",
        "remediation",
        "blocker",
    }
)

#: Reason classes that are explicitly non-actionable and must never create a card.
NON_ACTIONABLE_REASON_CLASSES: frozenset[str] = frozenset(
    {
        "needs_input",
        "credential_missing",
        "observation",
        "duplicate_diagnosis",
        "dirty_workspace",
    }
)

# Active statuses for which an existing successor is still useful and should be
# reused rather than replaced.
_REUSABLE_STATUSES: frozenset[str] = frozenset(
    {"todo", "scheduled", "ready", "running"}
)

# Terminal statuses — successor in these states is NOT reused (history preserved).
_TERMINAL_STATUSES: frozenset[str] = frozenset({"done", "archived", "cancelled"})

# Successor in "blocked" is not reused (it may be permanently stuck).
_NON_REUSE_STATUSES: frozenset[str] = _TERMINAL_STATUSES | {"blocked"}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class SuccessorRequest:
    """Describes a desired successor card."""

    source_task_id: str
    reason_class: str
    reason_detail: str
    assignee: str
    title: str
    body: str
    target_ref: str = ""
    policy_version: str = "v1"

    def idempotency_key(self) -> str:
        parts = [
            self.source_task_id,
            self.reason_class,
            self.target_ref or "",
            self.policy_version,
        ]
        return ":".join(parts)


@dataclass
class SuccessorDecision:
    """Result of evaluate_successor."""

    allowed: bool
    action: str  # "create" | "reuse" | "defer" | "reject"
    existing_id: Optional[str] = None
    reason: str = ""


@dataclass
class SuccessorPolicy:
    """Tunable budget/cooldown parameters."""

    #: Maximum number of active successors created from the same source task.
    max_per_source_total: int = 10
    #: Maximum new successors per tick globally (across all sources).
    max_per_tick_global: int = 5
    #: Maximum new successors per reason-class per tick.
    max_per_reason_per_tick: int = 3
    #: Cooldown in seconds after a terminal-state successor before a new one
    #: can be created with the same idempotency key.
    cooldown_seconds: int = 300


#: Module-level default policy; callers may pass their own.
DEFAULT_POLICY = SuccessorPolicy()


@dataclass
class SuccessorPolicyState:
    """Mutable per-tick counters. Create one per dispatch tick, share across calls."""

    #: Total new cards created this tick.
    tick_total_created: int = 0
    #: Cards created per reason_class this tick.
    tick_per_reason: dict[str, int] = field(default_factory=dict)
    #: Cards created per source_task_id this tick (separate from DB totals).
    tick_per_source: dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Core evaluation logic
# ---------------------------------------------------------------------------


def evaluate_successor(
    conn,
    req: SuccessorRequest,
    *,
    state: Optional[SuccessorPolicyState] = None,
    policy: Optional[SuccessorPolicy] = None,
) -> SuccessorDecision:
    """Decide whether to create, reuse, defer, or reject a successor card.

    This function reads the DB but makes no writes.

    Decision order:
    1. Reject if reason_class is unknown or non-actionable.
    2. Reuse if an active (non-terminal, non-blocked) successor exists.
    3. Defer if cooldown window is active (same key, terminal recently).
    4. Defer if any budget cap is exceeded.
    5. Allow create.
    """
    if policy is None:
        policy = DEFAULT_POLICY
    if state is None:
        state = SuccessorPolicyState()

    # --- 1. Reason-class gate (fail-closed) ---
    rc = req.reason_class
    if rc in NON_ACTIONABLE_REASON_CLASSES:
        return SuccessorDecision(
            allowed=False,
            action="reject",
            reason=f"reason_class {rc!r} is non-actionable",
        )
    if rc not in ACTIONABLE_REASON_CLASSES:
        return SuccessorDecision(
            allowed=False,
            action="reject",
            reason=f"reason_class {rc!r} is unknown (fail-closed)",
        )

    # --- 2. Reuse active successor ---
    ikey = req.idempotency_key()
    existing = _find_successor(conn, ikey)
    if existing is not None:
        status = existing["status"]
        if status in _REUSABLE_STATUSES:
            return SuccessorDecision(
                allowed=True,
                action="reuse",
                existing_id=existing["id"],
                reason=f"active successor {existing['id']} in status {status!r}",
            )
        # Terminal or blocked — do NOT reuse (fall through to cooldown check).

    # --- 3. Cooldown check (same key was terminal recently) ---
    if existing is not None and existing["status"] in _TERMINAL_STATUSES:
        last_ts = _last_creation_timestamp(conn, ikey)
        if last_ts is not None:
            elapsed = time.time() - last_ts
            if elapsed < policy.cooldown_seconds:
                return SuccessorDecision(
                    allowed=False,
                    action="defer",
                    reason=(
                        f"cooldown: {policy.cooldown_seconds - elapsed:.0f}s remaining"
                        f" (last created {elapsed:.0f}s ago)"
                    ),
                )

    # --- 4. Budget checks ---
    # 4a. Global per-tick cap.
    if policy.max_per_tick_global >= 0 and state.tick_total_created >= policy.max_per_tick_global:
        return SuccessorDecision(
            allowed=False,
            action="defer",
            reason=(
                f"global per-tick budget exhausted "
                f"({state.tick_total_created}/{policy.max_per_tick_global})"
            ),
        )

    # 4b. Per-reason per-tick cap.
    reason_count = state.tick_per_reason.get(rc, 0)
    if reason_count >= policy.max_per_reason_per_tick:
        return SuccessorDecision(
            allowed=False,
            action="defer",
            reason=(
                f"per-reason-per-tick budget exhausted for {rc!r} "
                f"({reason_count}/{policy.max_per_reason_per_tick})"
            ),
        )

    # 4c. Per-source total cap (DB count of active non-terminal successors).
    source_active = _count_active_successors(conn, req.source_task_id)
    if source_active >= policy.max_per_source_total:
        return SuccessorDecision(
            allowed=False,
            action="defer",
            reason=(
                f"per-source budget exhausted for {req.source_task_id!r} "
                f"({source_active}/{policy.max_per_source_total})"
            ),
        )

    # --- 5. Allow create ---
    return SuccessorDecision(allowed=True, action="create", reason="")


# ---------------------------------------------------------------------------
# Apply (write phase)
# ---------------------------------------------------------------------------


def apply_successor(
    conn,
    req: SuccessorRequest,
    decision: SuccessorDecision,
    *,
    dry_run: bool = False,
    state: Optional[SuccessorPolicyState] = None,
) -> Optional[str]:
    """Execute the decision and record audit events.

    Returns:
        The card id (new or reused) on create/reuse.
        None on defer or reject.
    """
    from hermes_cli.kanban_db import (
        TERMINAL_STATUSES,
        _append_event,
        create_task,
        write_txn,
    )

    ikey = req.idempotency_key()

    if decision.action == "reuse":
        if not dry_run:
            with write_txn(conn):
                _append_event(
                    conn,
                    req.source_task_id,
                    "successor_reused",
                    {
                        "successor_id": decision.existing_id,
                        "reason_class": req.reason_class,
                        "idempotency_key": ikey,
                    },
                )
        return decision.existing_id

    if decision.action == "create":
        if dry_run:
            return None
        # create_task already handles idempotency_key dedup at DB level.
        card_id = create_task(
            conn,
            title=req.title,
            body=req.body,
            assignee=req.assignee,
            idempotency_key=ikey,
        )
        with write_txn(conn):
            _append_event(
                conn,
                req.source_task_id,
                "successor_created",
                {
                    "successor_id": card_id,
                    "reason_class": req.reason_class,
                    "reason_detail": req.reason_detail,
                    "idempotency_key": ikey,
                },
            )
        # Update in-tick state counters.
        if state is not None:
            state.tick_total_created += 1
            state.tick_per_reason[req.reason_class] = (
                state.tick_per_reason.get(req.reason_class, 0) + 1
            )
            state.tick_per_source[req.source_task_id] = (
                state.tick_per_source.get(req.source_task_id, 0) + 1
            )
        return card_id

    # defer or reject — record audit event against source_task_id.
    if not dry_run:
        try:
            with write_txn(conn):
                _append_event(
                    conn,
                    req.source_task_id,
                    "successor_deferred",
                    {
                        "action": decision.action,
                        "reason": decision.reason,
                        "reason_class": req.reason_class,
                        "idempotency_key": ikey,
                    },
                )
        except Exception:
            # Audit must never block the caller.
            _log.warning(
                "kanban_successor_policy: failed to record %s event for source %s",
                decision.action,
                req.source_task_id,
                exc_info=True,
            )
    return None


# ---------------------------------------------------------------------------
# Internal DB helpers
# ---------------------------------------------------------------------------


def _find_successor(conn, idempotency_key: str) -> Optional[dict]:
    """Return the most-recent non-archived task row with the given key, or None."""
    row = conn.execute(
        "SELECT id, status FROM tasks "
        "WHERE idempotency_key = ? "
        "AND status NOT IN ('archived') "
        "ORDER BY created_at DESC LIMIT 1",
        (idempotency_key,),
    ).fetchone()
    if row is None:
        return None
    return {"id": row["id"] if hasattr(row, "__getitem__") else row[0],
            "status": row["status"] if hasattr(row, "__getitem__") else row[1]}


def _last_creation_timestamp(conn, idempotency_key: str) -> Optional[float]:
    """Return the created_at epoch of the most-recent task with this key (any status)."""
    row = conn.execute(
        "SELECT created_at FROM tasks "
        "WHERE idempotency_key = ? "
        "ORDER BY created_at DESC LIMIT 1",
        (idempotency_key,),
    ).fetchone()
    if row is None:
        return None
    return float(row["created_at"] if hasattr(row, "__getitem__") else row[0])


def _count_active_successors(conn, source_task_id: str) -> int:
    """Count non-terminal successor cards that carry this source in their idempotency key."""
    prefix = f"{source_task_id}:"
    # Use LIKE pattern — idempotency_key starts with "{source_task_id}:"
    row = conn.execute(
        "SELECT COUNT(*) FROM tasks "
        "WHERE idempotency_key LIKE ? "
        "AND status NOT IN ('done', 'archived', 'cancelled')",
        (prefix + "%",),
    ).fetchone()
    return int(row[0]) if row else 0
