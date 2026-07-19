"""Safe, deterministic retention cleanup for finalized project work.

This module is a narrow policy boundary over the existing HOF-002 persistence
interfaces.  It never infers ownership from titles or board-wide task data,
never deletes rows, and never mutates anything during planning or dry-runs.
Applying a plan archives only explicitly registered project members and writes
one idempotent cleanup journal record through the existing recorder boundary.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from hermes_cli import kanban_db
from hermes_cli.project_finalization_contract import (
    TERMINAL_OUTCOMES,
    get_project_finalization,
    list_project_members,
)


TERMINAL_DELIVERY_STATES = frozenset({
    "accepted", "complete", "completed", "delivered", "sent", "success",
})
TERMINAL_TASK_STATUSES = frozenset({"done", "archived", "cancelled", "failed"})


@dataclass(frozen=True)
class CleanupAction:
    task_id: str
    action: str = "archive"

    def to_dict(self) -> dict[str, str]:
        return {"task_id": self.task_id, "action": self.action}


@dataclass(frozen=True)
class CleanupPlan:
    board_id: str
    root_task_id: str
    generation: int
    retention_cutoff: str | None
    eligible_task_ids: tuple[str, ...] = field(default_factory=tuple)
    excluded_task_ids: tuple[str, ...] = field(default_factory=tuple)
    actions: tuple[CleanupAction, ...] = field(default_factory=tuple)
    refusal_reasons: tuple[str, ...] = field(default_factory=tuple)
    evidence_paths: tuple[str, ...] = field(default_factory=tuple)
    plan_sha256: str = ""
    already_applied: bool = False

    @property
    def eligible(self) -> bool:
        return not self.refusal_reasons

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "board_id": self.board_id,
            "root_task_id": self.root_task_id,
            "generation": self.generation,
            "retention_cutoff": self.retention_cutoff,
            "eligible_task_ids": list(self.eligible_task_ids),
            "excluded_task_ids": list(self.excluded_task_ids),
            "actions": [action.to_dict() for action in self.actions],
            "refusal_reasons": list(self.refusal_reasons),
            "evidence_paths": list(self.evidence_paths),
            "already_applied": self.already_applied,
        }

    def to_dict(self) -> dict[str, Any]:
        result = self.canonical_payload()
        result["eligible"] = self.eligible
        result["plan_sha256"] = self.plan_sha256
        return result


@dataclass(frozen=True)
class CleanupResult:
    plan: CleanupPlan
    dry_run: bool
    applied_task_ids: tuple[str, ...] = field(default_factory=tuple)
    journal: Any = None

    @property
    def changed(self) -> bool:
        return bool(self.applied_task_ids)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dry_run": self.dry_run,
            "applied_task_ids": list(self.applied_task_ids),
            "changed": self.changed,
            "plan": self.plan.to_dict(),
            "journal": _to_dict(self.journal),
        }


def _to_dict(item: Any) -> Any:
    if item is None:
        return None
    if isinstance(item, Mapping):
        return dict(item)
    if hasattr(item, "__dataclass_fields__"):
        return {key: _to_dict(getattr(item, key)) for key in item.__dataclass_fields__}
    return item


def _value(item: Any, key: str, default: Any = None) -> Any:
    if isinstance(item, Mapping):
        return item.get(key, default)
    return getattr(item, key, default)


def _aware_now(now: datetime | str | None) -> datetime:
    value = datetime.now(timezone.utc) if now is None else now
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value)
        except ValueError as exc:
            raise ValueError("now must be an ISO-8601 timestamp") from exc
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("now must be timezone-aware")
    return value


def _cutoff(finalization: Any, *, retention_duration: timedelta | None) -> datetime:
    raw = _value(finalization, "cleanup_after")
    if raw:
        if not isinstance(raw, str):
            raise ValueError("cleanup_after must be an ISO-8601 timestamp")
        try:
            result = datetime.fromisoformat(raw)
        except ValueError as exc:
            raise ValueError("cleanup_after must be an ISO-8601 timestamp") from exc
        if result.tzinfo is None or result.utcoffset() is None:
            raise ValueError("cleanup_after must be timezone-aware")
        return result
    if retention_duration is None:
        retention_days = _value(finalization, "retention_days")
        if (
            isinstance(retention_days, bool)
            or not isinstance(retention_days, int)
            or retention_days < 0
        ):
            raise ValueError(
                "retention_days is required when cleanup_after is absent"
            )
        retention_duration = timedelta(days=retention_days)
    if not isinstance(retention_duration, timedelta):
        raise ValueError("retention_duration must be a timedelta")
    if retention_duration.total_seconds() < 0:
        raise ValueError("retention_duration must not be negative")
    finalized_at = _value(finalization, "finalized_at")
    if not isinstance(finalized_at, (int, float)):
        raise ValueError("finalized_at is required when cleanup_after is absent")
    return datetime.fromtimestamp(finalized_at, tz=timezone.utc) + retention_duration


def _artifact_paths(finalization: Any) -> tuple[Path, ...]:
    report = _value(finalization, "final_report_path")
    manifest = _value(finalization, "manifest_path")
    if not isinstance(report, str) or not report or not isinstance(manifest, str) or not manifest:
        return ()
    manifest_path = Path(manifest)
    return (Path(report), manifest_path, manifest_path.with_name("usage-summary.json"))


def _artifact_reasons(finalization: Any) -> tuple[tuple[str, ...], tuple[str, ...]]:
    paths = _artifact_paths(finalization)
    if not paths:
        return ("missing_artifacts",), ()
    reasons: list[str] = []
    for path in paths:
        if not path.is_file():
            reasons.append(f"missing_artifact:{path}")
    for attr, path in (("final_report_sha256", paths[0]), ("manifest_sha256", paths[1])):
        expected = _value(finalization, attr)
        if isinstance(expected, str) and path.is_file():
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != expected:
                reasons.append(f"artifact_hash_mismatch:{path}")
    return tuple(reasons), tuple(str(path) for path in paths)


def _delivery_reasons(conn: sqlite3.Connection, *, board_id: str, root_task_id: str, generation: int) -> tuple[str, ...]:
    try:
        rows = conn.execute(
            "SELECT delivery_state, accepted FROM project_delivery_attempts "
            "WHERE board_id = ? AND root_task_id = ? AND generation = ? "
            "ORDER BY attempt_number ASC, id ASC",
            (board_id, root_task_id, generation),
        ).fetchall()
    except sqlite3.OperationalError:
        return ("missing_delivery_ledger",)
    terminal = [row for row in rows if str(row["delivery_state"]).casefold() in TERMINAL_DELIVERY_STATES and row["accepted"] == 1]
    if not terminal:
        return ("nonterminal_delivery",)
    if len(terminal) != 1:
        return ("ambiguous_delivery",)
    return ()


def _existing_plan_applied(conn: sqlite3.Connection, *, board_id: str, root_task_id: str, generation: int) -> bool:
    try:
        row = conn.execute(
            "SELECT 1 FROM project_cleanup_journal WHERE board_id = ? AND root_task_id = ? "
            "AND generation = ? AND status = 'applied' LIMIT 1",
            (board_id, root_task_id, generation),
        ).fetchone()
    except sqlite3.OperationalError:
        return False
    return row is not None


def _journal_for_plan(
    conn: sqlite3.Connection,
    *,
    board_id: str,
    root_task_id: str,
    generation: int,
    plan_sha256: str,
) -> Any:
    """Return an existing applied journal row for an exact plan, if present."""
    try:
        return conn.execute(
            "SELECT * FROM project_cleanup_journal WHERE board_id = ? "
            "AND root_task_id = ? AND generation = ? AND plan_sha256 = ? "
            "AND status = 'applied' ORDER BY id LIMIT 1",
            (board_id, root_task_id, generation, plan_sha256),
        ).fetchone()
    except sqlite3.OperationalError:
        return None


def _plan_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def plan_project_cleanup(
    conn: sqlite3.Connection,
    *,
    board_id: str,
    root_task_id: str,
    generation: int | None = None,
    now: datetime | str | None = None,
    retention_duration: timedelta | None = None,
) -> CleanupPlan:
    """Return a mutation-free cleanup plan for one finalized project.

    Refusals are fail-closed.  The only possible actions are archive operations
    for the root and tasks named by ``project_finalization_members``.
    """
    finalization = get_project_finalization(
        conn, board_id=board_id, root_task_id=root_task_id, generation=generation
    )
    resolved_generation = int(generation or _value(finalization, "generation", 0) or 0)
    reasons: list[str] = []
    evidence_paths: tuple[str, ...] = ()
    cutoff: datetime | None = None
    current = _aware_now(now)
    if finalization is None:
        reasons.append("missing_finalization")
    else:
        terminal_outcome = _value(finalization, "terminal_outcome")
        state = _value(finalization, "state")
        expected_states = {"COMPLETE": "complete", "BLOCKED": "blocked", "FAILED": "failed"}
        if terminal_outcome not in TERMINAL_OUTCOMES:
            reasons.append("active_project")
        elif state not in {expected_states[terminal_outcome], "cleanup_scheduled", "cleaned"}:
            reasons.append("contradictory_finalization")
        if state in {"cleaned"}:
            reasons.append("already_cleaned")
        try:
            cutoff = _cutoff(finalization, retention_duration=retention_duration)
            if current < cutoff:
                reasons.append("retention_not_expired")
        except ValueError as exc:
            reasons.append(f"invalid_retention:{exc}")
        artifact_error, evidence_paths = _artifact_reasons(finalization)
        reasons.extend(artifact_error)
        reasons.extend(_delivery_reasons(conn, board_id=board_id, root_task_id=root_task_id, generation=resolved_generation))

    members: tuple[Any, ...] = ()
    if finalization is not None:
        try:
            members = tuple(list_project_members(
                conn, board_id=board_id, root_task_id=root_task_id, generation=resolved_generation
            ))
        except (sqlite3.OperationalError, ValueError) as exc:
            reasons.append(f"invalid_membership:{exc}")
    explicit_ids = {root_task_id}
    explicit_ids.update(
        task_id for task_id in (_value(member, "task_id") for member in members)
        if isinstance(task_id, str) and task_id
    )
    tasks: dict[str, Any] = {}
    for task_id in sorted(explicit_ids):
        task = kanban_db.get_task(conn, task_id)
        if task is None:
            reasons.append(f"missing_member_task:{task_id}")
        else:
            tasks[task_id] = task
            if str(_value(task, "status", "")) not in TERMINAL_TASK_STATUSES:
                reasons.append(f"active_member_task:{task_id}")

    if not members:
        reasons.append("missing_explicit_membership")
    reasons = sorted(set(reasons))
    already_applied = _existing_plan_applied(
        conn, board_id=board_id, root_task_id=root_task_id, generation=resolved_generation
    )
    if already_applied:
        # An applied journal is the durable idempotency marker.  Re-planning is
        # a no-op even though the finalization row may still say scheduled.
        reasons = []

    eligible_ids = tuple(sorted(tasks)) if not reasons else ()
    actions = tuple(CleanupAction(task_id=task_id) for task_id in eligible_ids if str(_value(tasks[task_id], "status")) != "archived")
    cutoff_text = cutoff.isoformat() if cutoff is not None else None
    payload = {
        "board_id": board_id,
        "root_task_id": root_task_id,
        "generation": resolved_generation,
        "retention_cutoff": cutoff_text,
        "eligible_task_ids": list(eligible_ids),
        "excluded_task_ids": [],
        "actions": [action.to_dict() for action in actions],
        "refusal_reasons": reasons,
        "evidence_paths": list(evidence_paths),
        "already_applied": already_applied,
    }
    return CleanupPlan(
        board_id=board_id,
        root_task_id=root_task_id,
        generation=resolved_generation,
        retention_cutoff=cutoff_text,
        eligible_task_ids=eligible_ids,
        actions=actions,
        refusal_reasons=tuple(reasons),
        evidence_paths=evidence_paths,
        plan_sha256=_plan_hash(payload),
        already_applied=already_applied,
    )


def _validate_untrusted_plan_shape(plan: CleanupPlan) -> None:
    """Reject malformed plan input before reading or mutating durable state."""
    if not isinstance(plan, CleanupPlan):
        raise ValueError("cleanup plan must be a CleanupPlan")
    if not isinstance(plan.board_id, str) or not plan.board_id:
        raise ValueError("cleanup plan board_id is required")
    if not isinstance(plan.root_task_id, str) or not plan.root_task_id:
        raise ValueError("cleanup plan root_task_id is required")
    if not isinstance(plan.generation, int) or plan.generation < 1:
        raise ValueError("cleanup plan generation is invalid")
    if not isinstance(plan.eligible_task_ids, tuple) or any(
        not isinstance(task_id, str) or not task_id for task_id in plan.eligible_task_ids
    ):
        raise ValueError("cleanup plan eligible_task_ids are invalid")
    if len(set(plan.eligible_task_ids)) != len(plan.eligible_task_ids):
        raise ValueError("cleanup plan eligible_task_ids are duplicated")
    if not isinstance(plan.actions, tuple) or any(
        not isinstance(action, CleanupAction)
        or not isinstance(action.task_id, str)
        or not action.task_id
        or not isinstance(action.action, str)
        for action in plan.actions
    ):
        raise ValueError("cleanup plan actions are invalid")
    action_ids = tuple(action.task_id for action in plan.actions)
    if len(set(action_ids)) != len(action_ids):
        raise ValueError("cleanup plan actions are duplicated")
    if not isinstance(plan.plan_sha256, str) or plan.plan_sha256 != _plan_hash(plan.canonical_payload()):
        raise ValueError("cleanup plan hash is invalid")


def _require_current_generation(conn: sqlite3.Connection, plan: CleanupPlan) -> None:
    """Require the supplied identity to name the durable current generation."""
    current = get_project_finalization(
        conn,
        board_id=plan.board_id,
        root_task_id=plan.root_task_id,
    )
    expected = get_project_finalization(
        conn,
        board_id=plan.board_id,
        root_task_id=plan.root_task_id,
        generation=plan.generation,
    )
    if current is None or expected is None or _value(current, "generation") != plan.generation:
        raise ValueError("cleanup plan does not name the current finalization generation")


def _require_current_canonical_plan(conn: sqlite3.Connection, plan: CleanupPlan) -> CleanupPlan:
    """Replan under the write lock and reject stale or caller-authored input."""
    canonical = plan_project_cleanup(
        conn,
        board_id=plan.board_id,
        root_task_id=plan.root_task_id,
        generation=plan.generation,
        now=_aware_now(None),
    )
    if not canonical.eligible:
        raise ValueError(
            "cleanup plan is not eligible and is not currently authorized: "
            + "; ".join(canonical.refusal_reasons)
        )
    if (
        canonical.already_applied
        or plan.canonical_payload() != canonical.canonical_payload()
        or plan.plan_sha256 != canonical.plan_sha256
    ):
        raise ValueError("cleanup plan is stale or tampered")
    return canonical


def _archive_current_terminal_task(conn: sqlite3.Connection, task_id: str) -> bool:
    """Archive a task only if its status remains terminal inside this transaction."""
    cur = conn.execute(
        "UPDATE tasks SET status = 'archived', "
        "    claim_lock = NULL, claim_expires = NULL, worker_pid = NULL "
        "WHERE id = ? AND status IN ('cancelled', 'done', 'failed')",
        (task_id,),
    )
    if cur.rowcount != 1:
        return False
    run_id = kanban_db._end_run(
        conn,
        task_id,
        outcome="reclaimed",
        status="reclaimed",
        summary="task archived with run still active",
    )
    kanban_db._append_event(conn, task_id, "archived", None, run_id=run_id)
    return True


def _record_cleanup_journal_in_txn(conn: sqlite3.Connection, **kwargs: Any) -> sqlite3.Row:
    """Write the existing journal schema without opening a nested transaction."""
    cur = conn.execute(
        """
        INSERT INTO project_cleanup_journal
        (board_id, root_task_id, generation, plan_sha256, mode, status,
         retention_cutoff, eligible_task_count, excluded_task_count,
         deleted_task_count, archived_task_count, evidence_path, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            kwargs["board_id"], kwargs["root_task_id"], kwargs["generation"],
            kwargs["plan_sha256"], kwargs["mode"], kwargs["status"],
            kwargs["retention_cutoff"], kwargs["eligible_task_count"],
            kwargs["excluded_task_count"], kwargs["deleted_task_count"],
            kwargs["archived_task_count"], kwargs["evidence_path"],
            int(datetime.now(timezone.utc).timestamp()),
        ),
    )
    return conn.execute("SELECT * FROM project_cleanup_journal WHERE id = ?", (cur.lastrowid,)).fetchone()


def apply_cleanup_plan(
    conn: sqlite3.Connection,
    plan: CleanupPlan,
    *,
    dry_run: bool = False,
    archive_action: Callable[[sqlite3.Connection, str], bool] | None = None,
    journal_writer: Callable[..., Any] | None = None,
) -> CleanupResult:
    """Apply an eligible plan, or return it unchanged for a dry-run.

    ``archive_action`` and ``journal_writer`` are narrow injection points for
    disposable stores and tests.  The default path uses the existing Kanban
    archive operation and HOF-002 cleanup journal recorder.
    """
    if dry_run:
        return CleanupResult(plan=plan, dry_run=True)
    _validate_untrusted_plan_shape(plan)
    applied: list[str] = []
    journal: Any = None
    with kanban_db.write_txn(conn):
        _require_current_generation(conn, plan)
        existing_journal = _journal_for_plan(
            conn,
            board_id=plan.board_id,
            root_task_id=plan.root_task_id,
            generation=plan.generation,
            plan_sha256=plan.plan_sha256,
        )
        if existing_journal is not None:
            return CleanupResult(plan=plan, dry_run=False, journal=existing_journal)
        canonical = _require_current_canonical_plan(conn, plan)
        archive = archive_action or _archive_current_terminal_task
        for action in canonical.actions:
            if action.action != "archive":
                raise ValueError(f"unsupported cleanup action: {action.action!r}")
            if not archive(conn, action.task_id):
                raise ValueError("cleanup plan member changed before archive")
            applied.append(action.task_id)
        journal_kwargs = {
            "board_id": canonical.board_id,
            "root_task_id": canonical.root_task_id,
            "generation": canonical.generation,
            "plan_sha256": canonical.plan_sha256,
            "mode": "apply",
            "status": "applied",
            "retention_cutoff": (
                int(datetime.fromisoformat(canonical.retention_cutoff).timestamp())
                if canonical.retention_cutoff else None
            ),
            "eligible_task_count": len(canonical.eligible_task_ids),
            "excluded_task_count": len(canonical.excluded_task_ids),
            "deleted_task_count": 0,
            "archived_task_count": len(applied),
            "evidence_path": canonical.evidence_paths[0] if canonical.evidence_paths else None,
        }
        journal = (
            journal_writer(conn, **journal_kwargs)
            if journal_writer is not None
            else _record_cleanup_journal_in_txn(conn, **journal_kwargs)
        )
    if applied:
        kanban_db.recompute_ready(conn)
    return CleanupResult(plan=plan, dry_run=False, applied_task_ids=tuple(applied), journal=journal)


def cleanup_project(
    conn: sqlite3.Connection,
    *,
    board_id: str,
    root_task_id: str,
    generation: int | None = None,
    now: datetime | str | None = None,
    retention_duration: timedelta | None = None,
    dry_run: bool = False,
    **kwargs: Any,
) -> CleanupResult:
    """Plan and optionally apply one project's retention cleanup."""
    plan = plan_project_cleanup(
        conn,
        board_id=board_id,
        root_task_id=root_task_id,
        generation=generation,
        now=now,
        retention_duration=retention_duration,
    )
    return apply_cleanup_plan(conn, plan, dry_run=dry_run, **kwargs)


# Explicit planner/applicator aliases.
plan_retention_cleanup = plan_project_cleanup
apply_retention_cleanup = apply_cleanup_plan

__all__ = [
    "TERMINAL_DELIVERY_STATES", "CleanupAction", "CleanupPlan", "CleanupResult",
    "plan_project_cleanup", "apply_cleanup_plan", "cleanup_project",
    "plan_retention_cleanup", "apply_retention_cleanup",
]
