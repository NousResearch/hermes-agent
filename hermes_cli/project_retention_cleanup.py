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
    record_cleanup_journal,
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
    if not isinstance(retention_duration, timedelta):
        raise ValueError("retention_duration is required when cleanup_after is absent")
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


def _journal_for_plan(conn: sqlite3.Connection, plan_sha256: str) -> Any:
    """Return an existing applied journal row for an exact plan, if present."""
    try:
        return conn.execute(
            "SELECT * FROM project_cleanup_journal WHERE plan_sha256 = ? "
            "AND status = 'applied' ORDER BY id LIMIT 1",
            (plan_sha256,),
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
    if not plan.eligible:
        raise ValueError("cleanup plan is not eligible: " + "; ".join(plan.refusal_reasons))
    if plan.already_applied:
        return CleanupResult(plan=plan, dry_run=False)
    existing_journal = _journal_for_plan(conn, plan.plan_sha256)
    if existing_journal is not None:
        return CleanupResult(plan=plan, dry_run=False, journal=existing_journal)
    if plan.actions:
        current_tasks = [kanban_db.get_task(conn, action.task_id) for action in plan.actions]
        if any(task is None for task in current_tasks):
            raise ValueError("cleanup plan member disappeared before apply")
        current_statuses = [_value(task, "status") for task in current_tasks]
        if current_statuses and all(status == "archived" for status in current_statuses):
            return CleanupResult(plan=plan, dry_run=False)
    archive = archive_action or kanban_db.archive_task
    applied: list[str] = []
    for action in plan.actions:
        if action.action != "archive":
            raise ValueError(f"unsupported cleanup action: {action.action!r}")
        if archive(conn, action.task_id):
            applied.append(action.task_id)
    writer = journal_writer or record_cleanup_journal
    journal = writer(
        conn,
        board_id=plan.board_id,
        root_task_id=plan.root_task_id,
        generation=plan.generation,
        plan_sha256=plan.plan_sha256,
        mode="apply",
        status="applied",
        retention_cutoff=(
            int(datetime.fromisoformat(plan.retention_cutoff).timestamp())
            if plan.retention_cutoff else None
        ),
        eligible_task_count=len(plan.eligible_task_ids),
        excluded_task_count=len(plan.excluded_task_ids),
        deleted_task_count=0,
        archived_task_count=len(applied),
        evidence_path=plan.evidence_paths[0] if plan.evidence_paths else None,
    )
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
