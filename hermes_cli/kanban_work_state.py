"""work_state → Kanban run projection and authority-trail helpers.

CH-411 keeps Kanban as a shadow/run ledger while Linear remains the source of
truth.  This module therefore projects OMX/work_state outcomes into Kanban task
status plus ``task_runs.metadata``/outcome facts without dispatching executors,
mutating Linear, or writing generic task metadata.

CH-420 adds a narrow ingestion helper that persists those projections into
``task_runs`` as recoverable evidence.  The helper still refuses ambiguous task
correlation and never projects worker completion to Kanban ``done``/``closed``.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping, Optional

from hermes_cli import kanban_db as kb


KANBAN_RUN_METADATA_SCHEMA = "work_state_kanban_run_projection.v1"

_RECOVERABLE_WORK_STATES = frozenset({
    "blocked",
    "stale",
    "retry_needed",
    "handoff_needed",
    "continuation_needed",
    "worker_done_retry_needed",
    "failed",
})
# Keep task_runs.status/outcome aligned with hermes_cli.kanban_db schema
# comments. Distinct work_state outcomes stay in task_runs.metadata.work_state.
_KANBAN_RUN_STATUS_BY_WORK_STATE = {
    "blocked": "blocked",
    "stale": "blocked",
    "retry_needed": "blocked",
    "handoff_needed": "blocked",
    "continuation_needed": "blocked",
    "worker_done_retry_needed": "blocked",
    "failed": "failed",
}
_KANBAN_RUN_OUTCOME_BY_WORK_STATE = {
    "blocked": "blocked",
    "stale": "blocked",
    "retry_needed": "blocked",
    "handoff_needed": "blocked",
    "continuation_needed": "blocked",
    "worker_done_retry_needed": "blocked",
    "failed": "gave_up",
}
_RUNNING_WORK_STATES = frozenset({"active", "created", "running", "in_progress"})
_FINISHED_WORK_STATES = frozenset({
    "worker_done",
    "finished",
    "completed",
    "complete",
    "done",
    "succeeded",
    "success",
})
_FINISHED_USABLE_OUTCOMES = frozenset({
    "usable_output",
    "finished",
    "completed",
    "complete",
    "done",
    "succeeded",
    "success",
})


@dataclass(frozen=True)
class KanbanRunProjection:
    """Data-only projection suitable for Kanban dry-run/admission surfaces."""

    status: str
    reason: str
    task_status: str
    task_run_status: str
    task_run_outcome: Optional[str]
    task_run_summary: str
    task_runs_metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "reason": self.reason,
            "task_status": self.task_status,
            "task_run": {
                "status": self.task_run_status,
                "outcome": self.task_run_outcome,
                "summary": self.task_run_summary,
                "metadata": self.task_runs_metadata,
            },
            "side_effects": {
                "kanban_task_written": False,
                "executor_spawned": False,
                "linear_done_mutated": False,
                "kanban_done_projected_to_linear": False,
            },
        }


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _text(value: Any) -> Optional[str]:
    if value is None or isinstance(value, (dict, list, tuple, set)):
        return None
    text = str(value).strip()
    return text or None


def _attr(record: Any, name: str, default: Any = None) -> Any:
    if isinstance(record, Mapping):
        return record.get(name, default)
    return getattr(record, name, default)


def _norm(value: Any) -> Optional[str]:
    text = _text(value)
    return text.lower().replace("-", "_").replace(" ", "_").replace("/", "_") if text else None


def _resolution_failure_reason(resolution: Optional[Mapping[str, Any]]) -> Optional[str]:
    if not isinstance(resolution, Mapping):
        return None
    status = _norm(resolution.get("status"))
    matches = resolution.get("matches")
    matches_count = None
    if isinstance(matches, list):
        matches_count = len(matches)
    elif resolution.get("matches_count") is not None:
        try:
            matches_count = int(resolution.get("matches_count"))
        except (TypeError, ValueError):
            matches_count = None
    if status in {"ambiguous", "multiple_matches"} or (
        matches_count is not None and matches_count > 1
    ):
        return "ambiguous_recovery_metadata_fail_closed"
    if status in {"missing", "no_match", "not_found"} or matches_count == 0:
        return "missing_recovery_metadata_fail_closed"
    return None


def _recovery_fields(record: Any) -> dict[str, Optional[str]]:
    return {
        "work_id": _text(_attr(record, "work_id")),
        "owner_session_id": _text(_attr(record, "owner_session_id")),
        "executor_session_id": _text(_attr(record, "executor_session_id")),
        "tmux_session": _text(_attr(record, "tmux_session")),
        "repo_path": _text(_attr(record, "repo_path")),
        "worktree_path": _text(_attr(record, "worktree_path")),
        "next_action": _text(_attr(record, "next_action")),
        "proof": _text(_attr(record, "proof")),
    }


def _missing_recovery_fields(record: Any, *, require_execution_locator: bool) -> list[str]:
    recovery = _recovery_fields(record)
    missing = [
        name
        for name in ("work_id", "owner_session_id", "next_action", "proof")
        if not recovery.get(name)
    ]
    if require_execution_locator and not any(
        recovery.get(name)
        for name in ("executor_session_id", "tmux_session", "worktree_path", "repo_path")
    ):
        missing.append("executor_session_or_workspace_locator")
    return missing


def _base_metadata(record: Any, *, source: str, reason: str) -> dict[str, Any]:
    fields = {
        "work_id": _attr(record, "work_id"),
        "title": _attr(record, "title"),
        "objective": _attr(record, "objective"),
        "owner": _attr(record, "owner"),
        "executor": _attr(record, "executor"),
        "mode": _attr(record, "mode"),
        "state": _attr(record, "state"),
        "owner_session_id": _attr(record, "owner_session_id"),
        "executor_session_id": _attr(record, "executor_session_id"),
        "tmux_session": _attr(record, "tmux_session"),
        "repo_path": _attr(record, "repo_path"),
        "worktree_path": _attr(record, "worktree_path"),
        "task_branch": _attr(record, "task_branch"),
        "proof": _attr(record, "proof"),
        "usable_outcome": _attr(record, "usable_outcome"),
        "close_disposition": _attr(record, "close_disposition"),
        "blocked_reason": _attr(record, "blocked_reason"),
        "cleanup_required": _attr(record, "cleanup_required"),
        "cleanup_proof": _attr(record, "cleanup_proof"),
        "current_lane": _attr(record, "current_lane"),
        "planning_gate": _attr(record, "planning_gate"),
        "next_execution_branch": _attr(record, "next_execution_branch"),
        "close_authority": _attr(record, "close_authority"),
        "review_closeout": _attr(record, "review_closeout"),
        "reroute_recommendation": _attr(record, "reroute_recommendation"),
    }
    return {
        "schema": KANBAN_RUN_METADATA_SCHEMA,
        "source": source,
        "projection_reason": reason,
        "linear_is_ssot": True,
        "kanban_done_projection": "forbidden",
        "work_state": {key: value for key, value in fields.items() if value not in (None, "")},
    }


def _fail_closed_projection(
    record: Any,
    *,
    reason: str,
    source: str,
    missing: Optional[list[str]] = None,
) -> KanbanRunProjection:
    metadata = _base_metadata(record or {}, source=source, reason=reason)
    metadata["projection_failed_closed"] = True
    if missing:
        metadata["missing"] = missing
    summary = (
        "Work-state recovery metadata is missing or ambiguous; "
        "Kanban remains blocked pending operator review."
    )
    return KanbanRunProjection(
        status="fail_closed",
        reason=reason,
        task_status="blocked",
        task_run_status="blocked",
        task_run_outcome="blocked",
        task_run_summary=summary,
        task_runs_metadata=metadata,
    )


def project_work_state_to_kanban_run(
    record: Any,
    *,
    resolution: Optional[Mapping[str, Any]] = None,
    source: str = "work_state",
) -> dict[str, Any]:
    """Map one OMX/work_state record to Kanban task/run projection data.

    The mapping is intentionally fail-closed: missing records, ambiguous
    correlation, or missing recovery facts produce a blocked projection.  A
    finished/usable outcome records a completed run outcome but still leaves the
    Kanban task blocked because CH-411 forbids projecting Kanban Done while
    Linear is the source of truth.
    """

    if record is None:
        return _fail_closed_projection(
            {},
            reason="missing_work_state_record_fail_closed",
            source=source,
            missing=["record"],
        ).to_dict()

    resolution_reason = _resolution_failure_reason(resolution)
    if resolution_reason:
        return _fail_closed_projection(record, reason=resolution_reason, source=source).to_dict()

    state = _norm(_attr(record, "state"))
    usable_outcome = _norm(_attr(record, "usable_outcome"))
    close_disposition = _norm(_attr(record, "close_disposition"))
    owner = _norm(_attr(record, "owner"))
    executor = _norm(_attr(record, "executor"))
    mode = _norm(_attr(record, "mode"))
    require_execution_locator = (
        owner == "hermes" and executor in {"omx", "clawhip"} and mode == "delegated"
    )

    missing = _missing_recovery_fields(record, require_execution_locator=require_execution_locator)
    if missing:
        return _fail_closed_projection(
            record,
            reason="missing_recovery_metadata_fail_closed",
            source=source,
            missing=missing,
        ).to_dict()

    finished = state in _FINISHED_WORK_STATES or (
        close_disposition == "close" and usable_outcome in _FINISHED_USABLE_OUTCOMES
    )
    reason = "mapped_work_state_outcome"
    metadata = _base_metadata(record, source=source, reason=reason)
    metadata["recovery"] = _recovery_fields(record)
    metadata["projected_at"] = _utcnow_iso()

    next_action = _text(_attr(record, "next_action")) or "Operator review required."
    proof = _text(_attr(record, "proof")) or "work_state_projection"

    if finished:
        metadata["usable_output_recorded"] = True
        metadata["kanban_task_status_reason"] = "linear_ssot_no_kanban_done_projection"
        return KanbanRunProjection(
            status="mapped",
            reason=reason,
            task_status="blocked",
            task_run_status="done",
            task_run_outcome="completed",
            task_run_summary=(
                "Usable output recorded from work_state; "
                f"Linear remains SSOT. Proof: {proof}"
            ),
            task_runs_metadata=metadata,
        ).to_dict()

    if state in _RUNNING_WORK_STATES:
        return KanbanRunProjection(
            status="mapped",
            reason=reason,
            task_status="running",
            task_run_status="running",
            task_run_outcome=None,
            task_run_summary=f"Work is {state}; next action: {next_action}",
            task_runs_metadata=metadata,
        ).to_dict()

    if state in _RECOVERABLE_WORK_STATES:
        run_status = _KANBAN_RUN_STATUS_BY_WORK_STATE[state]
        outcome = _KANBAN_RUN_OUTCOME_BY_WORK_STATE[state]
        metadata["work_state_outcome_preserved_in_metadata"] = True
        return KanbanRunProjection(
            status="mapped",
            reason=reason,
            task_status="blocked",
            task_run_status=run_status,
            task_run_outcome=outcome,
            task_run_summary=f"Work-state outcome {state}; next action: {next_action}",
            task_runs_metadata=metadata,
        ).to_dict()

    return _fail_closed_projection(
        record,
        reason="unknown_work_state_outcome_fail_closed",
        source=source,
        missing=["recognized_state_or_usable_outcome"],
    ).to_dict()


def _task_ids_for_query(conn: Any, query: str, params: tuple[Any, ...]) -> set[str]:
    rows = conn.execute(query, params).fetchall()
    return {str(row["id"] if "id" in row.keys() else row["task_id"]) for row in rows}


def _active_task_ids_for_column(conn: Any, column: str, value: Optional[str]) -> set[str]:
    text = _text(value)
    if not text:
        return set()
    if column not in {"id", "public_id", "idempotency_key"}:
        raise ValueError(f"unsupported kanban task correlation column: {column}")
    return _task_ids_for_query(
        conn,
        f"SELECT id FROM tasks WHERE {column} = ? AND status != 'archived'",
        (text,),
    )


def _active_task_ids_for_alias(conn: Any, alias: Optional[str]) -> set[str]:
    text = _text(alias)
    if not text:
        return set()
    return _task_ids_for_query(
        conn,
        """
        SELECT a.task_id
          FROM task_aliases a
          JOIN tasks t ON t.id = a.task_id
         WHERE a.alias = ?
           AND t.status != 'archived'
        """,
        (text,),
    )


def _resolve_kanban_task_for_work_state(
    conn: Any,
    record: Any,
    *,
    task_id: Optional[str] = None,
    task_locator: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Resolve work_state evidence to exactly one active Kanban task.

    Correlation is intentionally conservative.  Explicit ``task_id`` or
    ``task_locator`` values are honored first.  Without an explicit locator,
    the work_state ``work_id`` may match a Kanban ``id``, ``public_id``,
    ``task_aliases.alias``, or the legacy/native idempotency keys used by
    admission helpers.  Multiple distinct matches fail closed rather than
    guessing which task should receive authority-trail evidence.
    """

    locator = dict(task_locator or {})
    if task_id is not None:
        locator["task_id"] = task_id

    matches: dict[str, set[str]] = {}

    if locator:
        locator_map = {
            "task_id": ("id", locator.get("task_id") or locator.get("id")),
            "public_id": ("public_id", locator.get("public_id")),
            "idempotency_key": ("idempotency_key", locator.get("idempotency_key")),
        }
        for reason, (column, value) in locator_map.items():
            task_ids = _active_task_ids_for_column(conn, column, _text(value))
            if task_ids:
                matches[reason] = task_ids
        alias_ids = _active_task_ids_for_alias(
            conn,
            locator.get("alias") or locator.get("work_id") or locator.get("legacy_ref"),
        )
        if alias_ids:
            matches["alias"] = alias_ids
    else:
        work_id = _text(_attr(record, "work_id"))
        if work_id:
            inferred = {
                "work_id_as_task_id": _active_task_ids_for_column(conn, "id", work_id),
                "work_id_as_public_id": _active_task_ids_for_column(conn, "public_id", work_id),
                "work_id_as_alias": _active_task_ids_for_alias(conn, work_id),
                "linear_idempotency_key": _active_task_ids_for_column(
                    conn,
                    "idempotency_key",
                    f"linear:{work_id}",
                ),
                "kanban_idempotency_key": _active_task_ids_for_column(
                    conn,
                    "idempotency_key",
                    f"kanban:{work_id}",
                ),
            }
            matches.update({reason: ids for reason, ids in inferred.items() if ids})

    all_matches = sorted({task_id for ids in matches.values() for task_id in ids})
    if not all_matches:
        return {
            "status": "missing",
            "reason": "missing_kanban_task_correlation_fail_closed",
            "matches": [],
        }
    if len(all_matches) > 1:
        return {
            "status": "ambiguous",
            "reason": "ambiguous_kanban_task_correlation_fail_closed",
            "matches": [
                {"task_id": matched, "matched_by": sorted(k for k, ids in matches.items() if matched in ids)}
                for matched in all_matches
            ],
        }
    selected = all_matches[0]
    return {
        "status": "single_match",
        "reason": "eligible",
        "task_id": selected,
        "matches": [
            {"task_id": selected, "matched_by": sorted(k for k, ids in matches.items() if selected in ids)}
        ],
    }


def _closed_kanban_task_fail_closed(task: Any) -> bool:
    return bool(task and _attr(task, "status") in {"done", "archived"})


def _insert_or_update_run_evidence(
    conn: Any,
    task_id: str,
    *,
    projection: Mapping[str, Any],
) -> int:
    run = projection["task_run"]
    run_status = str(run["status"])
    outcome = run.get("outcome")
    summary = run.get("summary")
    metadata = run.get("metadata") or {}
    metadata_json = json.dumps(metadata, ensure_ascii=False)
    now = int(time.time())
    task = kb.get_task(conn, task_id)
    profile = _attr(task, "assignee")
    step_key = _attr(task, "current_step_key")

    if run_status == "running":
        active = kb.active_run(conn, task_id)
        if active is not None:
            conn.execute(
                """
                UPDATE task_runs
                   SET status = 'running',
                       outcome = NULL,
                       summary = ?,
                       metadata = ?,
                       last_heartbeat_at = ?,
                       ended_at = NULL
                 WHERE id = ?
                """,
                (summary, metadata_json, now, active.id),
            )
            run_id = active.id
        else:
            cur = conn.execute(
                """
                INSERT INTO task_runs (
                    task_id, profile, step_key, status,
                    last_heartbeat_at, started_at, summary, metadata
                ) VALUES (?, ?, ?, 'running', ?, ?, ?, ?)
                """,
                (task_id, profile, step_key, now, now, summary, metadata_json),
            )
            run_id = int(cur.lastrowid or 0)
        conn.execute(
            """
            UPDATE tasks
               SET status = 'running',
                   started_at = COALESCE(started_at, ?),
                   current_run_id = ?,
                   claim_lock = NULL,
                   claim_expires = NULL,
                   worker_pid = NULL
             WHERE id = ?
               AND status != 'archived'
            """,
            (now, run_id, task_id),
        )
        return run_id

    active = kb.active_run(conn, task_id)
    if active is not None:
        conn.execute(
            """
            UPDATE task_runs
               SET status = ?,
                   outcome = ?,
                   summary = ?,
                   metadata = ?,
                   ended_at = ?,
                   claim_lock = NULL,
                   claim_expires = NULL,
                   worker_pid = NULL
             WHERE id = ?
            """,
            (run_status, outcome, summary, metadata_json, now, active.id),
        )
        run_id = active.id
    else:
        cur = conn.execute(
            """
            INSERT INTO task_runs (
                task_id, profile, step_key, status, outcome,
                summary, metadata, started_at, ended_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (task_id, profile, step_key, run_status, outcome, summary, metadata_json, now, now),
        )
        run_id = int(cur.lastrowid or 0)

    conn.execute(
        """
        UPDATE tasks
           SET status = ?,
               current_run_id = NULL,
               claim_lock = NULL,
               claim_expires = NULL,
               worker_pid = NULL
         WHERE id = ?
           AND status != 'archived'
        """,
        (projection["task_status"], task_id),
    )
    return run_id


def ingest_work_state_run_evidence(
    conn: Any,
    record: Any,
    *,
    task_id: Optional[str] = None,
    task_locator: Optional[Mapping[str, Any]] = None,
    resolution: Optional[Mapping[str, Any]] = None,
    source: str = "work_state_run_evidence",
) -> dict[str, Any]:
    """Persist one work_state evidence projection to Kanban ``task_runs``.

    This is an authority-trail helper, not a dispatcher or closeout controller:
    it never spawns executors, mutates Linear, writes generic task metadata, or
    projects worker completion to Kanban ``done``/``closed``.  Correlation must
    resolve to exactly one active Kanban task before any write occurs.
    """

    correlation = _resolve_kanban_task_for_work_state(
        conn,
        record,
        task_id=task_id,
        task_locator=task_locator,
    )
    if correlation["status"] != "single_match":
        return {
            "status": "fail_closed",
            "reason": correlation["reason"],
            "task_id": None,
            "correlation": correlation,
            "side_effects": {
                "kanban_task_written": False,
                "task_run_written": False,
                "executor_spawned": False,
                "linear_done_mutated": False,
                "kanban_done_projected_to_linear": False,
            },
        }

    selected_task_id = str(correlation["task_id"])
    task = kb.get_task(conn, selected_task_id)
    if _closed_kanban_task_fail_closed(task):
        return {
            "status": "fail_closed",
            "reason": "kanban_task_done_or_archived_fail_closed",
            "task_id": selected_task_id,
            "correlation": correlation,
            "side_effects": {
                "kanban_task_written": False,
                "task_run_written": False,
                "executor_spawned": False,
                "linear_done_mutated": False,
                "kanban_done_projected_to_linear": False,
            },
        }

    projection = project_work_state_to_kanban_run(record, resolution=resolution, source=source)
    with kb.write_txn(conn):
        run_id = _insert_or_update_run_evidence(conn, selected_task_id, projection=projection)
        kb._append_event(
            conn,
            selected_task_id,
            "work_state_evidence_ingested",
            {
                "source": source,
                "projection_status": projection["status"],
                "projection_reason": projection["reason"],
                "task_run_status": projection["task_run"]["status"],
                "task_run_outcome": projection["task_run"].get("outcome"),
                "work_state": _attr(record, "state"),
            },
            run_id=run_id,
        )

    return {
        "status": "ingested" if projection["status"] == "mapped" else "ingested_fail_closed",
        "reason": projection["reason"],
        "task_id": selected_task_id,
        "run_id": run_id,
        "correlation": correlation,
        "projection": projection,
        "side_effects": {
            "kanban_task_written": True,
            "task_run_written": True,
            "executor_spawned": False,
            "linear_done_mutated": False,
            "kanban_done_projected_to_linear": False,
        },
    }
